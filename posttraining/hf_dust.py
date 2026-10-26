"""DUST on native Hugging Face Llama/Mistral/Qwen2 causal language models.

The clean model, RoPE, attention implementation, SwiGLU, learned normalizations,
and tied weights are preserved. Only matrix weights are updated. This adapter
supports dense token losses, padding, separate gate/up populations, embedding
credit, and a stable uncapped-logit head estimator.
"""
from contextlib import nullcontext
from dataclasses import dataclass
import importlib
import json
import math
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from .allocation import resolve_allocation


SUPPORTED_MODELS = ('llama', 'mistral', 'qwen2', 'qwen3')


def autocast(device, precision):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' and precision == 'bf16' else nullcontext()


def token_losses(logits, labels):
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1),
                           ignore_index=-100, reduction='none').view_as(labels)


def native_losses(model, batch, precision='bf16'):
    with autocast(next(model.parameters()).device, precision):
        logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                       use_cache=False).logits
        return token_losses(logits, batch['labels'])


def average_ranks(tensor):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(dist.get_world_size())
    return tensor


class ModelLayout:
    def __init__(self, model, training):
        if model.config.model_type not in SUPPORTED_MODELS:
            raise ValueError(f'DUST adapter supports {SUPPORTED_MODELS}; {model.config.model_type!r} needs an architecture adapter')
        if model.config._attn_implementation != 'sdpa':
            raise ValueError('The verified DUST adapter requires attn_implementation="sdpa"')
        if getattr(model.config, 'attention_dropout', 0.) != 0.:
            raise ValueError('DUST requires zero attention dropout; stochastic dropout would change the reward estimator')
        self.model = model
        self.n_layers = len(model.model.layers)
        self.layers = list(range(self.n_layers)) if training.layers == 'all' else sorted(training.layers)
        if not self.layers or len(set(self.layers)) != len(self.layers) or min(self.layers) < 0 or max(self.layers) >= self.n_layers:
            raise ValueError('Invalid selected transformer layers')
        self.embedding = model.get_input_embeddings()
        self.head = model.get_output_embeddings()
        if not isinstance(self.embedding, nn.Embedding) or not isinstance(self.head, nn.Linear):
            raise ValueError('DUST requires ordinary, unquantized Embedding and Linear modules')
        self.tied = self.embedding.weight is self.head.weight
        if self.tied and training.train_embedding != training.train_head:
            raise ValueError('Tied embedding/head must be trained or frozen together')
        self.modules = {}
        self.families = {}
        for i in self.layers:
            block = model.model.layers[i]
            for suffix, family in [('self_attn.o_proj', 'writers'), ('mlp.down_proj', 'writers'),
                                   ('mlp.gate_proj', 'gate'), ('mlp.up_proj', 'up'),
                                   ('self_attn.q_proj', 'q'), ('self_attn.k_proj', 'k'), ('self_attn.v_proj', 'v')]:
                name = f'model.layers.{i}.{suffix}'
                module = model.get_submodule(name)
                if not isinstance(module, nn.Linear):
                    raise ValueError(f'{name} must be an unquantized Linear')
                self.modules[name] = module
                self.families[name] = family
            if model.config.model_type != 'qwen3' and (hasattr(block.self_attn, 'q_norm') or hasattr(block.self_attn, 'k_norm')):
                raise ValueError('Unsupported additional normalization')
        self.modules['model.embed_tokens'] = self.embedding
        self.modules['lm_head'] = self.head
        self.families.update({'model.embed_tokens': 'embedding', 'lm_head': 'head'})
        self.selected_sites = [s for s in self.modules if
                               (s != 'model.embed_tokens' or training.train_embedding) and
                               (s != 'lm_head' or training.train_head)]
        self.names = {id(p): name for name, p in model.named_parameters()}
        self.selected = {self.names[id(self.modules[s].weight)]: self.modules[s].weight for s in self.selected_sites}
        if any(p.dtype != torch.float32 for p in self.selected.values()):
            raise ValueError('Load FP32 master weights; BF16 is used only under forward autocast')
        self.train_embedding, self.train_head = training.train_embedding, training.train_head

    def canonical(self, site):
        return self.names[id(self.modules[site].weight)]


def select_parameters(model, training):
    """Use exactly the same parameter selection for DUST and its matched baseline."""
    if model.config.model_type in SUPPORTED_MODELS:
        layout = ModelLayout(model, training)
        return layout.selected
    if training.layers != 'all':
        raise ValueError('Partial-layer selection requires a supported architecture adapter')
    selected = {name: p for name, p in model.named_parameters() if p.ndim >= 2}
    embedding, head = model.get_input_embeddings().weight, model.get_output_embeddings().weight
    if embedding is head and training.train_embedding != training.train_head:
        raise ValueError('Tied embedding/head must be trained or frozen together')
    frozen_ids = ({id(embedding)} if not training.train_embedding else set()) | ({id(head)} if not training.train_head else set())
    return {name: p for name, p in selected.items() if id(p) not in frozen_ids}


def center_head_reward(reward, valid, mode):
    """Paper batch centering, with the loss mask reapplied after subtraction."""
    reward = reward * valid.unsqueeze(0)
    if mode == 'batch_reapply_mask' and reward.shape[1] > 1:
        reward = reward - reward.mean(1, keepdim=True)
    return reward * valid.unsqueeze(0)


@dataclass
class Estimate:
    loss_sum: float
    supervised_tokens: int
    gradients: dict[str, torch.Tensor]
    components: dict[str, torch.Tensor] | None = None
    probe_counts: dict | None = None


class HFDUST:
    def __init__(self, model, training, settings, seed=42):
        self.model = model.eval().requires_grad_(False)
        self.training, self.settings = training, settings
        self.layout = ModelLayout(model, training)
        self.layers = self.layout.layers
        self.device = next(model.parameters()).device
        self.dtype = torch.bfloat16 if self.device.type == 'cuda' and training.precision == 'bf16' else torch.float32
        self.world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        self.rank = dist.get_rank() if self.world > 1 else 0
        self.allocation = resolve_allocation(settings, self.layout.n_layers, self.layers, world=self.world,
                                            train_embedding=training.train_embedding, train_head=training.train_head)
        self.generator = torch.Generator(self.device).manual_seed(seed + 3 + 7777 * self.rank)
        self.inputs, self.outputs, self.block_inputs, self.block_kwargs = {}, {}, {}, {}
        self.jitter = {}
        self.direct_sampler = None
        self.probe_counts = {}
        self.capturing = False
        self.site_sigmas = dict(settings.noise.site_sigmas)
        if settings.noise.scales_file:
            saved = json.loads(Path(settings.noise.scales_file).read_text())
            self.site_sigmas = {**saved['site_sigmas'], **self.site_sigmas}
        allowed_sites = set(self.layout.modules) | {f'o@{i}' for i in self.layers}
        if set(self.site_sigmas) - allowed_sites or any(not math.isfinite(v) or v <= 0 for v in self.site_sigmas.values()):
            raise ValueError('Noise overrides must name supported sites and contain finite positive scales')
        self.hooks = [module.register_forward_hook(self.output_hook(name)) for name, module in self.layout.modules.items()]
        for i, block in enumerate(model.model.layers):
            self.hooks.append(block.register_forward_pre_hook(self.block_hook(i), with_kwargs=True))
            if i in self.layers:
                self.hooks.append(block.self_attn.o_proj.register_forward_pre_hook(self.hub_hook(i)))

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.inputs.clear(); self.outputs.clear(); self.block_inputs.clear(); self.block_kwargs.clear()
        self.last_attention_targets = {}

    def output_hook(self, name):
        def hook(module, args, output):
            if self.capturing:
                self.inputs[name] = args[0].detach().float() if args[0].is_floating_point() else args[0].detach()
                self.outputs[name] = output.detach()
            if name in self.jitter:
                perturbed = output + self.jitter[name].to(output.dtype)
                if self.direct_sampler is not None:
                    self.direct_sampler.observe_injection(name, output, perturbed)
                return perturbed
        return hook

    def hub_hook(self, layer):
        name = f'o@{layer}'
        def hook(module, args):
            if self.capturing:
                self.outputs[name] = args[0].detach()
            if name in self.jitter:
                return (args[0] + self.jitter[name].to(args[0].dtype),)
        return hook

    def block_hook(self, layer):
        def hook(module, args, kwargs):
            if self.capturing:
                if layer in self.layers:
                    self.block_inputs[layer] = args[0].detach()
                self.block_kwargs[layer] = kwargs.copy()
        return hook

    def repeat_kwargs(self, kwargs, copies):
        def repeat(value):
            if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[0] == self.batch:
                return value.repeat(copies, *([1] * (value.ndim - 1)))
            if isinstance(value, tuple):
                return tuple(repeat(v) for v in value)
            return value
        return {key: repeat(value) for key, value in kwargs.items()}

    def noise(self, *shape):
        return torch.randn(*shape, device=self.device, dtype=torch.float32, generator=self.generator)

    def sample_direct_noise(self, site, shape, clean_output, sigma):
        if self.direct_sampler is None:
            return self.noise(*shape)
        return self.direct_sampler.sample(site, shape, clean_output, sigma)

    def sigma(self, site):
        if site in self.site_sigmas:
            return self.site_sigmas[site]
        r = self.settings.noise
        if site == 'lm_head':
            return r.head_sigma
        for kind in ('q', 'k', 'v'):
            if site.endswith(f'self_attn.{kind}_proj'):
                return getattr(r, f'{kind}_sigma')
        if r.mode == 'activation_rms':
            raise ValueError(f'Missing frozen activation-RMS scale for {site}; run the scales calibration stage first')
        if site == 'model.embed_tokens':
            return r.embedding_sigma
        if site.startswith('o@'):
            return r.hub_sigma * (r.hub_deeper_multiplier if int(site[2:]) >= self.layout.n_layers // 2 else 1.)
        return r.direct_sigma

    def losses(self, x, y, *, force_full=False):
        copies = x.shape[0] // self.batch
        if x.shape[0] != copies * self.batch:
            raise ValueError('Perturbed batches must contain whole copies of the clean batch')
        with autocast(self.device, self.training.precision):
            if self.jitter and self.settings.cache_prefix and not force_full and 'model.embed_tokens' not in self.jitter:
                first = min(int(s[2:]) if s.startswith('o@') else int(s.split('.')[2]) for s in self.jitter)
                hidden = self.block_inputs[first].repeat(copies, 1, 1)
                for i in range(first, self.layout.n_layers):
                    hidden = self.model.model.layers[i](hidden, **self.repeat_kwargs(self.block_kwargs[i], copies))
                    if isinstance(hidden, tuple):
                        hidden = hidden[0]
                logits = self.model.lm_head(self.model.model.norm(hidden))
            else:
                logits = self.model(input_ids=x, attention_mask=self.mask.repeat(copies, 1), use_cache=False).logits
            return token_losses(logits, y)

    @torch.no_grad()
    def capture(self, batch):
        self.x, self.y, self.mask = batch['input_ids'], batch['labels'], batch['attention_mask']
        self.batch, self.length = self.x.shape
        self.valid = self.y.ne(-100)
        if not self.valid.any() or (self.valid & ~self.mask.bool()).any():
            raise ValueError('Supervised labels must be nonempty and fall on valid input positions')
        self.inputs.clear(); self.outputs.clear(); self.block_inputs.clear(); self.block_kwargs.clear()
        self.last_attention_targets = {}
        self.capturing = True
        try:
            self.clean_loss = self.losses(self.x, self.y)
        finally:
            self.capturing = False

    def direct_errors(self, sites, total_draws):
        n, b, t = self.settings.nrep, self.batch, self.length
        draws = total_draws // self.world
        errors = {s: torch.zeros_like(self.outputs[s], dtype=torch.float32) for s in sites}
        for _ in range(draws // n):
            noise = {s: self.sample_direct_noise(s, (n, b, t, self.outputs[s].shape[-1]), self.outputs[s], self.sigma(s))
                     * self.mask[None, :, :, None] for s in sites}
            self.jitter = {s: (self.sigma(s) * a).reshape(n*b, t, -1) for s, a in noise.items()}
            try:
                loss = self.losses(self.x.repeat(n, 1), self.y.repeat(n, 1)).view(n, b, t)
            finally:
                self.jitter = {}
            reward = self.clean_loss.unsqueeze(0) - loss
            raw_reward = reward
            if n > 1:
                reward = reward - reward.mean(0, keepdim=True)
            reward = reward * self.valid.unsqueeze(0)
            self.probe_counts['direct_draws'] = self.probe_counts.get('direct_draws', 0) + n
            self.probe_counts['perturbed_model_batches'] = self.probe_counts.get('perturbed_model_batches', 0) + 1
            self.probe_counts['perturbed_example_forwards'] = self.probe_counts.get('perturbed_example_forwards', 0) + n*b
            for site, a in noise.items():
                if self.direct_sampler is not None:
                    self.direct_sampler.observe_reward(site, reward, raw_reward)
                score = self.direct_sampler.score(site, a) if self.direct_sampler is not None and hasattr(self.direct_sampler, 'score') else a
                errors[site] -= torch.einsum('nbt,nbtd->btd', reward, score) / self.sigma(site)
        return {site: error / draws for site, error in errors.items()}

    def reassemble(self, site, error):
        module, x = self.layout.modules[site], self.inputs[site]
        if isinstance(module, nn.Embedding):
            return torch.zeros_like(module.weight).index_add_(0, x.reshape(-1), error.reshape(-1, error.shape[-1]))
        return torch.einsum('btd,bti->di', error, x)

    def attention_state(self, layer):
        attn = self.model.model.layers[layer].self_attn
        h, kv, d = self.model.config.num_attention_heads, self.model.config.num_key_value_heads, attn.head_dim
        prefix = f'model.layers.{layer}.self_attn.'
        q, k, v = [self.outputs[prefix + name + '_proj'].float().view(self.batch, self.length, heads, d)
                   for name, heads in [('q', h), ('k', kv), ('v', kv)]]
        cos, sin = self.block_kwargs[layer]['position_embeddings']
        rotary = importlib.import_module(type(attn).__module__).apply_rotary_pos_emb

        def process(qraw, kraw):
            copies = qraw.shape[0] // self.batch
            cq, sk = cos, sin
            if cos.shape[0] == self.batch:
                cq, sk = cos.repeat(copies, 1, 1), sin.repeat(copies, 1, 1)
            if hasattr(attn, 'q_norm'):
                qraw = attn.q_norm(qraw.to(self.outputs[prefix + "q_proj"].dtype))
                kraw = attn.k_norm(kraw.to(self.outputs[prefix + "k_proj"].dtype))
            qr, kr = rotary(qraw.transpose(1, 2), kraw.transpose(1, 2), cq, sk)
            return qr.to(self.dtype), kr.to(self.dtype)

        def attend(qr, kr, values):
            copies = qr.shape[0] // self.batch
            mask = self.repeat_kwargs(self.block_kwargs[layer], copies).get('attention_mask')
            with autocast(self.device, self.training.precision):
                output, _ = sdpa_attention_forward(attn, qr, kr, values.transpose(1, 2).to(self.dtype), mask,
                    dropout=0., scaling=attn.scaling, sliding_window=getattr(attn, 'sliding_window', None))
            return output.contiguous()

        qp, kp = process(q, k)
        return q, k, v, qp, kp, process, attend

    def credit(self, layer):
        positions = torch.arange(self.length, device=self.device)
        lag = positions[None, :] - positions[:, None]  # [perturbed t, scored s]
        allowed = (lag >= 0).unsqueeze(0) & self.mask[:, :, None].bool() & self.mask[:, None, :].bool()
        native_mask = self.block_kwargs[layer].get('attention_mask')
        if native_mask is not None:
            native_allowed = native_mask[:, 0] if native_mask.ndim == 4 else native_mask
            native_allowed = native_allowed if native_allowed.dtype == torch.bool else native_allowed == 0
            allowed = allowed & native_allowed.transpose(-1, -2)
        return allowed.float() * self.settings.gamma ** lag.clamp(min=0)

    def local_errors(self, layer, target, kinds=('q', 'k', 'v')):
        n, b, t = self.settings.nrep, self.batch, self.length
        attn = self.model.model.layers[layer].self_attn
        h, kv, d = self.model.config.num_attention_heads, self.model.config.num_key_value_heads, attn.head_dim
        q, k, v, qp, kp, process, attend = self.attention_state(layer)
        baseline = attend(qp, kp, v).float()
        credit = self.credit(layer)
        repeat = lambda value: value.repeat(n, *([1] * (value.ndim - 1)))
        errors = {}
        for kind in kinds:
            heads = h if kind == 'q' else kv
            site = f'model.layers.{layer}.self_attn.{kind}_proj'
            sigma = self.sigma(site)
            draws = self.allocation['local'][kind][layer] // self.world
            self.probe_counts['local_draws'] = self.probe_counts.get('local_draws', 0) + draws
            error = torch.zeros(b, t, heads, d, device=self.device)
            for _ in range(draws // n):
                a = self.noise(n, b, t, heads, d) * self.mask[None, :, :, None, None]
                if kind == 'v':
                    delta = attend(repeat(qp), repeat(kp), (sigma * a).reshape(n*b, t, kv, d)).float()
                else:
                    qr, kr = repeat(q), repeat(k)
                    if kind == 'q':
                        qr = qr + sigma * a.reshape(n*b, t, h, d)
                    else:
                        kr = kr + sigma * a.reshape(n*b, t, kv, d)
                    qa, ka = process(qr, kr)
                    delta = attend(qa, ka, repeat(v)).float().view(n, b, t, h, d) - baseline
                delta = delta.view(n, b, t, h, d)
                score = torch.einsum('bthd,nbthd->nbth', target.view(b, t, h, d), delta)
                if kind != 'q':
                    score = score.view(n, b, t, kv, h // kv).sum(-1)
                    score = torch.einsum('nbsh,bts->nbth', score, credit)
                error += torch.einsum('nbth,nbthd->bthd', score, a) / sigma
            errors[site] = error.flatten(-2) / draws
        return errors

    def head_slab_losses(self, lo, changed, other_lse, clean_target):
        """Exact uncapped-logit CE when only one vocabulary slab changes."""
        n, b, t, width = changed.shape
        targets = self.y.clamp_min(0)
        in_slab = (targets >= lo) & (targets < lo + width)
        chosen = changed.gather(-1, (targets - lo).clamp(0, width - 1)[None, :, :, None].expand(n, -1, -1, -1)).squeeze(-1)
        target = torch.where(in_slab[None], chosen, clean_target[None])
        partition = torch.logaddexp(other_lse[None], torch.logsumexp(changed, -1))
        return (partition - target) * self.valid[None]

    def head_gradient(self):
        n, b, t = self.settings.nrep, self.batch, self.length
        slab = self.settings.head_slab_size
        raw = self.outputs['lm_head'].float()
        chunks = list(raw.split(slab, dim=-1))
        slab_lse = torch.stack([torch.logsumexp(chunk, -1) for chunk in chunks], -1)
        clean_target = raw.gather(-1, self.y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        gradient = torch.zeros_like(self.layout.head.weight)
        draws = self.settings.head_draws_per_slab // self.world
        self.probe_counts['head_draws'] = draws * len(chunks)
        sigma = self.sigma('lm_head')
        for block, chunk in enumerate(chunks):
            lo, width = block * slab, chunk.shape[-1]
            others = torch.cat([slab_lse[..., :block], slab_lse[..., block+1:]], -1)
            other_lse = torch.logsumexp(others, -1) if others.shape[-1] else torch.full_like(clean_target, -torch.inf)
            error = torch.zeros_like(chunk)
            for _ in range(draws // n):
                a = self.noise(n, b, t, width)
                changed = chunk[None] + sigma * a
                loss = self.head_slab_losses(lo, changed, other_lse, clean_target)
                reward = center_head_reward(self.clean_loss[None] - loss, self.valid, self.settings.head_centering)
                error -= torch.einsum('nbt,nbtd->btd', reward / sigma, a)
            gradient[lo:lo+width] = torch.einsum('btd,bti->di', error / draws, self.inputs['lm_head'])
        return gradient

    @torch.no_grad()
    def estimate_gradients(self, batch, *, families=None, keep_components=False):
        self.probe_counts = {}
        self.capture(batch)
        all_families = {'writers', 'gate', 'up', 'attention', 'q', 'k', 'v', 'embedding', 'head'}
        wanted = all_families if families is None else set(families)
        if not wanted <= all_families:
            raise ValueError(f'Unknown estimator families: {wanted - all_families}')
        components = {}
        targets = {}
        for layer in self.layers:
            prefix = f'model.layers.{layer}.'
            for family, sites in [('writers', [prefix+'self_attn.o_proj', prefix+'mlp.down_proj']),
                                  ('gate', [prefix+'mlp.gate_proj']), ('up', [prefix+'mlp.up_proj'])]:
                if family in wanted:
                    for site, error in self.direct_errors(sites, self.allocation[family][layer]).items():
                        components[site] = self.reassemble(site, error)
            if wanted & {'attention', 'q', 'k', 'v'}:
                hub = f'o@{layer}'
                target = average_ranks(self.direct_errors([hub], self.allocation['attention'][layer])[hub])
                targets[hub] = target
                kinds = tuple(kind for kind in ('q', 'k', 'v') if kind in wanted)
                if kinds:
                    for site, error in self.local_errors(layer, target, kinds).items():
                        components[site] = self.reassemble(site, error)
        if self.layout.train_embedding and 'embedding' in wanted:
            site = 'model.embed_tokens'
            components[site] = self.reassemble(site, self.direct_errors([site], self.allocation['embedding'])[site])
        if self.layout.train_head and 'head' in wanted:
            components['lm_head'] = self.head_gradient()
        valid_tokens = int(self.valid.sum().item())
        gradients = {}
        for site, value in components.items():
            value = average_ranks(value) / valid_tokens
            components[site] = value
            name = self.layout.canonical(site)
            if name in gradients:
                gradients[name] = gradients[name] + value  # shared embedding/head contributions
            else:
                gradients[name] = value
        if families is None and gradients.keys() != self.layout.selected.keys():
            raise AssertionError('DUST must estimate every selected matrix exactly once after tying contributions')
        self.last_attention_targets = {name: value / valid_tokens for name, value in targets.items()}
        return Estimate(self.clean_loss.sum().item(), valid_tokens, gradients, components if keep_components else None,
                        dict(self.probe_counts))

    @torch.no_grad()
    def activation_statistics(self, batch):
        self.capture(batch)
        result = {}
        for site in self.layout.selected_sites + [f'o@{i}' for i in self.layers]:
            x = self.outputs[site].float()
            valid = self.mask.bool().unsqueeze(-1).expand_as(x)
            result[site] = dict(sum_squares=x.masked_select(valid).square().sum().item(), count=int(valid.sum().item()))
        return result
