"""Shared model, optimizer, evaluation, and provenance utilities."""
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random

import torch
from transformers import AutoModelForCausalLM

from .hf_dust import native_losses, select_parameters
from .sft_data import collate


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def atomic_save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    names = ['common/flops.py', *[f'posttraining/{name}.py' for name in
             ['configuration', 'allocation', 'sft_data', 'hf_dust', 'runtime', 'sft', 'calibrate']]]
    names += [str(path.relative_to(root)) for path in sorted((root/'posttraining/geometry').glob('*.py'))
              if not path.name.startswith('verify')]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names if (root / name).exists()}


def artifact_identity(config, manifest, model):
    local = Path(config.model.name_or_path)
    weights = {str(path.name): hashlib.sha256(path.read_bytes()).hexdigest()
               for pattern in ('*.safetensors', 'pytorch_model*.bin')
               for path in local.glob(pattern)} if local.is_dir() else {}
    return dict(model=asdict(config.model), model_commit=getattr(model.config, '_commit_hash', None),
                local_weights=weights, data_sha256=manifest['data_sha256'],
                layers=config.training.layers, train_embedding=config.training.train_embedding,
                train_head=config.training.train_head, precision=config.training.precision,
                allow_bf16_reduced_precision_reduction=config.training.allow_bf16_reduced_precision_reduction,
                torch_version=str(torch.__version__), sources=source_hashes())


def validate_scales(config, manifest, model):
    path = config.dust.noise.scales_file
    if path:
        artifact = json.loads(Path(path).read_text())
        if artifact.get('identity') != artifact_identity(config, manifest, model):
            raise ValueError('Calibration scales do not match model, data, precision, source, or parameter selection')
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return None


def load_model(config, method):
    device = torch.device(config.training.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = config.training.allow_bf16_reduced_precision_reduction
    torch.manual_seed(config.training.seed)
    model = AutoModelForCausalLM.from_pretrained(config.model.name_or_path, revision=config.model.revision,
        dtype=torch.float32, attn_implementation=config.model.attn_implementation,
        trust_remote_code=False, low_cpu_mem_usage=True).to(device)
    model.config.use_cache = False
    model.requires_grad_(False)
    selected = select_parameters(model, config.training)
    if not selected:
        raise ValueError('No matrix parameters selected for training')
    for parameter in selected.values():
        parameter.requires_grad_(method == 'backprop')
    model.eval()
    return model, selected


def make_optimizer(model, selected, config):
    settings = config.optimizer
    embedding, head = model.get_input_embeddings().weight, model.get_output_embeddings().weight
    if embedding is head and settings.embedding_lr_multiplier != settings.head_lr_multiplier:
        raise ValueError('Tied embedding/head must have one shared learning-rate multiplier')
    groups = {}
    qkv_count = 0
    for name, parameter in selected.items():
        multiplier = settings.embedding_lr_multiplier if parameter is embedding else settings.head_lr_multiplier if parameter is head else 1.
        if name.endswith(('self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight')):
            multiplier = settings.qkv_lr_multiplier
            qkv_count += 1
        group = groups.setdefault(multiplier, dict(params=[], names=[], lr=settings.lr * multiplier,
                                                   initial_lr=settings.lr * multiplier))
        group['params'].append(parameter)
        group['names'].append(name)
    if settings.qkv_lr_multiplier != 1. and not qkv_count:
        raise ValueError('A nondefault Q/K/V multiplier requires selected native q_proj/k_proj/v_proj weights')
    kwargs = dict(lr=settings.lr, weight_decay=settings.weight_decay)
    if settings.name == 'sgd':
        return torch.optim.SGD(list(groups.values()), momentum=settings.momentum,
                               nesterov=settings.nesterov, dampening=settings.dampening, **kwargs)
    return torch.optim.AdamW(list(groups.values()), betas=tuple(settings.betas), eps=settings.eps, **kwargs)


def set_learning_rate(optimizer, settings, step, total_steps):
    if settings.warmup_steps and step <= settings.warmup_steps:
        factor = step / settings.warmup_steps
    else:
        fraction = max(0., (step - settings.warmup_steps - 1) / max(1, total_steps - settings.warmup_steps))
        factor = 1. if settings.schedule == 'constant' else 1. - fraction if settings.schedule == 'linear' else .5 * (1. + math.cos(math.pi * fraction))
    for group in optimizer.param_groups:
        group['lr'] = group['initial_lr'] * factor


def update_plan(count, settings):
    batches = []
    size = settings.batch_size * settings.gradient_accumulation_steps
    for epoch in range(settings.epochs):
        order = list(range(count))
        random.Random(settings.seed + epoch).shuffle(order)
        for start in range(0, count, size):
            indices = order[start:start+size]
            if settings.drop_last and len(indices) != size:
                continue
            batches.append([indices[i:i+settings.batch_size] for i in range(0, len(indices), settings.batch_size)])
    if settings.max_steps is not None:
        batches = batches[:settings.max_steps]
    if not batches:
        raise ValueError('Not enough examples for one configured update')
    return batches


def to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def batch_examples(rows, indices, config, manifest):
    return collate([rows[i] for i in indices], manifest['pad_token_id'], multiple=config.data.pad_to_multiple_of)


@torch.no_grad()
def evaluate(model, rows, config, manifest):
    model.eval()
    loss_sum = 0.
    count = 0
    device = next(model.parameters()).device
    for start in range(0, len(rows), config.training.eval_batch_size):
        batch = batch_examples(rows, range(start, min(len(rows), start+config.training.eval_batch_size)), config, manifest)
        batch = to_device(batch, device)
        loss = native_losses(model, batch, config.training.precision)
        if not torch.isfinite(loss).all():
            raise FloatingPointError('Non-finite evaluation loss')
        loss_sum += loss.sum().item()
        count += batch['labels'].ne(-100).sum().item()
    if not count:
        raise ValueError('No supervised validation/test tokens')
    return dict(loss=loss_sum / count, supervised_tokens=int(count), examples=len(rows))


def optimizer_step(optimizer, selected, max_norm=None, *, return_details=False, record_families=False):
    if record_families and not return_details:
        raise ValueError('Family diagnostics require return_details=True')
    norm_squared = torch.zeros((), device=next(iter(selected.values())).device)
    gradient_squares = {}
    for name, parameter in selected.items():
        if parameter.grad is None:
            raise AssertionError(f'Missing gradient for selected parameter {name}')
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f'Non-finite gradient for {name}')
        squared = parameter.grad.float().square().sum()
        norm_squared += squared
        if record_families:
            gradient_squares[name] = squared
    norm = norm_squared.sqrt()
    if not torch.isfinite(norm):
        raise FloatingPointError('Non-finite gradient norm')
    if max_norm is not None:
        scale = (max_norm / (norm + 1e-6)).clamp(max=1.)
        for parameter in selected.values():
            parameter.grad.mul_(scale)
    before = {name:p.detach().clone() for name,p in selected.items()} if return_details else None
    optimizer.step()
    if any(not torch.isfinite(parameter).all() for parameter in selected.values()):
        raise FloatingPointError('Optimizer produced non-finite parameters')
    if return_details:
        update_squared = torch.zeros((), device=norm.device)
        families = {}
        rates = {name:group['lr'] for group in optimizer.param_groups for name in group['names']} if record_families else {}
        for name, parameter in selected.items():
            squared = (parameter.detach()-before[name]).square().sum()
            update_squared += squared
            if record_families:
                family = 'shared_embedding_head' if name=='model.embed_tokens.weight' else name.split('.')[-2]
                row = families.setdefault(family,dict(gradient_squared=torch.zeros_like(norm),update_squared=torch.zeros_like(norm),
                    weight_squared=torch.zeros_like(norm),momentum_squared=torch.zeros_like(norm),momentum_present=False,
                    parameters=0,learning_rates=set()))
                row['gradient_squared'] += gradient_squares[name]
                row['update_squared'] += squared
                row['weight_squared'] += before[name].square().sum()
                buffer = optimizer.state.get(parameter,{}).get('momentum_buffer')
                if buffer is not None:
                    row['momentum_squared'] += buffer.square().sum()
                    row['momentum_present'] = True
                row['parameters'] += parameter.numel()
                row['learning_rates'].add(rates[name])
        result = dict(gradient_l2=norm.item(), parameter_update_l2=update_squared.sqrt().item())
        if record_families:
            result['parameter_families'] = {name:dict(gradient_l2=row['gradient_squared'].sqrt().item(),
                parameter_update_l2=row['update_squared'].sqrt().item(),parameter_l2=row['weight_squared'].sqrt().item(),
                update_to_weight_norm=(row['update_squared']/row['weight_squared'].clamp_min(1e-30)).sqrt().item(),
                momentum_buffer_l2=row['momentum_squared'].sqrt().item() if row['momentum_present'] else None,
                parameters=row['parameters'],learning_rates=sorted(row['learning_rates'])) for name,row in families.items()}
        return result
    return norm.item()
