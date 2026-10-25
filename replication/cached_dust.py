"""DUST with the cached-prefix execution described in paper section 2.3.

The estimator, random draws, centering, allocation and optimizer are inherited
unchanged from the release. Only direct-loss forward execution is shortened.
"""
import torch
from torch.nn import functional as F

from .dust import DUST
from .model import norm


class CachedDUST(DUST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_inputs = {}
        for i, block in enumerate(self.model.transformer.h):
            self.hooks.append(block.register_forward_pre_hook(self.cache_block(i)))

    def cache_block(self, i):
        def hook(module, args):
            if self.capturing:
                self.block_inputs[i] = args[0].detach()
        return hook

    def capture(self, x, y):
        super().capture(x, y)
        self.clean_x0 = norm(self.outputs['transformer.wte'])

    def losses(self, idx, targets):
        # Embedding perturbations change x0 and every block. Residual-scalar ES
        # mutates weights and has no jitter, so it must also run the full model.
        if not self.jitter or 'transformer.wte' in self.jitter:
            return super().losses(idx, targets)
        starts = []
        for site in self.jitter:
            if site.startswith('o@'):
                starts.append(int(site.split('@')[1]))
            elif site.startswith('transformer.h.'):
                starts.append(int(site.split('.')[2]))
            else:
                return super().losses(idx, targets)
        first = min(starts)
        copies = idx.shape[0] // self.batch
        if idx.shape[0] != copies * self.batch:
            raise ValueError('Cached forwards require whole repeated clean batches')
        with self.autocast():
            x = self.block_inputs[first].repeat(copies, 1, 1)
            x0 = self.clean_x0.repeat(copies, 1, 1)
            cos_sin = self.model.cos[:, :self.length], self.model.sin[:, :self.length]
            for i in range(first, self.model.config.n_layer):
                if i != first:
                    x = self.model.resid_lambdas[i] * x + self.model.x0_lambdas[i] * x0
                ve = (self.outputs[f'value_embeds.{i}'].repeat(copies, 1, 1)
                      if str(i) in self.model.value_embeds else None)
                x = self.model.transformer.h[i](x, ve, cos_sin, self.model.window_sizes[i])
            logits = self.model.lm_head(norm(x))[..., :self.model.config.vocab_size].float()
            logits = 15 * torch.tanh(logits / 15)
            return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                   ignore_index=-1, reduction='none').view_as(targets).float()
