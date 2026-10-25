"""Arithmetic FLOP estimates for the eager PyTorch training loops in this repo.

Counts operations that actually execute, using explicit shape-based formulas.
These are dense-operator estimates, not GPU instruction or hardware counters.
The convention and exclusions are recorded in each run's ``flops.json`` file.
"""
from collections import Counter
from contextlib import contextmanager
import copy
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


CONVENTION = {
    'version': 'dust-arithmetic-v3',
    'kind': 'estimated_arithmetic_flops',
    'multiply_add': 2,
    'matmul': '2*m*n*k; dense shapes, including masked/padded attention positions',
    'scalar_math': 'One per add/mul/div/neg/abs/pow/exp/log/tanh/sqrt/rsqrt; sigmoid=4; silu=5; silu_backward=9; logaddexp=6.',
    'fused_attention': 'Dense QK/AV plus scale/softmax; backward includes score recomputation.',
    'precision_weighted': False,
    'excluded': ['RNG generation internals', 'integer/boolean operations and comparisons',
                 'memory movement, allocation, indexing, sorting, rounding and casts', 'communication collectives',
                 'model initialization, data preparation, checkpoint serialization'],
    'unsupported_floating_ops': 'raise an error; never silently count as zero',
    'resume': 'Counts follow the durable checkpoint trajectory; discarded/retried work is not reconstructed.',
}


def _tensors(value):
    return [x for x in tree_flatten(value)[0] if isinstance(x, torch.Tensor)]


def _numel(value):
    return sum(x.numel() for x in _tensors(value) if x.is_floating_point())


# These operators move/select values or perform comparisons, not floating arithmetic.
_ZERO = set('''_local_scalar_dense _to_copy _unsafe_view alias as_strided cat clone contiguous
copy_ detach empty empty_like empty_strided expand fill_ flatten full full_like gather index
index_select lift_fresh new_empty new_empty_strided new_ones new_zeros ones ones_like permute
repeat reshape scalar_tensor select select_backward slice slice_backward split split_with_sizes squeeze
stack t t_ to transpose transpose_ unbind unfold unsqueeze unsqueeze_ squeeze_ view view_as where zero_ zeros zeros_like
embedding eq ne gt ge lt le isfinite isinf isnan all any logical_and logical_or logical_not
masked_fill masked_fill_ masked_select constant_pad_nd scatter scatter_ select_scatter slice_scatter
bitwise_and bitwise_or bitwise_not clamp clamp_min clamp_max relu relu_ threshold_backward
maximum minimum max min argmax argmin detach_ _assert_async _assert_scalar
equal sort argsort floor floor_ ceil ceil_ round round_ trunc trunc_ multinomial
rand randn rand_like randn_like normal normal_ uniform_ random_ bernoulli bernoulli_
'''.split())
_UNARY = set('neg abs exp exp2 expm1 log log2 log10 log1p tanh sqrt rsqrt reciprocal sin cos pow square'.split())


def operation_flops(func, args, kwargs, out):
    """Return an integer estimate, or None for an unsupported floating operation."""
    packet = str(func._overloadpacket)
    if packet.startswith(('c10d.', '_c10d_functional.', '_c10d_functional_autograd.')):
        return 0
    if not packet.startswith('aten.'):
        return None
    name = packet[5:]
    if name in _ZERO:
        return 0
    n = _numel(out)
    if name in _UNARY:
        return n
    if name in ('add', 'add_', 'sub', 'sub_'):
        return n * (1 + int(kwargs.get('alpha', 1) != 1))
    if name in ('mul', 'mul_', 'div', 'div_', 'true_divide'):
        return n
    if name in ('addcmul', 'addcmul_', 'addcdiv', 'addcdiv_'):
        return n * (2 + int(kwargs.get('value', 1) != 1))
    if name in ('lerp', 'lerp_'):
        return 3 * n
    if name.startswith('_foreach_'):
        kind = name.removeprefix('_foreach_').rstrip('_')
        size = _numel(args[0])
        if kind in ('add', 'sub'):
            alpha = kwargs.get('alpha', args[2] if len(args) > 2 else 1)
            return size * (1 + int(alpha != 1))
        if kind in ('mul', 'div', 'neg', 'sqrt', 'abs'):
            return size
        if kind == 'lerp':
            return 3 * size
        if kind in ('addcmul', 'addcdiv'):
            value = kwargs.get('value', args[3] if len(args) > 3 else 1)
            return size * (2 + int(value != 1))
        return None
    if name == 'matmul':
        return 2 * n * args[0].shape[-1]
    if name == 'linear':
        bias = kwargs.get('bias', args[2] if len(args) > 2 else None)
        return 2 * n * args[1].shape[-1] + (n if bias is not None else 0)
    if name in ('mm', 'bmm', 'addmm', 'baddbmm'):
        a, b = args[:2] if name in ('mm', 'bmm') else args[1:3]
        batch = math.prod(a.shape[:-2])
        cost = 2 * batch * a.shape[-2] * a.shape[-1] * b.shape[-1]
        if name in ('addmm', 'baddbmm'):
            size = batch * a.shape[-2] * b.shape[-1]
            beta, alpha = kwargs.get('beta', 1), kwargs.get('alpha', 1)
            cost += size * (int(beta != 0) + int(beta not in (0, 1)) + int(alpha != 1))
        return cost
    if name == 'sum':
        return max(0, args[0].numel() - n)
    if name == 'mean':
        return args[0].numel()
    if name == 'logsumexp':
        return 3 * args[0].numel() + n
    if name == 'logaddexp':
        return 6 * n
    if name in ('_softmax', '_log_softmax', '_safe_softmax'):
        x, dim = args[:2]
        rows = x.numel() // x.shape[dim]
        return 4 * x.numel() - (0 if name == '_log_softmax' else rows)
    if name in ('_softmax_backward_data', '_log_softmax_backward_data'):
        grad, output, dim = args[:3]
        rows = grad.numel() // grad.shape[dim]
        return 4 * grad.numel() - rows
    if name == 'sigmoid':
        return 4 * n
    if name == 'silu':
        return 5 * n
    if name == 'silu_backward':
        # Reconstruct sigmoid (4), then g*s*(1+x*(1-s)): five scalar operations.
        return 9 * n
    if name in ('sigmoid_backward', 'tanh_backward'):
        return 3 * n
    if name == 'index_add_':
        return args[3].numel() * (1 + int(kwargs.get('alpha', 1) != 1))
    if name in ('index_put', 'index_put_'):
        return args[2].numel() if kwargs.get('accumulate', args[3] if len(args) > 3 else False) else 0
    if name in ('scatter_add', 'scatter_add_'):
        return args[3].numel()
    if name == 'embedding_dense_backward':
        return args[0].numel()
    if name == 'nll_loss_forward':
        x, target, weight, reduction = args[:4]
        rows = target.numel()
        return rows * (1 + int(weight is not None)) + (rows if reduction == 1 else max(0, rows - 1) if reduction == 2 else 0)
    if name == 'nll_loss_backward':
        grad, x, target, weight, reduction = args[:5]
        return target.numel() * (1 + int(weight is not None) + int(reduction == 1))
    if name == '_fused_rms_norm':
        x, normalized_shape, weight = args[:3]
        rows = x.numel() // math.prod(normalized_shape)
        return 3 * x.numel() + 2 * rows + (x.numel() if weight is not None else 0)
    if name == '_fused_rms_norm_backward':
        # Unweighted RMSNorm: reconstruct normalized x and apply its vector-Jacobian product.
        x = args[1]
        weight = args[4]
        if weight is not None:
            return None  # This repository's model has no learned RMSNorm weights.
        return 6 * x.numel()
    if name.startswith('_scaled_dot_product_') or name == 'scaled_dot_product_attention':
        dropout_index = next((i for i, argument in enumerate(func._schema.arguments)
                              if argument.name == 'dropout_p'), None)
        dropout = kwargs.get('dropout_p', args[dropout_index] if dropout_index is not None
                             and dropout_index < len(args) else 0)
        if dropout != 0:
            return None
        backward = name.endswith('_backward')
        q, k, v = args[1:4] if backward else args[:3]
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            return None
        batch, heads, queries, width = q.shape
        keys, value_width = k.shape[-2], v.shape[-1]
        scores = batch * heads * queries * keys
        rows = batch * heads * queries
        if backward:
            return 2 * scores * (3 * width + 2 * value_width) + 10 * scores - 2 * rows
        return 2 * scores * (width + value_width) + 5 * scores - rows
    return None


class ArithmeticFlopCounter(TorchDispatchMode):
    """Count every executed supported floating operator without changing its result."""
    def __init__(self):
        super().__init__()
        self.by_op = Counter()
        self.zero_ops = Counter()

    @property
    def total(self):
        return sum(self.by_op.values())

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if not any(x.is_floating_point() for x in _tensors((args, kwargs, out))):
            return out
        flops = operation_flops(func, args, kwargs, out)
        if flops is None:
            raise NotImplementedError(f'No FLOP formula for floating operation {func}; add a formula or explicit exclusion.')
        if flops:
            self.by_op[str(func)] += int(flops)
        else:
            self.zero_ops[str(func)] += 1
        return out


class ComputeLedger:
    """Persistent global counts; DUST's replicated data batch is counted once."""
    phases = ('train', 'val', 'test')

    def __init__(self, device='cpu', *, extra_phases=()):
        self.device = torch.device(device)
        self.phases = tuple(dict.fromkeys((*type(self).phases, *extra_phases)))
        self.totals = dict.fromkeys(self.phases, 0)
        self.by_op = {phase: Counter() for phase in self.phases}
        self.zero_ops = {phase: Counter() for phase in self.phases}
        self.train_tokens = 0
        self.train_steps = 0
        self.step_train_flops = 0

    @contextmanager
    def measure(self, phase, *, tokens=0, distributed=True):
        if phase not in self.phases or tokens < 0 or (tokens and phase != 'train'):
            raise ValueError(f'Use one of {self.phases} and nonnegative training tokens')
        counter = ArithmeticFlopCounter()
        with counter:
            yield counter
        counts, zeros = counter.by_op, counter.zero_ops
        if distributed and dist.is_available() and dist.is_initialized():
            # Counts are integers, and every rank may execute different operators.
            # Gather small dictionaries rather than assume identical rank workloads.
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, (dict(counts), dict(zeros), int(tokens)))
            if any(item[2] != tokens for item in gathered):
                raise ValueError('Replicated DUST batches must have the same token count on all ranks')
            counts, zeros = Counter(), Counter()
            for rank_counts, rank_zeros, _ in gathered:
                counts.update(rank_counts)
                zeros.update(rank_zeros)
        total = int(sum(counts.values()))
        self.totals[phase] += total
        self.by_op[phase].update(counts)
        self.zero_ops[phase].update(zeros)
        if phase == 'train':
            self.train_tokens += int(tokens)
            self.train_steps += 1
            self.step_train_flops = total

    def metrics(self):
        return dict(train_tokens=self.train_tokens, step_train_flops=self.step_train_flops,
                    **{f'{phase}_flops': value for phase, value in self.totals.items()},
                    total_flops=sum(self.totals.values()))

    def state_dict(self):
        return dict(convention=copy.deepcopy(CONVENTION), torch_version=str(torch.__version__),
                    train_steps=self.train_steps, **self.metrics(),
                    flops_by_op={k: dict(v) for k, v in self.by_op.items()},
                    zero_flop_operator_calls={k: dict(v) for k, v in self.zero_ops.items()})

    def load_state_dict(self, state):
        if state['convention'] != CONVENTION or state['torch_version'] != str(torch.__version__):
            raise ValueError('Cannot resume with a different FLOP convention or PyTorch version')
        if set(state['flops_by_op']) != set(self.phases):
            raise ValueError('Cannot resume with different compute phases')
        self.train_tokens = int(state['train_tokens'])
        self.train_steps = int(state['train_steps'])
        self.step_train_flops = int(state['step_train_flops'])
        self.totals = {phase: int(state[f'{phase}_flops']) for phase in self.phases}
        self.by_op = {phase: Counter(state['flops_by_op'][phase]) for phase in self.phases}
        self.zero_ops = {phase: Counter(state['zero_flop_operator_calls'][phase]) for phase in self.phases}
        if any(sum(self.by_op[p].values()) != self.totals[p] for p in self.phases):
            raise ValueError('Inconsistent saved FLOP totals')

    def save(self, path):
        path = Path(path)
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(self.state_dict(), indent=2) + '\n')
        os.replace(temporary, path)


def add_eval_argument(parser):
    parser.add_argument('--eval-interval', type=int,
                        help='Validation every N updates; default is the released budget-specific interval.')


def eval_interval(args, defaults, parser):
    interval = defaults['eval_interval'] if args.eval_interval is None else args.eval_interval
    if interval < 1:
        parser.error('--eval-interval must be positive')
    return interval
