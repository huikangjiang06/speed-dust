"""Strict, serializable configuration for the post-training experiments."""
from dataclasses import asdict, dataclass, field
import copy
import json
import math
from pathlib import Path


@dataclass
class ModelSettings:
    name_or_path: str = 'HuggingFaceTB/SmolLM2-360M'
    revision: str | None = None
    tokenizer_name_or_path: str | None = None
    tokenizer_revision: str | None = None
    attn_implementation: str = 'sdpa'


@dataclass
class DataSettings:
    dataset: str | None = 'HuggingFaceTB/smol-smoltalk'
    subset: str | None = None
    revision: str | None = None
    train_split: str = 'train'
    validation_split: str | None = None
    test_split: str | None = 'test'
    train_file: str | None = None
    validation_file: str | None = None
    test_file: str | None = None
    prepared_path: str = 'data/posttraining'
    cache_dir: str = 'data/cache/datasets'
    template_file: str | None = None
    objective: str = 'full'
    messages_field: str = 'messages'
    text_field: str = 'text'
    prompt_field: str = 'prompt'
    response_field: str = 'completion'
    source_field: str = 'source'
    train_examples: int = 512
    validation_examples: int = 64
    test_examples: int = 64
    max_length: int = 1024
    min_length: int = 8
    long_sequences: str = 'drop'
    seed: int = 123
    pad_to_multiple_of: int = 8


@dataclass
class TrainingSettings:
    seed: int = 42
    device: str = 'cuda:0'
    precision: str = 'bf16'
    # Prevent batch-shape-dependent BF16 GEMM intermediate truncation.
    allow_bf16_reduced_precision_reduction: bool = False
    batch_size: int = 2
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    drop_last: bool = True
    epochs: int = 1
    max_steps: int | None = 32
    eval_interval: int = 4
    checkpoint_interval: int = 4
    max_grad_norm: float | None = None
    record_parameter_families: bool = False
    layers: str | list[int] = 'all'
    train_embedding: bool = True
    train_head: bool = True
    freeze_norms: bool = True
    save_pretrained: bool = True
    evaluate_test: bool = True


@dataclass
class OptimizerSettings:
    name: str = 'sgd'
    lr: float = .001
    momentum: float = .9
    weight_decay: float = 0.
    nesterov: bool = False
    dampening: float = 0.
    betas: list[float] = field(default_factory=lambda: [.9, .95])
    eps: float = 1e-8
    embedding_lr_multiplier: float = 1.
    head_lr_multiplier: float = 1.
    qkv_lr_multiplier: float = 1.
    schedule: str = 'constant'
    warmup_steps: int = 0


@dataclass
class NoiseSettings:
    # Relative scales are measured once on fixed calibration batches, then frozen.
    mode: str = 'fixed'
    direct_sigma: float = .2
    embedding_sigma: float = .2
    hub_sigma: float = .2
    hub_deeper_multiplier: float = 1.
    q_sigma: float = .05
    k_sigma: float = .05
    v_sigma: float = .05
    head_sigma: float = .05
    relative_rho: float = .03
    min_sigma: float = 1e-6
    site_sigmas: dict[str, float] = field(default_factory=dict)
    scales_file: str | None = None


@dataclass
class DustSettings:
    direct_population: int = 4096
    depth_profile: str = 'transferred'
    nrep: int = 4
    # Optional exact allocations override the automatically transferred profile.
    allocation: dict = field(default_factory=dict)
    local_q: int | list[int] | None = None
    local_k: int | list[int] | None = None
    local_v: int | list[int] | None = None
    head_slab_size: int = 256
    head_draws_per_slab: int = 128
    head_centering: str = 'batch_reapply_mask'
    gamma: float = .98
    cache_prefix: bool = True
    noise: NoiseSettings = field(default_factory=NoiseSettings)


@dataclass
class GeometrySettings:
    enabled: bool = False
    bank_path: str | None = None
    covariance_type: str = 'diag'
    components: int = 8
    max_iter: int = 50
    tolerance: float = 1e-4
    variance_floor_relative: float = 1e-6
    variance_floor_absolute: float = 1e-12
    fit_chunk_rows: int = 8192
    cache_shard_rows: int = 4096
    cache_batch_size: int = 2
    storage_margin_fraction: float = .2
    storage_margin_bytes: int = 10737418240
    calibration_tokens: int = 16384
    distance_guard: float = 1e-12
    scale_mode: str = 'matched'
    rho_override: float | None = None
    norm_matching: str = 'none'
    seed: int = 271828
    log_samples_per_site: int = 256


@dataclass
class SFTConfig:
    model: ModelSettings = field(default_factory=ModelSettings)
    data: DataSettings = field(default_factory=DataSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    optimizer: OptimizerSettings = field(default_factory=OptimizerSettings)
    dust: DustSettings = field(default_factory=DustSettings)
    geometry: GeometrySettings = field(default_factory=GeometrySettings)
    description: str = ''

    @classmethod
    def from_dict(cls, raw):
        raw = copy.deepcopy(raw)
        unknown = set(raw) - {'model', 'data', 'training', 'optimizer', 'dust', 'geometry', 'description'}
        if unknown:
            raise ValueError(f'Unknown configuration sections: {sorted(unknown)}')
        dust = raw.pop('dust', {})
        noise = NoiseSettings(**dust.pop('noise', {}))
        result = cls(model=ModelSettings(**raw.pop('model', {})),
                     data=DataSettings(**raw.pop('data', {})),
                     training=TrainingSettings(**raw.pop('training', {})),
                     optimizer=OptimizerSettings(**raw.pop('optimizer', {})),
                     geometry=GeometrySettings(**raw.pop('geometry', {})),
                     dust=DustSettings(noise=noise, **dust), **raw)
        result.validate()
        return result

    def validate(self):
        t, d, o, e = self.training, self.data, self.optimizer, self.dust
        for value in [t.batch_size, t.eval_batch_size, t.gradient_accumulation_steps, t.epochs,
                      t.eval_interval, t.checkpoint_interval, d.train_examples, d.validation_examples,
                      d.max_length, d.pad_to_multiple_of, e.nrep, e.head_slab_size, e.head_draws_per_slab]:
            if not isinstance(value, int) or value < 1:
                raise ValueError('Batch sizes, intervals, populations and lengths must be positive integers')
        if t.max_steps is not None and t.max_steps < 1:
            raise ValueError('max_steps must be positive or null')
        if d.test_examples < 0 or not 2 <= d.min_length <= d.max_length:
            raise ValueError('Invalid example counts or sequence length limits')
        if d.objective not in ('full', 'assistant') or d.long_sequences not in ('drop', 'truncate'):
            raise ValueError('Use full/assistant objective and drop/truncate long_sequences')
        if t.precision not in ('fp32', 'bf16') or not t.freeze_norms:
            raise ValueError('This port supports fp32/bf16 with normalization gains frozen')
        if o.name not in ('sgd', 'adamw') or o.schedule not in ('constant', 'cosine', 'linear'):
            raise ValueError('Unsupported optimizer or schedule')
        if not math.isfinite(o.lr) or o.lr <= 0 or not 0 <= o.momentum < 1 or o.weight_decay < 0:
            raise ValueError('Invalid optimizer hyperparameters')
        if len(o.betas) != 2 or any(not 0 <= b < 1 for b in o.betas) or o.eps <= 0:
            raise ValueError('Invalid AdamW hyperparameters')
        multipliers = (o.embedding_lr_multiplier, o.head_lr_multiplier, o.qkv_lr_multiplier)
        if any(not math.isfinite(v) or v <= 0 for v in multipliers) or o.warmup_steps < 0:
            raise ValueError('Learning-rate multipliers must be positive; warmup nonnegative')
        if o.nesterov and (o.momentum <= 0 or o.dampening != 0):
            raise ValueError('Nesterov requires momentum and zero dampening')
        if t.max_grad_norm is not None and t.max_grad_norm <= 0:
            raise ValueError('max_grad_norm must be positive or null')
        if not 0 <= e.gamma <= 1 or e.depth_profile not in ('transferred', 'uniform'):
            raise ValueError('Use gamma in [0,1] and transferred/uniform depth')
        if e.head_centering not in ('batch_reapply_mask', 'none'):
            raise ValueError('Unsupported head centering')
        if e.noise.mode not in ('fixed', 'activation_rms'):
            raise ValueError('Noise mode must be fixed or activation_rms')
        for key, value in asdict(e.noise).items():
            if key.endswith('_sigma') or key in ('relative_rho', 'hub_deeper_multiplier'):
                if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f'Invalid noise parameter {key}')
        if any(not math.isfinite(value) or value <= 0 for value in e.noise.site_sigmas.values()):
            raise ValueError('Site noise scales must be finite and positive')
        g = self.geometry
        if g.covariance_type != 'diag' or g.scale_mode not in ('matched', 'intermediate', 'full'):
            raise ValueError('Geometry supports diagonal covariance and matched/intermediate/full fixed scales')
        if g.norm_matching not in ('none', 'isotropic_per_token'):
            raise ValueError('Geometry norm_matching must be none or isotropic_per_token')
        for value in (g.components, g.max_iter, g.fit_chunk_rows, g.cache_shard_rows, g.cache_batch_size,
                      g.calibration_tokens, g.log_samples_per_site):
            if not isinstance(value, int) or value < 1:
                raise ValueError('Geometry counts and chunk sizes must be positive integers')
        for value in (g.variance_floor_relative, g.variance_floor_absolute, g.distance_guard):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Geometry numerical guards must be finite and positive')
        if not math.isfinite(g.tolerance) or g.tolerance < 0 or g.storage_margin_fraction < 0 or g.storage_margin_bytes < 0:
            raise ValueError('Invalid geometry tolerance or storage margin')
        if g.rho_override is not None and (not math.isfinite(g.rho_override) or not 0 <= g.rho_override <= 1):
            raise ValueError('Explicit rho override must be in [0,1]')


def load_config(path, overrides=()):
    path = Path(path).resolve()
    raw = json.loads(path.read_text())
    for override in overrides:
        dotted, value = override.split('=', 1)
        parts = dotted.split('.')
        cursor = raw
        for key in parts[:-1]:
            cursor = cursor.setdefault(key, {})
        try:
            cursor[parts[-1]] = json.loads(value)
        except json.JSONDecodeError:
            cursor[parts[-1]] = value
    result = SFTConfig.from_dict(raw)
    for owner, names in [(result.data, ['template_file', 'train_file', 'validation_file', 'test_file', 'prepared_path']),
                         (result.dust.noise, ['scales_file']), (result.geometry, ['bank_path'])]:
        for name in names:
            value = getattr(owner, name)
            if value and not Path(value).is_absolute():
                setattr(owner, name, str((path.parent / value).resolve()))
    return result


def save_config(path, config):
    Path(path).write_text(json.dumps(asdict(config), indent=2) + '\n')
