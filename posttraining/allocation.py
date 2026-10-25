"""Transfer the paper's family shares and relative-depth profile without losing draws."""
import math


def apportion(total, weights, quantum, *, positive=True):
    """Largest-remainder rounding in scheduler-compatible units, preserving the total."""
    if total % quantum or any(w < 0 for w in weights) or not sum(weights):
        raise ValueError('Invalid allocation total, weights, or draw-chunk divisibility')
    units = total // quantum
    minimum = int(positive)
    if units < minimum * len(weights):
        raise ValueError(f'{total} draws cannot cover {len(weights)} groups with quantum={quantum}; increase population or reduce nrep/world')
    # Preserve exact profiles when their ordinary rounded quotas meet the minimum.
    quotas = [units * w / sum(weights) for w in weights]
    counts = [max(minimum, math.floor(q)) for q in quotas]
    while sum(counts) > units:
        candidates = [i for i, c in enumerate(counts) if c > minimum]
        i = max(candidates, key=lambda j: (counts[j] - quotas[j], -j))
        counts[i] -= 1
    while sum(counts) < units:
        i = max(range(len(counts)), key=lambda j: (quotas[j] - counts[j], -j))
        counts[i] += 1
    return [quantum * count for count in counts]


def resolve_allocation(settings, n_layers, selected_layers, *, world=1, train_embedding=True, train_head=True):
    quantum = world * settings.nrep
    selected = sorted(selected_layers)
    if not selected or selected[0] < 0 or selected[-1] >= n_layers or len(set(selected)) != len(selected):
        raise ValueError('Select unique, valid transformer layers')
    groups = ('writers', 'gate', 'up', 'attention')
    if settings.allocation:
        unknown = set(settings.allocation) - set(groups) - {'embedding'}
        if unknown:
            raise ValueError(f'Unknown allocation groups: {sorted(unknown)}')
        counts = {key: list(settings.allocation[key]) for key in groups}
        embedding = settings.allocation.get('embedding', 0)
    else:
        # Original shares are 110/110/26/10 at K=256. Split SwiGLU hidden into 55/55.
        family_weights = [110, 55, 55, 26] + ([10] if train_embedding else [])
        totals = apportion(settings.direct_population, family_weights, quantum)
        profiles = [38, 32, 16, 10, 6, 4, 2, 2], [6, 6, 4, 2, 2, 2, 2, 2]
        counts = {}
        for key, total in zip(groups, totals):
            profile = profiles[1 if key == 'attention' else 0]
            weights = [1 if settings.depth_profile == 'uniform' else profile[min(7, i * 8 // n_layers)] for i in selected]
            values = apportion(total, weights, quantum)
            counts[key] = [values[selected.index(i)] if i in selected else 0 for i in range(n_layers)]
        embedding = totals[-1] if train_embedding else 0
    for key, values in counts.items():
        if len(values) != n_layers:
            raise ValueError(f'{key} allocation must have {n_layers} entries')
        for i, value in enumerate(values):
            if not isinstance(value, int) or value < 0 or value % quantum or (i in selected) != (value > 0):
                raise ValueError(f'{key}[{i}] must cover exactly the selected layers and be divisible by {quantum}')
    if not isinstance(embedding, int) or embedding < 0 or embedding % quantum or (embedding > 0) != train_embedding:
        raise ValueError('Embedding draws must match train_embedding and scheduler divisibility')
    actual = sum(map(sum, counts.values())) + embedding
    if actual != settings.direct_population:
        raise ValueError(f'Direct draws total {actual}, expected {settings.direct_population}')
    local = {}
    for kind, base in [('q', 2), ('k', 12), ('v', 20)]:
        given = getattr(settings, f'local_{kind}')
        if given is None:
            total = round(base * 8 * settings.direct_population / 256 / quantum) * quantum
            values = apportion(total, [1] * len(selected), quantum)
            local[kind] = [values[selected.index(i)] if i in selected else 0 for i in range(n_layers)]
        elif isinstance(given, int):
            local[kind] = [given if i in selected else 0 for i in range(n_layers)]
        else:
            local[kind] = list(given)
        if len(local[kind]) != n_layers or any(not isinstance(v, int) or v < 0 or v % quantum or
                ((i in selected) != (v > 0)) for i, v in enumerate(local[kind])):
            raise ValueError(f'Local {kind} counts must cover selected layers and be divisible by {quantum}')
    if train_head and settings.head_draws_per_slab % quantum:
        raise ValueError('Head draws per slab must be divisible by world*nrep')
    return dict(**counts, embedding=embedding, local=local, direct_population=actual,
                nrep=settings.nrep, world=world, quantum=quantum,
                head_draws_per_slab=settings.head_draws_per_slab if train_head else 0,
                head_slab_size=settings.head_slab_size)
