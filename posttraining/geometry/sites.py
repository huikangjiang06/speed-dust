from dataclasses import asdict
import hashlib
from pathlib import Path

from torch import nn

from ..hf_dust import ModelLayout
from ..runtime import artifact_identity
from .common import FORMAT, fingerprint


SUFFIXES = ('self_attn.o_proj', 'mlp.down_proj', 'mlp.up_proj', 'mlp.gate_proj')


def resolve_sites(model, training):
    layout = ModelLayout(model, training)
    sites = []
    for layer in layout.layers:
        for suffix in SUFFIXES:
            name = f'model.layers.{layer}.{suffix}'
            module = model.get_submodule(name)
            if not isinstance(module, nn.Linear):
                raise TypeError(f'Geometry site must be an ordinary Linear: {name}')
            sites.append(dict(name=name, layer=layer, suffix=suffix, width=module.out_features,
                              filename=name+'.safetensors', family=layout.families[name]))
    return sites


def reference_identity(config, manifest, model, sites):
    return dict(format=FORMAT, baseline_reference=artifact_identity(config, manifest, model), sites=sites,
        data_configuration_fingerprint=manifest['configuration_fingerprint'],
        template_sha256=hashlib.sha256((manifest.get('chat_template') or '').encode()).hexdigest(),
        scope='all_valid_input_occurrences_in_processed_train_split',
        input_rule='same collator as SFT; input_ids[:-1], right padding, position IDs from native model',
        pooling='all roles/examples/positions within a site; no row deduplication',
        forward_precision=config.training.precision, storage_dtype='bfloat16')


def expected_input_tokens(data):
    return sum(len(row['input_ids'])-1 for row in data['train'])
