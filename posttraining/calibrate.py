"""Explicit, gradient-assisted calibration; never launches a training pilot implicitly."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import statistics
import time

import torch

from common.flops import ComputeLedger
from .configuration import load_config
from .hf_dust import HFDUST, autocast, native_losses
from .runtime import (artifact_identity, atomic_json, batch_examples, load_model, to_device,
                      validate_scales)
from .sft_data import load_prepared


def diagnostic_batches(rows, count, batch_size, seed):
    """Fixed quantiles of source/length-sorted training rows, then seeded mixing."""
    if count < 1:
        raise ValueError('Calibration batch count must be positive')
    order = sorted(range(len(rows)), key=lambda i: (rows[i]['source'], rows[i]['length'], rows[i]['content_sha256']))
    total = min(len(order), count * batch_size)
    selected = [order[min(len(order)-1, int((i+.5)*len(order)/total))] for i in range(total)]
    random.Random(seed).shuffle(selected)
    return [selected[i:i+batch_size] for i in range(0, len(selected), batch_size)]


def reference_gradients(model, estimator, batch):
    """Ordinary supervised gradient, with tied embedding/head contributions separated."""
    selected = estimator.layout.selected
    for p in selected.values():
        p.requires_grad_(True)
    saved, hooks = {}, []
    def output_hook(site):
        def hook(module, args, output):
            if output.requires_grad:
                output.retain_grad()
            saved[site] = (args[0].detach(), output)
        return hook
    def input_hook(site):
        def hook(module, args):
            if args[0].requires_grad:
                args[0].retain_grad()
            saved[site] = args[0]
        return hook
    for site in ('model.embed_tokens', 'lm_head'):
        hooks.append(estimator.layout.modules[site].register_forward_hook(output_hook(site)))
    for layer in estimator.layers:
        hooks.append(model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(input_hook(f'o@{layer}')))
    model.zero_grad(set_to_none=True)
    try:
        losses = native_losses(model, batch, estimator.training.precision)
        loss = losses.sum() / batch['labels'].ne(-100).sum()
        loss.backward()
        gradients = {name: p.grad.detach().float().cpu().clone() for name, p in selected.items()}
        components = {}
        for site in estimator.layout.selected_sites:
            if site == 'model.embed_tokens':
                indices, activation = saved[site]
                contribution = torch.zeros_like(estimator.layout.embedding.weight)
                contribution.index_add_(0, indices.reshape(-1), activation.grad.float().reshape(-1, activation.shape[-1]))
            elif site == 'lm_head':
                inputs, activation = saved[site]
                with autocast(estimator.device, estimator.training.precision):
                    contribution = activation.grad.flatten(0, 1).T @ inputs.flatten(0, 1)
            else:
                contribution = selected[estimator.layout.canonical(site)].grad
            components[site] = contribution.detach().float().cpu().clone()
        targets = {f'o@{i}': saved[f'o@{i}'].grad.detach().float().cpu().clone() for i in estimator.layers}
        return dict(loss=loss.item(), gradients=gradients, components=components, attention=targets)
    finally:
        for hook in hooks:
            hook.remove()
        model.zero_grad(set_to_none=True)
        model.requires_grad_(False)


def alignment(estimated, reference):
    x, y = estimated.float(), reference.to(estimated.device).float()
    dot = (x * y).sum()
    xx, yy = x.square().sum(), y.square().sum()
    denominator = (xx * yy).sqrt()
    return dict(cosine=(dot / denominator).item() if denominator.item() else None,
                estimate_norm=xx.sqrt().item(), reference_norm=yy.sqrt().item(),
                norm_ratio=(xx / yy).sqrt().item() if yy.item() else None,
                dot=dot.item(), estimate_squared=xx.item(), reference_squared=yy.item())


def group_alignments(sites, family_of):
    totals = defaultdict(lambda: dict(dot=0., estimate_squared=0., reference_squared=0.))
    for site, metric in sites.items():
        family = family_of.get(site, 'attention')
        for key in ('dot', 'estimate_squared', 'reference_squared'):
            totals[family][key] += metric[key]
    output = {}
    for family, values in totals.items():
        xx, yy = values['estimate_squared'], values['reference_squared']
        output[family] = dict(**values, cosine=values['dot']/math.sqrt(xx*yy) if xx*yy else None,
                              norm_ratio=math.sqrt(xx/yy) if yy else None)
    return output


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def calibrate(config, stage, output, *, batches=4, seeds=(42, 43, 44), families=None, trial_lr=None):
    """Measure scales or frozen-weight estimator diagnostics; optional reversible trial step."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Choose an empty calibration output: {output}')
    output.mkdir(parents=True, exist_ok=True)
    data, manifest = load_prepared(config)
    model, selected = load_model(config, 'dust')
    if stage == 'diagnostics':
        validate_scales(config, manifest, model)
    estimator = HFDUST(model, config.training, config.dust, seed=config.training.seed)
    ledger = ComputeLedger(estimator.device, extra_phases=('calibration',))
    indices = diagnostic_batches(data['train'], batches, config.training.batch_size, config.data.seed)
    identity = artifact_identity(config, manifest, model)
    metadata = dict(identity=identity, config=asdict(config), batch_indices=indices,
                    sources=[[data['train'][i]['source'] for i in batch] for batch in indices],
                    allocation=estimator.allocation, stage=stage)
    atomic_json(output/'config.json', metadata)
    if trial_lr is not None and (trial_lr <= 0 or families is not None):
        raise ValueError('A trial learning rate must be positive and requires the full estimator')
    statistics_by_site, records = {}, []
    try:
        for batch_index, chosen in enumerate(indices):
            batch = to_device(batch_examples(data['train'], chosen, config, manifest), estimator.device)
            if stage == 'scales':
                with ledger.measure('calibration'):
                    stats = estimator.activation_statistics(batch)
                for site, values in stats.items():
                    total = statistics_by_site.setdefault(site, dict(sum_squares=0., count=0))
                    for key in total:
                        total[key] += values[key]
                continue
            with ledger.measure('calibration'):
                reference = reference_gradients(model, estimator, batch)
            for seed in seeds:
                estimator.generator.manual_seed(seed + 3 + 100003 * batch_index)
                synchronize(estimator.device)
                started = time.monotonic()
                if estimator.device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(estimator.device)
                before = ledger.totals['calibration']
                with ledger.measure('calibration'):
                    estimate = estimator.estimate_gradients(batch, families=families, keep_components=True)
                synchronize(estimator.device)
                seconds = time.monotonic()-started
                cost = ledger.totals['calibration']-before
                with ledger.measure('calibration'):
                    sites = {site: alignment(value, reference['components'][site]) for site, value in estimate.components.items()}
                    targets = {site: alignment(value, reference['attention'][site]) for site, value in estimator.last_attention_targets.items()}
                    params = {site: alignment(value, reference['gradients'][site]) for site, value in estimate.gradients.items()}
                record = dict(batch=batch_index, seed=seed, reference_loss=reference['loss'], clean_loss=estimate.loss_sum/estimate.supervised_tokens,
                    seconds=seconds, estimator_flops=cost, sites=sites, attention_targets=targets,
                    families=group_alignments({**sites, **targets}, estimator.layout.families),
                    parameters=params if families is None else None,
                    max_memory_allocated=torch.cuda.max_memory_allocated(estimator.device) if estimator.device.type == 'cuda' else None)
                if trial_lr is not None:
                    originals = {name: p.detach().cpu().clone() for name, p in selected.items()}
                    try:
                        with ledger.measure('calibration'), torch.no_grad():
                            for name, p in selected.items():
                                p.add_(estimate.gradients[name], alpha=-trial_lr)
                            trial = native_losses(model, batch, config.training.precision).sum()/estimate.supervised_tokens
                        record['trial'] = dict(lr=trial_lr, loss=trial.item(), loss_change=trial.item()-reference['loss'], momentum=0.)
                    finally:
                        with torch.no_grad():
                            for name, p in selected.items():
                                p.copy_(originals[name])
                records.append(record)
                atomic_json(output/'diagnostics.json', dict(records=records, **ledger.metrics()))
                ledger.save(output/'flops.json')
                print(json.dumps(dict(batch=batch_index, seed=seed, seconds=seconds, estimator_flops=cost,
                    families=record['families'])), flush=True)
                del estimate
        if stage == 'scales':
            rms = {site: math.sqrt(values['sum_squares']/values['count']) for site, values in statistics_by_site.items()}
            direct = {site for site in rms if site.startswith('o@') or estimator.layout.families.get(site) in
                      ('writers', 'gate', 'up', 'embedding')}
            sigmas = {site: max(config.dust.noise.min_sigma, config.dust.noise.relative_rho*rms[site]) for site in sorted(direct)}
            atomic_json(output/'scales.json', dict(identity=identity, site_sigmas=sigmas, activation_rms=rms,
                relative_rho=config.dust.noise.relative_rho, batch_indices=indices, **ledger.metrics()))
        else:
            # Variation is across noise seeds on each fixed batch, not across unlike batches.
            variation = []
            for batch_index in range(len(indices)):
                rows = [r for r in records if r['batch'] == batch_index]
                for family in sorted(set().union(*(r['families'] for r in rows))):
                    values = [r['families'][family]['cosine'] for r in rows if r['families'][family]['cosine'] is not None]
                    if values:
                        variation.append(dict(batch=batch_index, family=family, cosine_mean=statistics.mean(values),
                            cosine_stdev=statistics.stdev(values) if len(values)>1 else None, seeds=len(values)))
            atomic_json(output/'diagnostics.json', dict(records=records, variation=variation, **ledger.metrics()))
        ledger.save(output/'flops.json')
    finally:
        estimator.close()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', choices=['scales', 'diagnostics'], required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--batches', type=int, default=4)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--families', nargs='+', help='Optional subset: writers gate up attention q k v embedding head')
    parser.add_argument('--trial-lr', type=float, help='Optional reversible plain-SGD trial; measured then restored')
    parser.add_argument('--set', action='append', default=[])
    args = parser.parse_args()
    calibrate(load_config(args.config, args.set), args.stage, args.output, batches=args.batches,
              seeds=args.seeds, families=args.families, trial_lr=args.trial_lr)


if __name__ == '__main__':
    main()
