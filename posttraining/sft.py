"""Configurable DUST or backpropagation SFT with identical data and matrix selection."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import torch
import torch.distributed as dist
import transformers

from common.flops import ComputeLedger, CONVENTION
from .configuration import load_config
from .hf_dust import HFDUST, native_losses
from .sft_data import load_prepared
from .runtime import (atomic_json, atomic_save, batch_examples, cpu_tree, evaluate, load_model,
                      make_optimizer, optimizer_step, set_learning_rate, source_hashes, to_device, update_plan,
                      validate_scales)


def train(config, method, output, *, resume=False, stop_after=None):
    config.validate()
    if config.geometry.enabled != (method == 'geometry_dust'):
        raise ValueError('geometry.enabled must be true exactly for method=geometry_dust')
    output = Path(output)
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    if world > 1:
        if method not in ('dust', 'geometry_dust'):
            raise ValueError('Run the matched Backprop baseline on one GPU; distributed DUST splits perturbations, not data')
        if not config.training.device.startswith('cuda'):
            raise ValueError('Distributed production runs require CUDA')
        config.training.device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
        dist.init_process_group('nccl', device_id=torch.device(config.training.device))
    if rank == 0:
        if not resume and output.exists() and any(output.iterdir()):
            raise FileExistsError(f'Choose a fresh output directory: {output}')
        output.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()
    data, manifest = load_prepared(config)
    plan = update_plan(len(data['train']), config.training)
    total_steps = len(plan)
    model, selected = load_model(config, method)
    if manifest['tokenizer_size'] > model.get_input_embeddings().num_embeddings:
        raise ValueError('Tokenizer vocabulary exceeds the pretrained embedding matrix')
    optimizer = make_optimizer(model, selected, config)
    scales_sha256 = validate_scales(config, manifest, model) if method in ('dust', 'geometry_dust') else None
    estimator = HFDUST(model, config.training, config.dust, config.training.seed) if method in ('dust', 'geometry_dust') else None
    if estimator:
        for site in estimator.layout.selected_sites + [f'o@{i}' for i in estimator.layers]:
            estimator.sigma(site)  # Fail before any training/evaluation if scales are missing.
    geometry_bank = None
    offline_costs = {}
    if estimator and config.dust.noise.scales_file:
        scales = json.loads(Path(config.dust.noise.scales_file).read_text())
        offline_costs['baseline_scale_calibration_flops'] = scales.get('calibration_flops', 0)
    if config.geometry.enabled:
        if not config.geometry.bank_path:
            raise ValueError('Geometry training requires a completed, calibrated bank')
        from .geometry.calibrate import validate_bank
        from .geometry.direct_noise import AdditiveGeometryNoise
        geometry_bank, bank_root = validate_bank(config.geometry.bank_path, config, manifest, model, estimator)
        estimator.direct_sampler = AdditiveGeometryNoise(estimator, geometry_bank, bank_root, config.geometry)
        offline_costs.update(geometry_bank['costs'])
    offline_flops = offline_costs.get('baseline_scale_calibration_flops', 0) + offline_costs.get('total_offline_flops', 0)
    device = next(model.parameters()).device
    compute = ComputeLedger(device)
    saved_config = asdict(config)
    # Rank-local device indices do not change the global experiment identity.
    saved_config['training']['device'] = 'cuda:local_rank' if world > 1 else str(device)
    identity = dict(config=saved_config, method=method, world=world,
                    model_commit=getattr(model.config, '_commit_hash', None),
                    data_sha256=manifest['data_sha256'], scales_sha256=scales_sha256, sources=source_hashes(),
                    geometry_bank_fingerprint=geometry_bank['fingerprint'] if geometry_bank else None,
                    transformers_version=transformers.__version__, torch_version=str(torch.__version__),
                    flop_convention=CONVENTION, plan_sha256=hashlib.sha256(json.dumps(plan).encode()).hexdigest())
    metadata = dict(identity=identity, total_steps=total_steps,
                    total_parameters=sum(p.numel() for p in model.parameters()),
                    updated_parameters=sum(p.numel() for p in selected.values()),
                    trainable_names=list(selected), frozen_names=[n for n, p in model.named_parameters() if n not in selected],
                    tied_embedding_head=model.get_input_embeddings().weight is model.get_output_embeddings().weight,
                    allocation=estimator.allocation if estimator else None,
                    geometry_rhos=estimator.direct_sampler.rhos if geometry_bank else None, offline_costs=offline_costs,
                    optimizer_groups=[dict(lr=g['lr'], parameters=sum(p.numel() for p in g['params']), names=g['names']) for g in optimizer.param_groups])
    last = output / 'last.pt'
    if resume:
        previous = json.loads((output / 'config.json').read_text())
        if previous['identity'] != identity:
            raise ValueError('Resume identity mismatch: configuration, data, source, dependency version or model revision changed')
        saved = torch.load(last, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        compute.load_state_dict(saved['compute'])
        step, best, best_step, history = saved['step'], saved['best_val'], saved['best_step'], saved['history']
        elapsed_before = saved['elapsed_seconds']
        if compute.train_steps != step:
            raise ValueError('Checkpoint step and compute ledger disagree')
        if rank == 0:
            atomic_save(output / 'best.pt', saved['best_model'])
            (output / 'metrics.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in history))
        rng = saved['rng'][rank]
        torch.set_rng_state(rng['cpu'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(rng['cuda'], device)
        if estimator:
            estimator.generator.set_state(rng['estimator'])
            if estimator.direct_sampler is not None:
                estimator.direct_sampler.load_rng_state(rng['geometry'])
                if saved['geometry_bank_fingerprint'] != geometry_bank['fingerprint'] or saved['geometry_rhos'] != estimator.direct_sampler.rhos:
                    raise ValueError('Checkpoint geometry fingerprint or fixed rho values changed')
        del saved
    else:
        if rank == 0:
            atomic_json(output / 'config.json', metadata)
        with compute.measure('val'):
            initial = evaluate(model, data['validation'], config, manifest)
        step, best, best_step, elapsed_before = 0, initial['loss'], 0, 0.
        history = [dict(step=0, val_loss=best, validation_tokens=initial['supervised_tokens'], **compute.metrics())]
        if rank == 0:
            atomic_save(output / 'best.pt', cpu_tree(model.state_dict()))
            (output / 'metrics.jsonl').write_text(json.dumps(history[0])+'\n')
    stopping = False
    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True
    old_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    started = time.monotonic()

    def checkpoint():
        rng = dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
                   estimator=estimator.generator.get_state() if estimator else None,
                   geometry=estimator.direct_sampler.rng_state() if geometry_bank else None)
        rng = cpu_tree(rng)
        states = [rng]
        if world > 1:
            states = [None] * world
            dist.all_gather_object(states, rng)
        if rank == 0:
            atomic_save(last, dict(step=step, best_val=best, best_step=best_step, history=history,
                model=cpu_tree(model.state_dict()), optimizer=cpu_tree(optimizer.state_dict()),
                best_model=torch.load(output/'best.pt', map_location='cpu', weights_only=True),
                geometry_bank_fingerprint=geometry_bank['fingerprint'] if geometry_bank else None,
                geometry_rhos=estimator.direct_sampler.rhos if geometry_bank else None,
                compute=compute.state_dict(), rng=states, elapsed_seconds=elapsed_before+time.monotonic()-started))
            compute.save(output / 'flops.json')
        if world > 1:
            dist.barrier()

    try:
        checkpoint()
        for next_step in range(step+1, total_steps+1):
            microbatches = [batch_examples(data['train'], indices, config, manifest) for indices in plan[next_step-1]]
            supervised = sum(int(batch['labels'].ne(-100).sum()) for batch in microbatches)
            input_tokens = sum(int(batch['attention_mask'].sum()) for batch in microbatches)
            set_learning_rate(optimizer, config.optimizer, next_step, total_steps)
            loss_sum = 0.
            probes = {}
            geometry_report = None
            if geometry_bank:
                estimator.direct_sampler.reset_logs()
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            update_started = time.monotonic()
            with compute.measure('train', tokens=supervised):
                optimizer.zero_grad(set_to_none=True)
                model.eval()  # Dropout-free matched objective; autograd remains enabled for Backprop.
                for batch in microbatches:
                    batch = to_device(batch, device)
                    if estimator:
                        estimate = estimator.estimate_gradients(batch)
                        for key, value in (estimate.probe_counts or {}).items():
                            probes[key] = probes.get(key, 0) + value
                        loss_sum += estimate.loss_sum
                        scale = estimate.supervised_tokens / supervised
                        for name, parameter in selected.items():
                            value = estimate.gradients[name] * scale
                            if parameter.grad is None:
                                parameter.grad = value
                            else:
                                parameter.grad.add_(value)
                        del estimate
                    else:
                        losses = native_losses(model, batch, config.training.precision)
                        loss_sum += losses.detach().sum().item()
                        (losses.sum() / supervised).backward()
                update_statistics = optimizer_step(optimizer, selected, config.training.max_grad_norm, return_details=True,
                    record_families=config.training.record_parameter_families)
                if geometry_bank:
                    geometry_report = estimator.direct_sampler.report()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            step = next_step
            record = dict(step=step, train_loss=loss_sum/supervised, **update_statistics,
                          supervised_tokens_this_step=supervised, input_tokens_this_step=input_tokens,
                          processed_examples_this_step=sum(len(indices) for indices in plan[next_step-1]),
                          probe_counts_on_rank=probes, diagnostics_rank=rank,
                          step_seconds=time.monotonic()-update_started,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                          input_tokens=sum(r.get('input_tokens_this_step', 0) for r in history)+input_tokens,
                          learning_rates=[group['lr'] for group in optimizer.param_groups],
                          elapsed_seconds=elapsed_before+time.monotonic()-started)
            if step == 1 or step % config.training.eval_interval == 0 or step == total_steps:
                with compute.measure('val'):
                    validation = evaluate(model, data['validation'], config, manifest)
                record.update(val_loss=validation['loss'], validation_tokens=validation['supervised_tokens'])
                if validation['loss'] < best:
                    best, best_step = validation['loss'], step
                    if rank == 0:
                        atomic_save(output/'best.pt', cpu_tree(model.state_dict()))
            record.update(compute.metrics())
            record.update(offline_flops=offline_flops,total_with_offline_flops=compute.metrics()['total_flops']+offline_flops)
            if geometry_report is not None:
                record['geometry'] = geometry_report
            history.append(record)
            if rank == 0:
                with (output/'metrics.jsonl').open('a') as handle:
                    handle.write(json.dumps(record)+'\n')
                atomic_json(output/'progress.json', dict(state='running', total_steps=total_steps, **record))
                print(json.dumps({key:value for key,value in record.items() if key not in ('geometry','parameter_families')}), flush=True)
            stop = torch.tensor(int(stopping or (stop_after is not None and step >= stop_after)), device=device)
            if world > 1:
                dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if step % config.training.checkpoint_interval == 0 or step == total_steps or stop.item():
                checkpoint()
            if stop.item():
                break
        if estimator:
            estimator.close()
        if rank == 0:
            complete = step == total_steps
            if complete:
                model.load_state_dict(torch.load(output/'best.pt', map_location=device, weights_only=True))
                test = None
                if config.training.evaluate_test and 'test' in data:
                    with compute.measure('test', distributed=False):
                        test = evaluate(model, data['test'], config, manifest)
                if config.training.save_pretrained:
                    model.save_pretrained(output/'best_model', safe_serialization=True)
                    from transformers import AutoTokenizer
                    AutoTokenizer.from_pretrained(Path(config.data.prepared_path)/'tokenizer').save_pretrained(output/'best_model')
                atomic_json(output/'result.json', dict(complete=True, method=method, step=step, best_step=best_step,
                    initial_val=history[0]['val_loss'], best_val=best, test_at_best_val=test['loss'] if test else None,
                    history=history, offline_costs=offline_costs,offline_flops=offline_flops,
                    total_with_offline_flops=compute.metrics()['total_flops']+offline_flops, **compute.metrics()))
            atomic_json(output/'progress.json', dict(state='complete' if complete else 'paused', step=step,
                total_steps=total_steps, best_val=best, best_step=best_step, **compute.metrics()))
            compute.save(output/'flops.json')
        if world > 1:
            dist.barrier()
    finally:
        if estimator:
            estimator.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--method', choices=['dust', 'backprop', 'geometry_dust'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--set', action='append', default=[])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int, help='Controlled partial run; saves resume state without claiming completion')
    args = parser.parse_args()
    train(load_config(args.config, args.set), args.method, args.output, resume=args.resume, stop_after=args.stop_after)


if __name__ == '__main__':
    main()
