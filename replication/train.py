"""Resumable execution of the released DUST/backprop configurations only."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import time

import torch

from .cached_dust import CachedDUST
from .dust import (DEFAULT_CONFIGS, GPTConfig, evaluate, file_hash, load_sequences,
                  make_model, make_optimizer, recipe)
from .backprop import DEFAULT_CONFIGS as BP_CONFIGS
from .backprop import make_optimizer as bp_optimizer, train_step
from common.flops import CONVENTION, ComputeLedger, add_eval_argument, eval_interval


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, path)


def atomic_save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['dust', 'backprop'], required=True)
    parser.add_argument('--tokens', choices=['1M', '10M'], required=True)
    parser.add_argument('--population', type=int, choices=[256, 1024, 4096, 16384])
    parser.add_argument('--seed', type=int, choices=[42, 43, 44], required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint-every', type=int, default=10)
    parser.add_argument('--max-steps', type=int, help='Validation-only early stop, never marks a run complete.')
    add_eval_argument(parser)
    args = parser.parse_args()
    if args.method == 'dust' and args.population is None:
        parser.error('DUST requires --population')
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if args.method == 'backprop' and world != 1:
        parser.error('Backprop uses one GPU, as in the release')
    if args.checkpoint_every < 1:
        parser.error('--checkpoint-every must be positive')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.set_float32_matmul_precision('high')
    if world > 1:
        torch.distributed.init_process_group('nccl', device_id=device)
    defaults = DEFAULT_CONFIGS[args.tokens] if args.method == 'dust' else BP_CONFIGS[args.tokens]
    interval = eval_interval(args, defaults, parser)
    total_steps = defaults['steps']
    stop_step = min(total_steps, args.max_steps or total_steps)
    if stop_step < 1:
        parser.error('--max-steps must be positive')
    paths = {name: args.data_dir / filename for name, filename in (
        ('train', f'train_{args.tokens}.pt'), ('val', 'val.pt'), ('test', 'test.pt'))}
    training = load_sequences(paths['train'], total_steps * 8)
    validation = load_sequences(paths['val'], 544)
    testing = load_sequences(paths['test'], 544)
    order = torch.randperm(len(training), generator=torch.Generator().manual_seed(args.seed + 1))
    training = training[order]
    model = make_model(GPTConfig(), device, args.seed)
    if args.method == 'dust':
        settings = recipe(args.tokens, args.population)
        optimizer = make_optimizer(model, settings)
        estimator = CachedDUST(model, settings, args.seed, rank, world)
    else:
        settings = None
        model.requires_grad_(True)
        optimizer = bp_optimizer(model, defaults['lr'], defaults['momentum'])
        estimator = None
    compute = ComputeLedger(device)
    identity = dict(method=args.method, tokens=args.tokens, population=args.population,
                    seed=args.seed, world=world)
    src = Path(__file__).resolve().parent
    root = src.parent
    sources = {str(p.relative_to(root)): file_hash(p) for p in (
        src / 'model.py', src / 'dust.py', src / 'cached_dust.py',
        src / 'backprop.py', root / 'common/flops.py', Path(__file__).resolve())}
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / 'config.json'
    metadata = dict(**identity, total_steps=total_steps, model=asdict(model.config),
                    training_tokens=total_steps * 8 * model.config.sequence_len, validation_sequences=544,
                    test_sequences=544, recipe=asdict(settings) if settings else defaults,
                    eval_interval=interval, flop_accounting=CONVENTION,
                    source_sha256=sources, torch_version=torch.__version__,
                    execution='cached-prefix' if estimator else 'backprop',
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    parameter_groups=[dict(lr=g['lr'], momentum=g['momentum'],
                                           parameters=sum(p.numel() for p in g['params']))
                                      for g in optimizer.param_groups])
    if rank == 0:
        metadata['data_sha256'] = {k: file_hash(v) for k, v in paths.items()}
        if config_path.exists():
            old = json.loads(config_path.read_text())
            for k in list(identity) + ['source_sha256', 'data_sha256', 'recipe', 'eval_interval', 'flop_accounting']:
                if old.get(k) != metadata.get(k):
                    raise ValueError(f'Resume configuration mismatch: {k}')
        else:
            atomic_json(config_path, metadata)
    if world > 1:
        torch.distributed.barrier()
    last = args.output / 'last.pt'
    if last.exists():
        saved = torch.load(last, map_location='cpu', weights_only=False)
        if saved['identity'] != identity:
            raise ValueError('Checkpoint identity mismatch')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        start_step, best, history = saved['step'], saved['best_val'], saved['history']
        elapsed_before = saved['elapsed_seconds']
        if 'compute' not in saved:
            raise ValueError('This checkpoint predates FLOP accounting; its historical compute is unknown.')
        compute.load_state_dict(saved['compute'])
        if compute.train_steps != start_step:
            raise ValueError('Checkpoint training step and FLOP accounting disagree')
        states = saved['rng'][rank]
        if rank == 0:
            # Restore the best checkpoint from the same durable boundary too.
            # A crash may have occurred after writing a newer best.pt.
            atomic_save(args.output / 'best.pt', saved['best_model'])
        torch.set_rng_state(states['cpu'])
        torch.cuda.set_rng_state(states['cuda'], device)
        if estimator:
            estimator.generator.set_state(states['estimator'])
        del saved
        if rank == 0:
            # Discard log entries after the last durable checkpoint on crash recovery.
            (args.output / 'metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in history))
            compute.save(args.output / 'flops.json')
    else:
        start_step = 0
        with compute.measure('val'):
            best = evaluate(model, validation, device)
        history = [dict(step=0, val_loss=best, **compute.metrics())]
        elapsed_before = 0.
        if rank == 0:
            atomic_save(args.output / 'best.pt', cpu_tree(model.state_dict()))
            (args.output / 'metrics.jsonl').write_text(json.dumps(history[0]) + '\n')
            compute.save(args.output / 'flops.json')
    start = time.monotonic()
    requested_stop = False

    def stop_handler(signum, frame):
        nonlocal requested_stop
        requested_stop = True

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)

    def checkpoint(step):
        states = dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device),
                      estimator=estimator.generator.get_state() if estimator else None)
        states = cpu_tree(states)
        if world > 1:
            collected = [None] * world if rank == 0 else None
            torch.distributed.gather_object(states, collected, dst=0)
        else:
            collected = [states]
        if rank == 0:
            atomic_save(last, dict(identity=identity, step=step, best_val=best,
                                  model=cpu_tree(model.state_dict()), optimizer=cpu_tree(optimizer.state_dict()),
                                  best_model=torch.load(args.output / 'best.pt', map_location='cpu', weights_only=True),
                                  rng=collected, history=history, compute=compute.state_dict(),
                                  elapsed_seconds=elapsed_before + time.monotonic() - start))
        if world > 1:
            torch.distributed.barrier()

    if rank == 0:
        print(json.dumps(dict(event='start', **identity, resume_step=start_step, total_steps=total_steps)), flush=True)
    checkpoint(start_step)
    step = start_step
    for step in range(start_step + 1, stop_step + 1):
        batch = training[(step - 1) * 8:step * 8].to(device)
        x, y = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
        with compute.measure('train', tokens=y.numel()):
            loss = estimator.step(x, y, optimizer) if estimator else train_step(model, optimizer, x, y)
        record = dict(step=step, train_loss=loss,
                      elapsed_seconds=elapsed_before + time.monotonic() - start)
        # max-steps never adds an evaluation absent from the released protocol.
        if step == 1 or step % interval == 0 or step == total_steps:
            with compute.measure('val'):
                value = evaluate(model, validation, device)
            record['val_loss'] = value
            if value < best:
                best = value
                if rank == 0:
                    atomic_save(args.output / 'best.pt', cpu_tree(model.state_dict()))
        record.update(compute.metrics())
        history.append(record)
        if rank == 0:
            print(json.dumps(record), flush=True)
            with (args.output / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            atomic_json(args.output / 'progress.json', dict(**identity, **record, best_val=best,
                                                          total_steps=total_steps, state='running'))
            if 'val_loss' in record:
                compute.save(args.output / 'flops.json')
        stop = torch.tensor(int(requested_stop), device=device)
        if world > 1:
            torch.distributed.all_reduce(stop, op=torch.distributed.ReduceOp.MAX)
        if step % args.checkpoint_every == 0 or step == stop_step or stop.item():
            checkpoint(step)
        if stop.item():
            break
    if estimator:
        estimator.close()
    if rank == 0:
        if step == total_steps:
            model.load_state_dict(torch.load(args.output / 'best.pt', map_location=device, weights_only=True))
            with compute.measure('test', distributed=False):
                test_loss = evaluate(model, testing, device)
            result = dict(**identity, complete=True, best_val=best,
                          test_at_best_val=test_loss, history=history, **compute.metrics(),
                          elapsed_seconds=elapsed_before + time.monotonic() - start)
            atomic_json(args.output / 'result.json', result)
            print(json.dumps({k: v for k, v in result.items() if k != 'history'}), flush=True)
        atomic_json(args.output / 'progress.json', dict(**identity, step=step, total_steps=total_steps,
                                                      best_val=best, **compute.metrics(),
                                                      state='complete' if step == total_steps else 'paused'))
        compute.save(args.output / 'flops.json')
    if world > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    if requested_stop:
        raise SystemExit(75)


if __name__ == '__main__':
    main()
