"""Minimal SGD backpropagation baseline with the same model, data, and evaluation as DUST."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import torch

from .dust import GPTConfig, evaluate, file_hash, load_sequences, make_model
from common.flops import CONVENTION, ComputeLedger, add_eval_argument, eval_interval


DEFAULT_CONFIGS = {
    '1M': dict(lr=0.175, momentum=0.98, steps=61, eval_interval=2),
    '10M': dict(lr=0.2, momentum=0.95, steps=610, eval_interval=10),
}


def make_optimizer(model, lr, momentum):
    groups = [
        {'params': [p for name, p in model.named_parameters()
                    if name != 'transformer.wte.weight'], 'lr': lr},
        {'params': [model.transformer.wte.weight], 'lr': 1000.0},
    ]
    return torch.optim.SGD(groups, lr=lr, momentum=momentum)


def train_step(model, optimizer, x, y):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
        loss = model(x, y)
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite backpropagation loss')
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f'Non-finite backpropagation gradient for {name}')
    optimizer.step()
    return loss.item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', choices=DEFAULT_CONFIGS, default='1M')
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('runs/backprop'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, help='Override the base learning rate for the selected token budget.')
    parser.add_argument('--momentum', type=float, help='Override SGD momentum for the selected token budget.')
    parser.add_argument('--steps', type=int, help='Stop early; keep the full selected training subset.')
    add_eval_argument(parser)
    args = parser.parse_args()
    defaults = DEFAULT_CONFIGS[args.tokens]
    interval = eval_interval(args, defaults, parser)
    args.lr = defaults['lr'] if args.lr is None else args.lr
    args.momentum = defaults['momentum'] if args.momentum is None else args.momentum

    if not torch.cuda.is_available():
        parser.error('Training requires a CUDA GPU.')
    if int(os.environ.get('WORLD_SIZE', 1)) != 1:
        parser.error('This baseline uses one GPU; launch it with python, not torchrun.')
    if args.lr <= 0 or not 0 <= args.momentum < 1:
        parser.error('Use a positive learning rate and momentum in [0, 1).')
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')

    config = GPTConfig()
    steps = defaults['steps']
    if args.steps is not None and not 1 <= args.steps <= steps:
        parser.error(f'--steps must be between 1 and {steps}')
    paths = dict(
        train=args.data_dir / f'train_{args.tokens}.pt',
        val=args.data_dir / 'val.pt',
        test=args.data_dir / 'test.pt',
    )
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        parser.error(
            f"Missing data: {', '.join(missing)}. Run python -m replication.prepare_data "
            f"--tokens {args.tokens} --output {args.data_dir}"
        )
    training = load_sequences(paths['train'], steps * 8)
    validation = load_sequences(paths['val'], 544)
    testing = load_sequences(paths['test'], 544)
    order = torch.randperm(len(training), generator=torch.Generator().manual_seed(args.seed + 1))
    training = training[order]
    steps = args.steps or steps

    model = make_model(config, device, args.seed).requires_grad_(True)
    optimizer = make_optimizer(model, args.lr, args.momentum)
    compute = ComputeLedger(device)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Choose an empty output directory: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = dict(
        method='backprop', tokens=args.tokens, seed=args.seed, steps=steps,
        lr=args.lr, momentum=args.momentum, weight_decay=0.0,
        training_tokens=steps * 8 * config.sequence_len,
        validation_sequences=544, test_sequences=544, model=asdict(config),
        parameter_groups=[dict(lr=g['lr'], momentum=g['momentum'],
                               parameters=sum(p.numel() for p in g['params']))
                          for g in optimizer.param_groups],
        source_sha256=file_hash(Path(__file__)),
        model_source_sha256=file_hash(Path(__file__).resolve().parent / 'dust.py'),
        model_sha256=file_hash(Path(__file__).resolve().parent / 'model.py'),
        torch_version=torch.__version__,
        eval_interval=interval, flop_accounting=CONVENTION,
        flop_accounting_sha256=file_hash(Path(__file__).resolve().parents[1] / 'common/flops.py'),
        data_sha256={name: file_hash(path) for name, path in paths.items()},
    )
    (args.output / 'config.json').write_text(json.dumps(metadata, indent=2) + '\n')

    with compute.measure('val'):
        best = evaluate(model, validation, device)

    def save_best():
        torch.save(
            {name: value.detach().cpu() for name, value in model.state_dict().items()},
            args.output / 'best.pt',
        )

    save_best()
    history = [dict(step=0, val_loss=best, **compute.metrics())]
    (args.output / 'metrics.jsonl').write_text(json.dumps(history[0]) + '\n')
    compute.save(args.output / 'flops.json')
    start = time.monotonic()
    print(f'Backprop SGD: {args.tokens}, {steps} steps, val={best:.4f}', flush=True)

    for step in range(1, steps + 1):
        batch = training[(step - 1) * 8:step * 8].to(device)
        x, y = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
        with compute.measure('train', tokens=y.numel()):
            loss = train_step(model, optimizer, x, y)
        record = dict(step=step, train_loss=loss, elapsed_seconds=time.monotonic() - start)
        if step == 1 or step % interval == 0 or step == steps:
            with compute.measure('val'):
                value = evaluate(model, validation, device)
            record['val_loss'] = value
            if value < best:
                best = value
                save_best()
        record.update(compute.metrics())
        history.append(record)
        print(json.dumps(record), flush=True)
        with (args.output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        if 'val_loss' in record:
            compute.save(args.output / 'flops.json')

    model.load_state_dict(torch.load(args.output / 'best.pt', map_location=device, weights_only=True))
    with compute.measure('test'):
        test_loss = evaluate(model, testing, device)
    result = dict(best_val=best, test_at_best_val=test_loss, history=history, **compute.metrics())
    compute.save(args.output / 'flops.json')
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f"Best validation: {best:.4f}; test: {result['test_at_best_val']:.4f}", flush=True)


if __name__ == '__main__':
    main()
