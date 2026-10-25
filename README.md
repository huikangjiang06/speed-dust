# DUST

DUST replication, pretrained-model SFT, and activation-GMM experiments.
Paper: <https://qlabs.sh/research/dust>. License: [MIT](LICENSE).

## Layout

```text
replication/     Paper model, DUST/backprop, data preparation, campaigns, plots
  tokenizer/     Fixed BPE tokenizer and optional tokenizer training
posttraining/    Hugging Face SFT and DUST
  geometry/      Activation caching, GMM fitting, calibration, perturbations
  configs/       Example experiment settings
  templates/     Chat templates
common/          Shared FLOP accounting
```

Run commands from the repository root with `python -m`.
Each pipeline has its own `requirements.txt`; use separate environments.

## Replication

```bash
python -m pip install -r replication/requirements.txt
python -m replication.prepare_data --tokens both --output data --revision 9bb295ddab0e05d785b879661af7260fed5140fc
python -m replication.dust --tokens 1M --population 256 --output runs/dust-1m-p256
python -m replication.backprop --tokens 1M --output runs/backprop-1m
```

`replication.train` provides resumable runs. `replication.campaign` coordinates
all seeded settings using `--gpus`; its data must be under `<root>/data`.
`replication.check`, `replication.summarize`, and `replication.plot` monitor,
aggregate, and plot saved campaign results. Each command supports `--help`.

## SFT and GMM DUST

The shared trainer supports `--method backprop`, `dust`, or `geometry_dust`.
Model, data, optimizer, population, and noise settings are in the example config.

```bash
python -m pip install -r posttraining/requirements.txt
python -m posttraining.sft_data --config posttraining/configs/smollm2_360m.json
python -m posttraining.calibrate --config posttraining/configs/smollm2_360m.json --stage scales --output runs/scales
python -m posttraining.sft --config posttraining/configs/smollm2_360m.json --method dust --set dust.noise.scales_file=../../runs/scales/scales.json --output runs/dust-sft
```

For activation caching, GMM fitting, and the three-method comparison:

```bash
python -m posttraining.geometry.campaign offline --config posttraining/configs/smollm2_360m.json --root runs/gmm-sft --gpus 0,1,2,3
python -m posttraining.geometry.campaign launch --root runs/gmm-sft --gpus 0,1,2,3
python -m posttraining.geometry.campaign check --root runs/gmm-sft
python -m posttraining.summarize_geometry --root runs/gmm-sft
```

Runs save configuration, checkpoints, losses, tokens, and estimated FLOPs.
