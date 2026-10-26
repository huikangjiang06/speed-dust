# OPDT feasibility pilot: sentiment-guided DUST

This experiment compares ordinary DUST with a fixed, training-derived sentiment
perturbation distribution on Qwen3-0.6B / SST-2. **There is no agent in this pilot.**
It tests one proposed component of OPDT, not the complete framework or a faithful
reproduction of AGZO's benchmark configuration.

## Recorded experiment

- Same pretrained checkpoint (pinned in `prepare()`), FP32 master weights, BF16 forwards.
- SST-2: 1,024 training rows and 128 disjoint development rows, split seed 20261026.
- Official SST-2 validation is stored by preparation but **not evaluated** in this pilot.
- Prompt: `<review> It was`; single-token answers ` terrible` and ` great`.
- Objective: two-candidate classification cross-entropy. This differs from full-vocabulary next-token CE.
- Last 64 prompt tokens, batch 2, 100 updates, seed 42, identical batch schedule (seed 90210).
- SGD, momentum 0.9, constant learning rates 1e-6 and 1e-7.
- Eight perturbation draws per direct/local/head site, evaluated in groups of two.
- Initial activation RMS times 0.03 sets fixed site noise scales; head sigma 0.05.
- All matrix weights updated; normalization parameters frozen. No checkpoint selection.
- Semantic arm extracts directions from 32 positive and 32 negative training reviews
  at the final prompt token. These 64 extra example forwards are additional cost.

For unit difference-of-class-means direction `u` in dimension `d`, sample
`z = sqrt(.5) epsilon_perp + sqrt(.5 + .5*d) epsilon_parallel`.
This has covariance `C = .5 I + .5 d u u^T` and trace `d`, matching isotropic
expected squared noise. Reconstruction uses `C^-1 z`, not the uncorrected sample.
This 50% allocation was chosen manually, not learned. Directions remain fixed.
Semantic sampling applies to direct activation sites with a nonzero direction;
local Q/K/V and the head retain isotropic sampling.

Before training, a single training minibatch is used to compare DUST's estimate
with autograd. Autograd never supplies a training update. This diagnostic consumes
estimator RNG before training in both arms. Native/cached forwards and local
attention are checked at startup. These checks do not prove estimator correctness.

## Reproduce

From the repository root, in a CUDA environment:

```bash
python -m pip install -r experiments/opdt_sst2/requirements.txt
python -m experiments.opdt_sst2.pilot prepare
OPDT_GPUS=0,1,2,3 OMP_NUM_THREADS=4 python -m experiments.opdt_sst2.campaign
```

Four workers run concurrently (one per GPU). The controller stops remaining
workers after 27 minutes, records exit codes, and writes `OPDT-pilot.html`,
`curves.png`, `summary.json`, and `Claude-handoff.zip` inside this directory.
A nonzero worker exit is a failure or time-limit termination, not a completed run.
The recorded run used four RTX 3090 24GB GPUs and finished in about 15 minutes.
Do not point this launcher at occupied GPUs. Existing metric files are rejected;
archive prior outputs before repeating. The pilot does not save final weights.

Individual arm:

```bash
CUDA_VISIBLE_DEVICES=0 OPDT_LR=1e-6 python -m experiments.opdt_sst2.pilot semantic --steps 100
```

## Sentiment-axis audit

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.opdt_sst2.sentiment_audit semantic
python -m experiments.opdt_sst2.plot_sentiment
```

This reloads the original pretrained checkpoint, makes no updates, and extracts
three diagnostic sites: embedding, middle MLP output, final MLP output. It saves
original vectors/examples and checks (1) in-sample alignment, (2) leave-one-out
alignment, and (3) a different balanced set of 64 training examples. It writes
PNG/SVG plots and statistics. Cosines use activations centered on the class-mean
midpoint. The embedding direction is zero because the final cue token is identical.
AUC 0.5 means chance ordering. The independent examples are not official test data.

## Results and limitations

| Method | LR | Final dev accuracy | Final dev CE |
|---|---:|---:|---:|
| Ordinary | 1e-6 | 68.75% | 0.6847 |
| Semantic | 1e-6 | 68.75% | 0.6128 |
| Ordinary | 1e-7 | 62.50% | 0.6584 |
| Semantic | 1e-7 | 64.84% | 0.6363 |

Initial accuracy was 59.375%. Neither arm ended in complete label collapse,
but predictions remained biased toward positive. Results are one seed and only
128 development examples, with no final held-out evaluation. Lower final loss
does not establish general superiority. Initial full-gradient cosine alignment
was near zero (ordinary 0.00136, semantic -0.000339), and estimate norms greatly
exceeded true-gradient norms. More estimator validation is needed.

Independent sentiment-axis AUC was 0.596 at the middle MLP output and 0.411 at
the final MLP output, versus in-sample 0.861 and 0.705. Thus apparent in-sample
separation did not reliably generalize. Do not treat these directions as validated
semantic axes or the pilot as evidence that an agent improves DUST.

Small recorded summaries are in `results/`. Dataset text, cached models,
checkpoints, machine paths, and credentials are not included in the repository.

## Shared adapter changes

`posttraining/hf_dust.py` adds Qwen3 Q/K normalization before rotary embedding,
including the native activation dtype conversion, and an optional sampler `score`
method for covariance-corrected direct reconstruction. Existing samplers without
`score` retain their previous behavior. The SST-2 script replaces the token-loss
function only within its worker process, and supplies a two-answer head estimator.
