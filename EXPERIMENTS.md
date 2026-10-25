# Biologically inspired mechanisms for three-factor credit assignment

Seven mechanisms tested against the node-perturbation baseline on CIFAR-10. Raw
per-run records: `experiments/results.jsonl`; reproduce any row with `bench.py`.

## Protocol

- **References** (3 seeds each):
  - CNN, top-k 5% + local rewards, Adam 3e-4, 3 epochs
  - CNN, layer-wise, same settings
  - ResNet-18, layer-wise, Adam 1e-4, 1 epoch
- **Mechanisms**: each was added on top of a reference behind a flag that is off by
  default, and benchmarked with 3 seeds on a local MPS GPU, from a frozen copy of
  the code.
- **Metrics**: final val accuracy; loss AUC (mean training loss over the run, lower =
  faster learning); and three alignments with the true gradient of the global loss,
  measured every 20 steps:
  - `align/total`: the estimate, over the layers updated this step
  - `align/full`: the estimate, over all layers
  - `align_upd`: **the step actually applied**, over all layers. This is the decisive
    one, because it includes the optimizer and anything a mechanism does to the update.
- **Decision rule**: a mechanism is kept only if accuracy beats the reference with
  non-overlapping seed ranges, or alignment improves clearly without hurting accuracy.
  Otherwise its code is reverted; its runs stay in the results file.
- **Noise floor**: seed spread (± = half the range) is ±1.9 points on the CNN and
  ±1.9 on 1-epoch ResNet-18. One-epoch ResNet-18 runs sit near chance, so they
  cannot rank mechanisms; see the 30-epoch cluster sweep below.

## Results

| # | Mechanism (bio analogue) | Implementation | CNN val acc (ref) | ResNet-18 val acc (ref 13.9) | `align_upd` CNN / ResNet (ref) | Verdict |
|---|---|---|---|---|---|---|
| 1 | Activity-driven adaptive scheduling (novelty-triggered plasticity) | `--layer-schedule adaptive`: perturb group g with p ∝ EMA of √E[R²] while g is perturbed (= σ‖∂L/∂z_g‖), plus a uniform floor | **28.3 ±1.9** (layer-wise 23.2 ±2.9) | 13.1 ±2.3 | **0.061 / 0.019** (0.002 / 0.004) | **Kept** |
| 2 | Dopaminergic gating of plasticity on surprise | `--surprise-gate 1`: use R_b only if \|R_b\| > 1 × running RMS(R) per reward stream | 34.8 ±0.9 (36.4 ±1.9) | 13.8 ±2.1 | 0.041 / 0.004 (0.049 / 0.004) | Reverted |
| 3 | Homeostatic synaptic scaling | `--homeostasis η`: rescale each unit's weights by (set-point / rate)^η | 30.6 ±3.6 (η=0.1), 33.9 ±0.4 (η=0.01) | 14.0 ±2.0 | 0.010, 0.025 / 0.000 | Reverted |
| 4 | Lingering eligibility (synaptic tag) | `--trace-decay 0.9`: every layer updated each step with a decaying R·E trace | 18.3 ±1.3, unstable (23.2) | 10.6 ±0.3 | 0.006 / 0.001 | Reverted |
| 5 | Lateral inhibition before top-k (GABAergic surround) | `--topk-score wta`: divisive normalization over a 3×3 × channel surround before top-k | 35.7 ±2.8 (36.4) | 12.7 ±1.1 (layer-wise+top-k ref 13.5 ±1.1) | 0.049 / 0.000 | Reverted (neutral) |
| 6 | Metaplasticity: fast + slow synaptic states | `--slow-every 5 --slow-alpha 0.5` (Lookahead); evaluate slow weights | 34.2 ±1.1; 35.5 ±2.2 at 2× lr | 12.8 ±1.9 | 0.054, 0.051 / 0.002 | Reverted (neutral) |
| 7 | Structural selection of consistent synapses | `--consistency-mask 0.4`: suppress updates where \|EMA(update)\| / EMA(\|update\|) < 0.4 | 32.1 ±3.8 | 13.1 ±2.2 | 0.023 / 0.004 | Reverted |
| – | Stack: #1 on the best CNN mix | layer-wise + top-k + local rewards, adaptive vs. round-robin | **33.5 ±0.9** vs. 27.3 ±1.0 | – | **0.063** vs. −0.003 | #1 replicates |

Controls:
- **#4:** a plain larger learning rate without the trace fails the same way (CNN
  19.9%, ResNet 11.9%). The trace acts as a larger effective step on noise.
- **#6:** with 2× learning rate it matches the reference. The α-averaging slows
  progress without improving the direction.

## What the results say

The bottleneck is **information per step**. Each step's update is one
random-direction probe whose signal-to-noise ratio scales like √(batch / perturbed
units). Mechanisms help only if they put that probe where the signal is.

- **Adaptive scheduling (#1) works for this reason.** The reward magnitude while a
  region is perturbed is a free, local readout of that region's error ‖∂L/∂z‖, so
  the perturbation budget goes where the expected signal is largest. On the CNN it
  raises the alignment of the applied step about 30× and accuracy by ~5–6 points,
  in two independent settings.
- **The ResNet-18 caveat.** There the score adds up over all units of a block, so
  it favours the high-resolution early blocks, which have the most units and the
  worst SNR. A per-unit score (√(E[R²] / units)) is the natural next variant.
- **Mechanisms that only filter or reshape the same probe add no information**:
  gating (#2), inhibition (#5), consistency masks (#7), fast/slow averaging (#6).
  For Gaussian probes, gating on |R| keeps the expected direction but discards signal
  power (predicted before running: SNR × ~0.89 at a 1σ threshold). Adam's m/√v
  already acts as a per-synapse consistency filter, which leaves #7 little to add.
- **Mechanisms that re-apply or constrain updates hurt.** Homeostasis (#3) fights
  learning: the growth in deep-layer activity it suppresses *is* the learned signal,
  and harm grows with η. Decaying traces (#4) re-apply the same noisy direction
  about 1/(1−τ) times.

## Longer runs

**50 epochs, existing ELSC sweep (array 63065917), 3-seed means:**

| | MLP (MNIST) | CNN | ResNet-10 | ResNet-18 |
|---|---|---|---|---|
| backprop | 97.9 | 82.0 | 82.9 | 87.5 |
| all | 87.8 | 29.5 | 25.7 | 18.2 |
| layer-wise | 92.2 | 32.5 | 23.5 | 21.1 |
| top-k | 93.9 | 42.5 | – | **25.3** |
| local rewards | 91.2 | 35.4 | – | 18.3 |
| top-k + local rewards | – | **44.9** | – | – |
| layer-wise + top-k | – | 40.0 | – | 21.5 |
| layer-wise + top-k + local rewards | – | – | 22.4 | – |

One-epoch rankings did not predict 50-epoch rankings on ResNet-18: there, top-k
alone wins.

**30 epochs, ResNet-18, mechanism #1 (array 64257362):** results pending; see
`slurm/grid_resnet_bio.txt`.
