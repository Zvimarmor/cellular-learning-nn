# Three-factor learning vs. backprop

Node perturbation + eligibility traces + global reward, compared with backprop
on MLP / CNN / ResNet for MNIST and CIFAR-10, plus three mechanisms that shrink
the dimensionality of the perturbation (`--perturb-mode`).

| file | contents |
|---|---|
| `layers.py` | `PerturbedLinear`, `PerturbedConv2d`: noise injection (dense or top-K), trace storage, fused trace contraction (the conv trace derivation is in the `PerturbedConv2d` docstring) |
| `models.py` | `MLP`, `ShallowCNN`, `ResNet` (CIFAR ResNet-18 / -10) as sequences of *stages*, plus `AuxHead` for local rewards |
| `data.py` | MNIST / CIFAR-10 loaders |
| `train.py` | three-factor and backprop trainers, wandb logging, checkpoint / auto-resume / pre-emption handling |
| `test_layers.py`, `test_new_modes.py` | estimator correctness for every mode (21 tests, ~10 s on CPU) |
| `slurm/` | ELSC cluster scripts (see below) |

## The rule

For each perturbed layer, `z~ = Wx + b + xi`, `xi ~ N(0, sigma^2)`; trace `E = xi x^T`;
reward `R_b = L_b(noisy) - L_b(noise-free)`; update `dW = -(lr / sigma^2) * mean_b R_b E_b`.
To first order, `E[R xi x^T] / sigma^2 = dL/dW`. The step runs under `torch.no_grad()`
and `.backward()` is never called, except by the `--align-every` diagnostic.

## Perturbation modes

The estimator's variance grows with the number of units perturbed at once:
~266 per sample for the MLP, ~50k for the CNN, ~0.5M for ResNet-18. Each mode
cuts that number differently. Modes can be combined, e.g. `--perturb-mode layerwise,topk`.

| mode | what is perturbed / credited | the estimate converges to |
|---|---|---|
| `all` | every unit, one global reward | `dL/dW` |
| `layerwise` | one *group* per step (MLP/CNN: a layer; ResNet: stem, one residual block, or fc), chosen by `--layer-schedule cycle\|random` | `dL/dW` for that group; the other groups are untouched |
| `topk` | only the `--topk-ratio` most active units per sample in each hidden layer (`--topk-score abs\|pos`); the output layer stays dense | the *masked* gradient `sum_b (m_b * dL/dz_b) x_b^T` |
| `blockwise_reward` | stage `s` is scored by its own local loss `CE(head_s(h_s), y)`, with `--aux-head fixed` (frozen random projection) or `learned` (a perturbed linear head trained by the same reward). Each stage is re-run from the *clean* output of the previous stage, so a stage's reward sees only its own noise. | the gradient of each stage's local loss |

`test_new_modes.py` checks each row of this table against autograd, using a
mini-batch repeated n times (= n noise draws averaged in one step). It also
checks that layerwise beats `all` on hidden layers, that stage rewards don't
leak noise across stages, and that layerwise steps change only the active group.

## Local quick start

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python -m pytest -q
python train.py --method three_factor --dataset mnist --model mlp            # ~87% after 1 epoch
python train.py --method three_factor --dataset cifar10 --model cnn --optimizer adam --lr 1e-4 \
                --perturb-mode layerwise,topk --align-every 100
```

## ELSC cluster (`ssh elsc-login-direct`)

Nothing heavy runs on the login node. `train.py` and `slurm/run_single.sh` both refuse
to start there outside a SLURM job.

```bash
# once, on the login node
git clone <repo> && cd cellular-learning-nn
PY=/ems/elsc-labs/apps/python/3.11.13               # `module load python/3.11.13`; system python3 is 3.9
$PY/bin/python3 -m venv .venv && source .venv/bin/activate   # or conda: edit ENV_ACTIVATE
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # sm_75..sm_120 (no V100)
pip install numpy wandb pytest patchelf
# The module's python has no RPATH to its libpython3.11.so: give the venv its own copy that does,
# so .venv/bin/* runs without `module load` (cluster.env also sets LD_LIBRARY_PATH for jobs).
(cd .venv/bin && rm python python3 python3.11 && cp $PY/bin/python3.11 . && ./patchelf --set-rpath $PY/lib python3.11 \
  && ln -s python3.11 python3 && ln -s python3.11 python)
./slurm/check_cluster.sh          # read-only: partitions, GPU type names, your accounts
$EDITOR slurm/cluster.env         # PARTITION, GPU_GRES, ACCOUNT, ENV_ACTIVATE, WANDB_MODE
wandb login
./slurm/prefetch_data.sh          # downloads MNIST + CIFAR-10 to data/ (download only)

# experiments
DRY_RUN=1 ./slurm/launch_sweep.sh                  # print the task list
./slurm/launch_sweep.sh                            # slurm/grid.txt x seeds "0 1 2", max 8 GPUs at once
SEEDS="0" GRID=slurm/grid_smoke.txt ./slurm/launch_sweep.sh   # 10-minute smoke test first
sbatch slurm/submit_experiments.sbatch --dataset cifar10 --model resnet18 --perturb-mode layerwise   # one run
./slurm/dev_session.sh                             # interactive GPU shell inside tmux
```

| script | runs on | does |
|---|---|---|
| `cluster.env` | - | all cluster-specific settings (partition, GRES, env activation, paths, wandb) |
| `check_cluster.sh` | login | read-only `sinfo` / `sacctmgr` / `squeue` summary |
| `prefetch_data.sh` | login | dataset download; compute jobs use `--no-download` |
| `launch_sweep.sh` | login | verifies data, submits `grid.txt` x seeds as one throttled array |
| `sweep.sbatch` | compute | maps array index to a (grid line, seed) pair |
| `submit_experiments.sbatch` | compute | single run, arguments forwarded to `train.py` |
| `run_single.sh` | compute | env activation, optional data staging, signal forwarding, `train.py` |
| `dev_session.sh` | login, then compute | `tmux` + `srun --pty` so a disconnect doesn't kill the session |

**Disconnects, time limits, pre-emption.** `sbatch` jobs do not depend on your SSH
session. Each job asks for `--signal=B:USR1@300`. Five minutes before the time
limit, `run_single.sh` forwards the signal and `train.py` writes `runs/<name>/last.pt`
at the next step boundary, calls `scontrol requeue <job>_<task>`, and exits 0.
Pre-emption (SIGTERM) checkpoints the same way, and SLURM requeues the job itself
(`--requeue`). On restart `--auto-resume` (on by default) restores model, optimizer,
trainer state (EMA baseline, layer-wise schedule position) and RNG, and continues the
same wandb run. Logs append across restarts (`--open-mode=append`).

**wandb.** Run names and groups come from method / dataset / model / modes, e.g.
group `cifar10-resnet18-three_factor-layerwise+topk`, so all seeds of a config share
a group. If compute nodes have no internet, set `WANDB_MODE=offline` and later run
`wandb sync runs/*/wandb/offline-*` from the login node.

**Data staging.** `STAGE_DATA=1` copies the dataset to node-local `$TMPDIR` (or `/tmp`)
and deletes it at job end. It is off by default: torchvision's MNIST / CIFAR-10 load
the whole dataset into RAM at start-up, so NFS is read once per job anyway.

## Notes

- **Baseline.** `--baseline clean` (default) costs one extra forward pass.
  `--baseline ema` (`R = L - moving average`) leaves per-sample difficulty in `R`:
  MNIST MLP after 1 epoch reaches 26% vs. 87% with the clean baseline.
- **Memory.** Per-sample traces are never materialised. Only `x` and `xi` are kept,
  contracted with `conv2d_weight` / `addmm` once `R` is known, and freed after each
  micro-batch. `--micro-batch-size` bounds peak memory.
- **Normalisation.** BN affine parameters stay at identity in three-factor mode.
  BN batch statistics couple samples in a batch; `--norm gn` avoids this.
