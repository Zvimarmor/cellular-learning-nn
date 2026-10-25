"""Train with the three-factor rule (node perturbation) or a backprop baseline.

Examples
--------
    python train.py --method three_factor --dataset mnist --model mlp
    python train.py --method backprop --dataset cifar10 --model cnn --optimizer adam --lr 1e-3
    python train.py --method three_factor --dataset cifar10 --model resnet18 --perturb-mode layerwise,topk
    python train.py --method three_factor --dataset cifar10 --model cnn --perturb-mode blockwise_reward

Perturbation modes (``--perturb-mode``, comma-separated, combinable):
    all               every unit of every layer is perturbed each step (the original rule)
    layerwise         one perturbation group (layer / residual block) per step
    topk              noise only on the --topk-ratio most active units of each hidden layer
    blockwise_reward  each stage is credited by a local loss from an auxiliary head
                      on its own output, with the stage's input taken from the clean pass

Run ``python train.py -h`` for all options.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import get_loaders
from models import build_model

try:
    import wandb
except ImportError:  # allow running without wandb installed
    wandb = None

PERTURB_MODES = ("all", "layerwise", "topk", "blockwise_reward")
ALIGN_KEYS = ("align/total", "align/full", "align_update/full")


# --------------------------------------------------------------------------- utils

def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def default_workers() -> int:
    # On SLURM respect the CPU allocation rather than the node's core count.
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    return max(0, min(8, cpus - 1))


def guard_login_node(pattern: str) -> None:
    """Refuse to train on a cluster login node (outside any SLURM allocation)."""
    host = socket.gethostname()
    if pattern and re.search(pattern, host) and "SLURM_JOB_ID" not in os.environ:
        sys.exit(f"refusing to train on login node {host!r}: submit with sbatch (slurm/) or run inside "
                 f"`srun --pty`. Override with --login-node-regex ''.")


class UpdateTracker:
    """Measures ||dW|| = ||W_after - W_before|| per perturbed layer.

    Taking the difference of parameters (rather than recomputing the rule) makes
    the metric identical for backprop and three-factor runs and for every optimizer.
    Snapshots are only taken on logging steps to keep the memory/time cost low.
    """

    def __init__(self, model):
        self.layers = model.named_perturbed_layers()
        self.before = None

    def snapshot(self):
        self.before = [m.weight.detach().clone() for _, m in self.layers]

    def measure(self) -> dict:
        logs, total_sq = {}, 0.0
        for (name, m), w0 in zip(self.layers, self.before):
            d = (m.weight.detach() - w0).norm().item()
            total_sq += d * d
            logs[f"update_norm/{name}"] = d
            logs[f"update_ratio/{name}"] = d / (w0.norm().item() + 1e-12)
        logs["update_norm/total"] = math.sqrt(total_sq)
        self.before = None
        return logs


class PreemptionHandler:
    """Turns SLURM's pre-timeout / pre-emption signals into a checkpoint + requeue.

    sbatch ``--signal=USR1@300`` delivers SIGUSR1 five minutes before the time
    limit; pre-emption delivers SIGTERM. The handler only sets a flag, and the
    training loop checkpoints at the next step boundary (never mid-update).
    """

    def __init__(self):
        self.signum = None
        for sig in (signal.SIGUSR1, signal.SIGTERM):
            signal.signal(sig, self._handle)

    def _handle(self, signum, frame):
        self.signum = signum

    @property
    def triggered(self) -> bool:
        return self.signum is not None

    def requeue(self) -> None:
        # On a time-limit warning the job must requeue itself; on pre-emption
        # SLURM requeues --requeue jobs on its own.
        if self.signum != signal.SIGUSR1 or "SLURM_JOB_ID" not in os.environ:
            return
        job = os.environ["SLURM_JOB_ID"]
        if "SLURM_ARRAY_TASK_ID" in os.environ:
            job = f"{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ['SLURM_ARRAY_TASK_ID']}"
        subprocess.run(["scontrol", "requeue", job], check=False)


def make_optimizer(args, model):
    if args.optimizer == "manual":
        return None
    params = [p for p in model.parameters() if p.requires_grad]
    if args.method == "three_factor":
        # Only perturbed layers (and learned aux heads) receive an update signal.
        params = [p for m in model.perturbed_layers() + model.aux_layers() for p in m.parameters(recurse=False)]
    if args.optimizer == "sgd":
        return torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.optimizer == "adam":
        return torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    raise ValueError(args.optimizer)


# --------------------------------------------------------------------------- three-factor step

class ThreeFactorTrainer:
    """Node perturbation with eligibility traces and a global (or stage-local) reward.

    No ``.backward()`` is called: the whole step runs under ``torch.no_grad()``,
    so autograd neither builds a graph nor touches the weights.
    """

    def __init__(self, model, args, optimizer):
        self.model, self.args, self.opt = model, args, optimizer
        self.layerwise = "layerwise" in args.perturb_modes
        self.blockwise = "blockwise_reward" in args.perturb_modes
        self.groups = model.perturbation_groups()
        self.stage_of = model.stage_index()
        self.loss_ema: float | None = None   # moving-average baseline L_bar
        self.t = 0                           # step counter (round-robin schedule)
        self.rng = random.Random(args.seed)
        # Adaptive schedule: running "local error" per group, EMA of sqrt(E[R^2]) measured
        # while that group was perturbed. None = not visited yet.
        self.group_score: list[float | None] = [None] * len(self.groups)

    def state_dict(self):
        return {"loss_ema": self.loss_ema, "t": self.t, "rng": self.rng.getstate(),
                "group_score": self.group_score}

    def load_state_dict(self, state):
        self.loss_ema, self.t = state["loss_ema"], state["t"]
        self.rng.setstate(state["rng"])
        self.group_score = state.get("group_score", self.group_score)

    # -- which layers are perturbed this step ---------------------------------
    def select_active(self):
        """Returns (perturbed main-network layers, group index or -1)."""
        if not self.layerwise:
            return self.model.perturbed_layers(), -1
        if self.args.layer_schedule == "cycle":
            g = self.t % len(self.groups)
        elif self.args.layer_schedule == "adaptive":
            g = self.rng.choices(range(len(self.groups)), weights=self.schedule_probs())[0]
        else:
            g = self.rng.randrange(len(self.groups))
        return self.groups[g], g

    def schedule_probs(self) -> list[float]:
        """Activity-driven scheduling: perturb a region with probability proportional to
        its recent local error, mixed with a uniform floor.

        When only group g is perturbed, R ~= xi . dL/dz_g, so E[R^2] = sigma^2 ||dL/dz_g||^2:
        the reward magnitude while a region is perturbed measures that region's error
        signal at no extra cost. Unvisited groups get the current maximum (optimism),
        so each group is tried at least once early on.
        """
        known = [s for s in self.group_score if s is not None]
        top = max(known) if known else 1.0
        scores = [top if s is None else s for s in self.group_score]
        total = sum(scores) or 1.0
        eps, n = self.args.adaptive_eps, len(scores)
        return [(1 - eps) * s / total + eps / n for s in scores]

    # -- rewards ----------------------------------------------------------------
    def _finalize_reward(self, r):
        if self.args.reward == "batch":
            # One scalar for the whole batch (the strictly "global" broadcast signal).
            r = r.mean().expand_as(r)
        return r

    def _global_rewards(self, xb, yb, active):
        """One reward for every layer: R = L(noisy) - baseline, from the network output."""
        a, model = self.args, self.model
        loss_clean = logits_clean = None
        if a.baseline == "clean":
            # Paired baseline: same sample, same weights, no noise. Removes the
            # sample-to-sample loss variance from R, leaving only the effect of xi.
            model.set_perturb(False)
            logits_clean = model(xb)
            loss_clean = F.cross_entropy(logits_clean, yb, reduction="none")
        model.set_perturb(True, active)
        logits = model(xb)
        loss_noisy = F.cross_entropy(logits, yb, reduction="none")
        if loss_clean is not None:
            r = loss_noisy - loss_clean
        else:
            r = loss_noisy - (self.loss_ema if self.loss_ema is not None else loss_noisy.mean().item())
        r = self._finalize_reward(r)
        # Report clean-network stats when available, otherwise the noisy ones.
        ref_logits, ref_loss = (logits_clean, loss_clean) if loss_clean is not None else (logits, loss_noisy)
        return {m: r for m in active}, ref_logits, ref_loss

    def _stage_rewards(self, xb, yb, active):
        """Blockwise local rewards.

        Clean pass: h_0 = x, h_s = stage_s(h_{s-1}), local loss L_s = CE(head_s(h_s), y)
        (the last stage uses the real classifier output). Noisy pass: every stage
        that has perturbed layers is re-run *from its clean input* h_{s-1}, so
        R_s = L_s(noisy) - L_s(clean) only contains noise injected inside stage s
        (plus its head). Credit-assignment paths are one stage long, and all
        stages are still evaluated in a single noisy sweep.
        """
        model, stages = self.model, self.model.stages
        last = len(stages) - 1

        def stage_loss(s, out):
            logits = out if s == last else model.aux_heads[s](out)
            return F.cross_entropy(logits, yb, reduction="none")

        active_stages = sorted({self.stage_of[m] for m in active})
        layers = list(active) + [m for m in model.aux_layers() if self.stage_of[m] in active_stages]

        model.set_perturb(False)
        h, clean_out, loss_clean = xb, [], {}
        for s, stage in enumerate(stages):
            h = stage(h)
            clean_out.append(h)
            if s in active_stages:
                loss_clean[s] = stage_loss(s, h)

        model.set_perturb(True, layers)
        rewards = {}
        for s in active_stages:
            inp = xb if s == 0 else clean_out[s - 1]
            r = self._finalize_reward(stage_loss(s, stages[s](inp)) - loss_clean[s])
            rewards.update({m: r for m in layers if self.stage_of[m] == s})
        logits_clean = clean_out[-1]
        return rewards, logits_clean, F.cross_entropy(logits_clean, yb, reduction="none")

    # -- estimate ------------------------------------------------------------------
    @torch.no_grad()
    def estimate(self, x, y, active) -> dict:
        """Fill weight_est / bias_est of the perturbed layers with mean_b R_b E_b / sigma^2.

        Returns the layers that received an estimate and training statistics.
        """
        a, model = self.args, self.model
        B = x.size(0)
        mb = a.micro_batch_size or B
        estimated: dict = {}

        loss_sum, correct, r_abs, r_sq, units = 0.0, 0, 0.0, 0.0, 0.0
        # Micro-batching: traces (x, xi) for every layer are alive only for one
        # micro-batch; the contracted estimate (parameter-sized) is accumulated.
        # Peak memory therefore scales with micro_batch_size, not batch_size.
        for i, (xb, yb) in enumerate(zip(x.split(mb), y.split(mb))):
            fn = self._stage_rewards if self.blockwise else self._global_rewards
            rewards, logits, loss = fn(xb, yb, active)
            if i == 0:
                for m in rewards:
                    m.zero_estimate()
                    estimated[m] = None
                # Effective dimensionality of the perturbation: noisy units per sample.
                units = sum((m._xi != 0).sum().item() for m in rewards) / xb.size(0)
            for m, r in rewards.items():
                # Per-layer 1/sigma_m^2: the estimate stays unbiased if layers use different sigma.
                m.accumulate_estimate(r, 1.0 / (B * m.sigma ** 2))
            model.clear_traces()

            loss_sum += loss.sum().item()
            correct += (logits.argmax(1) == yb).sum().item()
            unique = {id(r): r for r in rewards.values()}.values()
            r_abs += sum(r.abs().sum().item() for r in unique)
            r_sq += sum(r.pow(2).sum().item() for r in unique)
        model.set_perturb(False)

        batch_loss = loss_sum / B
        if a.baseline == "ema":
            self.loss_ema = batch_loss if self.loss_ema is None else (
                a.ema_decay * self.loss_ema + (1 - a.ema_decay) * batch_loss)
        logs = {"train/loss": batch_loss, "train/acc": correct / B, "tf/abs_reward": r_abs / B, "tf/reward_sq": r_sq / B,
                "tf/perturbed_units": units}
        if self.loss_ema is not None:
            logs["tf/loss_baseline"] = self.loss_ema
        return list(estimated), logs

    def step(self, x, y, measure_alignment: bool = False) -> dict:
        active, group = self.select_active()
        layers, logs = self.estimate(x, y, active)
        if group >= 0:
            logs["tf/active_group"] = group
            score = math.sqrt(logs["tf/reward_sq"])
            if self.args.adaptive_score == "per_unit":
                # sqrt(E[R^2] / units) = sigma * RMS of dL/dz over the group's units: big
                # blocks no longer win just by having more units.
                score /= math.sqrt(max(logs["tf/perturbed_units"], 1.0))
            old = self.group_score[group]
            d = self.args.adaptive_decay
            self.group_score[group] = score if old is None else d * old + (1 - d) * score

        if self.args.clip_update > 0:
            total = math.sqrt(sum(m.weight_est.pow(2).sum().item() for m in layers))
            if total > self.args.clip_update:
                for m in layers:
                    m.weight_est.mul_(self.args.clip_update / total)
                    if m.bias_est is not None:
                        m.bias_est.mul_(self.args.clip_update / total)
            logs["tf/estimate_norm"] = total

        grads = None
        if measure_alignment:
            grads = self.true_gradients(x, y)
            logs.update(self.alignment(grads, [m for m in layers if m in grads]))
            before = {m: m.weight.detach().clone() for m in grads}

        self.apply_update(layers)
        self.t += 1

        if grads is not None:
            # Alignment of the step actually taken (optimizer state, schedules and any
            # mechanism acting on the update included), over *all* layers: does the
            # whole network move down the true loss gradient?
            dot, n1, n2 = 0.0, 0.0, 0.0
            for m, g in grads.items():
                d = (before[m] - m.weight.detach()).flatten()   # -dW
                dot += (g * d).sum().item()
                n1 += g.pow(2).sum().item()
                n2 += d.pow(2).sum().item()
            logs["align_update/full"] = dot / (math.sqrt(n1 * n2) + 1e-12)
        return logs

    @torch.no_grad()
    def apply_update(self, layers):
        if self.opt is None:
            # The rule exactly as written: dW = -eta * R * E / sigma^2
            # (weight_est already holds mean_b R_b E_b / sigma^2).
            for m in layers:
                m.weight.add_(m.weight_est, alpha=-self.args.lr)
                if m.bias_est is not None:
                    m.bias.add_(m.bias_est, alpha=-self.args.lr)
        else:
            # Same estimate, fed into SGD-momentum / Adam as if it were a gradient.
            # Layers not perturbed this step keep grad=None, so the optimizer skips them.
            for m in layers:
                m.weight.grad = m.weight_est.clone()
                if m.bias is not None:
                    m.bias.grad = m.bias_est.clone()
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)

    def true_gradients(self, x, y) -> dict:
        """dL/dW of the *global* loss for every main-network layer, via autograd on the
        clean network. Diagnostic only, never used for learning."""
        model = self.model
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            F.cross_entropy(model(x), y).backward()
        grads = {m: m.weight.grad.flatten().clone() for m in model.perturbed_layers()}
        model.zero_grad(set_to_none=True)
        return grads

    def alignment(self, grads, layers) -> dict:
        """Cosine similarity between the estimate and the true global gradient.

        ``align/total``: over the layers updated this step (in layerwise mode, the
        active group). ``align/full``: over all layers, counting layers without an
        estimate this step as zero. In blockwise_reward mode the estimate targets local
        losses, so both measure agreement with end-to-end backprop.
        """
        names = {m: n for n, m in self.model.named_perturbed_layers()}
        logs, dot, n1, n2 = {}, 0.0, 0.0, 0.0
        for m in layers:
            g, e = grads[m], m.weight_est.flatten()
            logs[f"align/{names[m]}"] = F.cosine_similarity(g, e, dim=0).item()
            dot += (g * e).sum().item()
            n1 += g.pow(2).sum().item()
            n2 += e.pow(2).sum().item()
        g_all = sum(g.pow(2).sum().item() for g in grads.values())
        logs["align/total"] = dot / (math.sqrt(n1 * n2) + 1e-12)
        logs["align/full"] = dot / (math.sqrt(g_all * n2) + 1e-12)
        return logs


# --------------------------------------------------------------------------- backprop step

class BackpropTrainer:
    def __init__(self, model, args, optimizer):
        if optimizer is None:
            raise ValueError("backprop needs --optimizer sgd or adam")
        self.model, self.opt = model, optimizer

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass

    def step(self, x, y, measure_alignment: bool = False) -> dict:
        self.model.set_perturb(False)
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return {"train/loss": loss.item(), "train/acc": (logits.argmax(1) == y).float().mean().item()}


# --------------------------------------------------------------------------- eval / checkpoint

@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    model.set_perturb(False)
    loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        n += y.numel()
    model.train()
    return {"val/loss": loss / n, "val/acc": correct / n}


def save_checkpoint(path: Path, model, opt, trainer, epoch, step, wandb_id):
    """``epoch`` is the last *completed* epoch; resuming starts at epoch + 1."""
    tmp = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict() if opt is not None else None,
        "trainer": trainer.state_dict(),
        "epoch": epoch, "step": step, "wandb_id": wandb_id,
        "rng": torch.get_rng_state(),
    }, tmp)
    os.replace(tmp, path)  # atomic: a pre-empted job never leaves a half-written checkpoint


# --------------------------------------------------------------------------- main

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=["three_factor", "backprop"], default="three_factor")
    p.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    p.add_argument("--model", default="mlp", help="mlp | cnn | resnet18 | resnet10")
    p.add_argument("--width", type=int, default=None, help="hidden size (mlp) / base channels (cnn, resnet)")
    p.add_argument("--norm", choices=["bn", "gn", "none"], default="bn", help="ResNet normalisation")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (0 = no limit)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=None, help="default: 1e-2 (three_factor), 1e-3 (backprop)")
    p.add_argument("--optimizer", choices=["manual", "sgd", "adam"], default=None,
                   help="three_factor default: manual (dW = -lr R E / sigma^2); backprop default: adam")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--augment", action="store_true", help="CIFAR-10 random crop + flip")

    tf = p.add_argument_group("three-factor")
    tf.add_argument("--sigma", type=float, default=0.05, help="std of injected node noise")
    tf.add_argument("--baseline", choices=["ema", "clean"], default="clean",
                    help="clean: R = L(noisy) - L(noise-free), one extra forward pass; "
                         "ema: R = L - moving avg(L). With ema, R is dominated by per-sample difficulty "
                         "(O(1)) rather than the effect of the noise (O(sigma)), so it needs ~100x lower lr "
                         "and learns far slower (MNIST MLP, 1 epoch: 26%% vs 87%%)")
    tf.add_argument("--ema-decay", type=float, default=0.99)
    tf.add_argument("--reward", choices=["per_sample", "batch"], default="per_sample",
                    help="per_sample: R_b per example; batch: one scalar R for the whole batch")
    tf.add_argument("--perturb-mode", default="all",
                    help=f"comma-separated subset of {', '.join(PERTURB_MODES)} (e.g. layerwise,topk)")
    tf.add_argument("--layer-schedule", choices=["cycle", "random", "adaptive"], default="cycle",
                    help="layerwise: round-robin, uniformly random, or adaptive (probability "
                         "proportional to each group's recent local error sqrt(E[R^2]))")
    tf.add_argument("--adaptive-score", choices=["total", "per_unit"], default="total",
                    help="adaptive: group score sqrt(E[R^2]) or sqrt(E[R^2] / perturbed units)")
    tf.add_argument("--adaptive-eps", type=float, default=0.2,
                    help="adaptive schedule: weight of the uniform floor in the group probabilities")
    tf.add_argument("--adaptive-decay", type=float, default=0.9,
                    help="adaptive schedule: EMA decay of each group's error score (per visit)")
    tf.add_argument("--topk-ratio", type=float, default=0.05, help="topk: fraction of units perturbed per layer")
    tf.add_argument("--topk-score", choices=["abs", "pos"], default="abs",
                    help="topk: rank units by |z| or by z (most ReLU-active)")
    tf.add_argument("--aux-head", choices=["fixed", "learned"], default="fixed",
                    help="blockwise_reward: fixed random projection or a learned (perturbed) linear head")
    tf.add_argument("--micro-batch-size", type=int, default=0,
                    help="process the batch in chunks to bound trace memory (0 = whole batch)")
    tf.add_argument("--clip-update", type=float, default=0.0, help="clip global norm of the estimate (0 = off)")
    tf.add_argument("--align-every", type=int, default=0,
                    help="log cosine(estimate, true grad) every N steps (0 = off)")

    io = p.add_argument_group("infrastructure")
    io.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    io.add_argument("--seed", type=int, default=0)
    io.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "./data"))
    io.add_argument("--out-dir", default=os.environ.get("OUT_DIR", "./runs"))
    io.add_argument("--num-workers", type=int, default=None)
    io.add_argument("--no-download", action="store_true", help="fail instead of downloading datasets")
    io.add_argument("--log-every", type=int, default=50)
    io.add_argument("--run-name", default=None)
    io.add_argument("--tag", default="",
                    help="suffix for the derived run name and wandb group, to tell apart variants of "
                         "the same method/modes (e.g. a schedule) and separate sweeps")
    io.add_argument("--wandb-project", default="three-factor-learning")
    io.add_argument("--wandb-entity", default=None)
    io.add_argument("--wandb-mode", choices=["online", "offline", "disabled"],
                    default=os.environ.get("WANDB_MODE", "online"))
    io.add_argument("--wandb-group", default=None, help="default: <dataset>-<model>-<method>[-<modes>]")
    io.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=True,
                    help="resume from <out-dir>/<run-name>/last.pt if it exists")
    io.add_argument("--login-node-regex", default=os.environ.get("LOGIN_NODE_REGEX", r"^(bava|loginserver|login)"),
                    help="refuse to train on hosts matching this outside a SLURM job ('' disables)")
    args = p.parse_args(argv)

    modes = {m.strip() for m in args.perturb_mode.split(",") if m.strip()}
    if not modes or not modes <= set(PERTURB_MODES):
        p.error(f"--perturb-mode must be a comma-separated subset of {PERTURB_MODES}")
    if "all" in modes and len(modes) > 1:
        p.error("--perturb-mode 'all' cannot be combined with other modes")
    if args.method == "backprop" and modes != {"all"}:
        p.error("--perturb-mode applies to --method three_factor only")
    if "blockwise_reward" in modes and args.baseline != "clean":
        p.error("blockwise_reward needs --baseline clean (stage losses are paired with the clean pass)")
    if not 0 < args.topk_ratio <= 1:
        p.error("--topk-ratio must be in (0, 1]")
    args.perturb_modes = modes
    mode_tag = "+".join(m for m in PERTURB_MODES if m in modes)

    if args.optimizer is None:
        args.optimizer = "manual" if args.method == "three_factor" else "adam"
    if args.method == "backprop" and args.optimizer == "manual":
        p.error("backprop requires --optimizer sgd or adam")
    if args.lr is None:
        args.lr = 1e-2 if args.method == "three_factor" else 1e-3
    if args.num_workers is None:
        args.num_workers = default_workers()
    tag = args.method if args.method == "backprop" else f"{args.method}-{mode_tag}"
    if args.tag:
        tag = f"{tag}-{args.tag}"
    if args.wandb_group is None:
        args.wandb_group = f"{args.dataset}-{args.model}-{tag}"
    if args.run_name is None:
        args.run_name = f"{tag}-{args.dataset}-{args.model}-s{args.seed}"
    return args


def main(argv=None):
    args = parse_args(argv)
    guard_login_node(args.login_node_regex)
    seed_everything(args.seed)
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_loader, test_loader, in_shape, n_cls = get_loaders(
        args.dataset, args.data_dir, args.batch_size, args.num_workers, args.augment,
        pin_memory=device.type == "cuda", download=not args.no_download)

    model = build_model(args.model, in_shape, n_cls, sigma=args.sigma, width=args.width, norm=args.norm)
    if args.method == "three_factor":
        if "topk" in args.perturb_modes:
            model.set_topk(args.topk_ratio, args.topk_score)
        if "blockwise_reward" in args.perturb_modes:
            # Built after the main network so its initialisation matches the other modes.
            model.build_aux_heads(args.aux_head, n_cls, args.sigma)
    model = model.to(device)
    opt = make_optimizer(args, model)
    trainer = (ThreeFactorTrainer if args.method == "three_factor" else BackpropTrainer)(model, args, opt)
    tracker = UpdateTracker(model)

    # ---- checkpoint / resume (SLURM jobs get pre-empted and requeued)
    out_dir = Path(args.out_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "last.pt"
    start_epoch, step, wandb_id = 0, 0, None
    if ckpt_path.exists() and args.auto_resume:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if opt is not None and ckpt["opt"] is not None:
            opt.load_state_dict(ckpt["opt"])
        trainer.load_state_dict(ckpt["trainer"])
        torch.set_rng_state(ckpt["rng"].cpu())  # map_location moved it to the device
        start_epoch, step, wandb_id = ckpt["epoch"] + 1, ckpt["step"], ckpt["wandb_id"]
        print(f"resumed from {ckpt_path} at epoch {start_epoch}, step {step}")

    n_params = sum(p.numel() for p in model.parameters())
    config = {**vars(args), "perturb_modes": sorted(args.perturb_modes), "n_params": n_params,
              "device": str(device), "host": socket.gethostname(),
              "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID")}
    run = None
    if wandb is not None and args.wandb_mode != "disabled":
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.run_name,
                         group=args.wandb_group, config=config, mode=args.wandb_mode, dir=str(out_dir),
                         id=wandb_id, resume="allow")
        wandb_id = run.id
        wandb.define_metric("epoch")
        wandb.define_metric("val/*", step_metric="epoch")
    print(f"{args.run_name}: {n_params / 1e6:.2f}M params on {device}", flush=True)

    def log(d):
        if run is not None:
            run.log(d, step=step)

    preempt = PreemptionHandler()
    model.train()
    done, val = False, None
    history = {k: [] for k in ("train/loss",) + ALIGN_KEYS}
    val_acc = []
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            log_now = step % args.log_every == 0
            align_now = args.align_every > 0 and step % args.align_every == 0 and args.method == "three_factor"
            if log_now:
                tracker.snapshot()
            logs = trainer.step(x, y, measure_alignment=align_now)
            history["train/loss"].append(logs["train/loss"])
            if align_now:
                for k in ALIGN_KEYS:
                    history[k].append(logs[k])
                if not log_now:
                    log({k: v for k, v in logs.items() if k.startswith("align")})
            if log_now:
                logs.update(tracker.measure())
                if device.type == "cuda":
                    logs["sys/max_mem_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
                if not math.isfinite(logs["train/loss"]):
                    raise RuntimeError(f"loss diverged at step {step}: {logs['train/loss']}")
                log({**logs, "epoch": epoch})
                print(f"ep {epoch} step {step} loss {logs['train/loss']:.4f} acc {logs['train/acc']:.3f} "
                      f"|dW| {logs['update_norm/total']:.2e}"
                      + (f" align {logs['align/total']:.3f}" if "align/total" in logs else ""), flush=True)
            step += 1
            if preempt.triggered:
                # Mid-epoch: record epoch - 1 as completed, so the current epoch restarts on resume.
                save_checkpoint(ckpt_path, model, opt, trainer, epoch - 1, step, wandb_id)
                print(f"signal {preempt.signum}: checkpointed at step {step}, requeueing", flush=True)
                if run is not None:
                    run.mark_preempting()
                    run.finish(exit_code=0)
                preempt.requeue()
                sys.exit(0)
            if args.max_steps and step >= args.max_steps:
                done = True
                break

        val = evaluate(model, test_loader, device)
        val["epoch"] = epoch
        val["time/epoch_s"] = time.time() - t0
        log(val)
        print(f"== epoch {epoch}: val loss {val['val/loss']:.4f} val acc {val['val/acc']:.4f} "
              f"({val['time/epoch_s']:.0f}s)", flush=True)
        val_acc.append(val["val/acc"])
        save_checkpoint(ckpt_path, model, opt, trainer, epoch, step, wandb_id)
        if done:
            break

    # Run summary: final accuracy, per-epoch accuracy, alignment averaged over the run,
    # and mean training loss over the run (area under the loss curve: lower = learns faster).
    summary = {"final/val_acc": val["val/acc"] if val else None, "val_acc_per_epoch": val_acc,
               "train_loss_auc": float(np.mean(history["train/loss"])) if history["train/loss"] else None,
               "steps": step}
    for k in ALIGN_KEYS:
        if history[k]:
            summary[f"{k}_mean"] = float(np.mean(history[k]))
    if run is not None:
        for k, v in summary.items():
            if not isinstance(v, list):
                run.summary[k] = v
        run.finish()
    print("SUMMARY " + json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    main()
