"""Fast CIFAR-10 benchmark for mechanism experiments.

Runs reference configs (+ optional extra train.py flags) over seeds, appends one JSON
line per run to experiments/results.jsonl, and prints mean / spread per (tag, base).

    python bench.py --tag ref --bases cnn rn --seeds 0 1
    python bench.py --tag H1-adaptive --bases cnn-lw rn --extra "--layer-schedule adaptive"
    python bench.py --summarize
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "experiments" / "results.jsonl"
COMMON = ("--dataset cifar10 --method three_factor --align-every 20 --log-every 1000 "
          "--wandb-mode disabled --no-auto-resume --num-workers 4 --login-node-regex ''")
BASES = {
    # best CNN config so far: top-k 5% + local rewards
    "cnn": "--model cnn --epochs 3 --optimizer adam --lr 3e-4 --perturb-mode topk,blockwise_reward",
    # CNN reference for layerwise-specific mechanisms
    "cnn-lw": "--model cnn --epochs 3 --optimizer adam --lr 3e-4 --perturb-mode layerwise",
    # ResNet-18 reference for top-k mechanisms
    "rn-lwtk": "--model resnet18 --epochs 1 --optimizer adam --lr 1e-4 --perturb-mode layerwise,topk "
               "--micro-batch-size 64",
    # best ResNet-18 config so far
    "rn": "--model resnet18 --epochs 1 --optimizer adam --lr 1e-4 --perturb-mode layerwise --micro-batch-size 64",
}


def snapshot_code(tag) -> Path:
    """Freeze the code for this benchmark, so editing the repo while it runs is safe."""
    snap = Path("/tmp/3f-bench-code") / tag
    snap.mkdir(parents=True, exist_ok=True)
    for f in ("layers.py", "models.py", "data.py", "train.py"):
        shutil.copy2(ROOT / f, snap / f)
    return snap


def run(tag, base, seed, extra, out_dir, code_dir):
    cmd = [sys.executable, "train.py", *shlex.split(COMMON), *shlex.split(BASES[base]), *shlex.split(extra),
           "--seed", str(seed), "--out-dir", out_dir, "--run-name", f"{tag}-{base}-s{seed}",
           "--data-dir", str(ROOT / "data")]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=code_dir, capture_output=True, text=True)
    lines = [l for l in proc.stdout.splitlines() if l.startswith("SUMMARY ")]
    if proc.returncode != 0 or not lines:
        print(proc.stdout[-2000:], proc.stderr[-3000:], file=sys.stderr)
        rec = {"tag": tag, "base": base, "seed": seed, "extra": extra, "failed": True}
    else:
        rec = {"tag": tag, "base": base, "seed": seed, "extra": extra, **json.loads(lines[-1][8:])}
    rec["wall_s"] = round(time.time() - t0, 1)
    RESULTS.parent.mkdir(exist_ok=True)
    with RESULTS.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def fmt(vals, scale=1.0, digits=1):
    vals = [v * scale for v in vals if v is not None]
    if not vals:
        return "-"
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ±{(max(vals) - min(vals)) / 2:.{digits}f}"


def summarize(tags=None):
    groups = defaultdict(list)
    for line in RESULTS.read_text().splitlines():
        r = json.loads(line)
        if tags is None or r["tag"] in tags:
            groups[(r["tag"], r["base"])].append(r)
    print(f"{'tag':28s} {'base':7s} {'n':>2s} {'val acc %':>11s} {'loss AUC':>12s} "
          f"{'align/total':>13s} {'align/full':>13s} {'align_upd':>13s} {'wall s':>7s}")
    for (tag, base), rs in groups.items():
        ok = [r for r in rs if not r.get("failed")]
        print(f"{tag:28s} {base:7s} {len(ok):2d} {fmt([r['final/val_acc'] for r in ok], 100):>11s} "
              f"{fmt([r['train_loss_auc'] for r in ok], 1, 3):>12s} "
              f"{fmt([r.get('align/total_mean') for r in ok], 1, 3):>13s} "
              f"{fmt([r.get('align/full_mean') for r in ok], 1, 3):>13s} "
              f"{fmt([r.get('align_update/full_mean') for r in ok], 1, 3):>13s} "
              f"{fmt([r['wall_s'] for r in ok], 1, 0):>7s}" + ("  FAILED x%d" % (len(rs) - len(ok)) if len(ok) < len(rs) else ""))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag")
    p.add_argument("--bases", nargs="+", default=["cnn", "rn"], choices=list(BASES))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--extra", default="", help="extra train.py flags for this mechanism")
    p.add_argument("--out-dir", default="/tmp/3f-bench")
    p.add_argument("--summarize", action="store_true")
    a = p.parse_args()
    if a.summarize:
        summarize()
        return
    code_dir = snapshot_code(a.tag)
    for base in a.bases:
        for seed in a.seeds:
            r = run(a.tag, base, seed, a.extra, a.out_dir, code_dir)
            print(f"{a.tag} {base} s{seed}: acc {r.get('final/val_acc')} "
                  f"align {r.get('align/total_mean')} upd {r.get('align_update/full_mean')} ({r['wall_s']}s)",
                  flush=True)
    summarize({a.tag})


if __name__ == "__main__":
    main()
