#!/bin/bash
# One experiment. Arguments go to train.py:
#   sbatch slurm/submit_experiments.sbatch --dataset cifar10 --model resnet18 --perturb-mode layerwise,topk
# Partition / GPU here are defaults; slurm/cluster.env values are applied when
# submitting through launch_sweep.sh, or override on the command line:
#   sbatch --partition=gpu.q --gres=gpu:a100:1 slurm/submit_experiments.sbatch ...
#SBATCH --job-name=3f
#SBATCH --partition=gpu.q
#SBATCH --gres=gpu:1
#SBATCH --constraint=RTX2080Ti|A100|rtxpro
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

set -euo pipefail
export REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"   # sbatch runs a spooled copy of this file
exec "$REPO_DIR/slurm/run_single.sh" "$@"
