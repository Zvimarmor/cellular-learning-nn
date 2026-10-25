#!/bin/bash
# Interactive GPU shell for debugging, wrapped in tmux so an SSH disconnect does not kill it.
#   ./slurm/dev_session.sh          # attach to (or create) tmux session "3f-dev"
# Inside it, run e.g.:  python train.py --max-steps 50 --wandb-mode disabled ...
set -euo pipefail
export REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_DIR/slurm/cluster.env"
exec tmux new-session -A -s 3f-dev \
  "srun --partition=$PARTITION --gres=$GPU_GRES --constraint='$CONSTRAINT' --cpus-per-task=$CPUS --mem=$MEM --time=${DEV_TIME:-02:00:00} --pty bash -l"
