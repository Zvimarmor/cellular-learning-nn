#!/bin/bash
# Job array over experiments x seeds. Submit with slurm/launch_sweep.sh, which sets
# --array from the grid size. Task i runs grid line (i mod N) with seed SEEDS[i div N].
#SBATCH --job-name=3f-sweep
#SBATCH --partition=ss.gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

set -euo pipefail
export REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
GRID="${GRID:-$REPO_DIR/slurm/grid.txt}"
read -r -a seeds <<< "${SEEDS:-0 1 2}"
lines=()
while IFS= read -r line; do lines+=("$line"); done < <(grep -vE '^[[:space:]]*(#|$)' "$GRID")

n=${#lines[@]}
i=${SLURM_ARRAY_TASK_ID:?sweep.sbatch must run as an array job}
(( i < n * ${#seeds[@]} )) || { echo "task $i out of range ($n configs x ${#seeds[@]} seeds)"; exit 1; }
read -r -a cfg <<< "${lines[$((i % n))]}"
seed=${seeds[$((i / n))]}

# Run name and wandb group are derived by train.py from method/dataset/model/modes,
# so all seeds of a config share one wandb group.
exec "$REPO_DIR/slurm/run_single.sh" "${cfg[@]}" --seed "$seed" --align-every 200
