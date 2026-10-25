#!/bin/bash
# Login-node launcher: checks the datasets are prefetched, then submits the grid as one
# throttled job array. Lightweight (no training). Usage:
#   ./slurm/launch_sweep.sh                       # slurm/grid.txt, seeds "0 1 2"
#   GRID=slurm/grid_smoke.txt SEEDS="0" ./slurm/launch_sweep.sh
#   PARTITION=gpu.q GPU_GRES=gpu:a100:1 ./slurm/launch_sweep.sh
#   DRY_RUN=1 ./slurm/launch_sweep.sh             # print the task list only
set -euo pipefail
export REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_DIR/slurm/cluster.env"
cd "$REPO_DIR"

GRID="$(realpath "${GRID:-slurm/grid.txt}")"
SEEDS="${SEEDS:-0 1 2}"
lines=()
while IFS= read -r line; do lines+=("$line"); done < <(grep -vE '^[[:space:]]*(#|$)' "$GRID")
read -r -a seeds <<< "$SEEDS"
n_tasks=$(( ${#lines[@]} * ${#seeds[@]} ))

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  for (( i = 0; i < n_tasks; i++ )); do
    echo "task $i: ${lines[$((i % ${#lines[@]}))]} --seed ${seeds[$((i / ${#lines[@]}))]}"
  done
  exit 0
fi

# Compute nodes may have no internet: refuse to submit before prefetch_data.sh ran.
for need in "$DATA_DIR/MNIST/raw/train-images-idx3-ubyte" "$DATA_DIR/cifar-10-batches-py/data_batch_1"; do
  [[ -e "$need" ]] || { echo "missing $need -- run ./slurm/prefetch_data.sh first" >&2; exit 1; }
done
if [[ "$WANDB_MODE" == online ]] && ! grep -qE "api\.wandb\.ai|api\.forge\.coreweave\.com" ~/.netrc 2>/dev/null && [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "warning: no wandb credentials found; run 'wandb login' once on the login node" >&2
fi

mkdir -p logs "$OUT_DIR"
args=(--partition="$PARTITION" --gres="$GPU_GRES" --constraint="$CONSTRAINT" --cpus-per-task="$CPUS" --mem="$MEM"
      --time="$TIME_LIMIT" --array="0-$((n_tasks - 1))%${MAX_PARALLEL}"
      --export="ALL,REPO_DIR=$REPO_DIR,GRID=$GRID,SEEDS=$SEEDS")
[[ -z "$ACCOUNT" ]] || args+=(--account="$ACCOUNT")

job=$(sbatch --parsable "${args[@]}" slurm/sweep.sbatch)
echo "submitted array job $job: ${#lines[@]} configs x ${#seeds[@]} seeds = $n_tasks tasks" \
     "(max $MAX_PARALLEL at once) on $PARTITION"
echo "monitor:  squeue -u \$USER -j $job     logs: logs/3f-sweep-${job}_<task>.out"
echo "cancel:   scancel $job"
