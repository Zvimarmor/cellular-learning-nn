#!/bin/bash
# Runs ONE experiment inside a SLURM allocation. Called by submit_experiments.sbatch and
# sweep.sbatch; all arguments are forwarded to train.py. Refuses to run outside a job.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export REPO_DIR
source "$REPO_DIR/slurm/cluster.env"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "run_single.sh: not inside a SLURM job. Submit with sbatch, or use slurm/dev_session.sh." >&2
  exit 1
fi
cd "$REPO_DIR"
eval "$ENV_ACTIVATE"

echo "[$(date '+%F %T')] job ${SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID:+ task $SLURM_ARRAY_TASK_ID}" \
     "on $(hostname), restart #${SLURM_RESTART_COUNT:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

data_dir="$DATA_DIR"
if [[ "$STAGE_DATA" == 1 ]]; then
  data_dir="$SCRATCH_ROOT/$USER/3f-data-${SLURM_JOB_ID}"
  mkdir -p "$data_dir"
  trap 'rm -rf "$data_dir"' EXIT
  rsync -a --exclude '*.tar.gz' "$DATA_DIR/" "$data_dir/"
  echo "staged $(du -sh "$data_dir" | cut -f1) of data to $data_dir"
fi

export WANDB_MODE WANDB_PROJECT LOGIN_NODE_REGEX
if [[ -n "$WANDB_ENTITY" ]]; then export WANDB_ENTITY; fi

# Python runs in the background so this shell can forward SLURM's signals:
# sbatch --signal=B:USR1@300 hits this (batch) shell 5 min before the time limit,
# train.py then checkpoints and requeues the job; SIGTERM (pre-emption) checkpoints only.
python -u train.py --data-dir "$data_dir" --out-dir "$OUT_DIR" --no-download \
       --wandb-project "$WANDB_PROJECT" "$@" &
pid=$!
trap 'kill -USR1 "$pid" 2>/dev/null' USR1
trap 'kill -TERM "$pid" 2>/dev/null' TERM
status=0
wait "$pid" || status=$?
# A trapped signal interrupts `wait` with status > 128; waiting again returns python's
# real exit code. 127 means it was already reaped, i.e. the previous status was real.
while (( status > 128 )); do
  s=0
  wait "$pid" || s=$?
  (( s == 127 )) && break
  status=$s
done
exit "$status"
