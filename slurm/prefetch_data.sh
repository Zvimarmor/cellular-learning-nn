#!/bin/bash
# Run ONCE on the login node: downloads MNIST + CIFAR-10 to $DATA_DIR (shared NFS).
# Download only; no training happens here. Compute jobs then run with --no-download.
set -euo pipefail
export REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_DIR/slurm/cluster.env"
eval "$ENV_ACTIVATE"
mkdir -p "$DATA_DIR"
nice -n 10 python - "$DATA_DIR" <<'PY'
import sys
from torchvision import datasets
root = sys.argv[1]
for ds in (datasets.MNIST, datasets.CIFAR10):
    for train in (True, False):
        d = ds(root, train=train, download=True)
        print(f"{ds.__name__:8s} {'train' if train else 'test ':5s} {len(d):6d} samples OK")
PY
echo "datasets ready in $DATA_DIR"
