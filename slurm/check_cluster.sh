#!/bin/bash
# Read-only overview of the cluster, to fill in slurm/cluster.env. Safe on the login node.
echo "host: $(hostname)   user: $USER"
echo; echo "== GPU partitions (partition, state, time limit, nodes, GRES)"
sinfo -o "%.15P %.6a %.11l %.5D %.40G" | grep -iE "PARTITION|gpu"
echo; echo "== GPU types available (use as --gres=gpu:<type>:1)"
sinfo -h -o "%G" | tr ',' '\n' | grep -o 'gpu:[^:(]*' | sort -u
echo; echo "== your accounts / partitions"
sacctmgr -nP show assoc user="$USER" format=account,partition,qos 2>/dev/null || echo "(sacctmgr not available)"
echo; echo "== your jobs"
squeue -u "$USER" -o "%.12i %.12P %.30j %.8T %.10M %.6D %R"
echo; echo "== node-local scratch on a GPU node: run inside an allocation:  df -h /tmp \$TMPDIR"
