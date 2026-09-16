#!/bin/bash
# Submit with explicit account, qos, partition, gres, time and output arguments.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PIP_CACHE_DIR="$PWD/cache/pip"
export XDG_CACHE_HOME="$PWD/cache"
export OMP_NUM_THREADS=2
export PYTHONPATH="$PWD/src"
# Preserve Slurm CUDA_VISIBLE_DEVICES. No host-global GPU selection.
test -n "${SLURM_JOB_ID:-}"
date -u
hostname
test -f .local/source-commit.txt && cat .local/source-commit.txt
cat /proc/cpuinfo | head -25 || true
free -m
.venv/bin/python -m pip freeze > "logs/environment-${SLURM_JOB_ID}.txt"
if [[ "${ACL_DEVICE:-cpu}" == cuda ]]; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  .venv/bin/python -c 'import torch; assert torch.cuda.device_count() == 1; p=torch.cuda.get_device_properties(0); print(p)'
fi
srun .venv/bin/python -m pytest -q
if [[ "${ACL_DEVICE:-cpu}" == cpu ]]; then
  srun .venv/bin/python scripts/acquire_data.py --audit-only
fi
srun .venv/bin/python -m acl_hct.smoke --device "${ACL_DEVICE:-cpu}" --output "logs/smoke-${SLURM_JOB_ID}.json"
