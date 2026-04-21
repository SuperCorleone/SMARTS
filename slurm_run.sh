#!/usr/bin/env bash
# slurm_run.sh - Run experiments inside an interactive salloc session.
# For multi-node batch submission, use slurm_submit.sh instead.
#
# Usage (after salloc -n 1 -c 8 --time=48:00:00; srun --pty bash):
#   source ~/stacking_env/bin/activate
#   bash slurm_run.sh              # auto-detect cores
#   bash slurm_run.sh --quick      # quick test
#   bash slurm_run.sh --skip-rq3

set -euo pipefail
cd "$(dirname "$0")"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

NCPUS="${SLURM_CPUS_PER_TASK:-${SLURM_JOB_CPUS_PER_NODE:-$(nproc 2>/dev/null || echo 4)}}"
echo "CPUs: $NCPUS  Node: ${SLURM_NODELIST:-local}"

GA_WORKERS=$(( (NCPUS - 3) / 2 ))
[ "$GA_WORKERS" -lt 1 ] && GA_WORKERS=1

exec bash run_all.sh --ga-workers "$GA_WORKERS" "$@"
