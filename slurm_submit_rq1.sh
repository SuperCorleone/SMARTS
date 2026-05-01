#!/usr/bin/env bash
# slurm_submit_rq1.sh - Submit only drift-only RQ1 experiments to SLURM.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CPUS=2
TIME_RQ1="00:50:00"
PARTITION=""
ACCOUNT=""
THRESHOLDS="0.035"
DRIFT_SCENARIOS="1x 2x 3x 4x 5x 6x perm_late perm_mid perm_early recurring"

while [[ $# -gt 0 ]]; do
    case $1 in
        --thresholds)       THRESHOLDS="$2"; shift 2 ;;
        --drift-scenarios)  DRIFT_SCENARIOS="$2"; shift 2 ;;
        --cpus)             CPUS="$2"; shift 2 ;;
        --time)             TIME_RQ1="$2"; shift 2 ;;
        --partition)        PARTITION="-p $2"; shift 2 ;;
        --account)          ACCOUNT="-A $2"; shift 2 ;;
        *)                  echo "Unknown: $1"; exit 1 ;;
    esac
done

COMMON="--export=ALL -c $CPUS $PARTITION $ACCOUNT"
COMMON="$COMMON -e logs/slurm_%x_%A.err -o logs/slurm_%x_%A.out"
PREAMBLE="export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; source ~/stacking_env/bin/activate; cd $SCRIPT_DIR"
th_tag() { awk -v t="$1" 'BEGIN{printf "th%.3f", t}'; }

mkdir -p logs plots

N_TH=$(echo $THRESHOLDS | wc -w | tr -d ' ')
N_SC=$(echo $DRIFT_SCENARIOS | wc -w | tr -d ' ')

echo "============================================"
echo "  RQ1: Drift Experiment (SLURM)"
echo "  Thresholds: $THRESHOLDS"
echo "  Scenarios:  $DRIFT_SCENARIOS"
echo "  Jobs:       $((N_TH * N_SC))"
echo "============================================"

RQ1_JOBS=""
for TH in $THRESHOLDS; do
    TAG=$(th_tag "$TH")
    for SC in $DRIFT_SCENARIOS; do
        RQ1=$(sbatch --parsable $COMMON --time=$TIME_RQ1 -J rq1_${TAG}_${SC} \
            --wrap="$PREAMBLE; python run_rq1.py --drift-threshold $TH --drift-scenario $SC")
        RQ1_JOBS="$RQ1_JOBS $RQ1"
        echo "  Submitted: th=$TH scenario=$SC  jobid=$RQ1"
    done
done

PLOT_DEPS=$(echo $RQ1_JOBS | tr ' ' ':')
PLOT=$(sbatch --parsable $COMMON --time=00:10:00 -J rq1_plot \
    --dependency=afterok:$PLOT_DEPS \
    --wrap="$PREAMBLE; python plot_results.py --rq1-only")
echo ""
echo "  Plot job: $PLOT (after all RQ1 complete)"
echo "============================================"
echo "  Monitor: squeue -u \$USER"
echo "============================================"
