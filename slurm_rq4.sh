#!/usr/bin/env bash
# slurm_rq4.sh - Submit only RQ4 concept-drift experiments to SLURM.
#
# Usage:
#   bash slurm_rq4.sh
#   bash slurm_rq4.sh --thresholds "0.035"
#   bash slurm_rq4.sh --drift-scenarios "4x 5x 6x"
#   bash slurm_rq4.sh --partition gpu --cpus 4

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CPUS=2
TIME_RQ4="00:50:00"
PARTITION=""
ACCOUNT=""
THRESHOLDS="0.035"
DRIFT_SCENARIOS="1x 2x 3x 4x 5x 6x 1x_w10 2x_w10 3x_w10 4x_w10 5x_w10 6x_w10 1x_w30 2x_w30 3x_w30 4x_w30 5x_w30 6x_w30"

while [[ $# -gt 0 ]]; do
    case $1 in
        --thresholds)       THRESHOLDS="$2"; shift 2 ;;
        --drift-scenarios)  DRIFT_SCENARIOS="$2"; shift 2 ;;
        --cpus)             CPUS="$2"; shift 2 ;;
        --time)             TIME_RQ4="$2"; shift 2 ;;
        --partition)        PARTITION="-p $2"; shift 2 ;;
        --account)          ACCOUNT="-A $2"; shift 2 ;;
        *)                  echo "Unknown: $1"; exit 1 ;;
    esac
done

COMMON="--export=ALL -c $CPUS $PARTITION $ACCOUNT"
COMMON="$COMMON -e logs/slurm_%x_%A.err -o logs/slurm_%x_%A.out"

PREAMBLE="export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; \
source ~/stacking_env/bin/activate; cd $SCRIPT_DIR"

th_tag() { awk -v t="$1" 'BEGIN{printf "th%.3f", t}'; }

mkdir -p logs plots

N_TH=$(echo $THRESHOLDS | wc -w | tr -d ' ')
N_SC=$(echo $DRIFT_SCENARIOS | wc -w | tr -d ' ')

echo "============================================"
echo "  RQ4: Concept Drift (SLURM)"
echo "  Thresholds: $THRESHOLDS"
echo "  Scenarios:  $DRIFT_SCENARIOS"
echo "  Jobs:       $((N_TH * N_SC))"
echo "============================================"

RQ4_JOBS=""
for TH in $THRESHOLDS; do
    TAG=$(th_tag "$TH")
    for SC in $DRIFT_SCENARIOS; do
        RQ4=$(sbatch --parsable $COMMON --time=$TIME_RQ4 -J rq4_${TAG}_${SC} \
            --wrap="$PREAMBLE; python run_rq4.py --drift-threshold $TH --drift-scenario $SC")
        RQ4_JOBS="$RQ4_JOBS $RQ4"
        echo "  Submitted: th=$TH scenario=$SC  jobid=$RQ4"
    done
done

# Plot job after all RQ4 finish
PLOT_DEPS=$(echo $RQ4_JOBS | tr ' ' ':')
PLOT=$(sbatch --parsable $COMMON --time=00:10:00 -J rq4_plot \
    --dependency=afterok:$PLOT_DEPS \
    --wrap="$PREAMBLE; python plot_results.py --rq4-only")
echo ""
echo "  Plot job: $PLOT (after all RQ4 complete)"
echo "============================================"
echo "  Monitor: squeue -u \$USER"
echo "============================================"
