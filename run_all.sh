#!/usr/bin/env bash
# run_all.sh - Single-node convenience runner (Setting A, threshold sweep).
#
# For the cluster, prefer slurm_submit.sh. This script runs each threshold
# sequentially on one machine.
#
# Usage:
#   ./run_all.sh                                          # defaults
#   ./run_all.sh --thresholds "0.05 0.10 0.15 0.20 0.25"
#   ./run_all.sh --ga-workers 4
#   ./run_all.sh --skip-rq3
#   ./run_all.sh --quick

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

GA_WORKERS=2
SKIP_RQ3=false
RQ2_EXTRA=""
RQ3_EXTRA=""
THRESHOLDS="0.035"
DRIFT_SCENARIOS="6x"

while [[ $# -gt 0 ]]; do
    case $1 in
        --ga-workers) GA_WORKERS="$2"; shift 2 ;;
        --thresholds) THRESHOLDS="$2"; shift 2 ;;
        --skip-rq3)   SKIP_RQ3=true; shift ;;
        --quick)      RQ2_EXTRA="--limit 20"; RQ3_EXTRA="--quick"; shift ;;
        *)            echo "Unknown: $1"; exit 1 ;;
    esac
done

mkdir -p logs plots

echo "============================================"
echo "  Stacking Experiment Suite (Setting A)"
echo "  Thresholds: $THRESHOLDS"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  GA workers: $GA_WORKERS"
echo "============================================"
START=$SECONDS

for TH in $THRESHOLDS; do
    TAG=$(awk -v t="$TH" 'BEGIN{printf "th%.3f", t}')
    echo ""
    echo "[RQ1] threshold=$TH ..."
    python run_rq1.py --drift-threshold "$TH" > "logs/run_rq1_${TAG}.log" 2>&1

    echo "[RQ2] threshold=$TH (ga-workers=$GA_WORKERS) ..."
    python run_rq2.py --drift-threshold "$TH" --ga-workers "$GA_WORKERS" $RQ2_EXTRA \
        > "logs/run_rq2_${TAG}.log" 2>&1
done

# RQ4: Concept Drift experiments
echo ""
echo "============================================"
echo "  RQ4: Concept Drift Experiments"
echo "============================================"
for TH in $THRESHOLDS; do
    TAG=$(awk -v t="$TH" 'BEGIN{printf "th%.3f", t}')
    for SC in $DRIFT_SCENARIOS; do
        echo "[RQ4] threshold=$TH scenario=$SC ..."
        python run_rq4.py --drift-threshold "$TH" --drift-scenario "$SC" > "logs/run_rq4_${TAG}_drift${SC}.log" 2>&1
        echo "  Done (see logs/run_rq4_${TAG}_drift${SC}.log)"
    done
done

if [ "$SKIP_RQ3" = false ]; then
    echo ""
    echo "[RQ3] ..."
    python run_rq3.py --seed 42 $RQ3_EXTRA > logs/run_rq3.log 2>&1
else
    echo ""
    echo "[RQ3] skipped"
fi

ELAPSED=$((SECONDS - START))
echo ""
echo "============================================"
echo "  Done in ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "============================================"

echo ""
echo "Generating plots..."
python plot_results.py > logs/plot_results.log 2>&1 \
    && echo "Plots saved to plots/" \
    || echo "Plot generation failed, check logs/plot_results.log"

echo ""
echo "Summaries:"
ls -1 logs/rq*_summary.tsv 2>/dev/null || echo "  (none yet)"
