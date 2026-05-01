#!/usr/bin/env bash
# slurm_submit_rq2.sh — Submit ONLY the RQ2 pipeline (no RQ1/RQ3/RQ4).
#
# Setting A. RQ2 is swept along three dimensions in parallel:
#   - drift scenario   (selection A: trigger AND online_update follow drifted stream)
#   - drift-detection threshold
#   - oracle kind
#
# Per-RQ2-drift-scenario:    setup, bma_online prequential, best_logit GA (per oracle).
# Per-(drift, threshold):    online prequential.
# Per-(drift, threshold, O): online_ga, bma_online_ga, merge.
#
# Job DAG (DR = drift, T = threshold, O = oracle):
#   for each DR:
#     setup_DR ──┬── bl_DR[O][0..N]                                    ┐
#                ├── online_DR[T] ──── online_ga_DR[T][O][0..N] ───────┤
#                └── bma_online_DR ─── bma_online_ga_DR[T][O][0..N] ───┴── merge_DR[T][O]
#                                                                              │
#                                                                              ▼
#                                                                            plot_rq2
#
# Usage:
#   bash slurm_submit_rq2.sh
#   bash slurm_submit_rq2.sh --rq2-drifts "none 1x 3x 6x"
#   bash slurm_submit_rq2.sh --thresholds "0.035 0.10 0.15" --oracles "logit rf"
#   bash slurm_submit_rq2.sh --ga-chunks 20 --cpus 4 --time 6:00:00
#   bash slurm_submit_rq2.sh --quick
#
# IMPORTANT: 'scancel -u $USER' before resubmitting to avoid stale-dependency
# DependencyNeverSatisfied failures.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
GA_CHUNKS=16
CPUS=4
TIME_SETUP="00:20:00"
TIME_ONLINE="00:20:00"
TIME_GA="6:00:00"
TIME_MERGE="00:10:00"
PARTITION=""
ACCOUNT=""
EXTRA_RQ2=""
THRESHOLDS="0.035" # 0.035
ORACLES="logit"
RQ2_DRIFTS="6x" # 1x 3x 
RETRAIN_EVERY=1

ALPHA=0.001
ONLINE_LR=0.01
MINI_BATCH=1

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --thresholds)     THRESHOLDS="$2"; shift 2 ;;
        --oracles)        ORACLES="$2"; shift 2 ;;
        --ga-chunks)      GA_CHUNKS="$2"; shift 2 ;;
        --cpus)           CPUS="$2"; shift 2 ;;
        --time)           TIME_GA="$2"; shift 2 ;;
        --partition)      PARTITION="-p $2"; shift 2 ;;
        --account)        ACCOUNT="-A $2"; shift 2 ;;
        --rq2-drifts)     RQ2_DRIFTS="$2"; shift 2 ;;
        --retrain-every)  RETRAIN_EVERY="$2"; shift 2 ;;
        --alpha)          ALPHA="$2"; shift 2 ;;
        --lr)             ONLINE_LR="$2"; shift 2 ;;
        --mini-batch)     MINI_BATCH="$2"; shift 2 ;;
        --quick)          EXTRA_RQ2="--limit 20"; shift ;;
        *)                echo "Unknown: $1"; exit 1 ;;
    esac
done

COMMON="--export=ALL -c $CPUS $PARTITION $ACCOUNT"
COMMON="$COMMON -e logs/slurm_%x_%A_%a.err -o logs/slurm_%x_%A_%a.out"
mkdir -p logs cache plots

PREAMBLE="export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; \
source ~/stacking_env/bin/activate; cd $SCRIPT_DIR"
th_tag() { awk -v t="$1" 'BEGIN{printf "th%.3f", t}'; }

echo "============================================"
echo "  RQ2-only SLURM Submission (Setting A)"
echo "  Thresholds: $THRESHOLDS"
echo "  Oracles:    $ORACLES"
echo "  Drifts:     $RQ2_DRIFTS"
echo "  GA chunks:  $GA_CHUNKS    CPUs/job: $CPUS"
echo "  SGD:        alpha=$ALPHA lr=$ONLINE_LR mini_batch=$MINI_BATCH"
echo "  BMA retrain_every: $RETRAIN_EVERY (1 = TAAS2024 paper-strict)"
echo "============================================"
echo ""

SGD_ARGS="--alpha $ALPHA --online-lr $ONLINE_LR --mini-batch $MINI_BATCH"
BMA_ARGS="--retrain-every $RETRAIN_EVERY"
LAST_CHUNK=$((GA_CHUNKS - 1))
ALL_MERGE_JOBS=""

# ---------------------------------------------------------------------------
# RQ2 DAG (outer loop over drift scenarios)
# ---------------------------------------------------------------------------
for DR in $RQ2_DRIFTS; do
    DR_TAG="$DR"

    SETUP_DR=$(sbatch --parsable $COMMON --time=$TIME_SETUP -J setup_A_${DR_TAG} \
        --wrap="$PREAMBLE; python run_rq2.py --phase setup --drift-scenario $DR $SGD_ARGS $EXTRA_RQ2")
    echo ""
    echo "=== Drift scenario: $DR ==="
    echo "Setup:       $SETUP_DR"

    for OR in $ORACLES; do
        JID=$(sbatch --parsable $COMMON --time=$TIME_GA -J rq2bl_A_${DR_TAG}_$OR \
            --dependency=afterok:$SETUP_DR --array=0-$LAST_CHUNK \
            --wrap="$PREAMBLE; python run_rq2.py --phase best_logit --drift-scenario $DR --oracle $OR --chunk \$SLURM_ARRAY_TASK_ID --n-chunks $GA_CHUNKS $EXTRA_RQ2")
        eval "BL_${DR_TAG}_$OR=$JID"
        echo "RQ2 Best-Logit [$DR/$OR]: $JID [0..$LAST_CHUNK]"
    done

    BMAON_DR=$(sbatch --parsable $COMMON --time=$TIME_ONLINE -J rq2bmaon_A_${DR_TAG} \
        --dependency=afterok:$SETUP_DR \
        --wrap="$PREAMBLE; python run_rq2.py --phase bma_online --drift-scenario $DR $BMA_ARGS $EXTRA_RQ2")
    echo "RQ2 BMA-on [$DR]:  $BMAON_DR"

    for TH in $THRESHOLDS; do
        TAG=$(th_tag "$TH")

        ON=$(sbatch --parsable $COMMON --time=$TIME_ONLINE -J rq2on_A_${DR_TAG}_$TAG \
            --dependency=afterok:$SETUP_DR \
            --wrap="$PREAMBLE; python run_rq2.py --phase online --drift-scenario $DR --drift-threshold $TH $SGD_ARGS $EXTRA_RQ2")
        echo "  [$DR/$TH]  online: $ON"

        for OR in $ORACLES; do
            ONGA=$(sbatch --parsable $COMMON --time=$TIME_GA -J rq2onga_A_${DR_TAG}_${TAG}_$OR \
                --dependency=afterok:$ON --array=0-$LAST_CHUNK \
                --wrap="$PREAMBLE; python run_rq2.py --phase online_ga --drift-scenario $DR --oracle $OR --drift-threshold $TH --chunk \$SLURM_ARRAY_TASK_ID --n-chunks $GA_CHUNKS $EXTRA_RQ2")

            BMAONGA=$(sbatch --parsable $COMMON --time=$TIME_GA -J rq2bmaonga_A_${DR_TAG}_${TAG}_$OR \
                --dependency=afterok:$BMAON_DR --array=0-$LAST_CHUNK \
                --wrap="$PREAMBLE; python run_rq2.py --phase bma_online_ga --drift-scenario $DR --oracle $OR --drift-threshold $TH $BMA_ARGS --chunk \$SLURM_ARRAY_TASK_ID --n-chunks $GA_CHUNKS $EXTRA_RQ2")

            BL_VAR="BL_${DR_TAG}_$OR"
            BL_OR=${!BL_VAR}
            MERGE=$(sbatch --parsable $COMMON --time=$TIME_MERGE -J merge_A_${DR_TAG}_${TAG}_$OR \
                --dependency=afterok:$ONGA:$BL_OR:$BMAONGA \
                --wrap="$PREAMBLE; python run_rq2.py --phase merge --drift-scenario $DR --oracle $OR --drift-threshold $TH")

            ALL_MERGE_JOBS="$ALL_MERGE_JOBS:$MERGE"
            echo "  [$DR/$TH/$OR]  online_ga=$ONGA  bma_online_ga=$BMAONGA  merge=$MERGE"
        done
    done
done

# ---------------------------------------------------------------------------
# Plot (RQ2 only)
# ---------------------------------------------------------------------------
PLOT_DEPS="${ALL_MERGE_JOBS#:}"
PLOT=$(sbatch --parsable $COMMON --time=$TIME_MERGE -J plot_rq2 \
    --dependency=afterok:$PLOT_DEPS \
    --wrap="$PREAMBLE; python plot_results.py --rq2-only")
echo ""
echo "Plot RQ2:    $PLOT"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
N_TH=$(echo $THRESHOLDS | wc -w | tr -d ' ')
N_OR=$(echo $ORACLES | wc -w | tr -d ' ')
N_DR=$(echo $RQ2_DRIFTS | wc -w | tr -d ' ')
PER_DRIFT=$((1 + N_OR*GA_CHUNKS + 1 + N_TH*(1 + N_OR*(GA_CHUNKS + GA_CHUNKS + 1))))
RQ2_TOTAL=$((N_DR * PER_DRIFT))
TOTAL_JOBS=$((RQ2_TOTAL + 1))
echo ""
echo "============================================"
echo "  Total RQ2 jobs: $TOTAL_JOBS  (incl. 1 plot job)"
echo "  Per-drift breakdown: setup(1) + best_logit($((N_OR*GA_CHUNKS)))"
echo "                      + bma_online(1)"
echo "                      + per_threshold($N_TH × [online(1)"
echo "                        + per_oracle($N_OR × (online_ga($GA_CHUNKS)"
echo "                        + bma_online_ga($GA_CHUNKS) + merge(1)))])"
echo "                      = $PER_DRIFT jobs/drift × $N_DR drifts = $RQ2_TOTAL"
echo ""
echo "  Outputs:"
echo "    logs/rq2_A_drift{DR}_th{TH}_oracle{O}.json/.tsv"
echo "    plots/rq2_A_drift{DR}_th{TH}_oracle{O}_metric_box.png"
echo "    plots/rq2_A_drift{DR}_th{TH}_oracle{O}_success_bars.png"
echo ""
echo "  Monitor: squeue -u \$USER  |  Cancel: scancel -u \$USER"
echo "============================================"
