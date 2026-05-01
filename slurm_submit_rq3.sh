#!/usr/bin/env bash
# slurm_submit_rq3.sh — Submit ONLY the RQ3 pipeline (RQ3-A + RQ3-B).
#
# RQ3-A: One-shot fit() cost on the synthetic (sample × vars) grid.
#        4 methods × N replicates × C chunks. Stacking + BMA + Best-Logit run
#        Python; MCMC runs the R sampler (bma_cost.R, TUNE-verbatim) so all
#        methods are timed on the same hardware.
# RQ3-B: online_update cost under drift.
#        Default mode is the new synthetic samples x vars grid benchmark
#        (`run_rq3.py --phase online --online-mode grid`), producing amortized per-sample
#        latency per cell. Legacy 450-row stream mode remains available via
#        `--rq3-online-mode legacy`.
#
# Job DAG:
#   rq3_<method>[0..C-1] (parallel, RQ3-A chunks)                ┐
#   rq3online_<DR>[0..OC-1] -> rq3online_merge_<DR> (RQ3-B grid) ┴── plot_rq3
#   or rq3online_<DR> (legacy mode)                              ┘
#
# Usage:
#   bash slurm_submit_rq3.sh
#   bash slurm_submit_rq3.sh --rq3-methods "stacking bma best_logit"
#   bash slurm_submit_rq3.sh --rq3-online-drifts "1x 6x"
#   bash slurm_submit_rq3.sh --rq3-chunks 16 --rq3-n-runs 20
#   bash slurm_submit_rq3.sh --quick
#
# IMPORTANT: 'scancel -u $USER' before resubmitting to avoid stale-dependency
# DependencyNeverSatisfied failures.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
RQ3_CHUNKS=16
RQ3_N_RUNS=1
RQ3_METHODS="stacking bma best_logit"   # BAS dropped per spec.
RQ3_ONLINE_DRIFTS="1x 3x 6x"
RQ3_ONLINE_MODE="grid"
RQ3_ONLINE_CHUNKS=16
RQ3_ONLINE_N_RUNS=1
CPUS=4
TIME_RQ3A="4:00:00"
TIME_RQ3B="4:00:00"
TIME_MERGE="00:10:00"
PARTITION=""
ACCOUNT=""
RETRAIN_EVERY=100

ALPHA=0.001
ONLINE_LR=0.05
MINI_BATCH=1
RQ3_ONLINE_EXTRA=""

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --rq3-chunks)         RQ3_CHUNKS="$2"; shift 2 ;;
        --rq3-n-runs)         RQ3_N_RUNS="$2"; shift 2 ;;
        --rq3-methods)        RQ3_METHODS="$2"; shift 2 ;;
        --rq3-online-drifts)  RQ3_ONLINE_DRIFTS="$2"; shift 2 ;;
        --rq3-online-mode)    RQ3_ONLINE_MODE="$2"; shift 2 ;;
        --rq3-online-chunks)  RQ3_ONLINE_CHUNKS="$2"; shift 2 ;;
        --rq3-online-n-runs)  RQ3_ONLINE_N_RUNS="$2"; shift 2 ;;
        --cpus)               CPUS="$2"; shift 2 ;;
        --time)               TIME_RQ3A="$2"; TIME_RQ3B="$2"; shift 2 ;;
        --time-rq3a)          TIME_RQ3A="$2"; shift 2 ;;
        --time-rq3b)          TIME_RQ3B="$2"; shift 2 ;;
        --partition)          PARTITION="-p $2"; shift 2 ;;
        --account)            ACCOUNT="-A $2"; shift 2 ;;
        --retrain-every)      RETRAIN_EVERY="$2"; shift 2 ;;
        --alpha)              ALPHA="$2"; shift 2 ;;
        --lr)                 ONLINE_LR="$2"; shift 2 ;;
        --mini-batch)         MINI_BATCH="$2"; shift 2 ;;
        --quick)              RQ3_CHUNKS=3; RQ3_ONLINE_CHUNKS=3; RQ3_N_RUNS=1; RQ3_ONLINE_EXTRA="--quick"; shift ;;
        *)                    echo "Unknown: $1"; exit 1 ;;
    esac
done

if [[ "$RQ3_ONLINE_MODE" != "grid" && "$RQ3_ONLINE_MODE" != "legacy" ]]; then
    echo "Invalid --rq3-online-mode: $RQ3_ONLINE_MODE (expected grid or legacy)"
    exit 1
fi

COMMON="--export=ALL -c $CPUS $PARTITION $ACCOUNT"
COMMON="$COMMON -e logs/slurm_%x_%A_%a.err -o logs/slurm_%x_%A_%a.out"
mkdir -p logs cache plots

PREAMBLE="export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; \
source ~/stacking_env/bin/activate; cd $SCRIPT_DIR"

echo "============================================"
echo "  RQ3-only SLURM Submission"
echo "  RQ3-A methods:        $RQ3_METHODS"
echo "  RQ3-A chunks/n_runs:  $RQ3_CHUNKS / $RQ3_N_RUNS"
echo "  RQ3-B drifts:         $RQ3_ONLINE_DRIFTS"
echo "  RQ3-B mode/chunks/n_runs: $RQ3_ONLINE_MODE / $RQ3_ONLINE_CHUNKS / $RQ3_ONLINE_N_RUNS"
echo "  CPUs/job:             $CPUS"
echo "  Time budgets:         RQ3-A=$TIME_RQ3A  RQ3-B=$TIME_RQ3B  merge=$TIME_MERGE"
echo "  SGD:                  alpha=$ALPHA lr=$ONLINE_LR mini_batch=$MINI_BATCH"
echo "  BMA retrain_every:    $RETRAIN_EVERY (1 = TAAS2024 paper-strict)"
echo "============================================"
echo ""

SGD_ARGS="--alpha $ALPHA --online-lr $ONLINE_LR --mini-batch $MINI_BATCH"
BMA_ARGS="--retrain-every $RETRAIN_EVERY"
RQ3_LAST=$((RQ3_CHUNKS - 1))
RQ3_ONLINE_LAST=$((RQ3_ONLINE_CHUNKS - 1))

# ---------------------------------------------------------------------------
# RQ3-A: chunked grid sweep, one method per array job.
# ---------------------------------------------------------------------------
RQ3_JOB_IDS=""
for M in $RQ3_METHODS; do
    JID=$(sbatch --parsable $COMMON --time=$TIME_RQ3A -J rq3_${M} \
        --array=0-$RQ3_LAST \
        --wrap="$PREAMBLE; python run_rq3.py --phase fit --method $M --n-runs $RQ3_N_RUNS --chunk \$SLURM_ARRAY_TASK_ID --n-chunks $RQ3_CHUNKS")
    RQ3_JOB_IDS="$RQ3_JOB_IDS:$JID"
    echo "RQ3-A $M:  $JID [0..$RQ3_LAST]  (n_runs=$RQ3_N_RUNS)"
done

# ---------------------------------------------------------------------------
# RQ3-B: per drift scenario. Grid mode uses chunked arrays + merge; legacy
# mode uses one job per drift.
# ---------------------------------------------------------------------------
RQ3_ONLINE_JOB_IDS=""
for DR in $RQ3_ONLINE_DRIFTS; do
    if [[ "$RQ3_ONLINE_MODE" == "grid" ]]; then
        CHUNK_JID=$(sbatch --parsable $COMMON --time=$TIME_RQ3B -J rq3online_${DR} \
            --array=0-$RQ3_ONLINE_LAST \
            --wrap="$PREAMBLE; python run_rq3.py --phase online --online-mode grid --drift-scenario $DR --n-runs $RQ3_ONLINE_N_RUNS --chunk \$SLURM_ARRAY_TASK_ID --n-chunks $RQ3_ONLINE_CHUNKS $SGD_ARGS $BMA_ARGS $RQ3_ONLINE_EXTRA")
        MERGE_JID=$(sbatch --parsable $COMMON --time=$TIME_MERGE -J rq3online_merge_${DR} \
            --dependency=afterok:$CHUNK_JID \
            --wrap="$PREAMBLE; python run_rq3.py --phase online --online-mode grid --drift-scenario $DR --merge $BMA_ARGS")
        RQ3_ONLINE_JOB_IDS="$RQ3_ONLINE_JOB_IDS:$MERGE_JID"
        echo "RQ3-B drift=$DR mode=grid: chunks=$CHUNK_JID [0..$RQ3_ONLINE_LAST] merge=$MERGE_JID"
    else
        JID=$(sbatch --parsable $COMMON --time=$TIME_RQ3B -J rq3online_$DR \
            --wrap="$PREAMBLE; python run_rq3.py --phase online --online-mode legacy --drift-scenario $DR --n-runs $RQ3_ONLINE_N_RUNS $SGD_ARGS $BMA_ARGS $RQ3_ONLINE_EXTRA")
        RQ3_ONLINE_JOB_IDS="$RQ3_ONLINE_JOB_IDS:$JID"
        echo "RQ3-B drift=$DR mode=legacy:  $JID"
    fi
done

# ---------------------------------------------------------------------------
# Final aggregation: per-method merge of RQ3-A chunk caches + RQ3 plots.
# RQ3-B writes a finalized TSV directly, so no merge step is needed for it.
# ---------------------------------------------------------------------------
PLOT_DEPS="${RQ3_JOB_IDS#:}${RQ3_ONLINE_JOB_IDS}"

RQ3_MERGE_CMD=""
for M in $RQ3_METHODS; do
    RQ3_MERGE_CMD="$RQ3_MERGE_CMD python run_rq3.py --phase fit --method $M --merge;"
done

PLOT=$(sbatch --parsable $COMMON --time=$TIME_MERGE -J plot_rq3 \
    --dependency=afterok:$PLOT_DEPS \
    --wrap="$PREAMBLE; $RQ3_MERGE_CMD python plot_results.py --rq3-only")
echo ""
echo "Plot RQ3:    $PLOT"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
N_RQ3M=$(echo $RQ3_METHODS | wc -w | tr -d ' ')
N_RQ3B=$(echo $RQ3_ONLINE_DRIFTS | wc -w | tr -d ' ')
RQ3A_TOTAL=$((N_RQ3M * RQ3_CHUNKS))
if [[ "$RQ3_ONLINE_MODE" == "grid" ]]; then
    RQ3B_TOTAL=$((N_RQ3B * (RQ3_ONLINE_CHUNKS + 1)))
else
    RQ3B_TOTAL=$N_RQ3B
fi
TOTAL_JOBS=$((RQ3A_TOTAL + RQ3B_TOTAL + 1))
echo ""
echo "============================================"
echo "  Total RQ3 jobs: $TOTAL_JOBS"
echo ""
echo "  Breakdown:"
echo "    RQ3-A (one-shot fit cost): $RQ3A_TOTAL jobs ($N_RQ3M methods × $RQ3_CHUNKS chunks)"
if [[ "$RQ3_ONLINE_MODE" == "grid" ]]; then
    echo "    RQ3-B (online_update latency): $RQ3B_TOTAL jobs ($N_RQ3B drifts × ($RQ3_ONLINE_CHUNKS chunks + 1 merge))"
else
    echo "    RQ3-B (online_update latency): $RQ3B_TOTAL jobs ($N_RQ3B drifts, legacy single-job mode)"
fi
echo "    Plot:                       1 job"
echo ""
echo "  Outputs (RQ3-A): logs/rq3_<method>.tsv → plots/rq3_boxplot.png"
echo "                   (mixed: SMARTS+BMA from RQ3-B drift=6x, Best-Logit from RQ3-A)"
if [[ "$RQ3_ONLINE_MODE" == "grid" ]]; then
    echo "  Outputs (RQ3-B): logs/rq3_online_grid_drift{DR}_re${RETRAIN_EVERY}.tsv →"
    echo "                   plots/rq3_online_boxplot_drift{DR}.png + plots/rq3_online_boxplot.png"
else
    echo "  Outputs (RQ3-B): logs/rq3_online_drift{DR}.tsv →"
    echo "                   plots/rq3_online_boxplot_drift{DR}.png + plots/rq3_online_boxplot.png"
fi
echo ""
echo "  Monitor: squeue -u \$USER  |  Cancel: scancel -u \$USER"
echo "============================================"
