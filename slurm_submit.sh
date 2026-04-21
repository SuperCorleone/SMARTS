#!/usr/bin/env bash
# slurm_submit.sh - Submit all experiments as a DAG of SLURM jobs.
#
# Setting A only. The drift-detection threshold is SWEPT in parallel: setup,
# offline GA, and BMA GA are threshold-independent (shared across the sweep);
# online + online_ga + merge run once per threshold.
#
# Job DAG (T = one threshold in the sweep):
#   setup_A ──┬── online_A[T] ── online_ga_A[T][0..N] ── merge_A[T]
#             ├── offline_A[0..N] ─────────────────────── merge_A[T]
#             └── bma_A[0..N] ─────────────────────────── merge_A[T]
#
#   rq1_A[T]  (independent, one per threshold)
#   rq3[0..M] (independent)
#
# Usage:
#   bash slurm_submit.sh                                          # defaults
#   bash slurm_submit.sh --thresholds "0.05 0.10 0.15 0.20 0.25"  # custom sweep
#   bash slurm_submit.sh --ga-chunks 20                           # more parallelism
#   bash slurm_submit.sh --cpus 4 --time 6:00:00
#   bash slurm_submit.sh --quick                                  # quick test
#   bash slurm_submit.sh --partition gpu                          # specify partition
#
# IMPORTANT: Always 'scancel -u $USER' before resubmitting to avoid
#            DependencyNeverSatisfied from stale failed jobs.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Defaults (tune to your cluster)
# ---------------------------------------------------------------------------
GA_CHUNKS=10           # chunks per batch (online_ga per threshold, offline, bma)
RQ3_CHUNKS=8           # chunks for RQ3 grid
CPUS=2                 # cores per job
TIME_SETUP="00:20:00"
TIME_RQ1="00:50:00"
TIME_ONLINE="00:20:00" # online is now fast (prequential loop only, no GA)
TIME_GA="6:00:00"      # per GA chunk
TIME_RQ3="4:00:00"     # per RQ3 chunk
TIME_MERGE="00:10:00"
PARTITION=""
ACCOUNT=""
EXTRA_RQ2=""
THRESHOLDS="0.05 0.10 0.15 0.20 0.25"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --thresholds) THRESHOLDS="$2"; shift 2 ;;
        --ga-chunks)  GA_CHUNKS="$2"; shift 2 ;;
        --rq3-chunks) RQ3_CHUNKS="$2"; shift 2 ;;
        --cpus)       CPUS="$2"; shift 2 ;;
        --time)       TIME_GA="$2"; shift 2 ;;
        --partition)  PARTITION="-p $2"; shift 2 ;;
        --account)    ACCOUNT="-A $2"; shift 2 ;;
        --quick)      EXTRA_RQ2="--limit 20"; RQ3_CHUNKS=3; shift ;;
        *)            echo "Unknown: $1"; exit 1 ;;
    esac
done

# Common sbatch flags
COMMON="--export=ALL -c $CPUS $PARTITION $ACCOUNT"
COMMON="$COMMON -e logs/slurm_%x_%A_%a.err -o logs/slurm_%x_%A_%a.out"

mkdir -p logs cache plots

# ---------------------------------------------------------------------------
# Environment preamble (injected into each job)
# ---------------------------------------------------------------------------
PREAMBLE="export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; \
source ~/stacking_env/bin/activate; cd $SCRIPT_DIR"

# Canonical tag builder for a threshold value (must match run_rq2.threshold_tag)
th_tag() { awk -v t="$1" 'BEGIN{printf "th%.3f", t}'; }

echo "============================================"
echo "  SLURM Experiment Submission (Setting A)"
echo "  Thresholds: $THRESHOLDS"
echo "  GA chunks: $GA_CHUNKS    RQ3 chunks: $RQ3_CHUNKS"
echo "  CPUs/job: $CPUS"
echo "============================================"
echo ""

LAST_CHUNK=$((GA_CHUNKS - 1))
ALL_MERGE_JOBS=""

# ---------------------------------------------------------------------------
# Setup A (train stacking, pickle)  -- shared across thresholds
# ---------------------------------------------------------------------------
SETUP_A=$(sbatch --parsable $COMMON --time=$TIME_SETUP -J setup_A \
    --wrap="$PREAMBLE; python run_rq2.py --phase setup $EXTRA_RQ2")
echo "Setup:       A=$SETUP_A"

# ---------------------------------------------------------------------------
# RQ2 Offline batch (shared across thresholds, depends on setup)
# ---------------------------------------------------------------------------
OFF_A=$(sbatch --parsable $COMMON --time=$TIME_GA -J rq2off_A \
    --dependency=afterok:$SETUP_A --array=0-$LAST_CHUNK \
    --wrap="$PREAMBLE; python run_rq2.py --phase offline --chunk \$SLURM_ARRAY_TASK_ID --n-chunks $GA_CHUNKS $EXTRA_RQ2")
echo "RQ2 Offline: $OFF_A [0..$LAST_CHUNK] (shared)"

# ---------------------------------------------------------------------------
# RQ2 BMA batch (shared across thresholds, depends on setup)
# ---------------------------------------------------------------------------
BMA_A=$(sbatch --parsable $COMMON --time=$TIME_GA -J rq2bma_A \
    --dependency=afterok:$SETUP_A --array=0-$LAST_CHUNK \
    --wrap="$PREAMBLE; python run_rq2.py --phase bma --chunk \$SLURM_ARRAY_TASK_ID --n-chunks $GA_CHUNKS $EXTRA_RQ2")
echo "RQ2 BMA:     $BMA_A [0..$LAST_CHUNK] (shared)"

# ---------------------------------------------------------------------------
# Per-threshold: RQ1, online, online_ga, merge
# ---------------------------------------------------------------------------
RQ1_JOBS=""
for TH in $THRESHOLDS; do
    TAG=$(th_tag "$TH")

    # RQ1 (independent, per threshold)
    RQ1=$(sbatch --parsable $COMMON --time=$TIME_RQ1 -J rq1_A_$TAG \
        --wrap="$PREAMBLE; python run_rq1.py --drift-threshold $TH")
    RQ1_JOBS="$RQ1_JOBS $RQ1"

    # Online prequential loop (fast: no GA), produces per-threshold snapshots
    ON=$(sbatch --parsable $COMMON --time=$TIME_ONLINE -J rq2on_A_$TAG \
        --dependency=afterok:$SETUP_A \
        --wrap="$PREAMBLE; python run_rq2.py --phase online --drift-threshold $TH $EXTRA_RQ2")

    # Online GA (embarrassingly parallel, per threshold)
    ONGA=$(sbatch --parsable $COMMON --time=$TIME_GA -J rq2onga_A_$TAG \
        --dependency=afterok:$ON --array=0-$LAST_CHUNK \
        --wrap="$PREAMBLE; python run_rq2.py --phase online_ga --drift-threshold $TH --chunk \$SLURM_ARRAY_TASK_ID --n-chunks $GA_CHUNKS $EXTRA_RQ2")

    # Merge (depends on this threshold's online GA + shared offline + shared bma)
    MERGE=$(sbatch --parsable $COMMON --time=$TIME_MERGE -J merge_A_$TAG \
        --dependency=afterok:$ONGA:$OFF_A:$BMA_A \
        --wrap="$PREAMBLE; python run_rq2.py --phase merge --drift-threshold $TH")

    ALL_MERGE_JOBS="$ALL_MERGE_JOBS:$MERGE"
    echo "Threshold $TH:"
    echo "  RQ1:       $RQ1"
    echo "  online:    $ON"
    echo "  online_ga: $ONGA [0..$LAST_CHUNK]"
    echo "  merge:     $MERGE"
done

# ---------------------------------------------------------------------------
# RQ3 (independent, chunked)
# ---------------------------------------------------------------------------
RQ3_LAST=$((RQ3_CHUNKS - 1))
RQ3=$(sbatch --parsable $COMMON --time=$TIME_RQ3 -J rq3 \
    --array=0-$RQ3_LAST \
    --wrap="$PREAMBLE; python run_rq3.py --chunk \$SLURM_ARRAY_TASK_ID --n-chunks $RQ3_CHUNKS")
echo "RQ3:         $RQ3 [0..$RQ3_LAST]"

# ---------------------------------------------------------------------------
# Final aggregation: RQ3 merge + plots (after all RQ2 merges + RQ3 + RQ1 done)
# ---------------------------------------------------------------------------
PLOT_DEPS="${ALL_MERGE_JOBS#:}:$RQ3"
for J in $RQ1_JOBS; do PLOT_DEPS="$PLOT_DEPS:$J"; done

PLOT=$(sbatch --parsable $COMMON --time=$TIME_MERGE -J plot \
    --dependency=afterok:$PLOT_DEPS \
    --wrap="$PREAMBLE; python run_rq3.py --merge; python plot_results.py")
echo "Plot:        $PLOT (depends on all merges + RQ3 + RQ1)"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
N_TH=$(echo $THRESHOLDS | wc -w | tr -d ' ')
TOTAL_JOBS=$((1 + GA_CHUNKS*2 + N_TH*(1+1+GA_CHUNKS+1) + RQ3_CHUNKS + 1))
echo ""
echo "============================================"
echo "  Total jobs submitted: $TOTAL_JOBS"
echo ""
echo "  Breakdown:"
echo "    Setup:          1 job"
echo "    RQ2 Offline:    $GA_CHUNKS jobs  (shared)"
echo "    RQ2 BMA:        $GA_CHUNKS jobs  (shared)"
echo "    Per threshold ($N_TH total):"
echo "      RQ1:          1 job"
echo "      RQ2 Online:   1 job"
echo "      RQ2 OnlineGA: $GA_CHUNKS jobs"
echo "      Merge:        1 job"
echo "    RQ3:            $RQ3_CHUNKS jobs"
echo "    Plot:           1 job"
echo ""
echo "  Monitor: squeue -u \$USER"
echo "  Cancel:  scancel -u \$USER"
echo ""
echo "  IMPORTANT: Always 'scancel -u \$USER' before resubmitting"
echo "  to avoid DependencyNeverSatisfied from stale jobs."
echo "============================================"
