#!/usr/bin/env bash
# slurm_submit.sh - Submit all experiments as a DAG of SLURM jobs.
#
# Setting A only. The drift-detection threshold AND the RQ2 oracle are SWEPT
# in parallel. Shared (threshold- AND oracle-independent): setup, bma_online
# prequential pass. Shared per-oracle (threshold-independent): best_logit GA.
# Per-(threshold, oracle): online_ga, bma_online_ga, merge.
#
# Job DAG (T = one threshold, O = one oracle):
#   setup_A ──┬── bl_A[O][0..N]  for each O ─────────────────────┐
#             ├── online_A[T] ──── online_ga_A[T][O][0..N] ──────┤
#             └── bma_online_A ─── bma_online_ga_A[T][O][0..N] ──┴── merge_A[T][O]
#                                                                       │
#                                                                       ▼
#                                                                     plot
#
#   rq1_A[T]  (independent, one per threshold; oracle-independent)
#   rq3[0..M] (independent)
#
# Usage:
#   bash slurm_submit.sh                                          # defaults: 4 oracles
#   bash slurm_submit.sh --thresholds "0.05 0.10 0.15 0.20 0.25"  # threshold sweep
#   bash slurm_submit.sh --oracles "logit gp"                     # subset of oracles
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
GA_CHUNKS=10           # chunks per batch (online_ga / offline / bma_online_ga)
RQ3_CHUNKS=8           # chunks for RQ3 grid
CPUS=2                 # cores per job
TIME_SETUP="00:20:00"
TIME_RQ1="00:50:00"
TIME_ONLINE="00:20:00" # online/bma_online prequential loops (no GA)
TIME_GA="6:00:00"      # per GA chunk
TIME_RQ3="4:00:00"     # per RQ3 chunk
TIME_MERGE="00:10:00"
PARTITION=""
ACCOUNT=""
EXTRA_RQ2=""
THRESHOLDS="0.035"
ORACLES="logit rf gp ensemble"   # RQ2 oracle sweep — runs all 4 in parallel by default
DRIFT_SCENARIOS="1x 3x 6x"   # transient-burst ladder; only these are plotted (plot_results.RQ4_SCENARIOS)
RETRAIN_EVERY=1        # BMA MAPE-K re-fit interval; 1 = TAAS2024 paper-strict eq.5

# Stacking SGD params (Pareto-optimal defaults from 4-scenario ablation;
# override via --alpha/--lr/--mini-batch).
ALPHA=0.001
ONLINE_LR=0.05
MINI_BATCH=1

# Ablation sweep mode: when set, skip RQ1/RQ2/RQ3 main pipeline and instead
# submit a grid of RQ1+RQ4 jobs sweeping (alpha, eta0, mini_batch).
ABLATION=""            # one of: "", "sgd", "sgd_quick"
ABLATION_ALPHAS="1e-4 1e-3 1e-2 1e-1"
ABLATION_LRS="0.005 0.01 0.05"
ABLATION_BATCHES="1 5 10"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --thresholds) THRESHOLDS="$2"; shift 2 ;;
        --oracles)    ORACLES="$2"; shift 2 ;;
        --ga-chunks)  GA_CHUNKS="$2"; shift 2 ;;
        --rq3-chunks) RQ3_CHUNKS="$2"; shift 2 ;;
        --cpus)       CPUS="$2"; shift 2 ;;
        --time)       TIME_GA="$2"; shift 2 ;;
        --partition)  PARTITION="-p $2"; shift 2 ;;
        --account)    ACCOUNT="-A $2"; shift 2 ;;
        --drift-scenarios) DRIFT_SCENARIOS="$2"; shift 2 ;;
        --retrain-every)   RETRAIN_EVERY="$2"; shift 2 ;;
        --alpha)      ALPHA="$2"; shift 2 ;;
        --lr)         ONLINE_LR="$2"; shift 2 ;;
        --mini-batch) MINI_BATCH="$2"; shift 2 ;;
        --ablation)   ABLATION="$2"; shift 2 ;;
        --ablation-alphas)  ABLATION_ALPHAS="$2"; shift 2 ;;
        --ablation-lrs)     ABLATION_LRS="$2"; shift 2 ;;
        --ablation-batches) ABLATION_BATCHES="$2"; shift 2 ;;
        --quick)      EXTRA_RQ2="--limit 20"; RQ3_CHUNKS=3; shift ;;
        *)            echo "Unknown: $1"; exit 1 ;;
    esac
done

# Quick ablation preset: smaller grid for sanity checks
if [[ "$ABLATION" == "sgd_quick" ]]; then
    ABLATION_ALPHAS="1e-3 1e-1"
    ABLATION_LRS="0.005 0.05"
    ABLATION_BATCHES="1 10"
    ABLATION="sgd"
fi

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
echo "  Oracles:    $ORACLES"
echo "  GA chunks: $GA_CHUNKS    RQ3 chunks: $RQ3_CHUNKS"
echo "  CPUs/job: $CPUS"
echo "  SGD: alpha=$ALPHA  lr=$ONLINE_LR  mini_batch=$MINI_BATCH"
echo "  BMA retrain_every: $RETRAIN_EVERY (1 = TAAS2024 paper-strict)"
[[ -n "$ABLATION" ]] && echo "  Ablation mode: $ABLATION"
echo "============================================"
echo ""

# Per-job stacking SGD flags (passed to every run_rq*.py invocation that
# instantiates StackingEnsemble). When --ablation sgd is active these are
# overridden inside the sweep loop.
SGD_ARGS="--alpha $ALPHA --online-lr $ONLINE_LR --mini-batch $MINI_BATCH"
BMA_ARGS="--retrain-every $RETRAIN_EVERY"

# ---------------------------------------------------------------------------
# Ablation mode (early exit): sweep RQ1 + RQ4 over (alpha × eta0 × mini_batch).
# RQ2/RQ3 are skipped because (a) RQ2 setup is expensive and shared, and
# (b) the SGD ablation question is "do prediction quality and drift recovery
# improve?", which RQ1 + RQ4 already answer end-to-end.
# ---------------------------------------------------------------------------
if [[ "$ABLATION" == "sgd" ]]; then
    echo ""
    echo "  >> ABLATION MODE: per-job (alpha, lr, mini_batch) flags OVERRIDE the"
    echo "     baseline above. The grid below is what each job will actually use."
    echo "     ABLATION_ALPHAS  = $ABLATION_ALPHAS"
    echo "     ABLATION_LRS     = $ABLATION_LRS"
    echo "     ABLATION_BATCHES = $ABLATION_BATCHES"
    echo ""
    N_AB=0
    for A in $ABLATION_ALPHAS; do
        for L in $ABLATION_LRS; do
            for M in $ABLATION_BATCHES; do
                # Sanitize the suffix for filename use
                SUFFIX="ab_a${A}_lr${L}_mb${M}"
                SUFFIX_SAFE=$(echo "$SUFFIX" | tr '.' 'p' | tr '-' 'm')
                JOB_ARGS="--alpha $A --online-lr $L --mini-batch $M --tag-suffix $SUFFIX_SAFE $BMA_ARGS"

                for TH in $THRESHOLDS; do
                    sbatch $COMMON --time=$TIME_RQ1 -J rq1_ab_${SUFFIX_SAFE}_$(th_tag $TH) \
                        --wrap="$PREAMBLE; python run_rq1.py --drift-threshold $TH $JOB_ARGS" >/dev/null
                    for SC in $DRIFT_SCENARIOS; do
                        sbatch $COMMON --time=$TIME_RQ1 -J rq4_ab_${SUFFIX_SAFE}_$(th_tag $TH)_drift${SC} \
                            --wrap="$PREAMBLE; python run_rq4.py --drift-threshold $TH --drift-scenario $SC $JOB_ARGS" >/dev/null
                    done
                    N_AB=$((N_AB + 1 + $(echo $DRIFT_SCENARIOS | wc -w)))
                done
                echo "  submitted: $SUFFIX  (× ${THRESHOLDS} × ${DRIFT_SCENARIOS})"
            done
        done
    done
    echo ""
    echo "============================================"
    echo "  Ablation jobs submitted: $N_AB"
    echo "  Outputs: logs/rq1_A_<th>_<suffix>.json, logs/rq4_A_<th>_<suffix>_drift<sc>.json"
    echo "  Monitor: squeue -u \$USER  |  Cancel: scancel -u \$USER"
    echo "============================================"
    exit 0
fi

LAST_CHUNK=$((GA_CHUNKS - 1))
ALL_MERGE_JOBS=""

# ---------------------------------------------------------------------------
# Setup A (train stacking, pickle)  -- shared across thresholds
# ---------------------------------------------------------------------------
SETUP_A=$(sbatch --parsable $COMMON --time=$TIME_SETUP -J setup_A \
    --wrap="$PREAMBLE; python run_rq2.py --phase setup $SGD_ARGS $EXTRA_RQ2")
echo "Setup:       A=$SETUP_A"

# ---------------------------------------------------------------------------
# RQ2 Best-Logit batch (shared across thresholds; per-oracle to populate
# logs/rq2_A_th{TH}_oracle{O}.json for each O ∈ {logit,rf,gp,ensemble}).
# Replaces the deprecated Stacking-Offline batch.
# ---------------------------------------------------------------------------
# Indirect-variable map (portable to bash 3.2; `declare -A` is bash 4+).
# After this loop, BL_<oracle> holds the best_logit array's job id.
for OR in $ORACLES; do
    JID=$(sbatch --parsable $COMMON --time=$TIME_GA -J rq2bl_A_$OR \
        --dependency=afterok:$SETUP_A --array=0-$LAST_CHUNK \
        --wrap="$PREAMBLE; python run_rq2.py --phase best_logit --oracle $OR --chunk \$SLURM_ARRAY_TASK_ID --n-chunks $GA_CHUNKS $EXTRA_RQ2")
    eval "BL_$OR=$JID"
    echo "RQ2 Best-Logit [$OR]: $JID [0..$LAST_CHUNK]"
done

# ---------------------------------------------------------------------------
# RQ2 BMA-Online prequential pass (shared: produces snapshots used by all
# thresholds' bma_online_ga chunks). Threshold- AND oracle-independent —
# the prequential pass only updates BMA model state, no oracle math.
# ---------------------------------------------------------------------------
BMAON_A=$(sbatch --parsable $COMMON --time=$TIME_ONLINE -J rq2bmaon_A \
    --dependency=afterok:$SETUP_A \
    --wrap="$PREAMBLE; python run_rq2.py --phase bma_online $BMA_ARGS $EXTRA_RQ2")
echo "RQ2 BMA-on:  $BMAON_A (shared snapshots)"

# ---------------------------------------------------------------------------
# Per-threshold: RQ1, online prequential, then per-oracle: online_ga, bma_online_ga, merge
# ---------------------------------------------------------------------------
RQ1_JOBS=""
for TH in $THRESHOLDS; do
    TAG=$(th_tag "$TH")

    # RQ1 (independent, per threshold; oracle-independent — uses ground-truth labels)
    RQ1=$(sbatch --parsable $COMMON --time=$TIME_RQ1 -J rq1_A_$TAG \
        --wrap="$PREAMBLE; python run_rq1.py --drift-threshold $TH $SGD_ARGS $BMA_ARGS")
    RQ1_JOBS="$RQ1_JOBS $RQ1"

    # Online prequential loop (fast: no GA, no oracle), produces per-threshold
    # stacking meta-learner snapshots reused by every oracle's online_ga.
    ON=$(sbatch --parsable $COMMON --time=$TIME_ONLINE -J rq2on_A_$TAG \
        --dependency=afterok:$SETUP_A \
        --wrap="$PREAMBLE; python run_rq2.py --phase online --drift-threshold $TH $SGD_ARGS $EXTRA_RQ2")

    echo "Threshold $TH:"
    echo "  RQ1:           $RQ1"
    echo "  online:        $ON"

    # Per-oracle inner loop: online_ga / bma_online_ga / merge all carry
    # --oracle $OR so their outputs go to the per-(th, oracle) tagged files.
    for OR in $ORACLES; do
        ONGA=$(sbatch --parsable $COMMON --time=$TIME_GA -J rq2onga_A_${TAG}_$OR \
            --dependency=afterok:$ON --array=0-$LAST_CHUNK \
            --wrap="$PREAMBLE; python run_rq2.py --phase online_ga --oracle $OR --drift-threshold $TH --chunk \$SLURM_ARRAY_TASK_ID --n-chunks $GA_CHUNKS $EXTRA_RQ2")

        BMAONGA=$(sbatch --parsable $COMMON --time=$TIME_GA -J rq2bmaonga_A_${TAG}_$OR \
            --dependency=afterok:$BMAON_A --array=0-$LAST_CHUNK \
            --wrap="$PREAMBLE; python run_rq2.py --phase bma_online_ga --oracle $OR --drift-threshold $TH $BMA_ARGS --chunk \$SLURM_ARRAY_TASK_ID --n-chunks $GA_CHUNKS $EXTRA_RQ2")

        # Merge (this threshold + this oracle: needs onga[OR] + bmaonga[OR] + bl[OR])
        BL_VAR="BL_$OR"
        BL_OR=${!BL_VAR}
        MERGE=$(sbatch --parsable $COMMON --time=$TIME_MERGE -J merge_A_${TAG}_$OR \
            --dependency=afterok:$ONGA:$BL_OR:$BMAONGA \
            --wrap="$PREAMBLE; python run_rq2.py --phase merge --oracle $OR --drift-threshold $TH")

        ALL_MERGE_JOBS="$ALL_MERGE_JOBS:$MERGE"
        echo "  [$OR]  online_ga=$ONGA [0..$LAST_CHUNK]  bma_online_ga=$BMAONGA [0..$LAST_CHUNK]  merge=$MERGE"
    done
done

# ---------------------------------------------------------------------------
# RQ4: Concept Drift (independent, per threshold × scenario)
# ---------------------------------------------------------------------------
RQ4_JOBS=""
for TH in $THRESHOLDS; do
    TAG=$(th_tag "$TH")
    for SC in $DRIFT_SCENARIOS; do
        RQ4=$(sbatch --parsable $COMMON --time=$TIME_RQ1 -J rq4_A_${TAG}_drift${SC} \
            --wrap="$PREAMBLE; python run_rq4.py --drift-threshold $TH --drift-scenario $SC $SGD_ARGS $BMA_ARGS")
        RQ4_JOBS="$RQ4_JOBS $RQ4"
        echo "  RQ4: threshold=$TH scenario=$SC  jobid=$RQ4"
    done
done

# ---------------------------------------------------------------------------
# RQ3 (independent, chunked) — Stacking AND BMA cost benchmarks
# ---------------------------------------------------------------------------
RQ3_LAST=$((RQ3_CHUNKS - 1))
RQ3=$(sbatch --parsable $COMMON --time=$TIME_RQ3 -J rq3 \
    --array=0-$RQ3_LAST \
    --wrap="$PREAMBLE; python run_rq3.py --chunk \$SLURM_ARRAY_TASK_ID --n-chunks $RQ3_CHUNKS")
echo "RQ3 stack:   $RQ3 [0..$RQ3_LAST]"

RQ3BMA=$(sbatch --parsable $COMMON --time=$TIME_RQ3 -J rq3bma \
    --array=0-$RQ3_LAST \
    --wrap="$PREAMBLE; python run_rq3.py --method bma --chunk \$SLURM_ARRAY_TASK_ID --n-chunks $RQ3_CHUNKS")
echo "RQ3 bma:     $RQ3BMA [0..$RQ3_LAST]"

# ---------------------------------------------------------------------------
# Final aggregation: RQ3 merge + plots (after all RQ2 merges + RQ3 + RQ1 done)
# ---------------------------------------------------------------------------
PLOT_DEPS="${ALL_MERGE_JOBS#:}:$RQ3:$RQ3BMA"
for J in $RQ1_JOBS; do PLOT_DEPS="$PLOT_DEPS:$J"; done
for J in $RQ4_JOBS; do PLOT_DEPS="$PLOT_DEPS:$J"; done

PLOT=$(sbatch --parsable $COMMON --time=$TIME_MERGE -J plot \
    --dependency=afterok:$PLOT_DEPS \
    --wrap="$PREAMBLE; python run_rq3.py --merge; python run_rq3.py --method bma --merge; python plot_results.py")
echo "Plot:        $PLOT (depends on all merges + RQ3 + RQ1 + RQ4)"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
N_TH=$(echo $THRESHOLDS | wc -w | tr -d ' ')
N_OR=$(echo $ORACLES | wc -w | tr -d ' ')
N_SC=$(echo $DRIFT_SCENARIOS | wc -w | tr -d ' ')
# setup + bl_array_per_oracle + bma_online
# + per_threshold(rq1 + online + per_oracle(online_ga_array + bma_online_ga_array + merge))
# + rq4 + rq3_stack_array + rq3_bma_array + plot
TOTAL_JOBS=$((1 + N_OR*GA_CHUNKS + 1 + N_TH*(1 + 1 + N_OR*(GA_CHUNKS + GA_CHUNKS + 1)) + N_TH*N_SC + RQ3_CHUNKS + RQ3_CHUNKS + 1))
echo ""
echo "============================================"
echo "  Total jobs submitted: $TOTAL_JOBS"
echo ""
echo "  Breakdown:"
echo "    Setup:                  1 job"
echo "    RQ2 Best-Logit:         $((N_OR * GA_CHUNKS)) jobs  ($N_OR oracles × $GA_CHUNKS chunks, shared across thresholds)"
echo "    RQ2 BMA-Online prequential: 1 job (oracle-independent shared snapshots)"
echo "    Per threshold ($N_TH total):"
echo "      RQ1:                  1 job"
echo "      RQ2 Online prequential: 1 job (oracle-independent)"
echo "      Per oracle ($N_OR total):"
echo "        RQ2 OnlineGA:       $GA_CHUNKS jobs"
echo "        RQ2 BMA-OnlineGA:   $GA_CHUNKS jobs"
echo "        Merge:              1 job"
echo "    RQ3 stack:              $RQ3_CHUNKS jobs"
echo "    RQ3 bma:                $RQ3_CHUNKS jobs"
echo "    RQ4:                    $((N_TH * N_SC)) jobs  ($N_TH thresholds × $N_SC scenarios)"
echo "    Plot:                   1 job"
echo ""
echo "  Outputs (RQ2):"
echo "    logs/rq2_A_th{TH}_oracle{O}.json/.tsv  — one per (threshold, oracle)"
echo "    plots/rq2_A_th{TH}_oracle{O}_metric_box.png + _success_bars.png"
echo ""
echo "  Monitor: squeue -u \$USER"
echo "  Cancel:  scancel -u \$USER"
echo ""
echo "  IMPORTANT: Always 'scancel -u \$USER' before resubmitting"
echo "  to avoid DependencyNeverSatisfied from stale jobs."
echo "============================================"
