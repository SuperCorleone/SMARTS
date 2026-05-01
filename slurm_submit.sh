#!/usr/bin/env bash
# slurm_submit.sh - Submit all experiments as a DAG of SLURM jobs.
#
# Setting A only. RQ2 is swept along three dimensions in parallel:
#   - drift scenario (selection A: trigger AND online_update follow drifted stream)
#   - drift-detection threshold
#   - oracle kind
#
# Per-RQ2-drift-scenario: setup, bma_online prequential, best_logit GA (per oracle).
# Per-(drift, threshold): online prequential.
# Per-(drift, threshold, oracle): online_ga, bma_online_ga, merge.
#
# Job DAG (DR = drift scenario, T = threshold, O = oracle):
#   for each DR:
#     setup_DR ──┬── bl_DR[O][0..N]                                    ┐
#                ├── online_DR[T] ──── online_ga_DR[T][O][0..N] ───────┤
#                └── bma_online_DR ─── bma_online_ga_DR[T][O][0..N] ───┴── merge_DR[T][O]
#                                                                              │
#                                                                              ▼
#                                                                            plot
#
#   rq1[T][SC] (independent, per drift scenario)
#   rq3[method][0..M]  (independent; one method per call: stacking/bma/best_logit/mcmc)
#
# Usage:
#   bash slurm_submit.sh                                          # defaults
#   bash slurm_submit.sh --rq2-drifts "none 1x 3x 6x"             # RQ2 drift sweep (default)
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
RQ3_N_RUNS=10          # replicates per (sample,vars) cell — needed for box plots
RQ3_METHODS="stacking bma best_logit"   # 4 methods on local hardware
                                              # (BAS dropped per spec; MCMC kept
                                              # as reference data — not plotted.)
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
ORACLES="logit"   #rf gp  RQ2 oracle sweep — runs all 4 in parallel by default
RQ2_DRIFTS="1x 3x 6x"        # RQ2 drift sweep — paper narrative is drift-only;
                              # 'none' is supported by run_rq2 but not scheduled here.
DRIFT_SCENARIOS="1x 3x 6x"   # RQ1 drift ladder; only these are plotted
RQ3_ONLINE_DRIFTS="1x 3x 6x" # RQ3-B: per-sample online_update latency under each drift scenario
RETRAIN_EVERY=1        # BMA MAPE-K re-fit interval; 1 = TAAS2024 paper-strict eq.5

# Stacking SGD params (Pareto-optimal defaults from 4-scenario ablation;
# override via --alpha/--lr/--mini-batch).
ALPHA=0.001
ONLINE_LR=0.05
MINI_BATCH=1

# Ablation sweep mode: when set, skip RQ2/RQ3 main pipeline and instead
# submit a grid of drift-only RQ1 jobs sweeping (alpha, eta0, mini_batch).
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
        --rq3-n-runs) RQ3_N_RUNS="$2"; shift 2 ;;
        --rq3-methods) RQ3_METHODS="$2"; shift 2 ;;
        --cpus)       CPUS="$2"; shift 2 ;;
        --time)       TIME_GA="$2"; shift 2 ;;
        --partition)  PARTITION="-p $2"; shift 2 ;;
        --account)    ACCOUNT="-A $2"; shift 2 ;;
        --drift-scenarios) DRIFT_SCENARIOS="$2"; shift 2 ;;
        --rq2-drifts) RQ2_DRIFTS="$2"; shift 2 ;;
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
echo "  Thresholds:  $THRESHOLDS"
echo "  Oracles:     $ORACLES"
echo "  RQ2 drifts:  $RQ2_DRIFTS"
echo "  RQ1 drifts:  $DRIFT_SCENARIOS"
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
# Ablation mode (early exit): sweep drift-only RQ1 over (alpha × eta0 × mini_batch).
# RQ2/RQ3 are skipped because (a) RQ2 setup is expensive and shared, and
# (b) the SGD ablation question is whether drift-time prediction quality and
# recovery improve, which drift-only RQ1 already answers end-to-end.
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
                    for SC in $DRIFT_SCENARIOS; do
                        sbatch $COMMON --time=$TIME_RQ1 -J rq1_ab_${SUFFIX_SAFE}_$(th_tag $TH)_drift${SC} \
                            --wrap="$PREAMBLE; python run_rq1.py --drift-threshold $TH --drift-scenario $SC $JOB_ARGS" >/dev/null
                    done
                    N_AB=$((N_AB + $(echo $DRIFT_SCENARIOS | wc -w)))
                done
                echo "  submitted: $SUFFIX  (RQ1 drift × ${THRESHOLDS} × ${DRIFT_SCENARIOS})"
            done
        done
    done
    echo ""
    echo "============================================"
    echo "  Ablation jobs submitted: $N_AB"
    echo "  Outputs: logs/rq1_A_<th>_<suffix>_drift<sc>.json"
    echo "  Monitor: squeue -u \$USER  |  Cancel: scancel -u \$USER"
    echo "============================================"
    exit 0
fi

LAST_CHUNK=$((GA_CHUNKS - 1))
ALL_MERGE_JOBS=""

# ---------------------------------------------------------------------------
# RQ1: drift-only prediction/recovery experiment (independent, per threshold x scenario)
# ---------------------------------------------------------------------------
RQ1_JOBS=""
for TH in $THRESHOLDS; do
    TAG=$(th_tag "$TH")
    for SC in $DRIFT_SCENARIOS; do
        RQ1=$(sbatch --parsable $COMMON --time=$TIME_RQ1 -J rq1_A_${TAG}_drift${SC} \
            --wrap="$PREAMBLE; python run_rq1.py --drift-threshold $TH --drift-scenario $SC $SGD_ARGS $BMA_ARGS")
        RQ1_JOBS="$RQ1_JOBS $RQ1"
        echo "RQ1: threshold=$TH scenario=$SC  jobid=$RQ1"
    done
done

# ---------------------------------------------------------------------------
# RQ2 pipeline — outer loop over drift scenarios. Each drift scenario gets
# its own setup, best_logit, bma_online prequential, online prequential and
# per-oracle online_ga / bma_online_ga / merge. Per spec (selection A):
# trigger AND online_update both follow the drifted stream, so EVERY phase
# must carry --drift-scenario.
#
# Job DAG per drift scenario DR:
#   setup_DR ──┬── bl_DR[O][0..N]                ───────────┐
#              ├── online_DR[T] ── online_ga_DR[T][O][0..N] ┤
#              └── bma_online_DR ── bma_online_ga_DR[T][O][0..N] ┴── merge_DR[T][O]
# ---------------------------------------------------------------------------
for DR in $RQ2_DRIFTS; do
    DR_TAG="$DR"  # alphanumeric: 'none', '1x', '3x', '6x'

    # Setup (per-drift: hazard0_indices/rows depend on drifted labels)
    SETUP_DR=$(sbatch --parsable $COMMON --time=$TIME_SETUP -J setup_A_${DR_TAG} \
        --wrap="$PREAMBLE; python run_rq2.py --phase setup --drift-scenario $DR $SGD_ARGS $EXTRA_RQ2")
    echo ""
    echo "=== Drift scenario: $DR ==="
    echo "Setup:       $SETUP_DR"

    # Best-Logit batch — per (drift, oracle); shared across thresholds.
    for OR in $ORACLES; do
        JID=$(sbatch --parsable $COMMON --time=$TIME_GA -J rq2bl_A_${DR_TAG}_$OR \
            --dependency=afterok:$SETUP_DR --array=0-$LAST_CHUNK \
            --wrap="$PREAMBLE; python run_rq2.py --phase best_logit --drift-scenario $DR --oracle $OR --chunk \$SLURM_ARRAY_TASK_ID --n-chunks $GA_CHUNKS $EXTRA_RQ2")
        eval "BL_${DR_TAG}_$OR=$JID"
        echo "RQ2 Best-Logit [$DR/$OR]: $JID [0..$LAST_CHUNK]"
    done

    # BMA-Online prequential pass — per drift (online_update uses drifted labels).
    # Threshold- and oracle-independent within a drift.
    BMAON_DR=$(sbatch --parsable $COMMON --time=$TIME_ONLINE -J rq2bmaon_A_${DR_TAG} \
        --dependency=afterok:$SETUP_DR \
        --wrap="$PREAMBLE; python run_rq2.py --phase bma_online --drift-scenario $DR $BMA_ARGS $EXTRA_RQ2")
    echo "RQ2 BMA-on [$DR]:  $BMAON_DR"

    # Per-threshold inner loop
    for TH in $THRESHOLDS; do
        TAG=$(th_tag "$TH")

        # Stacking online prequential — per (drift, threshold).
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
# RQ3 (independent, chunked) — 4 methods × N replicates, all on local hardware
# Stacking + BMA fit via run_rq3.py's Python path; MCMC + BAS via Rscript
# bma_cost.R (TUNE's verbatim sampler) — putting all four on the same machine
# is what makes the cross-method comparison defensible (the previous run
# imported MCMC/BAS times measured on TUNE's old i5 hardware).
# Each cell repeats RQ3_N_RUNS times so plots can show variability via box
# plots / median+IQR shading.
# ---------------------------------------------------------------------------
RQ3_LAST=$((RQ3_CHUNKS - 1))
RQ3_JOB_IDS=""
for M in $RQ3_METHODS; do
    JID=$(sbatch --parsable $COMMON --time=$TIME_RQ3 -J rq3_${M} \
        --array=0-$RQ3_LAST \
        --wrap="$PREAMBLE; python run_rq3.py --phase fit --method $M --n-runs $RQ3_N_RUNS --chunk \$SLURM_ARRAY_TASK_ID --n-chunks $RQ3_CHUNKS")
    RQ3_JOB_IDS="$RQ3_JOB_IDS:$JID"
    echo "RQ3-A $M:  $JID [0..$RQ3_LAST]  (n_runs=$RQ3_N_RUNS)"
done

# ---------------------------------------------------------------------------
# RQ3-B: online_update latency under drift. Companion to RQ3-A.
# One chunked grid sweep + merge per drift scenario.
# ---------------------------------------------------------------------------
RQ3_ONLINE_JOB_IDS=""
for DR in $RQ3_ONLINE_DRIFTS; do
    CHUNK_JID=$(sbatch --parsable $COMMON --time=$TIME_RQ3 -J rq3online_$DR \
        --array=0-$RQ3_LAST \
        --wrap="$PREAMBLE; python run_rq3.py --phase online --online-mode grid --drift-scenario $DR --chunk \$SLURM_ARRAY_TASK_ID --n-chunks $RQ3_CHUNKS $SGD_ARGS $BMA_ARGS")
    MERGE_JID=$(sbatch --parsable $COMMON --time=$TIME_MERGE -J rq3online_merge_$DR \
        --dependency=afterok:$CHUNK_JID \
        --wrap="$PREAMBLE; python run_rq3.py --phase online --online-mode grid --drift-scenario $DR --merge $BMA_ARGS")
    RQ3_ONLINE_JOB_IDS="$RQ3_ONLINE_JOB_IDS:$MERGE_JID"
    echo "RQ3-B drift=$DR: chunks=$CHUNK_JID [0..$RQ3_LAST] merge=$MERGE_JID"
done

# ---------------------------------------------------------------------------
# Final aggregation: RQ3-A merge per method + plots
# ---------------------------------------------------------------------------
PLOT_DEPS="${ALL_MERGE_JOBS#:}${RQ3_JOB_IDS}${RQ3_ONLINE_JOB_IDS}"
for J in $RQ1_JOBS; do PLOT_DEPS="$PLOT_DEPS:$J"; done

# Build the RQ3-A merge command (RQ3-B writes a final TSV directly, no merge needed).
RQ3_MERGE_CMD=""
for M in $RQ3_METHODS; do
    RQ3_MERGE_CMD="$RQ3_MERGE_CMD python run_rq3.py --phase fit --method $M --merge;"
done

PLOT=$(sbatch --parsable $COMMON --time=$TIME_MERGE -J plot \
    --dependency=afterok:$PLOT_DEPS \
    --wrap="$PREAMBLE; $RQ3_MERGE_CMD python plot_results.py")
echo "Plot:        $PLOT (depends on RQ1 + RQ2 merges + RQ3-A[$RQ3_METHODS] + RQ3-B[$RQ3_ONLINE_DRIFTS])"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
N_TH=$(echo $THRESHOLDS | wc -w | tr -d ' ')
N_OR=$(echo $ORACLES | wc -w | tr -d ' ')
N_SC=$(echo $DRIFT_SCENARIOS | wc -w | tr -d ' ')
N_DR=$(echo $RQ2_DRIFTS | wc -w | tr -d ' ')
N_RQ3M=$(echo $RQ3_METHODS | wc -w | tr -d ' ')
N_RQ3B=$(echo $RQ3_ONLINE_DRIFTS | wc -w | tr -d ' ')
# Per RQ2 drift scenario: setup(1) + bl_per_oracle(N_OR*GA_CHUNKS) + bma_online(1)
#   + per_threshold(online(1) + per_oracle(online_ga(GA_CHUNKS) + bma_online_ga(GA_CHUNKS) + merge(1)))
PER_DRIFT=$((1 + N_OR*GA_CHUNKS + 1 + N_TH*(1 + N_OR*(GA_CHUNKS + GA_CHUNKS + 1))))
RQ2_TOTAL=$((N_DR * PER_DRIFT))
RQ3A_TOTAL=$((N_RQ3M * RQ3_CHUNKS))
RQ3B_TOTAL=$((N_RQ3B * (RQ3_CHUNKS + 1)))
TOTAL_JOBS=$((N_TH*N_SC + RQ2_TOTAL + RQ3A_TOTAL + RQ3B_TOTAL + 1))
echo ""
echo "============================================"
echo "  Total jobs submitted: $TOTAL_JOBS"
echo ""
echo "  Breakdown:"
echo "    RQ1 (drift-only):      $((N_TH * N_SC)) jobs  ($N_TH thresholds × $N_SC scenarios)"
echo "    RQ2 (per drift scenario, $N_DR total = $RQ2_DRIFTS):"
echo "      Setup:                  1 job"
echo "      RQ2 Best-Logit:         $((N_OR * GA_CHUNKS)) jobs  ($N_OR oracles × $GA_CHUNKS chunks)"
echo "      RQ2 BMA-Online prequential: 1 job (per drift, oracle-independent)"
echo "      Per threshold ($N_TH total):"
echo "        RQ2 Online prequential: 1 job"
echo "        Per oracle ($N_OR total):"
echo "          RQ2 OnlineGA:       $GA_CHUNKS jobs"
echo "          RQ2 BMA-OnlineGA:   $GA_CHUNKS jobs"
echo "          Merge:              1 job"
echo "      Per-drift subtotal:     $PER_DRIFT jobs"
echo "    RQ2 grand subtotal:     $RQ2_TOTAL jobs"
echo "    RQ3-A (one-shot fit cost; $N_RQ3M methods × $RQ3_CHUNKS chunks): $RQ3A_TOTAL jobs"
echo "    RQ3-B (online_update latency under drift; chunked): $RQ3B_TOTAL jobs"
echo "    Plot:                   1 job"
echo ""
echo "  Outputs (RQ2):"
echo "    logs/rq2_A_drift{DR}_th{TH}_oracle{O}.json/.tsv  — one per (drift, threshold, oracle)"
echo "    plots/rq2_A_drift{DR}_th{TH}_oracle{O}_metric_box.png + _success_bars.png"
echo "  Outputs (RQ3-A): logs/rq3_<method>.tsv → plots/rq3_boxplot.png"
echo "  Outputs (RQ1): logs/rq1_A_th{TH}_drift{SC}.json/.tsv"
echo "  Outputs (RQ3-B): logs/rq3_online_grid_drift{DR}_re${RETRAIN_EVERY}.tsv →"
echo "                  plots/rq3_online_boxplot_drift{DR}.png + plots/rq3_online_boxplot.png"
echo ""
echo "  Monitor: squeue -u \$USER"
echo "  Cancel:  scancel -u \$USER"
echo ""
echo "  IMPORTANT: Always 'scancel -u \$USER' before resubmitting"
echo "  to avoid DependencyNeverSatisfied from stale jobs."
echo "============================================"
