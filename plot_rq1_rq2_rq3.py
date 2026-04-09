"""
plot_rq1_rq2_rq3.py
Generate comparison plots for RQ1-3: Stacking vs BMA vs Logit.

Data sources:
  - logs/precision_recall.log     — Baseline RQ1 (Logit1-8)
  - logs/stacking_rq1_results.tsv — Stacking RQ1
  - logs/re_bma_logit.log         — Baseline RQ2 (BMA + Logit1-8)
  - logs/stacking_rq2_results.tsv — Stacking RQ2
  - logs/baseline_rq3_results.tsv — Baseline RQ3 (MCMC + BAS)
  - logs/stacking_rq3_results.tsv — Stacking RQ3

Usage:
    python plot_rq1_rq2_rq3.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
})

BASE_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.join(BASE_DIR, "logs")
PLOT_DIR = os.path.join(BASE_DIR, "plots")

COLORS = {
    "BMA": "#4C72B0",
    "Stacking": "#DD8452",
    "Logit": "#55A868",
    "MCMC": "#4C72B0",
    "BAS": "#55A868",
}


# ==================== Data Loading ====================

def load_rq1_data():
    """Load RQ1: Logit baseline (averaged) + Stacking + BMA.
    Returns list of dicts: [{model, precision, recall, F1}, ...]
    """
    entries = []

    # Stacking / BMA from stacking_rq1_results.tsv
    stacking_file = os.path.join(LOG_DIR, "stacking_rq1_results.tsv")
    if os.path.exists(stacking_file):
        df = pd.read_csv(stacking_file, sep="\t")
        for _, row in df.iterrows():
            entries.append({
                "model": row["model"] if row["model"] != "Stacking" else "Stacking",
                "precision": row["precision"],
                "recall": row["recall"],
                "F1": row["F1"],
            })
            print(f"  RQ1 {row['model']}: P={row['precision']:.4f} R={row['recall']:.4f} F1={row['F1']:.4f}")

    # Logit baseline (average across all models)
    logit_file = os.path.join(LOG_DIR, "precision_recall.log")
    if os.path.exists(logit_file):
        df = pd.read_csv(logit_file, sep="\t")
        # Find best Logit by mean F1
        best_model = None
        best_f1 = -1
        for m in df["model"].unique():
            mf1 = df[df["model"] == m]["F1"].mean()
            if mf1 > best_f1:
                best_f1 = mf1
                best_model = m
        if best_model:
            sub = df[df["model"] == best_model]
            entries.append({
                "model": f"Best Logit ({best_model})",
                "precision": sub["precision"].mean(),
                "recall": sub["recall"].mean(),
                "F1": sub["F1"].mean(),
            })
            print(f"  RQ1 Best Logit ({best_model}): P={sub['precision'].mean():.4f} R={sub['recall'].mean():.4f} F1={best_f1:.4f}")

    return entries


def load_rq2_data():
    """Load RQ2: BMA + Stacking + best Logit (by success rate).
    Returns dict: {model_name: DataFrame with RE, success columns}
    """
    result = {}

    # Baseline (BMA + Logit)
    baseline_file = os.path.join(LOG_DIR, "re_bma_logit.log")
    if os.path.exists(baseline_file):
        df = pd.read_csv(baseline_file, sep="\t")

        # BMA
        bma = df[df["type"] == "BMA"]
        if not bma.empty:
            result["BMA"] = bma
            print(f"  RQ2 BMA: {len(bma)} rows")

        # Find best Logit by success rate
        logit = df[df["type"] == "Logit"]
        best_model = None
        best_rate = -1
        for m in logit["model"].unique():
            sub = logit[logit["model"] == m]
            rate = sub["success"].apply(lambda x: x == True or x == "TRUE").mean()
            if rate > best_rate:
                best_rate = rate
                best_model = m
        if best_model:
            result[f"Best Logit ({best_model})"] = logit[logit["model"] == best_model]
            print(f"  RQ2 Best Logit ({best_model}): success_rate={best_rate:.4f}")

    # Stacking
    stacking_file = os.path.join(LOG_DIR, "stacking_rq2_results.tsv")
    if os.path.exists(stacking_file):
        df = pd.read_csv(stacking_file, sep="\t")
        stacking = df[df["type"] == "Stacking"]
        if not stacking.empty:
            result["Stacking"] = stacking
            print(f"  RQ2 Stacking: {len(stacking)} rows")

    return result


def load_rq3_data():
    """Load RQ3: MCMC/BAS baseline + Stacking.
    Returns combined DataFrame with columns: method, vars, sample, time
    """
    frames = []

    baseline_file = os.path.join(LOG_DIR, "baseline_rq3_results.tsv")
    if os.path.exists(baseline_file):
        df = pd.read_csv(baseline_file, sep=r"\s+")
        frames.append(df)
        print(f"  RQ3 baseline: {len(df)} rows ({df['method'].value_counts().to_dict()})")

    stacking_file = os.path.join(LOG_DIR, "stacking_rq3_results.tsv")
    if os.path.exists(stacking_file):
        df = pd.read_csv(stacking_file, sep=r"\s+")
        frames.append(df)
        print(f"  RQ3 Stacking: {len(df)} rows")

    return pd.concat(frames, ignore_index=True) if frames else None


# ==================== RQ1 Plots ====================

def plot_rq1(entries, plot_dir):
    """RQ1: Three separate bar charts (one per metric)."""
    if len(entries) < 2:
        print("  Skipping RQ1: insufficient data")
        return

    metrics = [("precision", "Precision"), ("recall", "Recall"), ("F1", "F1 Score")]
    model_names = [e["model"] for e in entries]

    for col, label in metrics:
        fig, ax = plt.subplots(figsize=(6, 5))
        values = [e[col] for e in entries]
        bar_colors = []
        for name in model_names:
            if "BMA" in name and "Logit" not in name:
                bar_colors.append(COLORS["BMA"])
            elif "Stacking" in name:
                bar_colors.append(COLORS["Stacking"])
            else:
                bar_colors.append(COLORS["Logit"])

        bars = ax.bar(model_names, values, color=bar_colors, alpha=0.8,
                      edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax.set_ylabel(label)
        ax.set_title(f"RQ1: {label} Comparison")
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        filename = os.path.join(plot_dir, f"rq1_{col}.png")
        plt.savefig(filename, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {filename}")


# ==================== RQ2 Plots ====================

def plot_rq2_re_boxplot(data, plot_dir):
    """RQ2: RE box plot — BMA, Stacking, Best Logit only."""
    order = ["BMA", "Stacking"] + [k for k in data if k.startswith("Best Logit")]
    present = [k for k in order if k in data]
    if not present:
        print("  Skipping RQ2 RE plot: no data")
        return

    box_data = [data[k]["RE"].values for k in present]
    box_colors = []
    for k in present:
        if "BMA" in k and "Logit" not in k:
            box_colors.append(COLORS["BMA"])
        elif "Stacking" in k:
            box_colors.append(COLORS["Stacking"])
        else:
            box_colors.append(COLORS["Logit"])

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(box_data, tick_labels=present, patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", markersize=3, alpha=0.5))
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_yscale("log")
    ax.set_ylabel("Relative Error (log scale)")
    ax.set_title("RQ2: Relative Error Comparison")
    ax.grid(True, alpha=0.3, axis="y")

    for i, k in enumerate(present):
        median = np.median(data[k]["RE"].values)
        ax.text(i + 1, median * 1.5, f"{median:.4f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    plt.tight_layout()
    filename = os.path.join(plot_dir, "rq2_relative_error.png")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def plot_rq2_success_rate(data, plot_dir):
    """RQ2: Success rate bar — BMA, Stacking, Best Logit only."""
    order = ["BMA", "Stacking"] + [k for k in data if k.startswith("Best Logit")]
    present = [k for k in order if k in data]
    if not present:
        print("  Skipping RQ2 success plot: no data")
        return

    rates = []
    bar_colors = []
    for k in present:
        sub = data[k]
        rate = sub["success"].apply(lambda x: x == True or x == "TRUE").mean()
        rates.append(rate)
        if "BMA" in k and "Logit" not in k:
            bar_colors.append(COLORS["BMA"])
        elif "Stacking" in k:
            bar_colors.append(COLORS["Stacking"])
        else:
            bar_colors.append(COLORS["Logit"])

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(present, rates, color=bar_colors, alpha=0.8,
                  edgecolor="black", linewidth=0.5)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{rate:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Success Rate")
    ax.set_title("RQ2: Adaptation Success Rate Comparison")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    filename = os.path.join(plot_dir, "rq2_success_rate.png")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


# ==================== RQ3 Plots ====================

def plot_rq3_heatmap(df, plot_dir):
    """RQ3: Training time heatmap (sample x vars) per method."""
    TIMEOUT_THRESHOLD = 200
    methods = df["method"].unique()

    for method in methods:
        subset = df[df["method"] == method]
        pivot = subset.pivot_table(index="sample", columns="vars", values="time", aggfunc="mean")
        pivot = pivot.sort_index(ascending=True)

        data = pivot.values
        display_data = np.where(data >= TIMEOUT_THRESHOLD, TIMEOUT_THRESHOLD, data)

        plt.figure(figsize=(6, 5))
        im = plt.imshow(display_data, aspect="auto", cmap="YlOrRd", origin="lower",
                        vmin=np.nanmin(display_data), vmax=TIMEOUT_THRESHOLD)

        plt.xticks(range(len(pivot.columns)), pivot.columns.astype(int), fontsize=9)
        plt.yticks(range(len(pivot.index)), pivot.index.astype(int), fontsize=9)
        plt.xlabel("Number of Variables")
        plt.ylabel("Number of Samples")
        plt.title(f"{method}")

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if np.isnan(val):
                    continue
                text = "TO" if val >= TIMEOUT_THRESHOLD else (f"{val:.1f}" if val < 10 else f"{int(val)}")
                color_val = display_data[i, j] if not np.isnan(display_data[i, j]) else 0
                plt.text(j, i, text, ha="center", va="center", fontsize=7,
                         color="white" if color_val > TIMEOUT_THRESHOLD * 0.6 else "black")

        cbar = plt.colorbar(im, label="Time (seconds)")
        if np.nanmax(display_data) <= 10:
            cbar.set_ticks(np.arange(0, 11, 2))
        else:
            cbar.set_ticks(np.arange(0, int(TIMEOUT_THRESHOLD) + 1, 50))

        plt.tight_layout()
        filename = os.path.join(plot_dir, f"rq3_heatmap_{method}.png")
        plt.savefig(filename, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {filename}")


# ==================== Main ====================

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    print("=== Generating Plots ===\n")

    # ---- RQ1 ----
    print("RQ1: Prediction Accuracy")
    rq1_entries = load_rq1_data()
    if rq1_entries:
        plot_rq1(rq1_entries, PLOT_DIR)
    else:
        print("  No RQ1 data available, skipping.")

    # ---- RQ2 ----
    print("\nRQ2: Adaptation Decisions")
    rq2_data = load_rq2_data()
    if rq2_data:
        plot_rq2_re_boxplot(rq2_data, PLOT_DIR)
        plot_rq2_success_rate(rq2_data, PLOT_DIR)
    else:
        print("  No RQ2 data available, skipping.")

    # ---- RQ3 ----
    print("\nRQ3: Computational Cost")
    rq3_df = load_rq3_data()
    if rq3_df is not None:
        plot_rq3_heatmap(rq3_df, PLOT_DIR)
    else:
        print("  No RQ3 data available, skipping.")

    print(f"\nAll plots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
