#!/usr/bin/env python3
"""
plot_results.py
Generate publication-quality figures from RQ1/RQ2/RQ3/RQ4 experiment outputs.

Setting A only. Compared models: Stacking-Online, BMA, Best-Logit
(Stacking-Offline is intentionally NOT plotted). Brier-rate metrics are not
displayed (the metric is normalized and the storyline is precision/recall/F1).

Final figures keep ONLY axis labels — no titles or in-figure annotations.

Usage:
    python plot_results.py             # all plots
    python plot_results.py --rq1-only
    python plot_results.py --rq2-only
    python plot_results.py --rq3-only
    python plot_results.py --rq4-only
"""

import json
import argparse
import os
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
    "legend.fontsize": 10,
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
PLOT_DIR = os.path.join(BASE_DIR, "plots")

COLORS = {
    'Stacking-Online': '#E24A33',
    'BMA':             '#988ED5',
    'Best-Logit':      '#8EBA42',
    # Kept for incidental references; not used in any chart that we ship.
    'Stacking-Offline': '#348ABD',
    'MCMC': '#4C72B0',
    'BAS':  '#55A868',
    'Stacking': '#DD8452',
}

# All published charts compare these three models — no Stacking-Offline.
METHOD_ORDER = ['Stacking-Online', 'BMA', 'Best-Logit']
RQ2_METHODS  = ['Stacking-Online', 'BMA', 'Best-Logit']
TIMEOUT = 200

_RE_RQ1 = re.compile(r"^rq1_A_th(\d+\.\d+)\.json$")
# RQ2 logs may be oracle-tagged (rq2_A_th0.150_oraclelogit.json) — capture
# the oracle name optionally so legacy untagged logs still work.
_RE_RQ2 = re.compile(r"^rq2_A_th(\d+\.\d+)(?:_oracle(\w+))?\.json$")
_RE_RQ4 = re.compile(r"^rq4_A_th(\d+\.\d+)_drift(\w+)\.json$")


def _color(name):
    for key in COLORS:
        if key in name:
            return COLORS[key]
    return '#777777'


def discover_thresholds(regex):
    """Return sorted list of (threshold_float, filename) for matching outputs."""
    out = []
    if not os.path.isdir(LOG_DIR):
        return out
    for fname in sorted(os.listdir(LOG_DIR)):
        m = regex.match(fname)
        if m:
            out.append((float(m.group(1)), fname))
    return sorted(out, key=lambda x: x[0])


def discover_rq2_runs():
    """Discover all RQ2 logs as (threshold, oracle_or_None, filename).
    oracle is None for legacy un-tagged logs (treated as 'logit' in plot tags).
    """
    out = []
    if not os.path.isdir(LOG_DIR):
        return out
    for fname in sorted(os.listdir(LOG_DIR)):
        m = _RE_RQ2.match(fname)
        if m:
            out.append((float(m.group(1)), m.group(2), fname))
    return sorted(out, key=lambda x: (x[0], x[1] or 'zzz_legacy'))


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ==================== RQ1 ====================

def _plot_rq1_single_metric(data, th, metric_key, ylabel, fname_suffix):
    methods = [m for m in METHOD_ORDER if m in data['methods']]
    if not methods:
        return
    values = [data['methods'][m][metric_key] for m in methods]
    colors = [_color(m) for m in methods]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(methods, values, color=colors, alpha=0.85,
           edgecolor='black', linewidth=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    _save(fig, os.path.join(PLOT_DIR, f"rq1_A_th{th:.3f}_{fname_suffix}.png"))


def plot_rq1_metrics(data, th):
    """Three separate per-metric bar charts (precision, recall, f1)."""
    _plot_rq1_single_metric(data, th, 'precision', 'Precision', 'precision')
    _plot_rq1_single_metric(data, th, 'recall',    'Recall',    'recall')
    _plot_rq1_single_metric(data, th, 'f1',        'F1',        'f1')


# ==================== RQ2 ====================

def _rq2_metric_key(data):
    """Return (key, ylabel) — auto-detected from per_sample, falls back to abs_err."""
    for m in RQ2_METHODS:
        samples = data.get('methods', {}).get(m, {}).get('per_sample', [])
        if samples:
            keys = samples[0].keys()
            if 'RE' in keys:
                return 'RE', 'Relative Error'
            if 'abs_err' in keys:
                return 'abs_err', '|pred − oracle|'
    declared = data.get('metric_key', 'abs_err')
    return declared, ('Relative Error' if declared == 'RE' else '|pred − oracle|')


def _rq2_suffix(oracle):
    return f"_oracle{oracle}" if oracle else ""


def plot_rq2_metric_box(data, th, oracle=None):
    """Single-panel boxplot of the oracle-distance metric (RE or abs_err)."""
    methods = [m for m in RQ2_METHODS if m in data['methods']]
    if not methods:
        return
    key, ylabel = _rq2_metric_key(data)

    box_data = [
        [float(s[key]) for s in data['methods'][m]['per_sample']
         if key in s and s[key] is not None]
        for m in methods
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(box_data, patch_artist=True, showfliers=True,
                    flierprops=dict(marker='o', markersize=3, alpha=0.5))
    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels(methods)
    for patch, m in zip(bp['boxes'], methods):
        patch.set_facecolor(_color(m))
        patch.set_alpha(0.7)
    if key == 'RE':
        ax.set_yscale('log')
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    _save(fig, os.path.join(
        PLOT_DIR, f"rq2_A_th{th:.3f}{_rq2_suffix(oracle)}_metric_box.png"))


def plot_rq2_success_bars(data, th, oracle=None):
    """Bar chart of success rate (a.k.a. precision in this RQ)."""
    methods = [m for m in RQ2_METHODS if m in data['methods']]
    if not methods:
        return
    # Backward compat: older logs used 'success_rate'.
    rates = [
        data['methods'][m].get('precision',
                               data['methods'][m].get('success_rate', np.nan))
        for m in methods
    ]
    colors = [_color(m) for m in methods]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(methods, rates, color=colors, alpha=0.8,
           edgecolor='black', linewidth=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Model")
    ax.set_ylabel("Success Rate")
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    _save(fig, os.path.join(
        PLOT_DIR, f"rq2_A_th{th:.3f}{_rq2_suffix(oracle)}_success_bars.png"))


# ==================== RQ3 ====================

def load_rq3():
    frames = []
    for fname in ['rq3_stacking.tsv', 'baseline_rq3_results.tsv']:
        path = os.path.join(LOG_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path, sep=r"\s+")
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def plot_rq3_heatmap(df):
    for method in df['method'].unique():
        subset = df[df['method'] == method]
        pivot = subset.pivot_table(index='sample', columns='vars',
                                    values='time', aggfunc='mean')
        pivot = pivot.sort_index(ascending=True)
        data = pivot.values
        display = np.where(data >= TIMEOUT, TIMEOUT, data)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(display, aspect='auto', cmap='YlOrRd', origin='lower',
                       vmin=np.nanmin(display), vmax=TIMEOUT)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns.astype(int), fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index.astype(int), fontsize=9)
        ax.set_xlabel("Number of Variables")
        ax.set_ylabel("Number of Samples")

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if np.isnan(val):
                    continue
                text = "TO" if val >= TIMEOUT else (f"{val:.1f}" if val < 10 else f"{int(val)}")
                cval = display[i, j] if not np.isnan(display[i, j]) else 0
                ax.text(j, i, text, ha='center', va='center', fontsize=7,
                        color='white' if cval > TIMEOUT * 0.6 else 'black')

        cbar = fig.colorbar(im, ax=ax, label="Time (seconds)")
        if np.nanmax(display) <= 10:
            cbar.set_ticks(np.arange(0, 11, 2))
        else:
            cbar.set_ticks(np.arange(0, int(TIMEOUT) + 1, 50))

        plt.tight_layout()
        _save(fig, os.path.join(PLOT_DIR, f"rq3_heatmap_{method}.png"))


# ==================== RQ4 ====================

def discover_rq4():
    """Return sorted list of (threshold_float, scenario_str, filename)."""
    out = []
    if not os.path.isdir(LOG_DIR):
        return out
    for fname in sorted(os.listdir(LOG_DIR)):
        m = _RE_RQ4.match(fname)
        if m:
            out.append((float(m.group(1)), m.group(2), fname))
    return sorted(out, key=lambda x: (x[0], x[1]))


def _sliding_window_pr(samples, window=50):
    """Compute sliding-window precision, recall, and F1 from per_sample data."""
    preds_bin = [1 if s['pred'] >= 0.5 else 0 for s in samples]
    trues = [s['true'] for s in samples]
    ts = [s['t'] for s in samples]
    precisions, recalls, f1s = [], [], []
    for i in range(len(samples)):
        start = max(0, i - window + 1)
        p = preds_bin[start:i + 1]
        y = trues[start:i + 1]
        tp = sum(a == 1 and b == 1 for a, b in zip(p, y))
        fp = sum(a == 1 and b == 0 for a, b in zip(p, y))
        fn = sum(a == 0 and b == 1 for a, b in zip(p, y))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    return ts, precisions, recalls, f1s


# Only these drift scenarios are plotted; logs for other scenarios are ignored.
# Rationale: 1x / 3x / 6x form a clean ladder of transient drift intensity (1, 3,
# 6 bursts of 50 samples each), giving a controlled story for the paper.
RQ4_SCENARIOS = {'1x', '3x', '6x'}

_RQ4_METRIC_LABELS = {0: 'Precision', 1: 'Recall', 2: 'F1'}


def _plot_rq4_metric_over_time(data, th, scenario, metric_name, metric_idx):
    """Per-metric over-time plot for RQ4. Drift intervals are shaded; figure
    carries no title — only axis labels (project convention)."""
    methods = [m for m in METHOD_ORDER if m in data['methods']]
    if not methods:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for m in methods:
        samples = data['methods'][m]['per_sample']
        ts, precs, recs, f1s = _sliding_window_pr(samples)
        values = [precs, recs, f1s][metric_idx]
        ax.plot(ts, values, label=m, color=_color(m), alpha=0.85, linewidth=1.5)

    drift_intervals = data.get('drift_intervals', [])
    for start, end in drift_intervals:
        ax.axvline(x=start, color='red',   linestyle='--', alpha=0.7)
        ax.axvline(x=end,   color='green', linestyle='--', alpha=0.7)
        ax.axvspan(start, end, alpha=0.15, color='red')

    ax.set_xlabel(f"Stream sample index  (drift = {scenario})")
    ax.set_ylabel(f"{_RQ4_METRIC_LABELS[metric_idx]} (sliding window=50)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, os.path.join(
        PLOT_DIR,
        f"rq4_A_th{th:.3f}_drift{scenario}_{metric_name}_over_time.png"))


def _plot_rq4_metric_box(data, th, scenario, metric_name, metric_idx):
    """Per-metric boxplot for RQ4: aggregates the sliding-window distribution
    across the entire stream, one box per method. Complements the time-series:
    while time-series shows recovery dynamics, boxplot shows central tendency,
    spread and outliers across the whole run."""
    methods = [m for m in METHOD_ORDER if m in data['methods']]
    if not methods:
        return

    box_data = []
    for m in methods:
        samples = data['methods'][m]['per_sample']
        _ts, precs, recs, f1s = _sliding_window_pr(samples)
        box_data.append([precs, recs, f1s][metric_idx])

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(box_data, patch_artist=True, showfliers=True,
                    flierprops=dict(marker='o', markersize=3, alpha=0.5))
    for patch, m in zip(bp['boxes'], methods):
        patch.set_facecolor(_color(m))
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels(methods)
    ax.set_xlabel(f"Model  (drift = {scenario})")
    ax.set_ylabel(f"{_RQ4_METRIC_LABELS[metric_idx]} (sliding window=50)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    _save(fig, os.path.join(
        PLOT_DIR,
        f"rq4_A_th{th:.3f}_drift{scenario}_{metric_name}_box.png"))


def run_rq4():
    print("\nRQ4: Concept Drift")
    runs = discover_rq4()
    if not runs:
        print("  No RQ4 data found in logs/")
        return
    selected = [(th, sc, fn) for th, sc, fn in runs if sc in RQ4_SCENARIOS]
    skipped  = sorted({sc for _, sc, _ in runs if sc not in RQ4_SCENARIOS})
    if skipped:
        print(f"  Skipping non-target scenarios: {skipped}")
    if not selected:
        print(f"  No RQ4 data for target scenarios {sorted(RQ4_SCENARIOS)}")
        return
    for th, scenario, fn in selected:
        data = _load_json(os.path.join(LOG_DIR, fn))
        for metric_name, idx in [('precision', 0), ('recall', 1), ('f1', 2)]:
            _plot_rq4_metric_over_time(data, th, scenario, metric_name, idx)
            _plot_rq4_metric_box       (data, th, scenario, metric_name, idx)


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description='Generate result plots')
    parser.add_argument('--rq1-only', action='store_true')
    parser.add_argument('--rq2-only', action='store_true')
    parser.add_argument('--rq3-only', action='store_true')
    parser.add_argument('--rq4-only', action='store_true')
    args = parser.parse_args()

    do_all = not (args.rq1_only or args.rq2_only or args.rq3_only or args.rq4_only)
    os.makedirs(PLOT_DIR, exist_ok=True)
    print("=== Generating Plots ===\n")

    if do_all or args.rq1_only:
        print("RQ1: Prediction Accuracy")
        runs = discover_thresholds(_RE_RQ1)
        if not runs:
            print("  No RQ1 data found in logs/")
        for th, fn in runs:
            data = _load_json(os.path.join(LOG_DIR, fn))
            plot_rq1_metrics(data, th)

    if do_all or args.rq2_only:
        print("\nRQ2: Adaptation Decisions")
        runs = discover_rq2_runs()
        if not runs:
            print("  No RQ2 data found in logs/")
        for th, oracle, fn in runs:
            data = _load_json(os.path.join(LOG_DIR, fn))
            print(f"  [{fn}] threshold={th:.3f} oracle={oracle or 'legacy'}")
            plot_rq2_metric_box(data, th, oracle=oracle)
            plot_rq2_success_bars(data, th, oracle=oracle)

    if do_all or args.rq3_only:
        print("\nRQ3: Computational Cost")
        df = load_rq3()
        if df is not None:
            plot_rq3_heatmap(df)
        else:
            print("  No RQ3 data")

    if do_all or args.rq4_only:
        run_rq4()

    print(f"\nAll plots in: {PLOT_DIR}")


if __name__ == '__main__':
    main()
