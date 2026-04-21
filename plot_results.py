#!/usr/bin/env python3
"""
plot_results.py
Generate publication-quality figures from RQ1/RQ2/RQ3 experiment outputs.

Setting A only. RQ1 and RQ2 outputs are per drift-detection threshold:
    logs/rq1_A_th{TH}.json
    logs/rq2_A_th{TH}.json

For each threshold we render the per-threshold plots. We also render a
cross-threshold comparison showing how drift sensitivity affects the online
model.

Usage:
    python plot_results.py             # all plots
    python plot_results.py --rq1-only
    python plot_results.py --rq2-only
    python plot_results.py --rq3-only
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
    'Stacking-Offline': '#348ABD',
    'BMA': '#988ED5',
    'Best-Logit': '#8EBA42',
    'MCMC': '#4C72B0',
    'BAS': '#55A868',
    'Stacking': '#DD8452',
}

METHOD_ORDER = ['Stacking-Online', 'Stacking-Offline', 'BMA', 'Best-Logit']
TIMEOUT = 200

_RE_RQ1 = re.compile(r"^rq1_A_th(\d+\.\d+)\.json$")
_RE_RQ2 = re.compile(r"^rq2_A_th(\d+\.\d+)\.json$")


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


def _load_json(path):
    with open(path) as f:
        return json.load(f)


# ==================== RQ1 ====================

def plot_rq1_bars(data, th):
    methods = [m for m in METHOD_ORDER if m in data['methods']]
    if not methods:
        return
    metrics = ['precision', 'recall', 'f1']
    labels = ['Precision', 'Recall', 'F1']
    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        values = [data['methods'][m][metric] for m in methods]
        bars = ax.bar(x + i * width, values, width, label=label, alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x + width)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(f"RQ1 (drift_threshold={th:.3f}): Prediction Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"rq1_A_th{th:.3f}_bars.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_rq1_f1_over_time(data, th):
    methods = [m for m in METHOD_ORDER if m in data['methods']]
    fig, ax = plt.subplots(figsize=(10, 5))
    for m in methods:
        samples = data['methods'][m]['per_sample']
        ts = [s['t'] for s in samples]
        f1s = [s['f1_window'] for s in samples]
        ax.plot(ts, f1s, label=m, color=_color(m), alpha=0.85, linewidth=1.5)

    ax.set_xlabel("Stream sample index")
    ax.set_ylabel("F1 (sliding window=50)")
    ax.set_title(f"RQ1 (drift_threshold={th:.3f}): F1 Over Stream")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"rq1_A_th{th:.3f}_f1_over_time.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_rq1_brier_over_time(data, th):
    methods = [m for m in METHOD_ORDER if m in data['methods']]
    fig, ax = plt.subplots(figsize=(10, 5))
    for m in methods:
        samples = data['methods'][m]['per_sample']
        ts = [s['t'] for s in samples]
        preds = [s['pred'] for s in samples]
        trues = [s['true'] for s in samples]
        window = 50
        briers = []
        for i in range(len(ts)):
            start = max(0, i - window + 1)
            p = np.array(preds[start:i+1])
            y = np.array(trues[start:i+1])
            briers.append(float(np.mean((p - y) ** 2)))
        ax.plot(ts, briers, label=m, color=_color(m), alpha=0.85, linewidth=1.5)

    ax.set_xlabel("Stream sample index")
    ax.set_ylabel("Brier Score (sliding window=50)")
    ax.set_title(f"RQ1 (drift_threshold={th:.3f}): Calibration Over Stream")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"rq1_A_th{th:.3f}_brier_over_time.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_rq1_threshold_sweep(runs):
    """Compare F1/Brier across drift thresholds for the online model."""
    if len(runs) < 2:
        return
    ths = [th for th, _ in runs]
    datasets = [_load_json(os.path.join(LOG_DIR, fn)) for _, fn in runs]

    fig, (ax_f1, ax_br) = plt.subplots(1, 2, figsize=(12, 5))
    for m in METHOD_ORDER:
        f1s = [d['methods'][m]['f1'] if m in d['methods'] else np.nan for d in datasets]
        brs = [d['methods'][m]['brier'] if m in d['methods'] else np.nan for d in datasets]
        ax_f1.plot(ths, f1s, marker='o', label=m, color=_color(m), linewidth=1.5)
        ax_br.plot(ths, brs, marker='o', label=m, color=_color(m), linewidth=1.5)

    ax_f1.set_xlabel("Drift threshold")
    ax_f1.set_ylabel("F1")
    ax_f1.set_title("RQ1: F1 vs Drift Threshold")
    ax_f1.grid(True, alpha=0.3)
    ax_f1.legend()

    ax_br.set_xlabel("Drift threshold")
    ax_br.set_ylabel("Brier Score")
    ax_br.set_title("RQ1: Brier vs Drift Threshold")
    ax_br.grid(True, alpha=0.3)
    ax_br.legend()

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "rq1_A_threshold_sweep.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ==================== RQ2 ====================

def plot_rq2_re_box(data, th):
    methods = [m for m in ['Stacking-Online', 'Stacking-Offline', 'BMA']
               if m in data['methods']]
    if not methods:
        return
    box_data = []
    for m in methods:
        re_vals = [s['RE'] for s in data['methods'][m]['per_sample']
                   if s['RE'] is not None]
        box_data.append(re_vals)

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(box_data, patch_artist=True,
                    showfliers=True,
                    flierprops=dict(marker='o', markersize=3, alpha=0.5))
    ax.set_xticklabels(methods)
    for patch, m in zip(bp['boxes'], methods):
        patch.set_facecolor(_color(m))
        patch.set_alpha(0.7)

    ax.set_yscale('log')
    ax.set_ylabel("Relative Error (log scale)")
    ax.set_title(f"RQ2 (drift_threshold={th:.3f}): Relative Error")
    ax.grid(True, alpha=0.3, axis='y')

    for i, m in enumerate(methods):
        med = np.median(box_data[i]) if box_data[i] else 0
        ax.text(i + 1, med * 1.5, f"{med:.4f}", ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"rq2_A_th{th:.3f}_re_box.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_rq2_success_bars(data, th):
    methods = [m for m in ['Stacking-Online', 'Stacking-Offline', 'BMA']
               if m in data['methods']]
    rates = [data['methods'][m]['success_rate'] for m in methods]
    colors = [_color(m) for m in methods]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(methods, rates, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=0.5)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{rate:.4f}", ha='center', va='bottom', fontsize=11,
                fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Success Rate")
    ax.set_title(f"RQ2 (drift_threshold={th:.3f}): Adaptation Success Rate")
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"rq2_A_th{th:.3f}_success_bars.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_rq2_re_over_time(data, th):
    methods = [m for m in ['Stacking-Online', 'Stacking-Offline', 'BMA']
               if m in data['methods']]
    window = 30

    fig, ax = plt.subplots(figsize=(10, 5))
    for m in methods:
        samples = data['methods'][m]['per_sample']
        re_vals = [s['RE'] for s in samples if s['RE'] is not None]
        if not re_vals:
            continue
        medians = []
        for i in range(len(re_vals)):
            start = max(0, i - window + 1)
            medians.append(np.median(re_vals[start:i+1]))
        ax.plot(range(len(medians)), medians, label=m, color=_color(m),
                alpha=0.85, linewidth=1.5)

    ax.set_xlabel("Hazard=0 sample index")
    ax.set_ylabel("Median RE (sliding window=30)")
    ax.set_title(f"RQ2 (drift_threshold={th:.3f}): RE Over Stream")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"rq2_A_th{th:.3f}_re_over_time.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_rq2_threshold_sweep(runs):
    """Compare median RE and success rate across thresholds."""
    if len(runs) < 2:
        return
    ths = [th for th, _ in runs]
    datasets = [_load_json(os.path.join(LOG_DIR, fn)) for _, fn in runs]
    methods = ['Stacking-Online', 'Stacking-Offline', 'BMA']

    fig, (ax_sr, ax_re) = plt.subplots(1, 2, figsize=(12, 5))
    for m in methods:
        srs = [d['methods'][m]['success_rate'] if m in d['methods'] else np.nan
               for d in datasets]
        mre = [d['methods'][m]['median_RE'] if m in d['methods'] else np.nan
               for d in datasets]
        ax_sr.plot(ths, srs, marker='o', label=m, color=_color(m), linewidth=1.5)
        ax_re.plot(ths, mre, marker='o', label=m, color=_color(m), linewidth=1.5)

    ax_sr.set_xlabel("Drift threshold")
    ax_sr.set_ylabel("Success Rate")
    ax_sr.set_ylim(0, 1.05)
    ax_sr.set_title("RQ2: Success Rate vs Drift Threshold")
    ax_sr.grid(True, alpha=0.3)
    ax_sr.legend()

    ax_re.set_xlabel("Drift threshold")
    ax_re.set_ylabel("Median Relative Error")
    ax_re.set_yscale('log')
    ax_re.set_title("RQ2: Median RE vs Drift Threshold")
    ax_re.grid(True, alpha=0.3)
    ax_re.legend()

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "rq2_A_threshold_sweep.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_rq2_drift_count(runs):
    """How many drifts fire at each threshold (for the Stacking-Online run)."""
    if len(runs) < 2:
        return
    ths = [th for th, _ in runs]
    drifts = []
    for _, fn in runs:
        d = _load_json(os.path.join(LOG_DIR, fn))
        drifts.append(d.get('diagnostics', {}).get('drift_count', 0))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([f"{th:.3f}" for th in ths], drifts,
                  color=_color('Stacking-Online'), alpha=0.8,
                  edgecolor='black', linewidth=0.5)
    for bar, n in zip(bars, drifts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                str(n), ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_xlabel("Drift threshold")
    ax.set_ylabel("Warm restarts triggered")
    ax.set_title("RQ2: Drift Detections per Threshold")
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "rq2_A_drift_counts.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


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

        plt.figure(figsize=(6, 5))
        im = plt.imshow(display, aspect='auto', cmap='YlOrRd', origin='lower',
                        vmin=np.nanmin(display), vmax=TIMEOUT)
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
                text = "TO" if val >= TIMEOUT else (f"{val:.1f}" if val < 10 else f"{int(val)}")
                cval = display[i, j] if not np.isnan(display[i, j]) else 0
                plt.text(j, i, text, ha='center', va='center', fontsize=7,
                         color='white' if cval > TIMEOUT * 0.6 else 'black')

        cbar = plt.colorbar(im, label="Time (seconds)")
        if np.nanmax(display) <= 10:
            cbar.set_ticks(np.arange(0, 11, 2))
        else:
            cbar.set_ticks(np.arange(0, int(TIMEOUT) + 1, 50))

        plt.tight_layout()
        path = os.path.join(PLOT_DIR, f"rq3_heatmap_{method}.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description='Generate result plots')
    parser.add_argument('--rq1-only', action='store_true')
    parser.add_argument('--rq2-only', action='store_true')
    parser.add_argument('--rq3-only', action='store_true')
    args = parser.parse_args()

    do_all = not (args.rq1_only or args.rq2_only or args.rq3_only)
    os.makedirs(PLOT_DIR, exist_ok=True)
    print("=== Generating Plots ===\n")

    if do_all or args.rq1_only:
        print("RQ1: Prediction Accuracy")
        runs = discover_thresholds(_RE_RQ1)
        if not runs:
            print("  No RQ1 data found in logs/")
        for th, fn in runs:
            data = _load_json(os.path.join(LOG_DIR, fn))
            plot_rq1_bars(data, th)
            plot_rq1_f1_over_time(data, th)
            plot_rq1_brier_over_time(data, th)
        plot_rq1_threshold_sweep(runs)

    if do_all or args.rq2_only:
        print("\nRQ2: Adaptation Decisions")
        runs = discover_thresholds(_RE_RQ2)
        if not runs:
            print("  No RQ2 data found in logs/")
        for th, fn in runs:
            data = _load_json(os.path.join(LOG_DIR, fn))
            plot_rq2_re_box(data, th)
            plot_rq2_success_bars(data, th)
            plot_rq2_re_over_time(data, th)
        plot_rq2_threshold_sweep(runs)
        plot_rq2_drift_count(runs)

    if do_all or args.rq3_only:
        print("\nRQ3: Computational Cost")
        df = load_rq3()
        if df is not None:
            plot_rq3_heatmap(df)
        else:
            print("  No RQ3 data")

    print(f"\nAll plots in: {PLOT_DIR}")


if __name__ == '__main__':
    main()
