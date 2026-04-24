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
        p = preds_bin[start:i+1]
        y = trues[start:i+1]
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


def _plot_rq4_metric_over_time(data, th, scenario, metric_name, metric_idx):
    """Generic per-metric over-time plot for RQ4.
    metric_idx: 0=precision, 1=recall, 2=f1"""
    methods = [m for m in METHOD_ORDER if m in data['methods']]
    fig, ax = plt.subplots(figsize=(10, 5))
    label_map = {0: 'Precision', 1: 'Recall', 2: 'F1'}
    for m in methods:
        samples = data['methods'][m]['per_sample']
        ts, precs, recs, f1s = _sliding_window_pr(samples)
        values = [precs, recs, f1s][metric_idx]
        ax.plot(ts, values, label=m, color=_color(m), alpha=0.85, linewidth=1.5)

    drift_intervals = data.get('drift_intervals', [])
    for start, end in drift_intervals:
        ax.axvline(x=start, color='red', linestyle='--', alpha=0.7)
        ax.axvline(x=end, color='green', linestyle='--', alpha=0.7)
        ax.axvspan(start, end, alpha=0.15, color='red')

    ax.set_xlabel("Stream sample index")
    ax.set_ylabel(f"{label_map[metric_idx]} (sliding window=50)")
    ax.set_title(f"RQ4 (th={th:.3f}, drift={scenario}): {label_map[metric_idx]} Over Stream")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR,
                        f"rq4_A_th{th:.3f}_drift{scenario}_{metric_name}_over_time.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_rq4_recovery_bars(data, th, scenario):
    # segment_metrics format: {method: {interval_key: {segment: float}}}
    segment_metrics = data.get('segment_metrics', {})
    if not segment_metrics:
        return

    methods = [m for m in METHOD_ORDER if m in segment_metrics]
    if not methods:
        return
    segments = ['pre_drift', 'during_drift', 'post_drift']
    seg_labels = ['Pre-Drift', 'During Drift', 'Post-Drift']

    interval_keys = sorted(segment_metrics[methods[0]].keys())
    n_intervals = len(interval_keys)
    n_methods = len(methods)

    fig, axes = plt.subplots(1, max(n_intervals, 1), figsize=(7 * n_intervals, 6),
                             squeeze=False)
    for idx, interval_key in enumerate(interval_keys):
        ax = axes[0][idx]
        x = np.arange(len(segments))
        width = 0.8 / max(n_methods, 1)

        for j, m in enumerate(methods):
            vals = []
            for seg in segments:
                v = segment_metrics[m].get(interval_key, {}).get(seg)
                vals.append(v if v is not None else 0.0)
            bars = ax.bar(x + j * width, vals, width, label=m, color=_color(m), alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x + width * (n_methods - 1) / 2)
        ax.set_xticklabels(seg_labels, rotation=15, ha='right')
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("F1")
        ax.set_title(interval_key.replace('_', ' ').title())
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f"RQ4 (th={th:.3f}, drift={scenario}): Recovery Analysis", fontsize=14)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"rq4_A_th{th:.3f}_drift{scenario}_recovery.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_rq4_comparison(rq4_runs):
    """Per-width cross-scenario comparison (one plot per metric per threshold)."""
    if not rq4_runs:
        return
    thresholds = sorted(set(th for th, _, _ in rq4_runs))
    counts = ['1x', '2x', '3x', '4x', '5x', '6x']
    # widths present in the data (suffix after count, '' means w50/original)
    width_styles = {
        '':    ('w=50', '-',  'o'),
        '_w10': ('w=10', '--', 's'),
        '_w30': ('w=30', ':',  '^'),
    }
    metric_info = [('f1', 'F1'), ('precision', 'Precision'), ('recall', 'Recall')]

    for metric_key, metric_label in metric_info:
        for th in thresholds:
            th_runs = [(sc, fn) for t, sc, fn in rq4_runs if t == th]
            sc_map = {sc: _load_json(os.path.join(LOG_DIR, fn)) for sc, fn in th_runs}

            # Two focus methods: Online vs Offline — clearest story
            focus = ['Stacking-Online', 'Stacking-Offline']
            method_colors = {'Stacking-Online': '#E24A33', 'Stacking-Offline': '#348ABD'}

            fig, ax = plt.subplots(figsize=(10, 5))
            x_ticks = list(range(1, 7))

            for suffix, (wlabel, ls, marker) in width_styles.items():
                for m in focus:
                    vals = []
                    for cnt in counts:
                        key = cnt + suffix
                        if key in sc_map and m in sc_map[key]['methods']:
                            vals.append(sc_map[key]['methods'][m][metric_key])
                        else:
                            vals.append(np.nan)
                    if all(np.isnan(v) for v in vals):
                        continue
                    ax.plot(x_ticks, vals, linestyle=ls, marker=marker,
                            color=method_colors[m], linewidth=1.5, alpha=0.85,
                            label=f"{m} ({wlabel})")

            ax.set_xticks(x_ticks)
            ax.set_xticklabels([f"{c}" for c in counts])
            ax.set_xlabel("Number of drift injections")
            ax.set_ylabel(metric_label)
            ax.set_title(f"RQ4 (th={th:.3f}): {metric_label} — Online vs Offline across drift counts & widths")
            ax.legend(fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)
            plt.tight_layout()
            path = os.path.join(PLOT_DIR, f"rq4_cross_scenario_{metric_key}_th{th:.3f}.png")
            plt.savefig(path, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {path}")


def plot_rq4_width_gap(rq4_runs):
    """Show Online−Offline F1 gap as a function of drift count, one line per width."""
    if not rq4_runs:
        return
    thresholds = sorted(set(th for th, _, _ in rq4_runs))
    counts = ['1x', '2x', '3x', '4x', '5x', '6x']
    width_info = [('', 'w=50 (strong)', '-'), ('_w30', 'w=30 (medium)', '--'), ('_w10', 'w=10 (mild)', ':')]

    for th in thresholds:
        th_runs = [(sc, fn) for t, sc, fn in rq4_runs if t == th]
        sc_map = {sc: _load_json(os.path.join(LOG_DIR, fn)) for sc, fn in th_runs}

        fig, ax = plt.subplots(figsize=(9, 5))
        x_ticks = list(range(1, 7))
        for suffix, wlabel, ls in width_info:
            gaps = []
            for cnt in counts:
                key = cnt + suffix
                if key not in sc_map:
                    gaps.append(np.nan)
                    continue
                m_on = sc_map[key]['methods'].get('Stacking-Online', {}).get('f1', np.nan)
                m_off = sc_map[key]['methods'].get('Stacking-Offline', {}).get('f1', np.nan)
                gaps.append(m_on - m_off if not np.isnan(m_on) and not np.isnan(m_off) else np.nan)
            if all(np.isnan(g) for g in gaps):
                continue
            ax.plot(x_ticks, gaps, linestyle=ls, marker='o', linewidth=2, label=wlabel)

        ax.axhline(0, color='black', linewidth=0.8, linestyle='-')
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(counts)
        ax.set_xlabel("Number of drift injections")
        ax.set_ylabel("F1 gap (Online − Offline)")
        ax.set_title(f"RQ4 (th={th:.3f}): Online advantage grows with drift frequency & intensity")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(PLOT_DIR, f"rq4_online_offline_gap_th{th:.3f}.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")


def run_rq4():
    print("\nRQ4: Concept Drift")
    runs = discover_rq4()
    if not runs:
        print("  No RQ4 data found in logs/")
        return
    for th, scenario, fn in runs:
        data = _load_json(os.path.join(LOG_DIR, fn))
        _plot_rq4_metric_over_time(data, th, scenario, 'f1', 2)
        _plot_rq4_metric_over_time(data, th, scenario, 'precision', 0)
        _plot_rq4_metric_over_time(data, th, scenario, 'recall', 1)
        plot_rq4_recovery_bars(data, th, scenario)
    plot_rq4_comparison(runs)
    plot_rq4_width_gap(runs)


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

    if do_all or args.rq4_only:
        run_rq4()

    print(f"\nAll plots in: {PLOT_DIR}")


if __name__ == '__main__':
    main()
