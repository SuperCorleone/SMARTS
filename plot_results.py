#!/usr/bin/env python3
"""
plot_results.py
Generate publication-quality figures from RQ1/RQ2/RQ3 experiment outputs.

Setting A only. Compared models: Stacking-Online, BMA, Best-Logit
(Stacking-Offline is intentionally NOT plotted). Brier-rate metrics are not
displayed (the metric is normalized and the storyline is precision/recall/F1).

Final figures keep ONLY axis labels — no titles or in-figure annotations.

Usage:
    python plot_results.py             # all plots
    python plot_results.py --rq1-only
    python plot_results.py --rq2-only
    python plot_results.py --rq3-only
    python plot_results.py --rq4-only   # compatibility alias for drift-only RQ1
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
    "font.size": 14,
    "axes.labelsize": 18,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "figure.dpi": 150,
    "legend.fontsize": 12,
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

# Display label override: data files (TSV/JSON) keep 'Stacking-Online' for
# back-compat; published figures show 'SMARTS' instead.
DISPLAY_NAME = {'Stacking-Online': 'SMARTS'}


def _disp(name):
    return DISPLAY_NAME.get(name, name)

_RE_RQ1 = re.compile(r"^rq1_A_th(?P<th>\d+\.\d+)_drift(?P<drift>\w+)\.json$")
# RQ2 logs are tagged with optional drift scenario, threshold, optional
# oracle name, and optional oracle-label-source segment, e.g.
# rq2_A_drift1x_th0.150_oraclelogit_oraclelabelsmatch-drift.json.
# The drift segment is absent for the no-drift baseline (legacy filenames).
_RE_RQ2 = re.compile(
    r"^rq2_A(?:_drift(?P<drift>[A-Za-z0-9]+))?"
    r"_th(?P<th>\d+\.\d+)"
    r"(?:_oracle(?P<oracle>\w+))?"
    r"(?:_oraclelabels(?P<oracle_label_source>[A-Za-z0-9-]+))?\.json$"
)
_RE_RQ1_LEGACY_RQ4 = re.compile(r"^rq4_A_th(?P<th>\d+\.\d+)_drift(?P<drift>\w+)\.json$")


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
    """Discover all RQ2 logs as
    (threshold, drift_or_None, oracle_or_None, oracle_label_source_or_None, filename).
    drift is None for the no-drift baseline (legacy filename); oracle is None
    for un-tagged legacy logs.
    """
    out = []
    if not os.path.isdir(LOG_DIR):
        return out
    for fname in sorted(os.listdir(LOG_DIR)):
        m = _RE_RQ2.match(fname)
        if m:
            out.append((
                float(m.group('th')),
                m.group('drift'),
                m.group('oracle'),
                m.group('oracle_label_source'),
                fname,
            ))
    return sorted(out, key=lambda x: (x[0], x[1] or '', x[2] or 'zzz_legacy', x[3] or 'zzz_legacy'))


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ==================== RQ1 ====================


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


def _rq2_filename(th, drift, oracle, oracle_label_source, suffix):
    """Compose the canonical RQ2 plot filename.

    Pattern: rq2_A[_drift{sc}]_th{th}[_oracle{o}][_oraclelabels{s}]_{suffix}.png
    """
    drift_part  = f"_drift{drift}"  if drift  else ""
    oracle_part = f"_oracle{oracle}" if oracle else ""
    source_part = (f"_oraclelabels{oracle_label_source}"
                   if oracle_label_source else "")
    return f"rq2_A{drift_part}_th{th:.3f}{oracle_part}{source_part}_{suffix}.png"


# RE is mathematically undefined when the oracle hazard probability approaches
# zero (TUNE's strict definition: |p-o|/o, no epsilon). For reporting we drop
# rows where oracle < RE_ORACLE_EPS so the boxplot is not dominated by 1e7-
# scale outliers from points where the GA-optimized features fall outside the
# 450-row warmup distribution and the BMA-over-Logit oracle saturates by
# extrapolation. Underlying JSON keeps every row (TUNE-faithful raw data).
RE_ORACLE_EPS = 1e-4


def plot_rq2_metric_box(data, th, drift=None, oracle=None, oracle_label_source=None):
    """Single-panel boxplot of the oracle-distance metric (RE or abs_err).

    For RE plots, rows with oracle < RE_ORACLE_EPS are excluded from the
    distribution; per-method retention counts are stamped onto the figure.
    """
    methods = [m for m in RQ2_METHODS if m in data['methods']]
    if not methods:
        return
    key, ylabel = _rq2_metric_key(data)

    box_data, n_kept, n_total = [], {}, {}
    for m in methods:
        rows = data['methods'][m]['per_sample']
        n_total[m] = len(rows)
        vals = []
        for s in rows:
            v = s.get(key)
            if v is None:
                continue
            v = float(v)
            if not np.isfinite(v):
                continue
            if key == 'RE':
                ov = float(s.get('oracle', 0.0))
                if ov < RE_ORACLE_EPS:
                    continue
            vals.append(v)
        n_kept[m] = len(vals)
        box_data.append(vals)

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(box_data, patch_artist=True, showfliers=True,
                    flierprops=dict(marker='o', markersize=3, alpha=0.5))
    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels([_disp(m) for m in methods])
    for patch, m in zip(bp['boxes'], methods):
        patch.set_facecolor(_color(m))
        patch.set_alpha(0.7)
    if key == 'RE':
        ax.set_yscale('log')
        note = "  ".join(f"{_disp(m)}: {n_kept[m]}/{n_total[m]}"
                         for m in methods)
        ax.text(0.02, 0.98,
                f"oracle ≥ {RE_ORACLE_EPS:g}: {note}",
                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    xlab = "Method"
    if drift:
        xlab = f"Method"
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    _save(fig, os.path.join(PLOT_DIR,
        _rq2_filename(th, drift, oracle, oracle_label_source, 'metric_box')))


def plot_rq2_success_bars(data, th, drift=None, oracle=None, oracle_label_source=None):
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
    ax.bar([_disp(m) for m in methods], rates, color=colors, alpha=0.8,
           edgecolor='black', linewidth=0.5)
    ax.set_ylim(0, 1.05)
    xlab = "Method"
    if drift:
        xlab = f"Method"
    ax.set_xlabel(xlab)
    ax.set_ylabel("Success Rate")
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    _save(fig, os.path.join(PLOT_DIR,
        _rq2_filename(th, drift, oracle, oracle_label_source, 'success_bars')))


# ==================== RQ3 ====================

# The published comparison includes only these three (BAS dropped per spec).
# MCMC remains in the dataset (loaded but not plotted) — kept as reference data.
RQ3_PLOT_ORDER    = ['Stacking', 'BMA', 'Best-Logit']
RQ3_METHOD_COLORS = {
    'Stacking':   '#E24A33',
    'BMA':        '#988ED5',
    'Best-Logit': '#8EBA42',
    'MCMC':       '#4C72B0',  # kept for incidental references; not plotted
    'BAS':        '#55A868',
}


def load_rq3():
    """Load all RQ3 method TSVs and normalize the schema.

    Reads (in order, later overrides earlier for same method):
      logs/rq3_stacking.tsv   (local Python sampler)
      logs/rq3_bma.tsv        (local Python sampler)
      logs/rq3_mcmc.tsv       (local R sampler — preferred over baseline copy)
      logs/rq3_bas.tsv        (local R sampler — preferred over baseline copy)
      logs/baseline_rq3_results.tsv  (TUNE-hardware copy; only used if a method
                                      has no local TSV — e.g. user hasn't run
                                      MCMC/BAS yet on their cluster)

    Returns DataFrame with columns: method, vars, sample, run_id, time.
    Missing run_id (old single-replicate TSVs) is filled with 0.
    """
    frames = {}  # method -> DataFrame
    primary_files = [
        'rq3_stacking.tsv', 'rq3_bma.tsv', 'rq3_best_logit.tsv',
        'rq3_mcmc.tsv', 'rq3_bas.tsv',
    ]
    for fname in primary_files:
        path = os.path.join(LOG_DIR, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, sep=r"\s+")
        if 'run_id' not in df.columns:
            df['run_id'] = 0
        for m in df['method'].unique():
            frames[m] = df[df['method'] == m]

    # Fall back to the TUNE-hardware copy ONLY for methods we don't have local
    # data for. We deliberately let local data win so the cross-method
    # comparison stays on shared hardware.
    fallback = os.path.join(LOG_DIR, 'baseline_rq3_results.tsv')
    if os.path.exists(fallback):
        df = pd.read_csv(fallback, sep=r"\s+")
        if 'run_id' not in df.columns:
            df['run_id'] = 0
        for m in df['method'].unique():
            if m not in frames:
                frames[m] = df[df['method'] == m].copy()
                frames[m]['_baseline_hardware'] = True  # tag for caption note

    if not frames:
        return None
    return pd.concat(frames.values(), ignore_index=True)


def plot_rq3_boxplot_single(df_a, df_b=None, drift_scenario='6x'):
    """Single canonical RQ3 figure mixing two cost semantics on a common
    'cost per adaptation event' axis:

      - SMARTS (Stacking-Online):  per-sample online_update latency under
                                   drift=`drift_scenario`, from RQ3-B TSVs.
      - BMA:                       per-sample online_update latency under
                                   drift=`drift_scenario`, from RQ3-B TSVs.
      - Best-Logit:                one-shot fit() time on the (sample×vars)
                                   grid, from RQ3-A TSV (no online_update,
                                   so the 'event' is a full re-fit).

    The two semantics share units (seconds) and log-scale; readers must rely
    on the figure caption for the per-method definition.
    """
    boxes, labels, colors = [], [], []

    # SMARTS + BMA from RQ3-B drift=6x (online_update per sample)
    if df_b is not None:
        sub = df_b[df_b['drift'] == drift_scenario]
        for m in ['Stacking-Online', 'BMA']:
            cell = sub[sub['method'] == m]['time_seconds'].values
            if len(cell):
                boxes.append(cell)
                labels.append(_disp(m))
                colors.append(_color(m))

    # Best-Logit from RQ3-A (one-shot fit on the synthetic grid)
    if df_a is not None:
        cell = df_a[df_a['method'] == 'Best-Logit']['time'].values
        if len(cell):
            boxes.append(cell)
            labels.append(_disp('Best-Logit'))
            colors.append(RQ3_METHOD_COLORS.get('Best-Logit', '#777777'))

    if not boxes:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(boxes, patch_artist=True, showfliers=True,
                    flierprops=dict(marker='o', markersize=3, alpha=0.5),
                    widths=0.35)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    for med in bp['medians']:
        med.set_color('black')

    # TIMEOUT reference applies only to the RQ3-A Best-Logit fit cells, but
    # the line is informative regardless (kept for visual cue).
    ax.axhline(TIMEOUT, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_yscale('log')
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Method")
    ax.set_ylabel("Time (seconds, log scale)")
    ax.grid(True, alpha=0.3, axis='y', which='both')
    plt.tight_layout()
    _save(fig, os.path.join(PLOT_DIR, "rq3_boxplot.png"))


# ---- RQ3-B: per-sample online_update latency under drift -----------------

# RQ3-B uses 2 methods only — Best-Logit has no online update by design.
RQ3B_PLOT_ORDER  = ['Stacking-Online', 'BMA']
RQ3B_DRIFT_ORDER = ['1x', '3x', '6x']

_RE_RQ3_ONLINE_LEGACY = re.compile(r"^rq3_online_drift([A-Za-z0-9]+)\.tsv$")
_RE_RQ3_ONLINE_GRID = re.compile(
    r"^rq3_online_grid_drift(?P<drift>[A-Za-z0-9]+)_re(?P<retrain>\d+)\.tsv$")


def discover_rq3_online():
    """Return sorted list of (drift_scenario, filename, mode) for RQ3-B TSVs."""
    out = []
    if not os.path.isdir(LOG_DIR):
        return out
    filenames = sorted(os.listdir(LOG_DIR))
    seen_grid = set()
    for fname in filenames:
        m = _RE_RQ3_ONLINE_GRID.match(fname)
        if m:
            drift = m.group('drift')
            seen_grid.add(drift)
    for fname in filenames:
        m = _RE_RQ3_ONLINE_GRID.match(fname)
        if m:
            out.append((m.group('drift'), fname, 'grid'))
            continue
        m = _RE_RQ3_ONLINE_LEGACY.match(fname)
        if m and m.group(1) not in seen_grid:
            out.append((m.group(1), fname, 'legacy'))
    # Sort by canonical drift order; unknown scenarios go to the end.
    def _key(item):
        sc = item[0]
        return (RQ3B_DRIFT_ORDER.index(sc) if sc in RQ3B_DRIFT_ORDER else 99, sc)
    return sorted(out, key=_key)


def load_rq3_online():
    """Concatenate all RQ3-B TSVs into one DataFrame.
    Normalized schema includes: method drift time_seconds.
    """
    runs = discover_rq3_online()
    if not runs:
        return None
    frames = []
    for sc, fname, mode in runs:
        df = pd.read_csv(os.path.join(LOG_DIR, fname), sep=r"\s+")
        if mode == 'grid':
            if 'time' in df.columns and 'time_seconds' not in df.columns:
                df = df.rename(columns={'time': 'time_seconds'})
            df['online_mode'] = 'grid'
        else:
            df['online_mode'] = 'legacy'
        if 'drift' not in df.columns:
            df['drift'] = sc
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def plot_rq3_online_per_drift(df):
    """One figure per drift scenario, X = method (2 boxes), Y = log latency."""
    for sc in RQ3B_DRIFT_ORDER:
        sub = df[df['drift'] == sc]
        methods = [m for m in RQ3B_PLOT_ORDER if m in sub['method'].unique()]
        if not methods:
            continue
        box_data = [sub[sub['method'] == m]['time_seconds'].values
                    for m in methods]
        fig, ax = plt.subplots(figsize=(6, 5))
        bp = ax.boxplot(box_data, patch_artist=True, showfliers=True,
                        flierprops=dict(marker='o', markersize=3, alpha=0.5),
                        widths=0.6)
        for patch, m in zip(bp['boxes'], methods):
            patch.set_facecolor(_color(m))
            patch.set_alpha(0.7)
        for med in bp['medians']:
            med.set_color('black')
        ax.set_yscale('log')
        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels([_disp(m) for m in methods])
        ax.set_xlabel(f"Method")
        ax.set_ylabel("Per-sample online_update latency (s, log scale)")
        ax.grid(True, alpha=0.3, axis='y', which='both')
        plt.tight_layout()
        _save(fig, os.path.join(
            PLOT_DIR, f"rq3_online_boxplot_drift{sc}.png"))


def plot_rq3_online_combined(df):
    """Single combined figure: X = method, grouped by drift scenario.
    2 methods × 3 drifts = 6 boxes side by side, color = method, x-label = drift.
    """
    methods = [m for m in RQ3B_PLOT_ORDER if m in df['method'].unique()]
    drifts  = [d for d in RQ3B_DRIFT_ORDER if d in df['drift'].unique()]
    if not methods or not drifts:
        return

    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(drifts) * len(methods)), 5))
    n_methods = len(methods)
    width = 0.8 / n_methods
    for j, m in enumerate(methods):
        positions, data = [], []
        for i, d in enumerate(drifts):
            cell = df[(df['method'] == m) & (df['drift'] == d)]['time_seconds'].values
            if not len(cell):
                continue
            positions.append(i + 1 + (j - (n_methods - 1) / 2) * width)
            data.append(cell)
        if not data:
            continue
        bp = ax.boxplot(data, positions=positions, widths=width * 0.85,
                        patch_artist=True, showfliers=False, manage_ticks=False)
        color = _color(m)
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor(color)
        for med in bp['medians']:
            med.set_color('black')

    ax.set_yscale('log')
    ax.set_xticks(range(1, len(drifts) + 1))
    ax.set_xticklabels([f"drift = {d}" for d in drifts])
    ax.set_xlabel("Drift scenario")
    ax.set_ylabel("Per-sample online_update latency (s, log scale)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=_color(m), alpha=0.7,
                              label=_disp(m))
               for m in methods]
    ax.legend(handles=handles, loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y', which='both')
    plt.tight_layout()
    _save(fig, os.path.join(PLOT_DIR, "rq3_online_boxplot.png"))


def plot_rq3_boxplots_by_vars(df):
    """One panel per method (2x2). X = vars, Y = log time, one box per
    vars-bucket aggregating across (sample-sizes × replicates). Shows
    scaling-with-vars distribution for each method on the same axis range.
    Logs tagged `_baseline_hardware` are noted in the panel ylabel.
    """
    methods = [m for m in RQ3_PLOT_ORDER if m in df['method'].unique()]
    if not methods:
        return
    n = len(methods)
    cols = 2 if n > 2 else n
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows),
                             squeeze=False, sharey=True)

    all_vars = sorted(df['vars'].unique())
    for idx, m in enumerate(methods):
        ax = axes[idx // cols][idx % cols]
        sub = df[df['method'] == m]
        is_baseline = bool(sub.get('_baseline_hardware', pd.Series([False])).any())

        groups = [sub[sub['vars'] == v]['time'].values for v in all_vars]
        groups = [g if len(g) else np.array([np.nan]) for g in groups]

        bp = ax.boxplot(groups, patch_artist=True, showfliers=True,
                        flierprops=dict(marker='o', markersize=3, alpha=0.5),
                        widths=0.6)
        for patch in bp['boxes']:
            patch.set_facecolor(RQ3_METHOD_COLORS.get(m, '#777777'))
            patch.set_alpha(0.65)

        ax.axhline(TIMEOUT, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.set_yscale('log')
        ax.set_xticks(range(1, len(all_vars) + 1))
        ax.set_xticklabels([str(v) for v in all_vars])
        ax.set_xlabel("Number of variables")
        ylabel = "Time (seconds, log scale)"
        if is_baseline:
            ylabel += "\n[baseline hardware]"
        ax.set_ylabel(ylabel)
        ax.set_title(m)
        ax.grid(True, alpha=0.3, axis='y', which='both')

    # Hide unused axes
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis('off')

    plt.tight_layout()
    _save(fig, os.path.join(PLOT_DIR, "rq3_boxplots_by_vars.png"))


def plot_rq3_boxplots_by_samples(df):
    """One panel per method, X = sample size, Y = log time. Same idea as
    by_vars, but for the sample-size axis."""
    methods = [m for m in RQ3_PLOT_ORDER if m in df['method'].unique()]
    if not methods:
        return
    n = len(methods)
    cols = 2 if n > 2 else n
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows),
                             squeeze=False, sharey=True)
    all_samples = sorted(df['sample'].unique())
    for idx, m in enumerate(methods):
        ax = axes[idx // cols][idx % cols]
        sub = df[df['method'] == m]
        is_baseline = bool(sub.get('_baseline_hardware', pd.Series([False])).any())
        groups = [sub[sub['sample'] == s]['time'].values for s in all_samples]
        groups = [g if len(g) else np.array([np.nan]) for g in groups]
        bp = ax.boxplot(groups, patch_artist=True, showfliers=True,
                        flierprops=dict(marker='o', markersize=3, alpha=0.5),
                        widths=0.6)
        for patch in bp['boxes']:
            patch.set_facecolor(RQ3_METHOD_COLORS.get(m, '#777777'))
            patch.set_alpha(0.65)
        ax.axhline(TIMEOUT, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.set_yscale('log')
        ax.set_xticks(range(1, len(all_samples) + 1))
        ax.set_xticklabels([str(s) for s in all_samples], rotation=30, ha='right')
        ax.set_xlabel("Number of observations")
        ylabel = "Time (seconds, log scale)"
        if is_baseline:
            ylabel += "\n[baseline hardware]"
        ax.set_ylabel(ylabel)
        ax.set_title(m)
        ax.grid(True, alpha=0.3, axis='y', which='both')
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis('off')
    plt.tight_layout()
    _save(fig, os.path.join(PLOT_DIR, "rq3_boxplots_by_samples.png"))


def plot_rq3_combined_box(df):
    """Single figure, X = vars (categorical), grouped boxes (4 methods per
    vars-bucket). Most useful for a paper RQ3 'cost' figure that lets readers
    compare methods at each problem size head-to-head."""
    methods = [m for m in RQ3_PLOT_ORDER if m in df['method'].unique()]
    if not methods:
        return
    all_vars = sorted(df['vars'].unique())
    n_methods = len(methods)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(all_vars) * n_methods), 5))
    for j, m in enumerate(methods):
        sub = df[df['method'] == m]
        positions, data = [], []
        for i, v in enumerate(all_vars):
            cell = sub[sub['vars'] == v]['time'].values
            if not len(cell):
                continue
            positions.append(i + 1 + (j - (n_methods - 1) / 2) * width)
            data.append(cell)
        if not data:
            continue
        bp = ax.boxplot(data, positions=positions, widths=width * 0.85,
                        patch_artist=True, showfliers=False, manage_ticks=False)
        color = RQ3_METHOD_COLORS.get(m, '#777777')
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor(color)
        for med in bp['medians']:
            med.set_color('black')

    ax.axhline(TIMEOUT, color='red', linestyle='--', linewidth=0.8, alpha=0.6,
               label=f"timeout = {TIMEOUT}s")
    ax.set_yscale('log')
    ax.set_xticks(range(1, len(all_vars) + 1))
    ax.set_xticklabels([str(v) for v in all_vars])
    ax.set_xlabel("Number of variables")
    ax.set_ylabel("Time (seconds, log scale)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=RQ3_METHOD_COLORS.get(m, '#777777'),
                             alpha=0.7, label=_disp(m)) for m in methods]
    handles.append(plt.Line2D([0], [0], color='red', linestyle='--',
                              label=f"timeout = {TIMEOUT}s"))
    ax.legend(handles=handles, loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y', which='both')
    plt.tight_layout()
    _save(fig, os.path.join(PLOT_DIR, "rq3_boxplot_combined.png"))


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


# ==================== RQ1 Drift ====================

def discover_rq1_runs():
    """Return sorted list of (threshold_float, scenario_str, filename)."""
    out = []
    if not os.path.isdir(LOG_DIR):
        return out
    for fname in sorted(os.listdir(LOG_DIR)):
        m = _RE_RQ1.match(fname)
        if m:
            out.append((float(m.group('th')), m.group('drift'), fname))
            continue
        m = _RE_RQ1_LEGACY_RQ4.match(fname)
        if m:
            out.append((float(m.group('th')), m.group('drift'), fname))
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
RQ1_SCENARIOS = {'1x', '3x', '6x'}

_RQ1_METRIC_LABELS = {0: 'Precision', 1: 'Recall', 2: 'F1'}


def _plot_rq1_metric_over_time(data, th, scenario, metric_name, metric_idx):
    """Per-metric over-time plot for drift-only RQ1. Drift intervals are shaded; figure
    carries no title — only axis labels (project convention)."""
    methods = [m for m in METHOD_ORDER if m in data['methods']]
    if not methods:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for m in methods:
        samples = data['methods'][m]['per_sample']
        ts, precs, recs, f1s = _sliding_window_pr(samples)
        values = [precs, recs, f1s][metric_idx]
        ax.plot(ts, values, label=_disp(m), color=_color(m), alpha=0.85, linewidth=1.5)

    drift_intervals = data.get('drift_intervals', [])
    for start, end in drift_intervals:
        ax.axvline(x=start, color='red',   linestyle='--', alpha=0.7)
        ax.axvline(x=end,   color='green', linestyle='--', alpha=0.7)
        ax.axvspan(start, end, alpha=0.15, color='red')

    ax.set_xlabel(f"Stream sample index")
    ax.set_ylabel(_RQ1_METRIC_LABELS[metric_idx])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, os.path.join(
        PLOT_DIR,
        f"rq1_A_th{th:.3f}_drift{scenario}_{metric_name}_over_time.png"))


def _plot_rq1_metric_box(data, th, scenario, metric_name, metric_idx):
    """Per-metric boxplot for drift-only RQ1: aggregates the sliding-window distribution
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
    ax.set_xticklabels([_disp(m) for m in methods])
    ax.set_xlabel(f"Method")
    ax.set_ylabel(_RQ1_METRIC_LABELS[metric_idx])
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    _save(fig, os.path.join(
        PLOT_DIR,
        f"rq1_A_th{th:.3f}_drift{scenario}_{metric_name}_box.png"))


def run_rq1():
    print("\nRQ1: Drift")
    runs = discover_rq1_runs()
    if not runs:
        print("  No RQ1 drift data found in logs/")
        return
    selected = [(th, sc, fn) for th, sc, fn in runs if sc in RQ1_SCENARIOS]
    skipped  = sorted({sc for _, sc, _ in runs if sc not in RQ1_SCENARIOS})
    if skipped:
        print(f"  Skipping non-target scenarios: {skipped}")
    if not selected:
        print(f"  No RQ1 data for target scenarios {sorted(RQ1_SCENARIOS)}")
        return
    for th, scenario, fn in selected:
        data = _load_json(os.path.join(LOG_DIR, fn))
        for metric_name, idx in [('precision', 0), ('recall', 1), ('f1', 2)]:
            # _plot_rq1_metric_over_time(data, th, scenario, metric_name, idx)
            _plot_rq1_metric_box(data, th, scenario, metric_name, idx)


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description='Generate result plots')
    parser.add_argument('--rq1-only', action='store_true')
    parser.add_argument('--rq2-only', action='store_true')
    parser.add_argument('--rq3-only', action='store_true')
    parser.add_argument('--rq4-only', action='store_true',
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    do_all = not (args.rq1_only or args.rq2_only or args.rq3_only or args.rq4_only)
    os.makedirs(PLOT_DIR, exist_ok=True)
    print("=== Generating Plots ===\n")

    if do_all or args.rq1_only or args.rq4_only:
        run_rq1()

    if do_all or args.rq2_only:
        print("\nRQ2: Adaptation Decisions")
        runs = discover_rq2_runs()
        if not runs:
            print("  No RQ2 data found in logs/")
        for th, drift, oracle, oracle_label_source, fn in runs:
            data = _load_json(os.path.join(LOG_DIR, fn))
            print(f"  [{fn}] threshold={th:.3f} "
                  f"drift={drift or 'none'} oracle={oracle or 'legacy'} "
                  f"oracle_labels={oracle_label_source or data.get('oracle_label_source', 'legacy')}")
            plot_rq2_metric_box(
                data, th, drift=drift, oracle=oracle,
                oracle_label_source=oracle_label_source or data.get('oracle_label_source'))
            plot_rq2_success_bars(
                data, th, drift=drift, oracle=oracle,
                oracle_label_source=oracle_label_source or data.get('oracle_label_source'))

    if do_all or args.rq3_only:
        print("\nRQ3: cost (mixed RQ3-A Best-Logit + RQ3-B drift=6x SMARTS/BMA)")
        df_a = load_rq3()
        df_b = load_rq3_online()
        if df_a is None and df_b is None:
            print("  No RQ3 data found.")
        else:
            plot_rq3_boxplot_single(df_a, df_b, drift_scenario='6x')
            if df_a is not None:
                plot_rq3_heatmap(df_a)
            if df_b is not None:
                plot_rq3_online_per_drift(df_b)
                plot_rq3_online_combined(df_b)

    print(f"\nAll plots in: {PLOT_DIR}")


if __name__ == '__main__':
    main()
