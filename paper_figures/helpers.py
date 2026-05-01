"""
helpers.py — paper-figure helpers, aligned with plot_results.py.

Each plotting function:
  * returns a matplotlib `Figure` (so notebook cells render inline)
  * accepts style kwargs (figsize, colors, alphas, fliers, ylim, ...) for fine
    tuning without forking the function
  * optionally writes a PNG when `output_dir` is given (filename matches
    plot_results.py exactly)
  * does NOT call plt.close — leave that to the caller / notebook
"""
from __future__ import annotations

import json
import os
import re
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# ===========================================================================
# Style / constants (mirrors plot_results.py)
# ===========================================================================

DEFAULT_COLORS = {
    'Stacking-Online':  '#E24A33',
    'BMA':              '#988ED5',
    'Best-Logit':       '#8EBA42',
    'Stacking-Offline': '#348ABD',
    'MCMC':             '#4C72B0',
    'BAS':              '#55A868',
    'Stacking':         '#DD8452',
}

METHOD_ORDER = ['Stacking-Online', 'BMA', 'Best-Logit']
RQ2_METHODS  = ['Stacking-Online', 'BMA', 'Best-Logit']
DEFAULT_TIMEOUT = 200

DISPLAY_NAME = {'Stacking-Online': 'SMARTS'}

DEFAULT_RE_ORACLE_EPS = 1e-4
RQ1_SCENARIOS = {'1x', '3x', '6x'}
RQ1_METRIC_LABELS = {0: 'Precision', 1: 'Recall', 2: 'F1'}

RQ3_PLOT_ORDER = ['Stacking', 'BMA', 'Best-Logit']
RQ3B_PLOT_ORDER  = ['Stacking-Online', 'BMA']
RQ3B_DRIFT_ORDER = ['1x', '3x', '6x']

# Filename label remap for the RQ3 online heatmap so the produced PNGs match
# the canonical names in /plots/ (Stacking-Online → Stacking).
RQ3_ONLINE_HEATMAP_LABELS = {'Stacking-Online': 'Stacking', 'BMA': 'BMA'}

_RE_RQ1 = re.compile(r"^rq1_A_th(?P<th>\d+\.\d+)_drift(?P<drift>\w+)\.json$")
_RE_RQ1_LEGACY_RQ4 = re.compile(r"^rq4_A_th(?P<th>\d+\.\d+)_drift(?P<drift>\w+)\.json$")
_RE_RQ2 = re.compile(
    r"^rq2_A(?:_drift(?P<drift>[A-Za-z0-9]+))?"
    r"_th(?P<th>\d+\.\d+)"
    r"(?:_oracle(?P<oracle>\w+))?"
    r"(?:_oraclelabels(?P<oracle_label_source>[A-Za-z0-9-]+))?\.json$"
)
_RE_RQ3_ONLINE_LEGACY = re.compile(r"^rq3_online_drift([A-Za-z0-9]+)\.tsv$")
_RE_RQ3_ONLINE_GRID = re.compile(
    r"^rq3_online_grid_drift(?P<drift>[A-Za-z0-9]+)_re(?P<retrain>\d+)\.tsv$")


def _disp(name: str) -> str:
    return DISPLAY_NAME.get(name, name)


def _color(name: str, colors: Dict[str, str]) -> str:
    for key in colors:
        if key in name:
            return colors[key]
    return '#777777'


def _maybe_save(fig: plt.Figure, output_dir: Optional[str], filename: str,
                dpi: int = 150) -> None:
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, bbox_inches='tight', dpi=dpi)
    print(f"  saved: {path}")


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ===========================================================================
# Discovery
# ===========================================================================

def discover_rq1_runs(log_dir: str) -> List[Tuple[float, str, str]]:
    out = []
    if not os.path.isdir(log_dir):
        return out
    for fname in sorted(os.listdir(log_dir)):
        m = _RE_RQ1.match(fname) or _RE_RQ1_LEGACY_RQ4.match(fname)
        if m:
            out.append((float(m.group('th')), m.group('drift'), fname))
    return sorted(out, key=lambda x: (x[0], x[1]))


def discover_rq2_runs(log_dir: str
                      ) -> List[Tuple[float, Optional[str], Optional[str], Optional[str], str]]:
    out = []
    if not os.path.isdir(log_dir):
        return out
    for fname in sorted(os.listdir(log_dir)):
        m = _RE_RQ2.match(fname)
        if m:
            out.append((
                float(m.group('th')),
                m.group('drift'),
                m.group('oracle'),
                m.group('oracle_label_source'),
                fname,
            ))
    return sorted(out, key=lambda x: (
        x[0], x[1] or '', x[2] or 'zzz', x[3] or 'zzz'))


def discover_rq3_online(log_dir: str) -> List[Tuple[str, str, str]]:
    out = []
    if not os.path.isdir(log_dir):
        return out
    filenames = sorted(os.listdir(log_dir))
    seen_grid = set()
    for fname in filenames:
        m = _RE_RQ3_ONLINE_GRID.match(fname)
        if m:
            seen_grid.add(m.group('drift'))
    for fname in filenames:
        m = _RE_RQ3_ONLINE_GRID.match(fname)
        if m:
            out.append((m.group('drift'), fname, 'grid'))
            continue
        m = _RE_RQ3_ONLINE_LEGACY.match(fname)
        if m and m.group(1) not in seen_grid:
            out.append((m.group(1), fname, 'legacy'))
    def _key(item):
        sc = item[0]
        return (RQ3B_DRIFT_ORDER.index(sc) if sc in RQ3B_DRIFT_ORDER else 99, sc)
    return sorted(out, key=_key)


# ===========================================================================
# Sliding-window PR (replicates plot_results._sliding_window_pr)
# ===========================================================================

def sliding_window_pr(samples: List[dict], window: int = 50,
                      threshold: float = 0.5
                      ) -> Tuple[List[int], List[float], List[float], List[float]]:
    preds_bin = [1 if s['pred'] >= threshold else 0 for s in samples]
    trues = [int(s['true']) for s in samples]
    ts = [s['t'] for s in samples]
    P, R, F = [], [], []
    for i in range(len(samples)):
        start = max(0, i - window + 1)
        p = preds_bin[start:i + 1]
        y = trues[start:i + 1]
        tp = sum(a == 1 and b == 1 for a, b in zip(p, y))
        fp = sum(a == 1 and b == 0 for a, b in zip(p, y))
        fn = sum(a == 0 and b == 1 for a, b in zip(p, y))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        P.append(prec); R.append(rec); F.append(f1)
    return ts, P, R, F


# ===========================================================================
# RQ1 — sliding-window box per (scenario × metric)
# ===========================================================================

def plot_rq1_metric_box(log_dir: str, threshold: float, scenario: str,
                        metric_name: str,
                        *,
                        methods: Sequence[str] = METHOD_ORDER,
                        colors: Dict[str, str] = DEFAULT_COLORS,
                        window: int = 50,
                        figsize: Tuple[float, float] = (7, 5),
                        showfliers: bool = True,
                        box_alpha: float = 0.7,
                        ylim: Tuple[float, float] = (0.0, 1.05),
                        grid_alpha: float = 0.3,
                        output_dir: Optional[str] = None,
                        dpi: int = 150) -> plt.Figure:
    """One sliding-window distribution box per method, for one metric in one
    drift scenario. metric_name in {'precision','recall','f1'}."""
    metric_idx = {'precision': 0, 'recall': 1, 'f1': 2}[metric_name]

    fname_candidates = [
        f"rq1_A_th{threshold:.3f}_drift{scenario}.json",
        f"rq4_A_th{threshold:.3f}_drift{scenario}.json",
    ]
    path = next((os.path.join(log_dir, c) for c in fname_candidates
                 if os.path.exists(os.path.join(log_dir, c))), None)
    if path is None:
        raise FileNotFoundError(f"No RQ1 log for th={threshold:.3f} sc={scenario}")
    data = _load_json(path)

    methods = [m for m in methods if m in data['methods']]
    box_data = []
    for m in methods:
        _, P, R, F = sliding_window_pr(data['methods'][m]['per_sample'], window=window)
        box_data.append([P, R, F][metric_idx])

    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(box_data, patch_artist=True, showfliers=showfliers,
                    flierprops=dict(marker='o', markersize=3, alpha=0.5))
    for patch, m in zip(bp['boxes'], methods):
        patch.set_facecolor(_color(m, colors))
        patch.set_alpha(box_alpha)
    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels([_disp(m) for m in methods])
    ax.set_xlabel("Method")
    ax.set_ylabel(RQ1_METRIC_LABELS[metric_idx])
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=grid_alpha, axis='y')
    fig.tight_layout()

    _maybe_save(fig, output_dir,
        f"rq1_A_th{threshold:.3f}_drift{scenario}_{metric_name}_box.png", dpi=dpi)
    return fig


# ===========================================================================
# RQ2 — abs_err / RE box  +  success-rate bars
# ===========================================================================

def _rq2_metric_key(data: dict) -> Tuple[str, str]:
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


def _rq2_filename(th: float, drift: Optional[str], oracle: Optional[str],
                  oracle_label_source: Optional[str], suffix: str) -> str:
    drift_part  = f"_drift{drift}"  if drift  else ""
    oracle_part = f"_oracle{oracle}" if oracle else ""
    source_part = f"_oraclelabels{oracle_label_source}" if oracle_label_source else ""
    return f"rq2_A{drift_part}_th{th:.3f}{oracle_part}{source_part}_{suffix}.png"


def _resolve_rq2_log(log_dir: str, threshold: float, drift: Optional[str],
                     oracle: Optional[str], oracle_label_source: Optional[str]) -> str:
    drift_part  = f"_drift{drift}"  if drift  else ""
    oracle_part = f"_oracle{oracle}" if oracle else ""
    src_part    = f"_oraclelabels{oracle_label_source}" if oracle_label_source else ""
    candidates = [
        f"rq2_A{drift_part}_th{threshold:.3f}{oracle_part}{src_part}.json",
        f"rq2_A{drift_part}_th{threshold:.3f}{oracle_part}.json",
        f"rq2_A_th{threshold:.3f}{oracle_part}.json",
    ]
    for c in candidates:
        p = os.path.join(log_dir, c)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No RQ2 log for {candidates}")


def plot_rq2_metric_box(log_dir: str, threshold: float,
                        *,
                        drift: Optional[str] = None,
                        oracle: Optional[str] = None,
                        oracle_label_source: Optional[str] = None,
                        methods: Sequence[str] = RQ2_METHODS,
                        colors: Dict[str, str] = DEFAULT_COLORS,
                        re_oracle_eps: float = DEFAULT_RE_ORACLE_EPS,
                        figsize: Tuple[float, float] = (7, 5),
                        showfliers: bool = True,
                        box_alpha: float = 0.7,
                        grid_alpha: float = 0.3,
                        annotate_kept: bool = True,
                        output_dir: Optional[str] = None,
                        dpi: int = 150) -> plt.Figure:
    path = _resolve_rq2_log(log_dir, threshold, drift, oracle, oracle_label_source)
    data = _load_json(path)
    methods = [m for m in methods if m in data['methods']]
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
                if ov < re_oracle_eps:
                    continue
            vals.append(v)
        n_kept[m] = len(vals)
        box_data.append(vals)

    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(box_data, patch_artist=True, showfliers=showfliers,
                    flierprops=dict(marker='o', markersize=3, alpha=0.5))
    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels([_disp(m) for m in methods])
    for patch, m in zip(bp['boxes'], methods):
        patch.set_facecolor(_color(m, colors))
        patch.set_alpha(box_alpha)
    if key == 'RE':
        ax.set_yscale('log')
        if annotate_kept:
            note = "  ".join(f"{_disp(m)}: {n_kept[m]}/{n_total[m]}" for m in methods)
            ax.text(0.02, 0.98, f"oracle ≥ {re_oracle_eps:g}: {note}",
                    transform=ax.transAxes, fontsize=8, va='top',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    ax.set_xlabel("Method")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=grid_alpha, axis='y')
    fig.tight_layout()
    _maybe_save(fig, output_dir,
        _rq2_filename(threshold, drift, oracle, oracle_label_source, 'metric_box'),
        dpi=dpi)
    return fig


def plot_rq2_success_bars(log_dir: str, threshold: float,
                          *,
                          drift: Optional[str] = None,
                          oracle: Optional[str] = None,
                          oracle_label_source: Optional[str] = None,
                          methods: Sequence[str] = RQ2_METHODS,
                          colors: Dict[str, str] = DEFAULT_COLORS,
                          figsize: Tuple[float, float] = (7, 5),
                          bar_alpha: float = 0.8,
                          bar_width: float = 0.8,
                          edge_color: str = 'black',
                          edge_width: float = 0.5,
                          ylim: Tuple[float, float] = (0.0, 1.05),
                          grid_alpha: float = 0.3,
                          output_dir: Optional[str] = None,
                          dpi: int = 150) -> plt.Figure:
    path = _resolve_rq2_log(log_dir, threshold, drift, oracle, oracle_label_source)
    data = _load_json(path)
    methods = [m for m in methods if m in data['methods']]
    rates = [
        data['methods'][m].get('precision',
                               data['methods'][m].get('success_rate', np.nan))
        for m in methods
    ]
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar([_disp(m) for m in methods], rates,
           width=bar_width,
           color=[_color(m, colors) for m in methods],
           alpha=bar_alpha, edgecolor=edge_color, linewidth=edge_width)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Method")
    ax.set_ylabel("Success Rate")
    ax.grid(True, alpha=grid_alpha, axis='y')
    fig.tight_layout()
    _maybe_save(fig, output_dir,
        _rq2_filename(threshold, drift, oracle, oracle_label_source, 'success_bars'),
        dpi=dpi)
    return fig


# ===========================================================================
# RQ3 — cost figures
# ===========================================================================

def load_rq3(log_dir: str) -> Optional[pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    primary_files = [
        'rq3_stacking.tsv', 'rq3_bma.tsv', 'rq3_best_logit.tsv',
        'rq3_mcmc.tsv', 'rq3_bas.tsv',
    ]
    for fname in primary_files:
        path = os.path.join(log_dir, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, sep=r"\s+")
        if 'run_id' not in df.columns:
            df['run_id'] = 0
        for m in df['method'].unique():
            frames[m] = df[df['method'] == m]
    fallback = os.path.join(log_dir, 'baseline_rq3_results.tsv')
    if os.path.exists(fallback):
        df = pd.read_csv(fallback, sep=r"\s+")
        if 'run_id' not in df.columns:
            df['run_id'] = 0
        for m in df['method'].unique():
            if m not in frames:
                frames[m] = df[df['method'] == m].copy()
                frames[m]['_baseline_hardware'] = True
    if not frames:
        return None
    return pd.concat(frames.values(), ignore_index=True)


def load_rq3_online(log_dir: str,
                    retrain_every: Optional[int] = 100) -> Optional[pd.DataFrame]:
    runs = discover_rq3_online(log_dir)
    if not runs:
        return None
    if retrain_every is not None:
        filtered = []
        for sc, fname, mode in runs:
            if mode == 'grid':
                m = _RE_RQ3_ONLINE_GRID.match(fname)
                if m and int(m.group('retrain')) != retrain_every:
                    continue
            filtered.append((sc, fname, mode))
        if filtered:
            runs = filtered
    frames = []
    for sc, fname, mode in runs:
        df = pd.read_csv(os.path.join(log_dir, fname), sep=r"\s+")
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


def plot_rq3_boxplot_single(log_dir: str,
                            *,
                            drift_scenario: str = '6x',
                            retrain_every: Optional[int] = 100,
                            colors: Dict[str, str] = DEFAULT_COLORS,
                            timeout_s: float = DEFAULT_TIMEOUT,
                            figsize: Tuple[float, float] = (7, 5),
                            box_widths: float = 0.35,
                            box_alpha: float = 0.7,
                            showfliers: bool = True,
                            grid_alpha: float = 0.3,
                            output_dir: Optional[str] = None,
                            dpi: int = 150) -> Optional[plt.Figure]:
    df_a = load_rq3(log_dir)
    df_b = load_rq3_online(log_dir, retrain_every=retrain_every)
    boxes, labels, box_colors = [], [], []
    if df_b is not None:
        sub = df_b[df_b['drift'] == drift_scenario]
        for m in ['Stacking-Online', 'BMA']:
            cell = sub[sub['method'] == m]['time_seconds'].values
            if len(cell):
                boxes.append(cell)
                labels.append(_disp(m))
                box_colors.append(_color(m, colors))
    if df_a is not None:
        cell = df_a[df_a['method'] == 'Best-Logit']['time'].values
        if len(cell):
            boxes.append(cell)
            labels.append(_disp('Best-Logit'))
            box_colors.append(colors.get('Best-Logit', '#777'))
    if not boxes:
        return None

    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(boxes, patch_artist=True, showfliers=showfliers,
                    flierprops=dict(marker='o', markersize=3, alpha=0.5),
                    widths=box_widths)
    for patch, c in zip(bp['boxes'], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(box_alpha)
    for med in bp['medians']:
        med.set_color('black')
    ax.axhline(timeout_s, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_yscale('log')
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Method")
    ax.set_ylabel("Time (seconds, log scale)")
    ax.grid(True, alpha=grid_alpha, axis='y', which='both')
    fig.tight_layout()
    _maybe_save(fig, output_dir, "rq3_boxplot.png", dpi=dpi)
    return fig


def plot_rq3_heatmap(log_dir: str,
                     *,
                     method: Optional[str] = None,
                     timeout_s: float = DEFAULT_TIMEOUT,
                     cmap: str = 'YlOrRd',
                     figsize: Tuple[float, float] = (6, 5),
                     annot_fontsize: int = 7,
                     output_dir: Optional[str] = None,
                     dpi: int = 150) -> Dict[str, plt.Figure]:
    """Build one heatmap per method (or just the requested one).
    Returns {method: fig} so the notebook can display each inline."""
    df = load_rq3(log_dir)
    if df is None:
        return {}
    targets = [method] if method else list(df['method'].unique())
    out: Dict[str, plt.Figure] = {}
    for mname in targets:
        sub = df[df['method'] == mname]
        if sub.empty:
            continue
        pivot = sub.pivot_table(index='sample', columns='vars',
                                values='time', aggfunc='mean')
        pivot = pivot.sort_index(ascending=True)
        data = pivot.values
        display = np.where(data >= timeout_s, timeout_s, data)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(display, aspect='auto', cmap=cmap, origin='lower',
                       vmin=np.nanmin(display), vmax=timeout_s)
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
                text = ("TO" if val >= timeout_s
                        else (f"{val:.1f}" if val < 10 else f"{int(val)}"))
                cval = display[i, j] if not np.isnan(display[i, j]) else 0
                ax.text(j, i, text, ha='center', va='center',
                        fontsize=annot_fontsize,
                        color='white' if cval > timeout_s * 0.6 else 'black')
        cbar = fig.colorbar(im, ax=ax, label="Time (seconds)")
        if np.nanmax(display) <= 10:
            cbar.set_ticks(np.arange(0, 11, 2))
        else:
            cbar.set_ticks(np.arange(0, int(timeout_s) + 1, 50))
        fig.tight_layout()
        _maybe_save(fig, output_dir, f"rq3_heatmap_{mname}.png", dpi=dpi)
        out[mname] = fig
    return out


def plot_rq3_online_heatmap(log_dir: str,
                            *,
                            scenario: str = '6x',
                            retrain_every: Optional[int] = 100,
                            methods: Sequence[str] = RQ3B_PLOT_ORDER,
                            value_col: str = 'total',
                            label_remap: Dict[str, str] = RQ3_ONLINE_HEATMAP_LABELS,
                            timeout_s: float = DEFAULT_TIMEOUT,
                            cmap: str = 'YlOrRd',
                            figsize: Tuple[float, float] = (6, 5),
                            annot_fontsize: int = 7,
                            output_dir: Optional[str] = None,
                            dpi: int = 150) -> Dict[str, plt.Figure]:
    """Heatmap (sample × vars) of online RQ3-B cost per method.

    `value_col` chooses which column to display:
      - 'total'  (default): full stream processing time, in seconds
      - 'time'              : per-sample latency
      - 'median' / 'p95'    : per-sample latency stats

    Output filenames use `label_remap` so 'Stacking-Online' → 'Stacking',
    matching the canonical names in /plots/.
    """
    df = load_rq3_online(log_dir, retrain_every=retrain_every)
    if df is None:
        return {}
    sub_all = df[df['drift'] == scenario]
    if sub_all.empty:
        return {}

    targets = [m for m in methods if m in sub_all['method'].unique()]
    out: Dict[str, plt.Figure] = {}
    for m in targets:
        sub = sub_all[sub_all['method'] == m]
        if sub.empty or value_col not in sub.columns:
            continue
        pivot = sub.pivot_table(index='sample', columns='vars',
                                values=value_col, aggfunc='mean')
        pivot = pivot.sort_index(ascending=True)
        data = pivot.values
        display = np.where(data >= timeout_s, timeout_s, data)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(display, aspect='auto', cmap=cmap, origin='lower',
                       vmin=np.nanmin(display), vmax=timeout_s)
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
                if val >= timeout_s:
                    text = "TO"
                elif val < 1:
                    text = f"{val:.1f}" if val >= 0.05 else f"{val:.2f}"
                elif val < 10:
                    text = f"{val:.1f}"
                else:
                    text = f"{int(val)}"
                cval = display[i, j] if not np.isnan(display[i, j]) else 0
                ax.text(j, i, text, ha='center', va='center',
                        fontsize=annot_fontsize,
                        color='white' if cval > timeout_s * 0.6 else 'black')
        cbar = fig.colorbar(im, ax=ax, label="Time (seconds)")
        if np.nanmax(display) <= 10:
            cbar.set_ticks(np.arange(0, 11, 2))
        else:
            cbar.set_ticks(np.arange(0, int(timeout_s) + 1, 50))
        fig.tight_layout()

        label = label_remap.get(m, m)
        _maybe_save(fig, output_dir, f"rq3_heatmap_{label}.png", dpi=dpi)
        out[label] = fig
    return out


def plot_rq3_online_per_drift(log_dir: str,
                              *,
                              scenario: str,
                              retrain_every: Optional[int] = 100,
                              methods: Sequence[str] = RQ3B_PLOT_ORDER,
                              colors: Dict[str, str] = DEFAULT_COLORS,
                              figsize: Tuple[float, float] = (6, 5),
                              box_widths: float = 0.6,
                              box_alpha: float = 0.7,
                              showfliers: bool = True,
                              grid_alpha: float = 0.3,
                              output_dir: Optional[str] = None,
                              dpi: int = 150) -> Optional[plt.Figure]:
    df = load_rq3_online(log_dir, retrain_every=retrain_every)
    if df is None:
        return None
    sub = df[df['drift'] == scenario]
    methods = [m for m in methods if m in sub['method'].unique()]
    if not methods:
        return None
    box_data = [sub[sub['method'] == m]['time_seconds'].values for m in methods]
    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(box_data, patch_artist=True, showfliers=showfliers,
                    flierprops=dict(marker='o', markersize=3, alpha=0.5),
                    widths=box_widths)
    for patch, m in zip(bp['boxes'], methods):
        patch.set_facecolor(_color(m, colors))
        patch.set_alpha(box_alpha)
    for med in bp['medians']:
        med.set_color('black')
    ax.set_yscale('log')
    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels([_disp(m) for m in methods])
    ax.set_xlabel("Method")
    ax.set_ylabel("Per-sample online_update latency (s, log scale)")
    ax.grid(True, alpha=grid_alpha, axis='y', which='both')
    fig.tight_layout()
    _maybe_save(fig, output_dir, f"rq3_online_boxplot_drift{scenario}.png", dpi=dpi)
    return fig


def plot_rq3_online_combined(log_dir: str,
                             *,
                             retrain_every: Optional[int] = 100,
                             methods: Sequence[str] = RQ3B_PLOT_ORDER,
                             drifts: Sequence[str] = tuple(RQ3B_DRIFT_ORDER),
                             colors: Dict[str, str] = DEFAULT_COLORS,
                             figsize: Optional[Tuple[float, float]] = None,
                             box_alpha: float = 0.7,
                             grid_alpha: float = 0.3,
                             legend_loc: str = 'upper left',
                             output_dir: Optional[str] = None,
                             dpi: int = 150) -> Optional[plt.Figure]:
    df = load_rq3_online(log_dir, retrain_every=retrain_every)
    if df is None:
        return None
    methods = [m for m in methods if m in df['method'].unique()]
    drifts  = [d for d in drifts if d in df['drift'].unique()]
    if not methods or not drifts:
        return None
    if figsize is None:
        figsize = (max(8, 1.6 * len(drifts) * len(methods)), 5)
    fig, ax = plt.subplots(figsize=figsize)
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
        c = _color(m, colors)
        for patch in bp['boxes']:
            patch.set_facecolor(c)
            patch.set_alpha(box_alpha)
            patch.set_edgecolor(c)
        for med in bp['medians']:
            med.set_color('black')
    ax.set_yscale('log')
    ax.set_xticks(range(1, len(drifts) + 1))
    ax.set_xticklabels([f"drift = {d}" for d in drifts])
    ax.set_xlabel("Drift scenario")
    ax.set_ylabel("Per-sample online_update latency (s, log scale)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=_color(m, colors),
                             alpha=box_alpha, label=_disp(m)) for m in methods]
    ax.legend(handles=handles, loc=legend_loc, fontsize=10)
    ax.grid(True, alpha=grid_alpha, axis='y', which='both')
    fig.tight_layout()
    _maybe_save(fig, output_dir, "rq3_online_boxplot.png", dpi=dpi)
    return fig


# ===========================================================================
# Bulk runners (kept for parity with plot_results.py)
# ===========================================================================

def run_all(log_dir: str = 'logs', output_dir: str = 'plots',
            retrain_every: Optional[int] = 100,
            threshold: float = 0.035) -> None:
    """Generate every figure plot_results.py would generate. No inline display."""
    os.makedirs(output_dir, exist_ok=True)
    print("RQ1: drift")
    for th, sc, _ in discover_rq1_runs(log_dir):
        if sc not in RQ1_SCENARIOS or abs(th - threshold) > 1e-9:
            continue
        for metric in ('precision', 'recall', 'f1'):
            fig = plot_rq1_metric_box(log_dir, th, sc, metric, output_dir=output_dir)
            plt.close(fig)
    print("RQ2: adaptation decisions")
    for th, drift, oracle, src, fn in discover_rq2_runs(log_dir):
        f1 = plot_rq2_metric_box(log_dir, th, drift=drift, oracle=oracle,
                                 oracle_label_source=src, output_dir=output_dir)
        f2 = plot_rq2_success_bars(log_dir, th, drift=drift, oracle=oracle,
                                   oracle_label_source=src, output_dir=output_dir)
        plt.close(f1); plt.close(f2)
    print("RQ3: cost")
    f = plot_rq3_boxplot_single(log_dir, output_dir=output_dir)
    if f is not None:
        plt.close(f)
    for _, fig in plot_rq3_heatmap(log_dir, output_dir=output_dir).items():
        plt.close(fig)
    for _, fig in plot_rq3_online_heatmap(
            log_dir, retrain_every=retrain_every,
            output_dir=output_dir).items():
        plt.close(fig)
    for sc in RQ3B_DRIFT_ORDER:
        f = plot_rq3_online_per_drift(log_dir, scenario=sc, output_dir=output_dir)
        if f is not None:
            plt.close(f)
    f = plot_rq3_online_combined(log_dir, output_dir=output_dir)
    if f is not None:
        plt.close(f)
    print(f"all plots in: {output_dir}")
