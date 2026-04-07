"""
绘图脚本：对 RQ1（预测精度）、RQ2（适应决策）、RQ3（计算花费）实验结果绘制对比图
代码注释使用中文，图表文字使用英文
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import os
import json
import sys

# 设置matplotlib参数
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.figsize': (10, 5),
    'figure.dpi': 150
})


def load_rq2_data():
    """加载 RQ2 实验数据（Stacking 和 BMA 的 RE 与成功率）"""
    base_dir = os.path.dirname(__file__)

    # 加载Stacking结果
    stacking_file = os.path.join(base_dir, 'logs', 'stacking_rq2_results.tsv')
    if not os.path.exists(stacking_file):
        print(f"警告: 未找到Stacking结果文件 {stacking_file}")
        return None, None

    stacking_df = pd.read_csv(stacking_file, sep='\t')

    # 加载BMA结果
    bma_file = os.path.join(base_dir, 'logs', 're_bma_logit.log')
    if not os.path.exists(bma_file):
        print(f"警告: 未找到BMA结果文件 {bma_file}")
        return stacking_df, None

    bma_df = pd.read_csv(bma_file, sep='\t')
    return stacking_df, bma_df


# ==================== RQ1 数据加载 ====================
def load_rq1_data():
    """加载RQ1实验结果（TSV文件 + precision_recall.log 中的 Logit 数据）"""
    base_dir = os.path.dirname(__file__)
    rq1_file = os.path.join(base_dir, 'logs', 'stacking_rq1_results.tsv')
    if not os.path.exists(rq1_file):
        print(f"警告: 未找到RQ1结果文件 {rq1_file}")
        return None
    df = pd.read_csv(rq1_file, sep='\t')

    # 从 precision_recall.log 加载 Logit 的 precision/recall/F1，取均值作为 Logit 基线
    logit_file = os.path.join(base_dir, 'logs', 'precision_recall.log')
    if os.path.exists(logit_file):
        logit_df = pd.read_csv(logit_file, sep='\t')
        logit_rows = logit_df[logit_df['type'] == 'Logit']
        if not logit_rows.empty:
            logit_row = pd.DataFrame([{
                'model': 'Logit',
                'precision': logit_rows['precision'].mean(),
                'recall': logit_rows['recall'].mean(),
                'f1': logit_rows['F1'].mean(),
                'train_time': 0.0
            }])
            df = pd.concat([df, logit_row], ignore_index=True)
    else:
        print(f"警告: 未找到 Logit 精度文件 {logit_file}")

    return df


def plot_re_comparison(stacking_df, bma_df, plot_dir, experiment_label=""):
    """绘制RE（相对误差）对比箱线图"""
    bma_re = bma_df[bma_df['type'] == 'BMA']['RE'].values
    logit_re = bma_df[bma_df['type'] == 'Logit']['RE'].values
    stacking_re = stacking_df[stacking_df['type'] == 'Stacking']['RE'].values

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(
        [bma_re, stacking_re, logit_re],
        labels=['BMA', 'Stacking', 'Logit'],
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker='o', markersize=3, alpha=0.5)
    )
    colors = ['#4C72B0', '#DD8452', '#55A868']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_yscale('log')
    ax.set_ylabel('RE (log scale)')
    ax.set_title(f'Relative Error Comparison{experiment_label}')
    ax.grid(True, alpha=0.3, axis='y')

    medians = [np.median(bma_re), np.median(stacking_re), np.median(logit_re)]
    for i, median in enumerate(medians):
        ax.text(i + 1, median * 1.3, f'median={median:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    filename = os.path.join(plot_dir, f'relative_error_comparison{experiment_label.replace(" ", "_")}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"RE对比图已保存: {filename}")


def plot_success_rate_comparison(stacking_df, bma_df, plot_dir, experiment_label=""):
    """绘制成功率对比柱状图"""
    bma_data = bma_df[bma_df['type'] == 'BMA']
    bma_success = bma_data['success'].apply(lambda x: x == True or x == 'TRUE').mean()

    logit_data = bma_df[bma_df['type'] == 'Logit']
    logit_success = logit_data['success'].apply(lambda x: x == True or x == 'TRUE').mean()

    stacking_data = stacking_df[stacking_df['type'] == 'Stacking']
    stacking_success = stacking_data['success'].apply(lambda x: x == True or x == 'TRUE').mean()

    fig, ax = plt.subplots(figsize=(7, 5))
    methods = ['BMA', 'Stacking', 'Logit']
    rates = [bma_success, stacking_success, logit_success]
    colors = ['#4C72B0', '#DD8452', '#55A868']

    bars = ax.bar(methods, rates, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{rate:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Success Rate')
    ax.set_title(f'Success Rate Comparison{experiment_label}')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    filename = os.path.join(plot_dir, f'success_rate_comparison{experiment_label.replace(" ", "_")}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"成功率对比图已保存: {filename}")


def plot_re_distribution(stacking_df, bma_df, plot_dir, experiment_label=""):
    """绘制RE分布直方图"""
    bma_re = bma_df[bma_df['type'] == 'BMA']['RE'].values
    stacking_re = stacking_df[stacking_df['type'] == 'Stacking']['RE'].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(bma_re, bins=30, color='#4C72B0', alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[0].axvline(np.median(bma_re), color='red', linestyle='--', label=f'Median={np.median(bma_re):.4f}')
    axes[0].set_xlabel('Relative Error')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('BMA RE Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(stacking_re, bins=30, color='#DD8452', alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[1].axvline(np.median(stacking_re), color='red', linestyle='--',
                     label=f'Median={np.median(stacking_re):.4f}')
    axes[1].set_xlabel('Relative Error')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Stacking RE Distribution{experiment_label}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    filename = os.path.join(plot_dir, f're_distribution{experiment_label.replace(" ", "_")}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"RE分布图已保存: {filename}")


# ==================== RQ1 绘图函数 ====================
def plot_rq1_metrics(df, plot_dir):
    """分别绘制 precision, recall, F1 对比柱状图"""
    metrics = ['precision', 'recall', 'f1']
    metric_names = ['Precision', 'Recall', 'F1 Score']
    colors = {'BMA': '#4C72B0', 'Stacking': '#DD8452', 'Logit': '#55A868'}

    # 确保 DataFrame 包含 BMA、Stacking、Logit，并按顺序排列
    df = df.set_index('model')
    order = [m for m in ['BMA', 'Stacking', 'Logit'] if m in df.index]
    df = df.reindex(order)

    for metric, name in zip(metrics, metric_names):
        fig, ax = plt.subplots(figsize=(6, 5))
        values = df[metric].values
        bars = ax.bar(order, values, color=[colors[m] for m in order],
                      alpha=0.8, edgecolor='black', linewidth=0.5)
        # 在柱上标注数值
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.set_ylabel(name)
        ax.set_title(f'{name} Comparison')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        filename = os.path.join(plot_dir, f'{metric}_comparison.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"{name}对比图已保存: {filename}")


# ==================== RQ3 数据加载 ====================
def load_rq3_data():
    """加载RQ3计算花费实验结果"""
    base_dir = os.path.dirname(__file__)
    rq3_file = os.path.join(base_dir, 'logs', 'stacking_rq3_results.tsv')
    if not os.path.exists(rq3_file):
        print(f"警告: 未找到RQ3结果文件 {rq3_file}")
        return None
    df = pd.read_csv(rq3_file, sep=r'\s+')
    return df


# ==================== RQ3 绘图函数 ====================
def plot_cost_vs_sample(df, plot_dir):
    """绘制训练时间随样本量变化的折线图（按变量数分面）"""
    methods = df['method'].unique()
    vars_list = sorted(df['vars'].unique())
    colors = {'MCMC': '#4C72B0', 'BAS': '#55A868', 'Stacking': '#DD8452'}
    markers = {'MCMC': 's', 'BAS': '^', 'Stacking': 'o'}

    # 选取有代表性的 vars 值绘制子图
    plot_vars = [v for v in [2, 8, 32, 64] if v in vars_list]
    if not plot_vars:
        plot_vars = vars_list[:4]

    n_plots = len(plot_vars)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5), sharey=True)
    if n_plots == 1:
        axes = [axes]

    for ax, v in zip(axes, plot_vars):
        for method in methods:
            subset = df[(df['method'] == method) & (df['vars'] == v)]
            if subset.empty:
                continue
            subset = subset.sort_values('sample')
            ax.plot(subset['sample'], subset['time'],
                    marker=markers.get(method, 'o'),
                    color=colors.get(method, '#999999'),
                    label=method, linewidth=1.5, markersize=5)
        ax.set_xlabel('Sample Size')
        ax.set_title(f'Variables = {v}')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    axes[0].set_ylabel('Training Time (s)')
    fig.suptitle('Training Cost vs Sample Size', fontsize=14, y=1.02)
    plt.tight_layout()
    filename = os.path.join(plot_dir, 'cost_vs_sample.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练时间-样本量对比图已保存: {filename}")


def plot_cost_vs_vars(df, plot_dir):
    """绘制训练时间随变量数变化的折线图（按样本量分面）"""
    methods = df['method'].unique()
    samples_list = sorted(df['sample'].unique())
    colors = {'MCMC': '#4C72B0', 'BAS': '#55A868', 'Stacking': '#DD8452'}
    markers = {'MCMC': 's', 'BAS': '^', 'Stacking': 'o'}

    # 选取有代表性的 sample 值
    plot_samples = [s for s in [200, 800, 3200, 12800] if s in samples_list]
    if not plot_samples:
        plot_samples = samples_list[:4]

    n_plots = len(plot_samples)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5), sharey=True)
    if n_plots == 1:
        axes = [axes]

    for ax, s in zip(axes, plot_samples):
        for method in methods:
            subset = df[(df['method'] == method) & (df['sample'] == s)]
            if subset.empty:
                continue
            subset = subset.sort_values('vars')
            ax.plot(subset['vars'], subset['time'],
                    marker=markers.get(method, 'o'),
                    color=colors.get(method, '#999999'),
                    label=method, linewidth=1.5, markersize=5)
        ax.set_xlabel('Number of Variables')
        ax.set_title(f'Samples = {s}')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    axes[0].set_ylabel('Training Time (s)')
    fig.suptitle('Training Cost vs Number of Variables', fontsize=14, y=1.02)
    plt.tight_layout()
    filename = os.path.join(plot_dir, 'cost_vs_vars.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练时间-变量数对比图已保存: {filename}")

# def plot_cost_heatmap(df, plot_dir):
#     """绘制各方法训练时间热力图（sample × vars），每个方法单独保存"""
#     methods = df['method'].unique()
    
#     for method in methods:
#         # 提取当前方法的数据
#         subset = df[df['method'] == method]
#         pivot = subset.pivot_table(index='sample', columns='vars', values='time', aggfunc='mean')
#         pivot = pivot.sort_index(ascending=True)
        
#         # 创建新图
#         plt.figure(figsize=(6, 5))
#         im = plt.imshow(np.log10(pivot.values + 1e-4), aspect='auto', cmap='YlOrRd',
#                         origin='lower')
        
#         # 设置坐标轴
#         plt.xticks(range(len(pivot.columns)), pivot.columns.astype(int), fontsize=9)
#         plt.yticks(range(len(pivot.index)), pivot.index.astype(int), fontsize=9)
#         plt.xlabel('Variables')
#         plt.ylabel('Samples')
#         plt.title(f'{method}')
        
#         # 在格子中标注时间值
#         for i in range(len(pivot.index)):
#             for j in range(len(pivot.columns)):
#                 val = pivot.values[i, j]
#                 if not np.isnan(val):
#                     text = f'{val:.1f}' if val < 10 else f'{val:.0f}'
#                     plt.text(j, i, text, ha='center', va='center', fontsize=7,
#                              color='white' if val > pivot.values[~np.isnan(pivot.values)].max() * 0.6 else 'black')
        
#         plt.colorbar(im, label='log10(time + 1e-4)')
#         plt.tight_layout()
        
#         # 保存为单独文件
#         filename = os.path.join(plot_dir, f'cost_heatmap_{method}.png')
#         plt.savefig(filename, dpi=150, bbox_inches='tight')
#         plt.close()
#         print(f"训练时间热力图已保存: {filename}")
def plot_cost_heatmap(df, plot_dir):
    """绘制各方法训练时间热力图（sample × vars），每个方法单独保存。
    时间≥200 的格子标注 'TO'，颜色统一使用最深色。
    """
    methods = df['method'].unique()
    TIMEOUT_THRESHOLD = 200   # 超时阈值
    
    for method in methods:
        # 提取当前方法的数据
        subset = df[df['method'] == method]
        pivot = subset.pivot_table(index='sample', columns='vars', values='time', aggfunc='mean')
        pivot = pivot.sort_index(ascending=True)
        
        # 获取原始数值矩阵（用于颜色映射）
        data = pivot.values
        # 为颜色映射准备裁剪后的数据（超过阈值的使用阈值，避免颜色条被拉得太长）
        display_data = np.where(data >= TIMEOUT_THRESHOLD, TIMEOUT_THRESHOLD, data)
        
        plt.figure(figsize=(6, 5))
        # 使用线性颜色映射
        im = plt.imshow(display_data, aspect='auto', cmap='YlOrRd', origin='lower',
                        vmin=np.nanmin(display_data), vmax=TIMEOUT_THRESHOLD)
        
        # 坐标轴
        plt.xticks(range(len(pivot.columns)), pivot.columns.astype(int), fontsize=9)
        plt.yticks(range(len(pivot.index)), pivot.index.astype(int), fontsize=9)
        plt.xlabel('Number of Variables')
        plt.ylabel('Number of Samples')
        plt.title(f'{method}')
        
        # 在格子中标注数值
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if np.isnan(val):
                    continue
                # 决定标注文本
                if val >= TIMEOUT_THRESHOLD:
                    text = 'TO'
                else:
                    # 小于10保留一位小数，否则取整
                    text = f'{val:.1f}' if val < 10 else f'{int(val)}'
                
                # 文本颜色：如果颜色较深则用白色，否则黑色
                color_val = display_data[i, j] if not np.isnan(display_data[i, j]) else 0
                is_dark = color_val > TIMEOUT_THRESHOLD * 0.6
                plt.text(j, i, text, ha='center', va='center', fontsize=7,
                         color='white' if is_dark else 'black')
        
        # 颜色条（线性，整数刻度）
        cbar = plt.colorbar(im, label='Time (seconds)')
        # 设置颜色条刻度为整数（如果范围较小则适当保留一位小数）
        if np.nanmax(display_data) <= 10:
            cbar.set_ticks(np.arange(0, 11, 2))
        else:
            cbar.set_ticks(np.arange(0, int(TIMEOUT_THRESHOLD)+1, 50))
        
        plt.tight_layout()
        
        # 保存
        filename = os.path.join(plot_dir, f'cost_heatmap_{method}.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"训练时间热力图已保存: {filename}")

def plot_cost_bar_comparison_UNUSED(df, plot_dir):
    """绘制固定配置下各方法训练时间柱状图对比"""
    methods = df['method'].unique()
    colors = {'MCMC': '#4C72B0', 'BAS': '#55A868', 'Stacking': '#DD8452'}

    # 选取几个代表性配置
    samples_list = sorted(df['sample'].unique())
    vars_list = sorted(df['vars'].unique())
    # 取中间和最大配置
    plot_configs = []
    for s in samples_list:
        for v in vars_list:
            if df[(df['sample'] == s) & (df['vars'] == v)].shape[0] == len(methods):
                plot_configs.append((s, v))

    if not plot_configs:
        # 如果没有所有方法都有数据的配置，选择有 Stacking 数据的配置
        for s in samples_list:
            for v in vars_list:
                if not df[(df['sample'] == s) & (df['vars'] == v)].empty:
                    plot_configs.append((s, v))

    # 最多选 8 个配置
    if len(plot_configs) > 8:
        step = len(plot_configs) // 8
        plot_configs = plot_configs[::step][:8]

    labels = [f's={s}\nv={v}' for s, v in plot_configs]
    x = np.arange(len(plot_configs))
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(max(10, len(plot_configs) * 1.5), 6))
    for i, method in enumerate(methods):
        times = []
        for s, v in plot_configs:
            subset = df[(df['method'] == method) & (df['sample'] == s) & (df['vars'] == v)]
            times.append(subset['time'].mean() if not subset.empty else 0)
        ax.bar(x + i * width - 0.4 + width / 2, times, width,
               label=method, color=colors.get(method, '#999999'), alpha=0.8,
               edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Training Time (s)')
    ax.set_title('Training Cost Comparison Across Configurations')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    filename = os.path.join(plot_dir, 'cost_bar_comparison.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练时间柱状图对比已保存: {filename}")


# ==================== RQ4 数据加载 ====================
def load_rq4_data():
    """加载RQ4在线学习对比实验结果"""
    base_dir = os.path.dirname(__file__)
    rq4_file = os.path.join(base_dir, 'logs', 'online_comparison_results.tsv')
    if not os.path.exists(rq4_file):
        print(f"警告: 未找到RQ4结果文件 {rq4_file}")
        return None
    df = pd.read_csv(rq4_file, sep='\t')
    return df


# ==================== RQ4 绘图函数 ====================
def plot_rq4_re_boxplot(df, plot_dir):
    """RQ4: 三种方法的RE箱线图对比"""
    methods = df['method'].unique()
    colors_map = {
        'Static Stacking': '#4C72B0',
        'Online Stacking (SGD+Drift)': '#DD8452',
        'EWA Baseline': '#55A868'
    }

    data = [df[df['method'] == m]['RE'].values for m in methods]
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(
        data, labels=[m.replace(' (SGD+Drift)', '\n(SGD+Drift)') for m in methods],
        patch_artist=True, showfliers=True,
        flierprops=dict(marker='o', markersize=3, alpha=0.5)
    )
    for patch, m in zip(bp['boxes'], methods):
        patch.set_facecolor(colors_map.get(m, '#999999'))
        patch.set_alpha(0.7)

    ax.set_yscale('log')
    ax.set_ylabel('Relative Error (log scale)')
    ax.set_title('RQ4: Online Learning — Relative Error Comparison')
    ax.grid(True, alpha=0.3, axis='y')

    for i, m in enumerate(methods):
        median = np.median(df[df['method'] == m]['RE'].values)
        ax.text(i + 1, median * 1.3, f'median={median:.4f}', ha='center', va='bottom', fontsize=9)

    # 添加BMA基线参考线
    ax.axhline(y=0.0242, color='red', linestyle='--', alpha=0.6, label='BMA baseline (0.0242)')
    ax.legend(fontsize=9)

    plt.tight_layout()
    filename = os.path.join(plot_dir, 'rq4_re_comparison.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"RQ4 RE对比图已保存: {filename}")


def plot_rq4_success_bar(df, plot_dir):
    """RQ4: 三种方法的成功率柱状图"""
    methods = df['method'].unique()
    colors_map = {
        'Static Stacking': '#4C72B0',
        'Online Stacking (SGD+Drift)': '#DD8452',
        'EWA Baseline': '#55A868'
    }

    rates = []
    for m in methods:
        subset = df[df['method'] == m]
        rates.append(subset['success'].apply(lambda x: x == True or x == 'TRUE').mean())

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = [m.replace(' (SGD+Drift)', '\n(SGD+Drift)') for m in methods]
    bars = ax.bar(labels, rates,
                  color=[colors_map.get(m, '#999999') for m in methods],
                  alpha=0.8, edgecolor='black', linewidth=0.5)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{rate:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # BMA基线
    ax.axhline(y=0.7582, color='red', linestyle='--', alpha=0.6, label='BMA baseline (0.7582)')
    ax.set_ylabel('Success Rate')
    ax.set_title('RQ4: Online Learning — Success Rate Comparison')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=9)
    plt.tight_layout()
    filename = os.path.join(plot_dir, 'rq4_success_rate.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"RQ4 成功率对比图已保存: {filename}")


def plot_rq4_rolling_re(df, plot_dir):
    """RQ4: 滑动平均RE随样本数变化的趋势图（展示在线学习效果）"""
    methods = df['method'].unique()
    colors_map = {
        'Static Stacking': '#4C72B0',
        'Online Stacking (SGD+Drift)': '#DD8452',
        'EWA Baseline': '#55A868'
    }
    window = 30  # 滑动窗口大小

    fig, ax = plt.subplots(figsize=(10, 5))
    for m in methods:
        subset = df[df['method'] == m].reset_index(drop=True)
        re_vals = subset['RE'].values
        # 计算滑动平均
        rolling = pd.Series(re_vals).rolling(window=window, min_periods=5).median()
        ax.plot(range(len(rolling)), rolling,
                color=colors_map.get(m, '#999999'),
                label=m, linewidth=1.5, alpha=0.85)

    ax.axhline(y=0.0242, color='red', linestyle='--', alpha=0.5, label='BMA baseline')
    ax.set_xlabel('Sample Index (online stream)')
    ax.set_ylabel(f'Rolling Median RE (window={window})')
    ax.set_title('RQ4: Online Learning — RE Trend Over Time')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = os.path.join(plot_dir, 'rq4_rolling_re.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"RQ4 RE趋势图已保存: {filename}")


# ==================== 主函数 ====================
def main():
    base_dir = os.path.dirname(__file__)
    plot_dir = os.path.join(base_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # 检查是否指定了实验标签（用于 RQ2 图表文件名）
    experiment_label = ""
    if len(sys.argv) > 1:
        experiment_label = f" ({sys.argv[1]})"

    print(f"\n=== 开始绘图{experiment_label} ===")

    # # ----------------- RQ2 绘图 -----------------
    # stacking_df, bma_df = load_rq2_data()
    # if stacking_df is not None and bma_df is not None:
    #     print("正在绘制 RQ2 结果...")
    #     plot_re_comparison(stacking_df, bma_df, plot_dir, experiment_label)
    #     plot_success_rate_comparison(stacking_df, bma_df, plot_dir, experiment_label)
    #     plot_re_distribution(stacking_df, bma_df, plot_dir, experiment_label)
    # else:
    #     print("RQ2 数据加载失败，跳过相关绘图。")

    # # ----------------- RQ1 绘图 -----------------
    # rq1_data = load_rq1_data()
    # if rq1_data is not None:
    #     print("正在绘制 RQ1 结果...")
    #     plot_rq1_metrics(rq1_data, plot_dir)
    # else:
    #     print("RQ1 数据加载失败，跳过相关绘图。")

    # ----------------- RQ3 绘图 -----------------
    rq3_data = load_rq3_data()
    if rq3_data is not None:
        print("正在绘制 RQ3 结果...")
        # plot_cost_vs_sample(rq3_data, plot_dir)
        # plot_cost_vs_vars(rq3_data, plot_dir)
        plot_cost_heatmap(rq3_data, plot_dir)
    else:
        print("RQ3 数据加载失败，跳过相关绘图。")

    # ----------------- RQ4 绘图 -----------------
    rq4_data = load_rq4_data()
    if rq4_data is not None:
        print("正在绘制 RQ4 结果...")
        plot_rq4_re_boxplot(rq4_data, plot_dir)
        plot_rq4_success_bar(rq4_data, plot_dir)
        plot_rq4_rolling_re(rq4_data, plot_dir)
    else:
        print("RQ4 数据加载失败，跳过相关绘图。（请先运行 experiment_online.py）")

    print(f"\n所有图表已保存到: {plot_dir}")


if __name__ == '__main__':
    main()