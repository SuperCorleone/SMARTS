# STACK-SAS: 基于 Stacking 集成学习的自适应系统模型不确定性管理

## 项目简介

本项目是论文 *"Taming Model Uncertainty in Self-Adaptive Systems using Bayesian Model Averaging"* 的扩展研究，提出用 **Stacking 集成学习** 替代 MAPE-K 反馈环中 Analyze 组件的 BMA（贝叶斯模型平均），并进一步引入**在线学习**机制，使模型在运行时持续适应环境变化。

### 核心思路

```
原始方法 (TUNE):  变量子集 GLM → BMA 加权平均 → 适应决策
本项目 (STACK-SAS): 变量子集 GLM → Stacking 元学习器组合 → 在线更新 → 适应决策
```

## 研究问题

| 编号 | 研究问题 | 对应脚本 |
|------|---------|---------|
| RQ1 | Stacking 与 BMA 的预测精度对比（Precision / Recall / F1） | `stacking_prediction.py` |
| RQ2 | Stacking 与 BMA 的适应决策质量对比（RE / 成功率） | `stacking_adaptation.py` |
| RQ3 | Stacking 与 BMA 的计算花费对比（训练时间 vs 样本量/变量数） | `stacking_cost.py` |
| RQ4 | 在线学习是否提升 Stacking 的适应性能（SGD partial_fit + 漂移检测 vs EWA baseline） | `experiment_online.py` |

## 项目结构

```
stacking-package/
├── stacking.py                 # 核心：StackingEnsemble 类（SGDClassifier 元学习器 + 在线 partial_fit + 漂移检测）
├── ewa_ensemble.py             # EWA 指数加权聚合（在线学习 baseline）
├── stacking_prediction.py      # RQ1 实验：预测精度对比
├── stacking_adaptation.py      # RQ2 实验：适应决策对比
├── stacking_cost.py            # RQ3 实验：计算花费对比（调用 bma_cost.R 获取 BMA 数据）
├── experiment_online.py        # RQ4 实验：在线学习对比（Static / SGD+Drift / EWA）
├── plot_rq1_rq2_rq3.py        # 绘图脚本（RQ1 精度柱状图 / RQ2 RE 箱线图 / RQ3 热力图）
├── data/                       # 数据集
│   ├── training_rescueRobot_450.csv       # RQ1/RQ2 训练集（450 样本，9 特征）
│   ├── validation_rescueRobot_450.csv     # RQ1/RQ2 验证集
│   ├── training_rescueRobot_25600_64.csv  # RQ3 大规模训练数据（25600 样本，64 变量）
│   └── cost_input.csv                     # RQ3 配置参数
├── logs/                       # 实验结果日志
│   ├── stacking_rq1_results.tsv           # RQ1：BMA/Stacking precision/recall/F1
│   ├── stacking_rq2_results.tsv           # RQ2：逐样本 RE 和成功标记
│   ├── stacking_rq2_summary.json          # RQ2：汇总指标
│   ├── stacking_rq3_results.tsv           # RQ3：MCMC/BAS/Stacking 各配置训练时间
│   ├── online_comparison_results.tsv      # RQ4：逐样本三方法对比
│   └── online_comparison_summary.json     # RQ4：汇总指标
├── plots/                      # 图表输出
│   ├── precision_comparison.png           # RQ1
│   ├── recall_comparison.png              # RQ1
│   ├── f1_comparison.png                  # RQ1
│   ├── relative_error_comparison.png      # RQ2
│   ├── success_rate_comparison.png        # RQ2
│   ├── re_distribution.png               # RQ2
│   ├── cost_heatmap_MCMC.png             # RQ3
│   ├── cost_heatmap_BAS.png              # RQ3
│   ├── cost_heatmap_Stacking.png         # RQ3
│   ├── rq4_re_comparison.png             # RQ4: RE箱线图
│   ├── rq4_success_rate.png              # RQ4: 成功率柱状图
│   └── rq4_rolling_re.png               # RQ4: RE随时间趋势
├── research proposal.docx      # 研究 proposal
└── readme_cn.md                # 本文件
```

## 方法说明

### 1. Stacking 集成（离线）

采用两层架构：

- **Level-0 基模型**：枚举所有变量子集的 Logistic 回归，通过 Occam's window 剪枝（BIC 似然比 < 1/20 的模型被淘汰）
- **Level-1 元学习器**：在 log-odds 空间上训练的 `SGDClassifier(loss='log_loss', penalty='l2', alpha=0.1, eta0=0.01)`，通过 10 折交叉验证生成元特征防止过拟合。使用 SGDClassifier 而非 LogisticRegression 的原因是 SGDClassifier 支持 `partial_fit()`，可以逐样本增量更新权重而无需全量重训。

### 2. SGD 增量更新 + 漂移检测（RQ4 主方法）

在线阶段每收到一个样本的真实反馈后：

1. 冻结基模型，仅计算该样本的元特征（29 维 log-odds 向量）
2. 计算预测误差 `|p_t − y_t|`，送入 ADWIN 风格的漂移检测器
3. **正常情况**：调用 `meta_learner.partial_fit(mf, [y_t])` 增量更新权重（无需缓冲区，无需全量重训）
4. **漂移检测触发**：当误差窗口的近期均值显著高于历史均值（差值 > 0.15）时，用近期缓冲数据（最多 150 条）全量重训 meta-learner，然后重置漂移检测器

**优势**：
- 基模型不需重训（耗时最大部分），仅更新轻量 meta-learner
- `partial_fit()` 实现真正的逐样本在线学习，权重逐步漂移向新模式
- 漂移检测确保只在环境真正变化时触发重训，避免固定周期的浪费
- 代码改动最小，自然融入现有框架

**潜在风险**：对学习率敏感，单个噪声样本可能将权重推向错误方向。因此使用保守学习率（eta0=0.01）并结合漂移检测器作为安全网。可通过实验经验性调整学习率获得更好效果。

### 3. 指数加权聚合 EWA（RQ4 baseline）

复用 Stacking 的基模型，用 Hedge 算法维护权重：

```
w_i ← w_i × exp(-η × loss_i)，归一化
```

其中 loss 为各基模型的交叉熵损失，学习率 η=0.5。

## 已有实验结果

### RQ1：预测精度

| 模型 | Precision | Recall | F1 | 训练时间 |
|------|-----------|--------|-------|---------|
| BMA | 0.6496 | 0.5563 | 0.5993 | 0.10s |
| Stacking | 0.6589 | 0.5313 | 0.5882 | 1.02s |

### RQ2：适应决策

| 模型 | Median RE | Mean RE | 成功率 |
|------|-----------|---------|--------|
| BMA (论文) | 0.0242 | - | 75.82% |
| Stacking | 0.0211 | 0.0268 | 79.14% |

### RQ3：计算花费

在 (sample, vars) 从 (200, 2) 到 (25600, 64) 的网格上比较三种方法的训练时间：

| 方法 | 说明 | 高维度瓶颈 |
|------|------|-----------|
| MCMC (BMA) | R 语言 BAS 包的 MCMC 采样 | 随样本和变量线性增长，最慢约 180s |
| BAS (BMA) | R 语言 BAS 包的确定性枚举 | 变量 >= 16 时超时（> 200s） |
| Stacking | Python StackingEnsemble | 变量 64 + 样本 25600 时约 179s |

详细热力图见 `plots/cost_heatmap_*.png`。

### RQ4：在线学习对比

**实验设计**：离线在训练集（462 样本）上训练 Stacking，然后在**验证集**（462 样本，模型从未见过）的 hazard=0 行上做在线适应。Oracle BMA 在验证集上训练，提供真实标签反馈。这确保在线阶段的数据对 Stacking 是全新的，在线学习能真正获取新信息。

运行 `experiment_online.py` 可得到三者的完整对比：

| 方法 | 说明 |
|------|------|
| Static Stacking | 纯离线，无更新（baseline） |
| Online Stacking (SGD+Drift) | SGDClassifier partial_fit + ADWIN 漂移检测（主方法） |
| EWA Baseline | 指数加权在线聚合（轻量 baseline） |

在线更新流程：

```
for each new (x_t, y_t):
    p_t ← meta_learner.predict_proba(x_t)
    error_t ← |p_t − y_t|
    feed error_t into drift detector (ADWIN-inspired)
    if drift_detected:
        retrain meta_learner on recent buffer
        reset drift detector
    else:
        meta_learner.partial_fit(x_t, [y_t])  # incremental update
```

绘图输出：`plots/rq4_re_comparison.png`（RE 箱线图）、`rq4_success_rate.png`（成功率）、`rq4_rolling_re.png`（RE 随时间趋势）。

## 快速开始

### 环境依赖

```bash
pip install numpy pandas statsmodels scikit-learn mpmath geneticalgorithm matplotlib
```

### 运行实验

```bash
# RQ1：预测精度对比
python stacking_prediction.py

# RQ2：适应决策对比
python stacking_adaptation.py

# RQ3：计算花费对比（需要 R 环境 + BAS 包）
python stacking_cost.py

# RQ4：在线学习对比
python experiment_online.py

# 绘图（RQ1 + RQ2 + RQ3）
python plot_rq1_rq2_rq3.py
```

### 数据要求

- RQ1/RQ2 数据：`../stacking-package/data/` 目录下的 `training_rescueRobot_450.csv` 和 `validation_rescueRobot_450.csv`
- RQ3 数据：本地 `data/training_rescueRobot_25600_64.csv`
- 如数据路径不同，请修改各脚本中的 `data_dir` 或 `DATA_FILE`

## 关键参数

| 参数 | 默认值 | 所属 | 说明 |
|------|--------|------|------|
| `n_folds` | 10（RQ1/2/4）, 3（RQ3） | StackingEnsemble | 交叉验证折数 |
| `alpha` | 0.1 | SGDClassifier | Meta-learner L2 正则化强度 |
| `eta0` | 0.01 | SGDClassifier | SGD 学习率（保守值，防止噪声样本不稳定） |
| `max_vars` | None (自动 min(nCols, 15)) | StackingEnsemble | 基模型最大变量子集大小 |
| `_retrain_buffer_max` | 150 | StackingEnsemble | 漂移重训时使用的最大近期样本数 |
| `drift threshold` | 0.15 | _detect_drift | 新旧误差均值差超过此值触发漂移 |
| `drift min_window` | 30 | _detect_drift | 漂移检测激活所需最少观测数 |
| `eta` | 0.5 | EWAEnsemble | EWA 学习率 |
| `use_log_odds` | True | 两者 | 是否在 log-odds 空间聚合 |
| `TIMEOUT` | 200s | stacking_cost.py | RQ3 单次训练超时阈值 |

## 参考文献

- Giese, H. et al. "Taming model uncertainty in self-adaptive systems using Bayesian Model Averaging." (基线论文)
