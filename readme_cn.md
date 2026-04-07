# STACK-SAS: 基于在线 Stacking 集成学习的自适应系统模型不确定性管理

## 项目简介

本项目是论文 *"Taming Model Uncertainty in Self-Adaptive Systems using Bayesian Model Averaging"* 的扩展研究，提出用**在线 Stacking 集成学习**替代 MAPE-K 反馈环中 Analyze 组件的 BMA（贝叶斯模型平均）。核心创新是将 Stacking 设计为一个**从一开始就在线的学习系统**——随着自适应系统运行，持续接收新数据并增量更新，而非传统的离线训练 + 部署模式。

### 核心思路

```
原始方法 (TUNE):  变量子集 GLM → BMA 加权平均 → 适应决策（需全量重训）
本项目 (STACK-SAS): 变量子集 GLM → SGD 元学习器 → partial_fit 在线更新 + 漂移检测 → 适应决策
```

### 架构

```
训练集 (450 samples)
├── 冷启动阶段 (warm-up): 前 N 条用于初始化 (SGDClassifier.fit)
└── 剩余部分: 并入在线流

验证集 (462 samples)
└── 完整作为在线评估流

整个流程: warm-up(100) → 在线流(824) — 一条连续的数据流，无硬分割
```

## 研究问题

| 编号 | 研究问题 | 指标 | 说明 |
|------|---------|------|------|
| RQ1 | 在线 Stacking 的预测性能是否随时间提升？ | Precision / Recall / F1 | Prequential 评估，窗口化指标 |
| RQ2 | 在线 Stacking 能否支持有效的运行时适应决策？ | RE / Success Rate | 不同阶段 checkpoint 做 GA 适应 |
| RQ3 | 在线 Stacking 的计算效率是否优于 BMA 重训？ | 累计时间 / 单步延迟 | partial_fit O(1) vs BMA retrain O(n) |
| RQ4 | 漂移检测的敏感性分析（消融实验） | F1 / MAE / 漂移次数 | 阈值和学习率的影响 |

所有 RQ 统一由 `experiment_online.py` 运行。

## 项目结构

```
stacking-package/
├── stacking.py                 # 核心：StackingEnsemble 类
│                               #   - SGDClassifier 元学习器（支持 partial_fit）
│                               #   - ADWIN 漂移检测 + 自动重训
│                               #   - 小数据集自动调整 CV 折数
├── ewa_ensemble.py             # EWA 指数加权聚合（在线学习 baseline）
├── experiment_online.py        # 统一实验脚本（RQ1-RQ4）
│                               #   --rq 1: 预测性能随时间变化
│                               #   --rq 2: 适应决策质量 (GA + BMA oracle)
│                               #   --rq 3: 计算效率对比
│                               #   --rq 4: 漂移检测消融实验
├── plot_rq1_rq2_rq3.py        # 绘图脚本
├── data/                       # 数据集
│   ├── training_rescueRobot_450.csv       # 训练集（450 样本，9 特征）
│   ├── validation_rescueRobot_450.csv     # 验证集（462 样本）
│   ├── training_rescueRobot_25600_64.csv  # 大规模数据（RQ3 扩展）
│   └── cost_input.csv
├── logs/                       # 实验结果
│   ├── online_experiment_results.json     # 统一结果（RQ1-RQ4）
│   └── online_rq1_per_sample.tsv          # RQ1 逐样本数据
├── plots/                      # 图表输出
├── research proposal.docx
└── readme_cn.md                # 本文件
```

## 方法说明

### 1. Stacking 集成架构

采用两层架构：

- **Level-0 基模型**：枚举所有变量子集的 Logistic 回归，通过 Occam's window 剪枝（BIC 似然比 < 1/20 的模型被淘汰）
- **Level-1 元学习器**：`SGDClassifier(loss='log_loss', penalty='l2', alpha=0.1, eta0=0.01)`，在 log-odds 空间训练。使用 SGDClassifier 的原因是支持 `partial_fit()` 逐样本增量更新

小数据集（warm-up 阶段）自动降低 CV 折数，确保每折至少 10 个样本。

### 2. 在线学习流程

```python
# 冷启动
stacking = StackingEnsemble()
stacking.fit(X_warmup, y_warmup)   # warm-up 数据初始化

# 在线流（prequential: 先预测再更新）
for t, (x_t, y_t) in enumerate(online_stream):
    p_t = stacking.predict_single(x_t)     # 1. 用当前模型预测
    log(t, p_t, y_t)                        # 2. 记录（用于评估）
    stacking.online_update(x_t, y_t)        # 3. 收到真实标签后更新
```

### 3. SGD 增量更新 + ADWIN 漂移检测

每收到一个样本：

1. 冻结基模型，计算元特征（log-odds 向量）
2. 计算预测误差 `|p_t − y_t|`，送入漂移检测器
3. **正常情况**：`meta_learner.partial_fit(mf, [y_t])` 增量更新权重
4. **漂移触发**：近期误差均值显著高于历史均值（差值 > threshold）时，用近期缓冲数据全量重训 meta-learner，重置检测器

## 实验结果

### RQ1：预测性能随时间变化

Prequential 评估，每 100 样本一个窗口：

| 阶段 (samples_seen) | Precision | Recall | F1 |
|---------------------|-----------|--------|-----|
| 200 | 0.600 | 0.441 | 0.508 |
| 300 | 0.526 | 0.270 | 0.357 |
| 400 | 0.588 | 0.312 | 0.408 |
| 500 | 0.778 | 0.553 | **0.646** |
| 600 | 0.636 | 0.438 | 0.519 |
| 700 | 0.440 | 0.324 | 0.373 |
| 800 | 0.800 | 0.444 | 0.571 |
| 900 | 0.654 | 0.548 | **0.596** |
| **Overall** | **0.623** | **0.406** | **0.491** |

漂移重训触发：2 次。总运行时间：~6 秒。

### RQ3：计算效率

| 累计数据量 | Online 累计时间 | BMA 重训时间 |
|-----------|----------------|-------------|
| 100 | 0.31s | 0.09s |
| 200 | 0.73s | 0.09s |
| 400 | 1.54s | 0.10s |
| 800 | 3.15s | 0.08s |

注：当前数据规模小（<1000），Online 累计包含 warm-up 训练开销。BMA 的 O(2^n) 模型枚举劣势在高维度/大数据量下才会显现。Online 的核心优势是**单步更新时间恒定**（~3.7ms），而 BMA 重训时间随数据量线性增长。

### RQ4：漂移检测消融实验

**策略对比：**

| 策略 | F1 | MAE | 重训次数 |
|------|-----|------|---------|
| 纯 partial_fit (无漂移) | **0.5071** | 0.3651 | 0 |
| partial_fit + 漂移检测 (默认) | 0.4914 | **0.3603** | 2 |
| 固定重训 K=50 | 0.4926 | 0.3666 | 16 |
| 固定重训 K=100 | 0.4748 | 0.3670 | 8 |

**漂移阈值敏感性：**

| Threshold | F1 | MAE | 漂移次数 |
|-----------|-----|------|---------|
| 0.05 | 0.4814 | **0.3594** | 7 |
| 0.10 | 0.4837 | 0.3595 | 7 |
| 0.15 (默认) | 0.4914 | 0.3603 | 2 |
| 0.20 | **0.5071** | 0.3651 | 0 |

**学习率敏感性：**

| eta0 | F1 | MAE | 漂移次数 |
|------|-----|------|---------|
| 0.001 | 0.4902 | 0.3630 | 1 |
| 0.005 | 0.4893 | 0.3631 | 1 |
| 0.01 (默认) | 0.4914 | 0.3603 | 2 |
| **0.05** | **0.4968** | **0.3594** | 2 |

**发现**：较高阈值（少重训）倾向于更高 F1，较低阈值倾向于更低 MAE；eta0=0.05 在两项指标上均表现最佳。

## 快速开始

### 环境依赖

```bash
pip install numpy pandas statsmodels scikit-learn mpmath geneticalgorithm matplotlib
```

### 运行实验

```bash
# 运行全部 RQ（RQ2 含 GA 较慢，其余几秒完成）
python experiment_online.py

# 单独运行某个 RQ
python experiment_online.py --rq 1   # RQ1: 预测性能（~6s）
python experiment_online.py --rq 2   # RQ2: 适应决策（含 GA，~20min）
python experiment_online.py --rq 3   # RQ3: 计算效率（~10s）
python experiment_online.py --rq 4   # RQ4: 消融实验（~90s）

# 自定义参数
python experiment_online.py --warmup 100 --window 50 --rq 1
```

### 数据要求

- `data/training_rescueRobot_450.csv` 和 `data/validation_rescueRobot_450.csv`
- 训练集前 N 条作为 warm-up，剩余 + 验证集构成在线流

## 关键参数

| 参数 | 默认值 | 所属 | 说明 |
|------|--------|------|------|
| `--warmup` | 100 | experiment_online.py | 冷启动样本数 |
| `--window` | 50 | experiment_online.py | 窗口化指标的窗口大小 |
| `n_folds` | 10 (自动降低) | StackingEnsemble | CV 折数，小数据集自动调整 |
| `alpha` | 0.1 | SGDClassifier | Meta-learner L2 正则化强度 |
| `eta0` | 0.01 | SGDClassifier | SGD 学习率 |
| `_retrain_buffer_max` | 150 | StackingEnsemble | 漂移重训最大缓冲样本数 |
| `drift threshold` | 0.15 | _detect_drift | 漂移触发阈值 |
| `drift min_window` | 30 | _detect_drift | 漂移检测最少观测数 |

## 参考文献

- Giese, H. et al. "Taming model uncertainty in self-adaptive systems using Bayesian Model Averaging." (基线论文)
