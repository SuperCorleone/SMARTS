# Online Stacking vs Static Stacking 对比报告

## 1. 核心架构差异

| 维度 | Static (`stacking_static.py`) | Online (`stacking.py`) |
|------|-------------------------------|------------------------|
| 元学习器 | `LogisticRegression` (L-BFGS) | `SGDClassifier` (log_loss) |
| 增量更新 | 不支持 | `partial_fit()` 逐样本更新 |
| 漂移检测 | 无 | ADWIN-inspired 双窗口检测 |
| 漂移恢复 | 无 | 保留近期 buffer，检测到漂移后 re-fit |
| Occam's window | 暴力枚举 `combinations()` | 逐层扩展剪枝，支持 `max_vars` 上限 |

## 2. `partial_fit` 的使用方式

Online 版本的 `online_update()` 方法（`stacking.py:209-272`）采用**两阶段策略**：

- **常态路径**：对每个新样本调用 `meta_learner.partial_fit(mf, [label])`，单步 SGD 更新元学习器权重，O(1) 时间复杂度。
- **漂移路径**：当 `_detect_drift()` 发现近期误差均值显著高于历史均值（阈值 0.15）时，用近期 buffer（最多 150 条）重新 `fit()` 一个全新的 SGDClassifier，相当于在新分布上冷启动。

注意：**基模型（statsmodels Logit）始终保持冻结**，`partial_fit` 仅作用于元学习器层。

## 3. 可优化空间

1. **基模型层无在线更新**：当前所有 statsmodels Logit 基模型在 `fit()` 后固定不变。若数据分布漂移较大，仅更新元权重可能不够。可考虑将基模型也替换为支持 `partial_fit` 的 SGDClassifier。
2. **漂移检测阈值硬编码**：`threshold=0.15` 和 `min_window=30` 为固定值，可引入自适应阈值（如真正的 ADWIN 算法的自适应分桶）。
3. **retrain buffer 策略粗糙**：当前漂移触发时直接在 buffer 上 full re-fit，丢弃所有历史权重。可改为对学习率做 warm-restart（提高 `eta0`）而非完全重建。

## 4. Sliding Window Online 方案可行性分析

**可行，但需权衡代价：**

- **元学习器层**：直接可行。维护一个固定大小的滑动窗口存储 `(meta_features, label)`，每次窗口滑动后在窗口内数据上 re-fit SGDClassifier（或持续 partial_fit + 样本权重衰减）。这与当前的 retrain buffer 机制本质相似，但更系统化。
- **基模型层**：这是主要瓶颈。当前基模型为 statsmodels MLE 拟合，不支持增量学习。若要在窗口内重训基模型，每次窗口滑动的开销为 O(k × window_size)（k = 基模型数量），在基模型较多时代价显著。
- **推荐折中方案**：元学习器用 sliding window 持续更新；基模型按固定间隔（如每 N 个窗口）或仅在漂移触发时批量重训，避免逐样本重训的开销。
