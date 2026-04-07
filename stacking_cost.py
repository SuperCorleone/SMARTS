#!/usr/bin/env python3
"""
stacking_cost.py
RQ3: 计算花费（训练时间）对比实验。
在不同 (sample, vars) 配置下分别测量 BMA (MCMC, BAS) 和 Stacking 的训练时间。
BMA 部分通过调用 R 脚本 bma_cost.R 完成；Stacking 使用 StackingEnsemble。
可直接运行: python stacking_cost.py
"""

import subprocess
import time
import pandas as pd
import numpy as np
from statsmodels.tools import add_constant
import warnings
import os
from datetime import datetime

from stacking import StackingEnsemble

warnings.filterwarnings('ignore')
os.environ["OMP_NUM_THREADS"] = "1"


# ============================================================
# 配置
# ============================================================
DATA_FILE = os.path.join(os.path.dirname(__file__), 'data', 'training_rescueRobot_25600_64.csv')
R_SCRIPT = os.path.join(os.path.dirname(__file__), 'bma_cost.R')
N_REPEATS = 1           # 每个配置重复次数
TIMEOUT = 200            # 超时阈值（秒）
N_FOLDS = 3              # Stacking 交叉验证折数（cost 实验中用较少折数以加速）

# 实验配置：(sample_size, num_vars)
# 与 cost_input.csv 中的配置保持一致
CONFIGS = [
    (200, 2), (200, 4), (200, 8), (200, 16), (200, 32), (200, 64),
    (400, 2), (400, 4), (400, 8), (400, 16), (400, 32), (400, 64),
    (800, 2), (800, 4), (800, 8), (800, 16), (800, 32), (800, 64),
    (1600, 2), (1600, 4), (1600, 8), (1600, 16), (1600, 32), (1600, 64),
    (3200, 2), (3200, 4), (3200, 8), (3200, 16), (3200, 32), (3200, 64),
    (6400, 2), (6400, 4), (6400, 8), (6400, 16), (6400, 32), (6400, 64),
    (12800, 2), (12800, 4), (12800, 8), (12800, 16), (12800, 32), (12800, 64),
    (25600, 2), (25600, 4), (25600, 8), (25600, 16), (25600, 32), (25600, 64),
]

# 是否运行 BMA（需要 R 环境和 bma_cost.R 脚本）
RUN_BMA = True


def run_bma(method, sample, vars_int, log_file):
    """通过 R 脚本运行 BMA (MCMC 或 BAS)，返回耗时"""
    for rep in range(N_REPEATS):
        try:
            p = subprocess.Popen(
                ['Rscript', R_SCRIPT, '-s', str(sample), '-v', str(vars_int), '-m', method],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, _ = p.communicate(timeout=TIMEOUT)
            line = stdout.decode('utf-8').strip()
            if line:
                log_file.write(line + '\n')
            else:
                log_file.write(f"{method} {vars_int} {sample} {TIMEOUT}\n")
            log_file.flush()
        except subprocess.TimeoutExpired:
            p.kill()
            log_file.write(f"{method} {vars_int} {sample} {TIMEOUT}\n")
            log_file.flush()
        except Exception as e:
            print(f"  Error in BMA {method} for sample={sample}, vars={vars_int}: {e}")
            log_file.write(f"{method} {vars_int} {sample} {TIMEOUT}\n")
            log_file.flush()


def run_stacking(sample, vars_int, data, log_file):
    """运行 Stacking 训练，返回耗时"""
    feature_cols = [f'x{i}' for i in range(1, vars_int + 1)]
    X_raw = data.loc[:sample - 1, feature_cols]
    y = data.loc[:sample - 1, 'hazard']

    for rep in range(N_REPEATS):
        try:
            start = time.time()
            stacking = StackingEnsemble(n_folds=N_FOLDS, random_state=42)
            stacking.fit(X_raw, y)
            elapsed = min(time.time() - start, TIMEOUT)
            log_file.write(f"Stacking {vars_int} {sample} {elapsed:.4f}\n")
            log_file.flush()
        except Exception as e:
            print(f"  Error in Stacking for sample={sample}, vars={vars_int}: {e}")
            log_file.write(f"Stacking {vars_int} {sample} {TIMEOUT}\n")
            log_file.flush()


def main():
    start_time = datetime.now()
    print(f"=== Stacking RQ3 (Cost) Experiment Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')} ===")

    # 创建日志目录
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    output_log = os.path.join(log_dir, 'stacking_rq3_results.tsv')

    # 加载数据
    print(f"Loading data from {DATA_FILE}...")
    data = pd.read_csv(DATA_FILE)
    print(f"Data shape: {data.shape}")

    total = len(CONFIGS)

    with open(output_log, 'w') as f:
        f.write("method vars sample time\n")

        for idx, (sample, vars_int) in enumerate(CONFIGS):
            print(f"\n[{idx + 1}/{total}] sample={sample}, vars={vars_int}")

            # 1. BMA (MCMC)
            if RUN_BMA and os.path.exists(R_SCRIPT):
                print(f"  Running BMA MCMC...")
                run_bma('MCMC', sample, vars_int, f)
                print(f"  Running BMA BAS...")
                run_bma('BAS', sample, vars_int, f)
            elif RUN_BMA:
                print(f"  Skipping BMA: R script not found at {R_SCRIPT}")

            # 2. Stacking
            print(f"  Running Stacking...")
            run_stacking(sample, vars_int, data, f)

    end_time = datetime.now()
    print(f"\n=== Experiment Complete ===")
    print(f"Duration: {end_time - start_time}")
    print(f"Results saved to: {output_log}")

    # 打印结果摘要
    print("\n=== Results Summary ===")
    results = pd.read_csv(output_log, sep=r'\s+')
    for method in results['method'].unique():
        subset = results[results['method'] == method]
        print(f"\n{method}:")
        print(f"  Configs: {len(subset)}")
        print(f"  Mean time: {subset['time'].mean():.4f}s")
        print(f"  Max time: {subset['time'].max():.4f}s")
        print(f"  Timeouts (>={TIMEOUT}s): {(subset['time'] >= TIMEOUT).sum()}")


if __name__ == '__main__':
    main()
