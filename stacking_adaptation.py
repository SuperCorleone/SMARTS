#!/usr/bin/env python3
"""
stacking_adaptation.py
RQ2: Adaptation Decision Quality — Stacking vs BMA vs individual Logit models.

Mirrors bma-package/bma_adaptation.py experiment design:
  - Train Stacking on training set (450 samples)
  - Build BMA oracle on validation set (ground truth)
  - For each hazard=0 sample, use GA to adapt (power, band, quality, speed)
  - Compute Relative Error (RE) and Success Rate vs oracle
  - Save results in same TSV format as baseline re_bma_logit.log

Baseline data (BMA + Logit1-8) is already in logs/re_bma_logit.log.

Usage:
    python stacking_adaptation.py                # all hazard=0 samples (~20min)
    python stacking_adaptation.py --limit 20     # first 20 samples (fast test)
"""

import numpy as np
import pandas as pd
from statsmodels.tools import add_constant
from geneticalgorithm import geneticalgorithm as ga
import warnings
import os
import argparse
from datetime import datetime

from stacking import StackingEnsemble
from bma import BMA

warnings.filterwarnings('ignore')


# ============================================================
# GA adaptation (same as baseline bma_adaptation.py)
# ============================================================
tmp_model = None
tmp_vars = None
tmp_data = None
tmp_index_set = None
tmp_initial_values = None


def fitness(X):
    global tmp_model, tmp_vars, tmp_data, tmp_index_set, tmp_initial_values
    i = 0
    for k in tmp_index_set:
        tmp_data.iloc[0, tmp_data.columns.get_loc(tmp_vars[k][0])] = X[i]
        i += 1
    prediction = tmp_model.predict_single(tmp_data)
    if prediction < 0.51:
        prediction = prediction / 10
    delta_change = 0.0
    i = 0
    for k in tmp_index_set:
        delta_change += tmp_vars[k][3] * abs(X[i] - tmp_initial_values[i]) / (
                tmp_vars[k][2][1] - tmp_vars[k][2][0])
        i += 1
    return delta_change - prediction


def run_adaptation(model, vars_dict, index_set, row_data, fitness_fn):
    vartype = np.array([vars_dict[k][1] for k in index_set])
    varbound = np.array([vars_dict[k][2] for k in index_set])
    params = {
        'max_num_iteration': 50,
        'population_size': 100,
        'mutation_probability': 0.1,
        'elit_ratio': 0.01,
        'crossover_probability': 0.5,
        'parents_portion': 0.3,
        'crossover_type': 'uniform',
        'max_iteration_without_improv': None
    }
    ga_model = ga(
        function=fitness_fn,
        dimension=len(index_set),
        variable_type_mixed=vartype,
        variable_boundaries=varbound,
        convergence_curve=False,
        progress_bar=False,
        algorithm_parameters=params
    )
    ga_model.run()
    assignment = ga_model.output_dict['variable']
    new_row = row_data.copy()
    for i, k in enumerate(index_set):
        new_row.iloc[0, new_row.columns.get_loc(vars_dict[k][0])] = assignment[i]
    return new_row


# ============================================================
# Main experiment
# ============================================================
def main():
    global tmp_model, tmp_vars, tmp_data, tmp_index_set, tmp_initial_values

    parser = argparse.ArgumentParser(description='RQ2: Adaptation Decision Quality')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of test samples (default: all)')
    args = parser.parse_args()

    start_time = datetime.now()
    print(f"=== RQ2: Adaptation Decision Quality ===")
    print(f"    Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    df = pd.read_csv(os.path.join(data_dir, 'training_rescueRobot_450.csv'))
    dfv = pd.read_csv(os.path.join(data_dir, 'validation_rescueRobot_450.csv'))
    df["firm"] = (df["firm"] == "Yes") * 1
    dfv["firm"] = (dfv["firm"] == "Yes") * 1

    LIMIT = 450
    X_oracle = dfv.drop(["hazard"], axis=1)
    y_oracle = dfv["hazard"]
    X_train = df.drop(["hazard"], axis=1)[0:LIMIT]
    y_train = df["hazard"][0:LIMIT]

    # Build oracle (BMA on validation data, same as baseline)
    print("\nBuilding Oracle model (BMA on validation data)...")
    oracle = BMA(y_oracle, add_constant(X_oracle), RegType='Logit', Verbose=False).fit()

    # Build Stacking model
    print("Building Stacking model...")
    stacking = StackingEnsemble(n_folds=10, random_state=42)
    stacking.fit(X_train, y_train)
    print(f"  Base models: {len(stacking.base_models)}")

    # Adaptable variables (same as baseline)
    vars_dict = {
        5: ('power', ['int'], [13, 78], 0.8),
        6: ('band', ['real'], [14.7, 46.58], 0.4),
        7: ('quality', ['real'], [0, 147.19], 0.2),
        8: ('speed', ['int'], [15, 64], 0.1)
    }

    # Select hazard=0 samples for adaptation
    selected_rows = [i for i in df.index if df.loc[i, 'hazard'] == 0]
    if args.limit is not None:
        selected_rows = selected_rows[:args.limit]
    print(f"\nAdaptation samples: {len(selected_rows)} (hazard=0)")

    results = []
    for count, r in enumerate(selected_rows):
        row_data = df.drop(["hazard"], axis=1)
        row_data = add_constant(row_data)[r:(r + 1)]

        # Set GA globals
        tmp_model = stacking
        tmp_vars = vars_dict
        tmp_data = row_data
        tmp_index_set = [5, 6, 7, 8]
        tmp_initial_values = [row_data[vars_dict[k][0]].values[0] for k in tmp_index_set]

        # GA adaptation
        new_data = run_adaptation(stacking, vars_dict, tmp_index_set, row_data, fitness)

        # Stacking prediction on adapted configuration
        prediction = stacking.predict_single(new_data)

        # Oracle prediction on adapted configuration
        pred_oracle = oracle.predict(row_data)[0]

        # RE and success (same criteria as baseline)
        re_val = abs(prediction - pred_oracle) / pred_oracle if pred_oracle > 1e-7 else float('nan')
        success = prediction > 0.5 and pred_oracle > 0.5

        results.append({'row': r, 'RE': re_val, 'success': success,
                        'prediction': prediction, 'oracle': pred_oracle})
        print(f'  [{count+1}/{len(selected_rows)}] RE={re_val:.4f} Success={success}')

        if (count + 1) % 50 == 0:
            elapsed = datetime.now() - start_time
            re_vals = [r['RE'] for r in results if not np.isnan(r['RE'])]
            succ_vals = [r['success'] for r in results]
            print(f"    Progress: Median RE={np.median(re_vals):.4f}, "
                  f"Success Rate={np.mean(succ_vals):.4f}, "
                  f"Elapsed={elapsed}")

    # ---- Save results ----
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    output_file = os.path.join(log_dir, 'stacking_rq2_results.tsv')
    with open(output_file, 'w') as f:
        f.write("type\tmodel\tRE\tsuccess\n")
        for r in results:
            f.write(f"Stacking\tStacking\t{r['RE']}\t{'TRUE' if r['success'] else 'FALSE'}\n")
    print(f"\nResults saved: {output_file}")

    # ---- Summary ----
    end_time = datetime.now()
    re_values = [r['RE'] for r in results if not np.isnan(r['RE'])]
    success_values = [r['success'] for r in results]

    print(f"\n=== Summary ===")
    print(f"  Samples: {len(results)}")
    print(f"  Median RE: {np.median(re_values):.6f}")
    print(f"  Mean RE: {np.mean(re_values):.6f}")
    print(f"  Success Rate: {np.mean(success_values):.4f} ({sum(success_values)}/{len(success_values)})")
    print(f"  Runtime: {end_time - start_time}")


if __name__ == '__main__':
    main()
