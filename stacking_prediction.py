#!/usr/bin/env python3
"""
stacking_prediction.py
RQ1: Prediction Accuracy — Stacking vs BMA vs individual Logit models.

Mirrors bma-package/bma_prec_recall.py experiment design:
  - Train Stacking on training set (450 samples)
  - Evaluate precision/recall/F1 on validation set
  - Save results in same TSV format as baseline precision_recall.log

Baseline data (Logit1-8) is already in logs/precision_recall.log.
Use --run-bma to also run BMA (optional, slow).

Usage:
    python stacking_prediction.py               # Stacking only (fast)
    python stacking_prediction.py --run-bma     # Stacking + BMA
"""

import numpy as np
import pandas as pd
from statsmodels.tools import add_constant
from sklearn.metrics import recall_score, precision_score, f1_score
import time
import warnings
import os
import argparse

from stacking import StackingEnsemble
from bma import BMA

warnings.filterwarnings('ignore')


def precision(predictions, true_labels):
    return precision_score(true_labels, (predictions > 0.5), zero_division=0)

def recall(predictions, true_labels):
    return recall_score(true_labels, (predictions > 0.5), zero_division=0)

def f_measure(predictions, true_labels):
    return f1_score(true_labels, (predictions > 0.5), zero_division=0)


def main():
    parser = argparse.ArgumentParser(description='RQ1: Prediction Accuracy')
    parser.add_argument('--run-bma', action='store_true', help='Also run BMA (baseline data already exists)')
    args = parser.parse_args()

    print("=== RQ1: Prediction Accuracy ===")

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    df = pd.read_csv(os.path.join(data_dir, 'training_rescueRobot_450.csv'))
    dfv = pd.read_csv(os.path.join(data_dir, 'validation_rescueRobot_450.csv'))
    df["firm"] = (df["firm"] == "Yes") * 1
    dfv["firm"] = (dfv["firm"] == "Yes") * 1

    LIMIT = 450
    X_train = df.drop(["hazard"], axis=1)[:LIMIT]
    y_train = df["hazard"][:LIMIT]
    X_val = dfv.drop(["hazard"], axis=1)
    y_val = dfv["hazard"]

    print(f"Training set: {len(X_train)}, Validation set: {len(X_val)}")

    results = []

    # ---- Stacking ----
    print("\nTraining Stacking model...")
    start = time.time()
    stacking = StackingEnsemble(n_folds=10, random_state=42)
    stacking.fit(X_train, y_train)
    stacking_time = time.time() - start

    pred_stacking = stacking.predict_proba(X_val)
    stack_prec = precision(pred_stacking, y_val)
    stack_rec = recall(pred_stacking, y_val)
    stack_f1 = f_measure(pred_stacking, y_val)
    print(f"Stacking: precision={stack_prec:.4f}, recall={stack_rec:.4f}, F1={stack_f1:.4f}, time={stacking_time:.2f}s")

    results.append({
        'type': 'Stacking', 'model': 'Stacking',
        'precision': stack_prec, 'recall': stack_rec, 'F1': stack_f1,
        'train_time': stacking_time
    })

    # ---- BMA (optional) ----
    if args.run_bma:
        print("\nTraining BMA model...")
        start = time.time()
        bma = BMA(y_train, add_constant(X_train), RegType='Logit', Verbose=False).fit()
        bma_time = time.time() - start
        pred_bma = bma.predict(add_constant(X_val))
        bma_prec = precision(pred_bma, y_val)
        bma_rec = recall(pred_bma, y_val)
        bma_f1 = f_measure(pred_bma, y_val)
        print(f"BMA: precision={bma_prec:.4f}, recall={bma_rec:.4f}, F1={bma_f1:.4f}, time={bma_time:.2f}s")

        results.append({
            'type': 'BMA', 'model': 'BMA',
            'precision': bma_prec, 'recall': bma_rec, 'F1': bma_f1,
            'train_time': bma_time
        })

    # ---- Save results ----
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    output_file = os.path.join(log_dir, 'stacking_rq1_results.tsv')
    results_df = pd.DataFrame(results)
    results_df[['type', 'model', 'precision', 'recall', 'F1']].to_csv(
        output_file, sep='\t', index=False
    )
    print(f"\nResults saved: {output_file}")

    # ---- Summary ----
    print(f"\n{'Model':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Time':>10}")
    print('-' * 55)
    for r in results:
        print(f"{r['model']:<12} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['F1']:>10.4f} {r['train_time']:>9.2f}s")


if __name__ == '__main__':
    main()
