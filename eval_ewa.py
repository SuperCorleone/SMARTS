"""
eval_ewa.py
Evaluate EWA (Exponentially Weighted Aggregation) on the validation set
using ACTUAL labels (prequential: test-then-train, no BMA oracle, no GA).
"""

import json
import os
import numpy as np
import pandas as pd
from statsmodels.tools import add_constant
from stacking import StackingEnsemble
from ewa_ensemble import EWAEnsemble

# ── 1. Load data ──────────────────────────────────────────────────────
train_df = pd.read_csv("data/training_rescueRobot_450.csv")
val_df = pd.read_csv("data/validation_rescueRobot_450.csv")

# ── 2. Convert "firm" column ──────────────────────────────────────────
train_df["firm"] = (train_df["firm"] == "Yes") * 1
val_df["firm"] = (val_df["firm"] == "Yes") * 1

y_col = "hazard"
feature_cols = [c for c in train_df.columns if c != y_col]

X_train = train_df[feature_cols]
y_train = train_df[y_col]
X_val = val_df[feature_cols]
y_val = val_df[y_col]

# ── 3. Train StackingEnsemble ─────────────────────────────────────────
print("Training StackingEnsemble...")
stacking = StackingEnsemble(n_folds=10, random_state=42).fit(X_train, y_train)

# ── 4. Create EWA ────────────────────────────────────────────────────
ewa = EWAEnsemble(stacking, eta=0.5, use_log_odds=True)

# ── 5. Prequential evaluation on validation set ─────────────────────
print("Running prequential EWA evaluation on validation set...")
X_val_const = add_constant(X_val, has_constant="add")

results = []
for r in range(len(X_val)):
    row = X_val_const.iloc[r : r + 1]

    # a. predict
    ewa_pred = ewa.predict_single(row)

    # b. actual label
    y_true = float(y_val.iloc[r])

    # c. error & accuracy
    ewa_error = abs(ewa_pred - y_true)
    ewa_correct = int((ewa_pred > 0.5) == y_true)

    results.append(
        {
            "row": r,
            "ewa_pred": round(ewa_pred, 6),
            "y_true": int(y_true),
            "ewa_error": round(ewa_error, 6),
            "ewa_correct": ewa_correct,
        }
    )

    # d. online update with actual label
    ewa.online_update(row, y_true)

# ── 6. Summary & save ───────────────────────────────────────────────
errors = [r["ewa_error"] for r in results]
corrects = [r["ewa_correct"] for r in results]

summary = {
    "ewa": {
        "mean_error": round(float(np.mean(errors)), 6),
        "median_error": round(float(np.median(errors)), 6),
        "accuracy": round(float(np.mean(corrects)), 6),
    }
}

print("\n=== EWA Evaluation Summary ===")
print(f"  Mean absolute error : {summary['ewa']['mean_error']}")
print(f"  Median absolute error: {summary['ewa']['median_error']}")
print(f"  Accuracy             : {summary['ewa']['accuracy']}")

os.makedirs("logs", exist_ok=True)
output = {"per_sample": results, "summary": summary}
with open("logs/eval_ewa_results.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nResults saved to logs/eval_ewa_results.json")
