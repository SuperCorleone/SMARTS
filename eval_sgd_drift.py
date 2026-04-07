"""
eval_sgd_drift.py
Evaluate Online Stacking (SGD+Drift) vs Static Stacking on validation set.
Prequential evaluation: test-then-train using actual labels.
"""

import copy
import json
import numpy as np
import pandas as pd
from statsmodels.tools import add_constant
from stacking import StackingEnsemble

# ---------- 1. Load data ----------
train_df = pd.read_csv("data/training_rescueRobot_450.csv")
val_df = pd.read_csv("data/validation_rescueRobot_450.csv")

# ---------- 2. Convert "firm" column ----------
train_df["firm"] = (train_df["firm"] == "Yes") * 1
val_df["firm"] = (val_df["firm"] == "Yes") * 1

# ---------- 3. Separate features / target ----------
target_col = "hazard"
feature_cols = [c for c in train_df.columns if c != target_col]

X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_val = val_df[feature_cols]
y_val = val_df[target_col]

# ---------- 4. Train StackingEnsemble ----------
print("Training StackingEnsemble...")
stacking = StackingEnsemble(n_folds=10, random_state=42)
stacking.fit(X_train, y_train)

# ---------- 5. Create static and online copies ----------
static_stacking = stacking  # static: no updates
online_stacking = copy.deepcopy(stacking)  # online: will be updated

# ---------- 6. Prequential evaluation ----------
print(f"\nRunning prequential evaluation on {len(y_val)} validation samples...")
results = []

for r in range(len(y_val)):
    row = add_constant(X_val, has_constant='add').iloc[r:r+1]

    # Predict
    static_pred = static_stacking.predict_single(row)
    online_pred = online_stacking.predict_single(row)

    # Actual label
    y_true = int(y_val.iloc[r])

    # Absolute error
    static_error = abs(static_pred - y_true)
    online_error = abs(online_pred - y_true)

    # Success: prediction > 0.5 matches y_true
    static_correct = int((static_pred > 0.5) == (y_true > 0.5))
    online_correct = int((online_pred > 0.5) == (y_true > 0.5))

    results.append({
        "row": r,
        "static_pred": float(static_pred),
        "online_pred": float(online_pred),
        "y_true": y_true,
        "static_error": float(static_error),
        "online_error": float(online_error),
        "static_correct": static_correct,
        "online_correct": online_correct,
    })

    # Online update (test-then-train)
    X_raw_row = X_val.iloc[r:r+1]
    online_stacking.online_update(X_raw_row, y_true, window_size=100)

    if (r + 1) % 50 == 0:
        print(f"  Processed {r + 1}/{len(y_val)} samples")

# ---------- 7. Compute summary ----------
static_errors = [x["static_error"] for x in results]
online_errors = [x["online_error"] for x in results]
static_correct_list = [x["static_correct"] for x in results]
online_correct_list = [x["online_correct"] for x in results]

summary = {
    "static": {
        "mean_error": float(np.mean(static_errors)),
        "median_error": float(np.median(static_errors)),
        "accuracy": float(np.mean(static_correct_list)),
    },
    "online_sgd_drift": {
        "mean_error": float(np.mean(online_errors)),
        "median_error": float(np.median(online_errors)),
        "accuracy": float(np.mean(online_correct_list)),
        "drift_count": online_stacking._drift_count,
    },
}

# ---------- 8. Save results ----------
output = {"per_sample": results, "summary": summary}
with open("logs/eval_sgd_drift_results.json", "w") as f:
    json.dump(output, f, indent=2)

print("\n=== SUMMARY ===")
print(f"Static Stacking:")
print(f"  Mean Error:   {summary['static']['mean_error']:.4f}")
print(f"  Median Error: {summary['static']['median_error']:.4f}")
print(f"  Accuracy:     {summary['static']['accuracy']:.4f}")
print(f"\nOnline Stacking (SGD+Drift):")
print(f"  Mean Error:   {summary['online_sgd_drift']['mean_error']:.4f}")
print(f"  Median Error: {summary['online_sgd_drift']['median_error']:.4f}")
print(f"  Accuracy:     {summary['online_sgd_drift']['accuracy']:.4f}")
print(f"  Drift Count:  {summary['online_sgd_drift']['drift_count']}")
print(f"\nResults saved to logs/eval_sgd_drift_results.json")
