"""
precompute_predictions.py
-------------------------
Loads the best in-play models and precomputes:
- Probabilities (classification)
- Expected margin (regression)
- SHAP values for the predicted class (top 5 features)
for all snapshots in the test set.
Saves to a Parquet file for fast API responses.
"""

import pandas as pd
import numpy as np
import joblib
import shap
import xgboost as xgb
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
import os

# -------------------- CONFIG --------------------
MODEL_CLF_PATH = 'best_inplay_clf.pkl'
MODEL_REG_PATH = 'best_inplay_reg.pkl'
SNAPSHOT_TEST_PATH = 'test_snapshots_for_app.csv'
OUTPUT_PATH = 'precomputed_predictions.parquet'
FEATURE_COLS = [c for c in pd.read_csv(SNAPSHOT_TEST_PATH).columns 
                if c not in ['match_id', 'snapshot_time', 'final_goal_diff', 'final_result']]

# -------------------- HELPER --------------------
def extract_tree_model(model):
    """Extract the underlying XGBoost booster from pipelines or IFX wrapper."""
    if hasattr(model, 'model') and isinstance(model.model, xgb.Booster):
        return model.model
    if hasattr(model, 'steps'):
        last_est = model.steps[-1][1]
        if hasattr(last_est, 'model') and isinstance(last_est.model, xgb.Booster):
            return last_est.model
        return last_est
    if hasattr(model, 'named_steps'):
        last_est = model.named_steps[list(model.named_steps.keys())[-1]]
        if hasattr(last_est, 'model') and isinstance(last_est.model, xgb.Booster):
            return last_est.model
        return last_est
    return model

# -------------------- LOAD MODELS --------------------
print("Loading models...")
clf_model = joblib.load(MODEL_CLF_PATH)
reg_model = joblib.load(MODEL_REG_PATH)

# Extract tree models for SHAP
shap_clf = extract_tree_model(clf_model)
shap_reg = extract_tree_model(reg_model)
explainer_clf = shap.TreeExplainer(shap_clf)

# -------------------- LOAD TEST SNAPSHOTS --------------------
print("Loading test snapshots...")
df_snap = pd.read_csv(SNAPSHOT_TEST_PATH)
df_snap = df_snap.sort_values(['match_id', 'snapshot_time'])
X = df_snap[FEATURE_COLS].values

# -------------------- PREDICT --------------------
print("Computing predictions...")
probs = clf_model.predict_proba(X)
margins = reg_model.predict(X)
pred_classes = np.argmax(probs, axis=1)

# -------------------- SHAP VALUES --------------------
print("Computing SHAP values for predicted class (this may take a few minutes)...")
shap_values = explainer_clf.shap_values(X)

# For each sample, extract SHAP for the predicted class
shap_class_vals = []
for i, cls in enumerate(pred_classes):
    if isinstance(shap_values, list):
        shap_class_vals.append(shap_values[cls][i])
    elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
        shap_class_vals.append(shap_values[i, :, cls])
    else:
        shap_class_vals.append(shap_values[i])
shap_class_vals = np.array(shap_class_vals)  # (n_samples, n_features)

# Get top 5 features by absolute SHAP value for each sample
top_k = 5
top_shap_features = []
top_shap_values = []
for i in range(len(shap_class_vals)):
    abs_shap = np.abs(shap_class_vals[i])
    top_idx = np.argsort(abs_shap)[-top_k:]
    top_shap_features.append([FEATURE_COLS[idx] for idx in top_idx])
    top_shap_values.append([shap_class_vals[i, idx] for idx in top_idx])

# -------------------- SAVE --------------------
print("Saving precomputed data...")
df_out = df_snap[['match_id', 'snapshot_time']].copy()
df_out['prob_H'] = probs[:, 0]
df_out['prob_D'] = probs[:, 1]
df_out['prob_A'] = probs[:, 2]
df_out['expected_margin'] = margins
df_out['pred_class'] = pred_classes

# Store top SHAP features and values as JSON strings (lists)
df_out['top_shap_features'] = [list(x) for x in top_shap_features]
df_out['top_shap_values'] = [list(x) for x in top_shap_values]

df_out.to_parquet(OUTPUT_PATH, index=False)
print(f"Done! Saved to {OUTPUT_PATH}")