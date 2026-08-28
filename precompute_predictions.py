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
import os

# -------------------- CONFIG --------------------
OUTPUT_DIR = "out"
MODEL_CLF_PATH = os.path.join(OUTPUT_DIR, 'best_inplay_clf.pkl')
MODEL_REG_PATH = os.path.join(OUTPUT_DIR, 'best_inplay_reg.pkl')
SNAPSHOT_TEST_PATH = os.path.join(OUTPUT_DIR, 'test_snapshots_for_app.csv')
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'precomputed_predictions.parquet')

# -------------------- LOAD SNAPSHOTS --------------------
df_snap = pd.read_csv(SNAPSHOT_TEST_PATH)

# -------------------- FEATURE SELECTION (exact 8 in-play features) --------------------
# These are the only features the in-play model expects.
FEATURE_COLS = [
    'time_norm',
    'current_home_score',
    'current_away_score',
    'red_card_diff',
    'shots_recent_5min',
    'passes_recent_5min',
    'pressures_recent_5min',
    'momentum'
]

# Ensure all columns exist; otherwise raise a clear error
missing = [col for col in FEATURE_COLS if col not in df_snap.columns]
if missing:
    raise ValueError(f"Missing required columns in test_snapshots_for_app.csv: {missing}")

X = df_snap[FEATURE_COLS].values

# -------------------- LOAD MODELS --------------------
print("Loading models...")
clf_model = joblib.load(MODEL_CLF_PATH)
reg_model = joblib.load(MODEL_REG_PATH)

# Extract tree models for SHAP
def extract_tree_model(model):
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

shap_clf = extract_tree_model(clf_model)
explainer_clf = shap.TreeExplainer(shap_clf)

# -------------------- PREDICT --------------------
print("Computing predictions...")
probs = clf_model.predict_proba(X)
margins = reg_model.predict(X)
pred_classes = np.argmax(probs, axis=1)

# -------------------- SHAP VALUES --------------------
print("Computing SHAP values (this may take a few minutes)...")
shap_values = explainer_clf.shap_values(X)

shap_class_vals = []
for i, cls in enumerate(pred_classes):
    if isinstance(shap_values, list):
        shap_class_vals.append(shap_values[cls][i])
    elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
        shap_class_vals.append(shap_values[i, :, cls])
    else:
        shap_class_vals.append(shap_values[i])
shap_class_vals = np.array(shap_class_vals)

# Top 5 features
top_k = 5
top_shap_features = []
top_shap_values = []
for i in range(len(shap_class_vals)):
    abs_shap = np.abs(shap_class_vals[i])
    top_idx = np.argsort(abs_shap)[-top_k:]
    top_shap_features.append([FEATURE_COLS[idx] for idx in top_idx])
    top_shap_values.append([shap_class_vals[i, idx] for idx in top_idx])

# -------------------- SAVE --------------------
df_out = df_snap[['match_id', 'snapshot_time']].copy()
df_out['prob_H'] = probs[:, 0]
df_out['prob_D'] = probs[:, 1]
df_out['prob_A'] = probs[:, 2]
df_out['expected_margin'] = margins
df_out['pred_class'] = pred_classes
df_out['top_shap_features'] = [list(x) for x in top_shap_features]
df_out['top_shap_values'] = [list(x) for x in top_shap_values]

df_out.to_parquet(OUTPUT_PATH, index=False)
print(f"Done! Saved to {OUTPUT_PATH}")