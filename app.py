import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import xgboost as xgb
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline

# ======================================================================
# Load saved models and data
# ======================================================================
st.set_page_config(page_title="In-Play Football Prediction", layout="wide")
st.title("Live In-Play Prediction & SHAP Explanation")

# Load models
try:
    best_inplay_clf = joblib.load('best_inplay_clf.pkl')
    best_inplay_reg = joblib.load('best_inplay_reg.pkl')
except FileNotFoundError:
    st.error("Models not found. Please run main.py first to save them.")
    st.stop()

# Load test data
test_snap = pd.read_csv('test_snapshots_for_app.csv')
test_pre = pd.read_csv('test_prematch_for_app.csv')
y_snap_test_cls = np.load('y_snap_test_cls.npy')
y_snap_test_reg = np.load('y_snap_test_reg.npy')

# Load feature names (you can also save them in a pickle)
# For simplicity, we define them here (they should match main)
snap_feat_cols = [c for c in test_snap.columns if c not in 
                  ['match_id', 'snapshot_time', 'final_goal_diff', 'final_result']]
pre_feat_cols = [c for c in test_pre.columns if c not in 
                 ['match_id', 'match_date', 'home_team', 'away_team', 
                  'label_goal_diff', 'label_result']]

# ======================================================================
# Helper to extract tree model from pipeline/IFX wrapper
# ======================================================================
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

# ======================================================================
# SHAP explainers (using extracted models)
# ======================================================================
shap_clf = extract_tree_model(best_inplay_clf)
shap_reg = extract_tree_model(best_inplay_reg)

explainer_clf = shap.TreeExplainer(shap_clf)
explainer_reg = shap.TreeExplainer(shap_reg)

# ======================================================================
# App UI
# ======================================================================
match_ids = test_snap['match_id'].unique()
selected_match = st.selectbox("Select a match ID", match_ids)

# Filter snapshots for this match
match_data = test_snap[test_snap['match_id'] == selected_match].sort_values('snapshot_time')
if match_data.empty:
    st.error("No snapshots found for this match.")
    st.stop()

times = match_data['snapshot_time'].values
snapshot_time = st.slider("Snapshot time (minute)", 
                          min_value=int(times[0]), 
                          max_value=int(times[-1]), 
                          step=5)

# Find the closest snapshot
closest_idx = np.argmin(np.abs(times - snapshot_time))
snap_row = match_data.iloc[closest_idx]
X_snap = snap_row[snap_feat_cols].values.reshape(1, -1)

# Predictions
clf_probs = best_inplay_clf.predict_proba(X_snap)[0]
reg_pred = best_inplay_reg.predict(X_snap)[0]

# Display
st.subheader(f"Match {selected_match} at minute {times[closest_idx]}")
col1, col2 = st.columns(2)
with col1:
    st.metric("Home Score", snap_row['current_home_score'])
    st.metric("Away Score", snap_row['current_away_score'])
    st.metric("Red Card Diff", snap_row['red_card_diff'])
with col2:
    st.metric("Predicted Home Win", f"{clf_probs[0]:.2%}")
    st.metric("Predicted Draw", f"{clf_probs[1]:.2%}")
    st.metric("Predicted Away Win", f"{clf_probs[2]:.2%}")
    st.metric("Expected Final Margin", f"{reg_pred:.2f}")

# SHAP explanation for classification
pred_class = np.argmax(clf_probs)
shap_values_clf = explainer_clf.shap_values(X_snap)

# Extract SHAP for predicted class
if isinstance(shap_values_clf, list):
    class_shap = shap_values_clf[pred_class][0]
    base = explainer_clf.expected_value[pred_class]
elif hasattr(shap_values_clf, 'ndim') and shap_values_clf.ndim == 3:
    class_shap = shap_values_clf[0, :, pred_class]
    base = explainer_clf.expected_value[pred_class]
else:
    class_shap = shap_values_clf[0]
    base = explainer_clf.expected_value

explanation = shap.Explanation(
    values=class_shap,
    base_values=base,
    data=X_snap[0],
    feature_names=snap_feat_cols
)

# Generate waterfall plot
fig, ax = plt.subplots()
shap.plots.waterfall(explanation, show=False)
# Get the current figure and display
fig.set_size_inches(6, 4)
st.pyplot(plt.gcf())