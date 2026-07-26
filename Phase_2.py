import os
import time
import gc
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.svm import SVC, SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, log_loss, brier_score_loss,
                             mean_absolute_error, mean_squared_error, confusion_matrix,
                             classification_report, roc_auc_score)
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


# Imbalanced-learn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# SHAP
import shap

# For memory measurement
import psutil

# ================ PHASE 1 IMPORTS (PF-SMOTE and IFX) ================
from PF_SMOTE import PF_SMOTE
from IFX_model import IFX_XGBoost

# ================ PHASE 2 MAIN WORKFLOW ================

# -------------------- 1. Load Data --------------------
DATA_DIR = "processed_data"  # adjust path if needed

def load_data():
    """Load pre-match and snapshot datasets, and match info."""
    train_pre = pd.read_parquet(os.path.join(DATA_DIR, 'train_prematch.parquet'))
    val_pre = pd.read_parquet(os.path.join(DATA_DIR, 'val_prematch.parquet'))
    test_pre = pd.read_parquet(os.path.join(DATA_DIR, 'test_prematch.parquet'))
    train_snap = pd.read_parquet(os.path.join(DATA_DIR, 'train_snapshots.parquet'))
    val_snap = pd.read_parquet(os.path.join(DATA_DIR, 'val_snapshots.parquet'))
    test_snap = pd.read_parquet(os.path.join(DATA_DIR, 'test_snapshots.parquet'))
    # matches_full for potential extra info (not needed for features)
    return (train_pre, val_pre, test_pre), (train_snap, val_snap, test_snap)

(train_pre, val_pre, test_pre), (train_snap, val_snap, test_snap) = load_data()
print("Data loaded. Pre-match train shape:", train_pre.shape)
print("Snapshots train shape:", train_snap.shape)
# PART 1 Done

# -------------------- 2. Define Features & Targets --------------------
# Pre-match classification features (exclude match_id, date, team names, label columns)
pre_feat_cols = [c for c in train_pre.columns if c not in 
                 ['match_id', 'match_date', 'home_team', 'away_team', 
                  'label_goal_diff', 'label_result']]
X_pre_train = train_pre[pre_feat_cols].values
y_pre_train_cls = train_pre['label_result'].map({'H':0, 'D':1, 'A':2}).values
y_pre_train_reg = train_pre['label_goal_diff'].values

X_pre_val = val_pre[pre_feat_cols].values
y_pre_val_cls = val_pre['label_result'].map({'H':0, 'D':1, 'A':2}).values
y_pre_val_reg = val_pre['label_goal_diff'].values

X_pre_test = test_pre[pre_feat_cols].values
y_pre_test_cls = test_pre['label_result'].map({'H':0, 'D':1, 'A':2}).values
y_pre_test_reg = test_pre['label_goal_diff'].values

# In-play snapshots features
snap_feat_cols = [c for c in train_snap.columns if c not in 
                  ['match_id', 'snapshot_time', 'final_goal_diff', 'final_result']]
X_snap_train = train_snap[snap_feat_cols].values
y_snap_train_cls = train_snap['final_result'].map({'H':0, 'D':1, 'A':2}).values
y_snap_train_reg = train_snap['final_goal_diff'].values

X_snap_val = val_snap[snap_feat_cols].values
y_snap_val_cls = val_snap['final_result'].map({'H':0, 'D':1, 'A':2}).values
y_snap_val_reg = val_snap['final_goal_diff'].values

X_snap_test = test_snap[snap_feat_cols].values
y_snap_test_cls = test_snap['final_result'].map({'H':0, 'D':1, 'A':2}).values
y_snap_test_reg = test_snap['final_goal_diff'].values

# Store snapshot times for evaluation
snap_times_train = train_snap['snapshot_time'].values
snap_times_test = test_snap['snapshot_time'].values

# PART 2 Done

# -------------------- 3. PF-SMOTE Integration (P1) --------------------
# We'll use PF-SMOTE as a resampler in the classification pipeline.
# For pre-match classification (Model 1) and in-play classification (Model 3).
# We'll wrap it in an imblearn Pipeline to ensure it only fits on training folds.

from imblearn.pipeline import Pipeline as ImbPipeline

def create_clf_pipeline(model, use_pf_smote=True, scaler=True):
    steps = []
    if scaler:
        steps.append(('scaler', StandardScaler()))
    if use_pf_smote:
        steps.append(('resampler', PF_SMOTE(random_state=42)))
    steps.append(('clf', model))
    return ImbPipeline(steps)

# Example usage:
# pipe = create_clf_pipeline(XGBClassifier(), use_pf_smote=True)
# pipe.fit(X_pre_train, y_pre_train_cls)

# PART 3 Done (integration logic defined)

# -------------------- 4. Model Suite & Hyperparameter Tuning --------------------
# We define a dictionary of model names and their parameter grids for tuning.
# We'll use GridSearchCV (or RandomizedSearchCV) with 3-fold CV on the training set,
# using the validation set as early stopping where applicable.

# Classification models
clf_models = {
    'Dummy': (DummyClassifier(strategy='most_frequent'), {}),
    'KernelSVM': (SVC(probability=True, random_state=42), 
                  {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}),
    'RandomForest': (RandomForestClassifier(random_state=42),
                     {'n_estimators': [100, 200], 'max_depth': [None, 5, 10]}),
    'GBM': (GradientBoostingClassifier(random_state=42),
            {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1, 0.2]}),
    'XGBoost': (XGBClassifier(random_state=42, eval_metric='mlogloss'),
                {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1, 0.2],
                 'max_depth': [3, 5, 7]}),
    'LightGBM': (LGBMClassifier(random_state=42, verbose=-1),
                 {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1, 0.2],
                  'num_leaves': [31, 63]}),
    'IFX-XGBoost': (IFX_XGBoost(random_state=42, n_iterations=3),
                    {'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]})
}

# Regression models (no resampling, no calibration)
reg_models = {
    'Dummy': (DummyRegressor(strategy='mean'), {}),
    'KernelRidge': (KernelRidge(),
                    {'alpha': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}),
    'KernelSVR': (SVR(),
                  {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}),
    'RandomForest': (RandomForestRegressor(random_state=42),
                     {'n_estimators': [100, 200], 'max_depth': [None, 5, 10]}),
    'GBM': (GradientBoostingRegressor(random_state=42),
            {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1, 0.2]}),
    'XGBoost': (XGBRegressor(random_state=42),
                {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1, 0.2],
                 'max_depth': [3, 5, 7]}),
    'LightGBM': (LGBMRegressor(random_state=42, verbose=-1),
                 {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1, 0.2],
                  'num_leaves': [31, 63]}),
    'IFX-XGBoost': (IFX_XGBoost(random_state=42, n_iterations=3, objective='reg:squarederror'),
                    {'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]})
}

# We will tune using GridSearchCV with 3-fold CV on the training set.
def tune_model(model, param_grid, X_train, y_train, X_val=None, scoring='neg_log_loss'):
    """Return best estimator after GridSearchCV."""
    if model.__class__.__name__ == 'IFX_XGBoost':
        # IFX has its own validation set; we won't use GridSearchCV.
        # Instead we will train with different params manually.
        # We'll pick a default set.
        return model
    # For other models, use GridSearchCV.
    # We'll also include early stopping for XGBoost and LightGBM via callbacks?
    # But we'll keep simple.
    from sklearn.model_selection import GridSearchCV
    if X_val is not None and hasattr(model, 'eval_set'):
        # Some models can use eval_set for early stopping, but GridSearchCV doesn't support it directly.
        # We'll just use CV.
        pass
    gs = GridSearchCV(model, param_grid, cv=3, scoring=scoring, n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    return gs.best_estimator_

# We'll train all models on pre-match classification and regression,
# and on in-play classification and regression. This will be a large loop.
# We'll store results in a dictionary.

# PART 4 Done (model definitions and tuning function)

# -------------------- 5. Calibration & Metrics --------------------
# We'll use Platt scaling (CalibratedClassifierCV with method='sigmoid')
# on all probabilistic classifiers. For classifiers that output probabilities,
# we can apply calibration on the validation set.
# We'll also compute ECE (Expected Calibration Error) and reliability diagrams.

from sklearn.calibration import calibration_curve

def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error."""
    bin_counts, bin_edges = np.histogram(y_prob, bins=n_bins, range=(0,1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_true = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i+1])
        if np.sum(mask) > 0:
            bin_true[i] = np.mean(y_true[mask])
    bin_accuracy = bin_true
    bin_confidence = bin_centers
    ece = np.sum(bin_counts * np.abs(bin_accuracy - bin_confidence)) / np.sum(bin_counts)
    return ece

def evaluate_classifier(model, X, y_true, model_name, task, calibrate=True, X_cal=None, y_cal=None):
    """Return metrics: log_loss, brier, accuracy, ECE, RPS (Ranked Probability Score)."""
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X)
    else:
        y_prob = None
    if y_prob is not None:
        # Calibrate if requested and calibration data provided
        if calibrate and X_cal is not None and y_cal is not None:
            cal_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
            cal_model.fit(X_cal, y_cal)
            y_prob_cal = cal_model.predict_proba(X)
        else:
            y_prob_cal = y_prob
        # Log-loss
        ll = log_loss(y_true, y_prob_cal)
        brier = brier_score_loss(y_true, y_prob_cal[:, 1] if y_prob_cal.shape[1]==2 else 
                                 # for multi-class, we need per-class brier; we'll compute average.
                                 np.mean([brier_score_loss(y_true==i, y_prob_cal[:, i]) for i in range(y_prob_cal.shape[1])])
                                 )
        # ECE (for the predicted class probability? Typically we compute per class; we'll average)
        ece = np.mean([compute_ece((y_true==i).astype(int), y_prob_cal[:, i]) for i in range(y_prob_cal.shape[1])])
        # RPS - Ranked Probability Score for 3 classes
        # For each sample, RPS = sum_{k=1}^{K-1} (CDF_pred - CDF_true)^2
        n_classes = y_prob_cal.shape[1]
        rps = []
        for i in range(len(y_true)):
            true_label = y_true[i]
            cum_pred = np.cumsum(y_prob_cal[i, :])
            cum_true = np.zeros(n_classes)
            cum_true[true_label:] = 1
            rps.append(np.sum((cum_pred[:-1] - cum_true[:-1])**2))
        rps = np.mean(rps)
        # Accuracy
        acc = accuracy_score(y_true, np.argmax(y_prob_cal, axis=1))
        return {'log_loss': ll, 'brier': brier, 'ece': ece, 'rps': rps, 'accuracy': acc}
    else:
        # For models without probabilities (e.g., some classifiers)
        return {'log_loss': np.nan, 'brier': np.nan, 'ece': np.nan, 'rps': np.nan, 
                'accuracy': accuracy_score(y_true, model.predict(X))}

def evaluate_regressor(model, X, y_true):
    y_pred = model.predict(X)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    corr = np.corrcoef(y_true, y_pred)[0,1]
    return {'mae': mae, 'rmse': rmse, 'corr': corr}

# PART 5 Done (calibration and metrics functions)

# -------------------- 6. Run All Models (Pre-match and In-play) --------------------
# We'll loop over the model dictionaries, fit, and evaluate.
# We'll store results in a DataFrame for reporting.

results_cls = []
results_reg = []

# Pre-match classification
for name, (model, param_grid) in clf_models.items():
    print(f"Training {name} (pre-match classification)...")
    if name == 'IFX-XGBoost':
        # IFX needs validation set for early stopping and SHAP
        # We'll set X_val = X_pre_val, y_val = y_pre_val_cls
        # But we still need to tune? We'll just use default params.
        model = IFX_XGBoost(random_state=42, n_iterations=3,
                            objective='multi:softprob', num_class=3)
        model.fit(X_pre_train, y_pre_train_cls, X_pre_val, y_pre_val_cls)
        best_model = model
    else:
        # For other models, we build a pipeline with PF-SMOTE and scaling.
        # We'll use GridSearchCV to find best parameters.
        from sklearn.model_selection import GridSearchCV
        pipe = create_clf_pipeline(model, use_pf_smote=True, scaler=True)
        # Since GridSearchCV doesn't handle imblearn Pipeline nicely, we'll use sklearn's Pipeline with PF_SMOTE as a custom transformer.
        # Actually we can use ImbPipeline which is compatible.
        # We'll use a simpler approach: define the pipeline with steps and use GridSearchCV.
        param_grid_adj = {}
        for k, v in param_grid.items():
            param_grid_adj['clf__' + k] = v
        gs = GridSearchCV(pipe, param_grid_adj, cv=3, scoring='neg_log_loss', n_jobs=-1, verbose=0)
        gs.fit(X_pre_train, y_pre_train_cls)
        best_model = gs.best_estimator_
    # Evaluate on test
    res = evaluate_classifier(best_model, X_pre_test, y_pre_test_cls, 
                              model_name=name, task='pre_cls', 
                              calibrate=True, X_cal=X_pre_val, y_cal=y_pre_val_cls)
    res['model'] = name
    results_cls.append(res)

# Pre-match regression
for name, (model, param_grid) in reg_models.items():
    print(f"Training {name} (pre-match regression)...")
    if name == 'IFX-XGBoost':
        model = IFX_XGBoost(random_state=42, n_iterations=3, objective='reg:squarederror')
        model.fit(X_pre_train, y_pre_train_reg, X_pre_val, y_pre_val_reg)
        best_model = model
    else:
        # For regression, we don't resample. We'll use a pipeline with scaling.
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([('scaler', StandardScaler()), ('reg', model)])
        # GridSearchCV
        from sklearn.model_selection import GridSearchCV
        param_grid_adj = {}
        for k, v in param_grid.items():
            param_grid_adj['reg__' + k] = v
        gs = GridSearchCV(pipe, param_grid_adj, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=0)
        gs.fit(X_pre_train, y_pre_train_reg)
        best_model = gs.best_estimator_
    res = evaluate_regressor(best_model, X_pre_test, y_pre_test_reg)
    res['model'] = name
    results_reg.append(res)

# In-play classification (model 3)
# We'll train on snapshot data, but we must respect match-level splitting.
# Already done by using separate train/val/test snapshots.
for name, (model, param_grid) in clf_models.items():
    print(f"Training {name} (in-play classification)...")
    if name == 'IFX-XGBoost':
        model = IFX_XGBoost(random_state=42, n_iterations=3,
                            objective='multi:softprob', num_class=3)
        model.fit(X_snap_train, y_snap_train_cls, X_snap_val, y_snap_val_cls)
        best_model = model
    else:
        pipe = create_clf_pipeline(model, use_pf_smote=True, scaler=True)
        param_grid_adj = {}
        for k, v in param_grid.items():
            param_grid_adj['clf__' + k] = v
        gs = GridSearchCV(pipe, param_grid_adj, cv=3, scoring='neg_log_loss', n_jobs=-1, verbose=0)
        gs.fit(X_snap_train, y_snap_train_cls)
        best_model = gs.best_estimator_
    res = evaluate_classifier(best_model, X_snap_test, y_snap_test_cls,
                              model_name=name, task='inplay_cls',
                              calibrate=True, X_cal=X_snap_val, y_cal=y_snap_val_cls)
    res['model'] = name
    results_cls.append(res)

# In-play regression
for name, (model, param_grid) in reg_models.items():
    print(f"Training {name} (in-play regression)...")
    if name == 'IFX-XGBoost':
        model = IFX_XGBoost(random_state=42, n_iterations=3, objective='reg:squarederror')
        model.fit(X_snap_train, y_snap_train_reg, X_snap_val, y_snap_val_reg)
        best_model = model
    else:
        pipe = Pipeline([('scaler', StandardScaler()), ('reg', model)])
        param_grid_adj = {}
        for k, v in param_grid.items():
            param_grid_adj['reg__' + k] = v
        gs = GridSearchCV(pipe, param_grid_adj, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=0)
        gs.fit(X_snap_train, y_snap_train_reg)
        best_model = gs.best_estimator_
    res = evaluate_regressor(best_model, X_snap_test, y_snap_test_reg)
    res['model'] = name
    results_reg.append(res)

# Convert results to DataFrames
df_cls_results = pd.DataFrame(results_cls)
df_reg_results = pd.DataFrame(results_reg)
print("Classification results:")
print(df_cls_results)
print("Regression results:")
print(df_reg_results)

# PART 6 Done (all models trained and evaluated)