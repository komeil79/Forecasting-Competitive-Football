import os
import time
import gc
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.linear_model import Ridge, SGDClassifier, SGDRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
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


# Helper function for approximating kernel models
def make_approx_kernel_pipeline(task_type='classification'):
    """Return a pipeline with Nystroem approximation + linear model."""
    if task_type == 'classification':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('kernel_approx', Nystroem(kernel='rbf', n_components=100, random_state=42)),
            ('clf', SGDClassifier(loss='log_loss', random_state=42, max_iter=1000, tol=1e-3))
        ])
    else:  # regression
        return Pipeline([
            ('scaler', StandardScaler()),
            ('kernel_approx', Nystroem(kernel='rbf', n_components=100, random_state=42)),
            ('reg', SGDRegressor(random_state=42, max_iter=1000, tol=1e-3))
        ])
    
# ================ PHASE 1 IMPORTS (PF-SMOTE and IFX) ================
from PF_SMOTE import PF_SMOTE
from IFX_model import IFX_XGBoost

# ================ PHASE 2 MAIN WORKFLOW ================

# -------------------- 1. Load Data --------------------
DATA_DIR = "processed_data"  # adjust path if needed
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, 'Forecasting-Competitive-Football\\figures')

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

from sklearn.calibration import CalibratedClassifierCV

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

def evaluate_classifier(model, X_train, y_train, X_test, y_test, model_name, task, calibrate=True, X_cal=None, y_cal=None):
    """
    Evaluate classifier on test set.
    Returns both uncalibrated and calibrated metrics (if calibrate=True).
    """
    # Get raw (uncalibrated) probabilities on test set
    probs_uncal = model.predict_proba(X_test)
    n_classes = probs_uncal.shape[1]

    # ----- Uncalibrated metrics -----
    ll_uncal = log_loss(y_test, probs_uncal)
    brier_uncal = np.mean([brier_score_loss((y_test == i).astype(int), probs_uncal[:, i]) for i in range(n_classes)])
    ece_uncal = np.mean([compute_ece((y_test == i).astype(int), probs_uncal[:, i]) for i in range(n_classes)])
    # RPS (uncalibrated)
    rps_list = []
    for i in range(len(y_test)):
        true_label = y_test[i]
        cum_pred = np.cumsum(probs_uncal[i, :])
        cum_true = np.zeros(n_classes)
        cum_true[true_label:] = 1
        rps_list.append(np.sum((cum_pred[:-1] - cum_true[:-1]) ** 2))
    rps_uncal = np.mean(rps_list)
    acc_uncal = accuracy_score(y_test, np.argmax(probs_uncal, axis=1))

    # ----- Calibrated probabilities (if requested) -----
    if calibrate and X_cal is not None and y_cal is not None:
        try:
            from sklearn.calibration import CalibratedClassifierCV
            calibrator = CalibratedClassifierCV(estimator=model, method='sigmoid', cv='prefit')
            calibrator.fit(X_cal, y_cal)
            probs_cal = calibrator.predict_proba(X_test)
        except (ValueError, TypeError, AttributeError):
            # Fallback: Isotonic per class
            probs_cal = model.predict_proba(X_cal)  # on calibration set
            calibrated_probs = np.zeros_like(probs_uncal)
            for i in range(n_classes):
                iso = IsotonicRegression(out_of_bounds='clip')
                iso.fit(probs_cal[:, i], (y_cal == i).astype(int))
                calibrated_probs[:, i] = iso.predict(probs_uncal[:, i])
            probs_cal = calibrated_probs / calibrated_probs.sum(axis=1, keepdims=True)
    else:
        probs_cal = probs_uncal  # use uncalibrated if calibration not performed

    # ----- Calibrated metrics -----
    ll_cal = log_loss(y_test, probs_cal)
    brier_cal = np.mean([brier_score_loss((y_test == i).astype(int), probs_cal[:, i]) for i in range(n_classes)])
    ece_cal = np.mean([compute_ece((y_test == i).astype(int), probs_cal[:, i]) for i in range(n_classes)])
    rps_list_cal = []
    for i in range(len(y_test)):
        true_label = y_test[i]
        cum_pred = np.cumsum(probs_cal[i, :])
        cum_true = np.zeros(n_classes)
        cum_true[true_label:] = 1
        rps_list_cal.append(np.sum((cum_pred[:-1] - cum_true[:-1]) ** 2))
    rps_cal = np.mean(rps_list_cal)
    acc_cal = accuracy_score(y_test, np.argmax(probs_cal, axis=1))

    return {
        # Uncalibrated
        'log_loss_uncal': ll_uncal,
        'brier_uncal': brier_uncal,
        'ece_uncal': ece_uncal,
        'rps_uncal': rps_uncal,
        'accuracy_uncal': acc_uncal,
        # Calibrated
        'log_loss': ll_cal,
        'brier': brier_cal,
        'ece': ece_cal,
        'rps': rps_cal,
        'accuracy': acc_cal
    }

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
    res = evaluate_classifier(best_model, X_pre_train, y_pre_train_cls,
                              X_pre_test, y_pre_test_cls,
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
# ======================================================================
# IN-PLAY LOOPS + PART 7
# ======================================================================

# -------------------- In-Play Classification Loop --------------------
best_inplay_clf = None
best_inplay_reg = None
best_inplay_clf_score = np.inf   # validation log-loss (lower is better)
best_inplay_reg_score = np.inf   # validation MAE (lower is better)

for name, (model, param_grid) in clf_models.items():
    print(f"Training {name} (in-play classification)...")
    if name == 'IFX-XGBoost':
        model = IFX_XGBoost(random_state=42, n_iterations=3)
        model.fit(X_snap_train, y_snap_train_cls, X_snap_val, y_snap_val_cls)
        best_model = model
        # Compute validation log-loss for IFX
        val_probs = model.predict_proba(X_snap_val)
        val_score = log_loss(y_snap_val_cls, val_probs)
    elif name in ['KernelSVM', 'KernelRidge']:
        print(f"   ***Using Nystroem approximation for {name} (large dataset)***")
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('kernel_approx', Nystroem(kernel='rbf', random_state=42)),
            ('clf', SGDClassifier(loss='log_loss', random_state=42, max_iter=1000, tol=1e-3))
        ])
        param_grid_adj = {
            'clf__alpha': [0.0001, 0.001, 0.01],
            'kernel_approx__n_components': [50, 100, 200]
        }
        gs = GridSearchCV(pipe, param_grid_adj, cv=3, scoring='neg_log_loss', n_jobs=-1, verbose=0)
        gs.fit(X_snap_train, y_snap_train_cls)
        best_model = gs.best_estimator_
        val_score = -gs.best_score_   # because scoring is neg_log_loss
    else:
        pipe = create_clf_pipeline(model, use_pf_smote=True, scaler=True)
        param_grid_adj = {}
        for k, v in param_grid.items():
            param_grid_adj['clf__' + k] = v
        gs = GridSearchCV(pipe, param_grid_adj, cv=3, scoring='neg_log_loss', n_jobs=-1, verbose=0)
        gs.fit(X_snap_train, y_snap_train_cls)
        best_model = gs.best_estimator_
        val_score = -gs.best_score_
    # Store if best so far
    if val_score < best_inplay_clf_score:
        best_inplay_clf_score = val_score
        best_inplay_clf = best_model
    # Evaluate on test
    res = evaluate_classifier(best_model, X_snap_train, y_snap_train_cls,
                              X_snap_test, y_snap_test_cls,
                              model_name=name, task='inplay_cls',
                              calibrate=True, X_cal=X_snap_val, y_cal=y_snap_val_cls)
    res['model'] = name
    results_cls.append(res)

# -------------------- In-Play Regression Loop --------------------
for name, (model, param_grid) in reg_models.items():
    print(f"Training {name} (in-play regression)...")
    if name == 'IFX-XGBoost':
        model = IFX_XGBoost(random_state=42, n_iterations=3, objective='reg:squarederror')
        model.fit(X_snap_train, y_snap_train_reg, X_snap_val, y_snap_val_reg)
        best_model = model
        val_preds = model.predict(X_snap_val)
        val_score = mean_absolute_error(y_snap_val_reg, val_preds)
    elif name in ['KernelRidge', 'KernelSVR']:
        print(f"   ***Using Nystroem approximation for {name} (large dataset)***")
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('kernel_approx', Nystroem(kernel='rbf', random_state=42)),
            ('reg', SGDRegressor(random_state=42, max_iter=1000, tol=1e-3))
        ])
        param_grid_adj = {
            'reg__alpha': [0.0001, 0.001, 0.01],
            'kernel_approx__n_components': [50, 100, 200]
        }
        gs = GridSearchCV(pipe, param_grid_adj, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=0)
        gs.fit(X_snap_train, y_snap_train_reg)
        best_model = gs.best_estimator_
        val_score = -gs.best_score_
    else:
        pipe = Pipeline([('scaler', StandardScaler()), ('reg', model)])
        param_grid_adj = {}
        for k, v in param_grid.items():
            param_grid_adj['reg__' + k] = v
        gs = GridSearchCV(pipe, param_grid_adj, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=0)
        gs.fit(X_snap_train, y_snap_train_reg)
        best_model = gs.best_estimator_
        val_score = -gs.best_score_
    # Store if best so far
    if val_score < best_inplay_reg_score:
        best_inplay_reg_score = val_score
        best_inplay_reg = best_model
    # Evaluate on test
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

# ======================================================================
# PART 7: IN-PLAY EVALUATION (USING THE BEST MODELS)
# ======================================================================

def eval_per_minute_cls(model, X, y, times):
    """Compute log-loss per 15-minute phase for classification."""
    bins = np.arange(0, 95, 15)
    metrics = []
    for i in range(len(bins)-1):
        mask = (times >= bins[i]) & (times < bins[i+1])
        if np.sum(mask) == 0:
            metrics.append(np.nan)
        else:
            y_prob = model.predict_proba(X[mask])
            metrics.append(log_loss(y[mask], y_prob))
    return bins[:-1], metrics

def eval_per_minute_reg(model, X, y, times):
    """Compute MAE per 15-minute phase for regression."""
    bins = np.arange(0, 95, 15)
    metrics = []
    for i in range(len(bins)-1):
        mask = (times >= bins[i]) & (times < bins[i+1])
        if np.sum(mask) == 0:
            metrics.append(np.nan)
        else:
            y_pred = model.predict(X[mask])
            metrics.append(mean_absolute_error(y[mask], y_pred))
    return bins[:-1], metrics

# ----------------------------------------------------------------------
# 7a. Per-phase metrics for the best in-play models
# ----------------------------------------------------------------------
# Use the best models stored during loops (or fallback to retrained XGBoost)
if best_inplay_clf is None:
    # Fallback: retrain a simple XGBoost (should not happen)
    best_inplay_clf = XGBClassifier(random_state=42, n_estimators=100, learning_rate=0.1)
    best_inplay_clf.fit(X_snap_train, y_snap_train_cls)
if best_inplay_reg is None:
    best_inplay_reg = XGBRegressor(random_state=42, n_estimators=100, learning_rate=0.1)
    best_inplay_reg.fit(X_snap_train, y_snap_train_reg)

# Log-loss per phase (classification)
bins, ll_per_phase = eval_per_minute_cls(best_inplay_clf, X_snap_test, y_snap_test_cls, snap_times_test)
plt.figure()
plt.plot(bins + 7.5, ll_per_phase, marker='o', label='In-play (best model)')
plt.xlabel('Match minute (phase)')
plt.ylabel('Log-Loss')
plt.title('In-play classification log-loss per game phase')
plt.legend()
plt.savefig(os.path.join(FIGURES_DIR, 'inplay_logloss_per_phase.png'))
plt.close()

# MAE per phase (regression)
bins, mae_per_phase = eval_per_minute_reg(best_inplay_reg, X_snap_test, y_snap_test_reg, snap_times_test)
plt.figure()
plt.plot(bins + 7.5, mae_per_phase, marker='o', label='In-play (best model)')
plt.xlabel('Match minute (phase)')
plt.ylabel('MAE')
plt.title('In-play regression MAE per game phase')
plt.legend()
plt.savefig(os.path.join(FIGURES_DIR, 'inplay_mae_per_phase.png'))
plt.close()

# ----------------------------------------------------------------------
# 7b. Frozen pre-match baseline (using the best pre-match model)
# ----------------------------------------------------------------------
# Train a pre-match model (we can use XGBoost with default params)
pre_model = XGBClassifier(random_state=42, n_estimators=100, learning_rate=0.1)
pre_model.fit(X_pre_train, y_pre_train_cls)
pre_reg = XGBRegressor(random_state=42, n_estimators=100, learning_rate=0.1)
pre_reg.fit(X_pre_train, y_pre_train_reg)

# Map each snapshot's match_id to its pre-match feature vector
# Use test_pre to get the features
test_pre_feat = test_pre[pre_feat_cols].values
test_pre_ids = test_pre['match_id'].values
pre_feat_dict = dict(zip(test_pre_ids, test_pre_feat))

snap_ids = test_snap['match_id'].values
pre_feat_for_snap = np.array([pre_feat_dict[mid] for mid in snap_ids])

# Pre-match predictions (probabilities and margin)
pre_prob = pre_model.predict_proba(pre_feat_for_snap)
pre_margin = pre_reg.predict(pre_feat_for_snap)

# Compute log-loss per phase for pre-match baseline
def eval_per_minute_probs(probs, y, times):
    bins = np.arange(0, 95, 15)
    metrics = []
    for i in range(len(bins)-1):
        mask = (times >= bins[i]) & (times < bins[i+1])
        if np.sum(mask) == 0:
            metrics.append(np.nan)
        else:
            metrics.append(log_loss(y[mask], probs[mask]))
    return bins[:-1], metrics

bins, ll_pre = eval_per_minute_probs(pre_prob, y_snap_test_cls, snap_times_test)

# Plot both curves (in-play vs frozen pre-match)
plt.figure()
plt.plot(bins + 7.5, ll_per_phase, marker='o', label='In-play (best model)')
plt.plot(bins + 7.5, ll_pre, marker='s', label='Frozen pre-match')
plt.xlabel('Match minute (phase)')
plt.ylabel('Log-Loss')
plt.title('In-play classification: In-play vs Frozen Pre-match')
plt.legend()
plt.savefig(os.path.join(FIGURES_DIR, 'inplay_vs_pre_logloss.png'))
plt.close()

# For regression, compute MAE per phase for pre-match
def eval_per_minute_preds(preds, y, times):
    bins = np.arange(0, 95, 15)
    metrics = []
    for i in range(len(bins)-1):
        mask = (times >= bins[i]) & (times < bins[i+1])
        if np.sum(mask) == 0:
            metrics.append(np.nan)
        else:
            metrics.append(mean_absolute_error(y[mask], preds[mask]))
    return bins[:-1], metrics

bins, mae_pre = eval_per_minute_preds(pre_margin, y_snap_test_reg, snap_times_test)

plt.figure()
plt.plot(bins + 7.5, mae_per_phase, marker='o', label='In-play (best model)')
plt.plot(bins + 7.5, mae_pre, marker='s', label='Frozen pre-match')
plt.xlabel('Match minute (phase)')
plt.ylabel('MAE')
plt.title('In-play regression: In-play vs Frozen Pre-match')
plt.legend()
plt.savefig(os.path.join(FIGURES_DIR, 'inplay_vs_pre_mae.png'))
plt.close()

# ----------------------------------------------------------------------
# 7c. Reliability diagrams for the best classification model
# ----------------------------------------------------------------------
from sklearn.calibration import calibration_curve

def plot_reliability_diagram(y_true, y_prob, model_name, task, n_bins=10):
    fig, axes = plt.subplots(1, y_prob.shape[1], figsize=(4*y_prob.shape[1], 4))
    if y_prob.shape[1] == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        fraction_of_positives, mean_predicted_value = calibration_curve(
            (y_true == i).astype(int), y_prob[:, i], n_bins=n_bins
        )
        ax.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        ax.plot([0, 1], [0, 1], "k--", label="Perfect")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"Class {i}")
        ax.legend()
    plt.suptitle(f"Reliability diagram – {model_name} ({task})")
    plt.savefig(os.path.join(FIGURES_DIR, f'reliability_{model_name}_{task}.png'))
    plt.close()

# Get probabilities from the best in-play classifier on test set
best_probs = best_inplay_clf.predict_proba(X_snap_test)
plot_reliability_diagram(y_snap_test_cls, best_probs, 
                         model_name='Best In-play Classifier', task='inplay')

# Also for the pre-match classifier (optional)
pre_probs = pre_model.predict_proba(X_pre_test)
plot_reliability_diagram(y_pre_test_cls, pre_probs,
                         model_name='Best Pre-match Classifier', task='pre')

print("Part 7 done: per-phase evaluation, frozen baseline comparison, and reliability diagrams saved.")

# -------------------- 8. Compute Cost & Kernel Scaling Analysis --------------------
# We'll measure wall-clock time and peak memory for each model on the full training set.
# We'll also demonstrate O(n^2) scaling of kernel methods by training on increasing subsets.

def measure_training(model, X, y):
    """Fit model and return time and memory."""
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024**2  # MB
    start = time.time()
    model.fit(X, y)
    end = time.time()
    mem_after = process.memory_info().rss / 1024**2
    return end-start, mem_after - mem_before

# We'll measure for a few models on pre-match training set.
# We'll also do kernel scaling by varying dataset size.
subsample_sizes = [100, 500, 1000, 2000, 5000]
kernel_times = []
for n in subsample_sizes:
    if n > len(X_pre_train):
        break
    X_sub = X_pre_train[:n]
    y_sub = y_pre_train_reg[:n]
    model = KernelRidge(alpha=1.0, kernel='rbf')
    t, _ = measure_training(model, X_sub, y_sub)
    kernel_times.append(t)
    gc.collect()

plt.figure()
plt.plot(subsample_sizes[:len(kernel_times)], kernel_times, marker='o')
plt.xlabel('Training sample size')
plt.ylabel('Time (seconds)')
plt.title('Kernel Ridge scaling (O(n^2))')
plt.savefig(os.path.join(FIGURES_DIR, 'kernel_scaling.png'))

# Approximate kernel scaling (Nystroem)
approx_times = []
for n in subsample_sizes:
    if n > len(X_pre_train):
        break
    X_sub = X_pre_train[:n]
    y_sub = y_pre_train_reg[:n]
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('kernel_approx', Nystroem(kernel='rbf', n_components=100, random_state=42)),
        ('reg', Ridge(alpha=1.0))
    ])
    t, _ = measure_training(pipe, X_sub, y_sub)
    approx_times.append(t)

plt.figure()
plt.plot(subsample_sizes[:len(kernel_times)], kernel_times, marker='o', label='Exact KernelRidge')
plt.plot(subsample_sizes[:len(approx_times)], approx_times, marker='s', label='Approx (Nystroem+Ridge)')
plt.xlabel('Training sample size')
plt.ylabel('Time (seconds)')
plt.title('Kernel Scaling: Exact vs. Approximate')
plt.legend()
plt.savefig(os.path.join(FIGURES_DIR, 'kernel_scaling_comparison.png'))
plt.close()

# PART 8 Done (compute & scaling)

# -------------------- 9. SHAP Analysis (Preliminary) --------------------
# We'll use a pre‑match classification model for SHAP (18 features).
# Retrain a simple XGBoost on pre‑match data (no tuning, just for demonstration).
shap_model = XGBClassifier(random_state=42, n_estimators=100, learning_rate=0.1)
shap_model.fit(X_pre_train, y_pre_train_cls)

# SHAP on first 100 test samples (to keep runtime manageable)
explainer = shap.TreeExplainer(shap_model)
shap_values = explainer.shap_values(X_pre_test[:100])

# Global summary plot (beeswarm)
shap.summary_plot(shap_values, X_pre_test[:100], feature_names=pre_feat_cols, show=False)
plt.savefig(os.path.join(FIGURES_DIR, 'shap_summary.png'))
plt.close()

# ----- Local waterfall plot for the first test sample (predicted class) -----
# Get predicted class for the first sample
pred_class = np.argmax(shap_model.predict_proba(X_pre_test[:1])[0])
# Create a shap.Explanation object for that class
explanation = shap.Explanation(
    values=shap_values[pred_class][0],
    base_values=explainer.expected_value[pred_class],
    data=X_pre_test[0],
    feature_names=pre_feat_cols
)
shap.plots.waterfall(explanation, show=False)
plt.savefig(os.path.join(FIGURES_DIR, 'shap_waterfall.png'))
plt.close()

print("SHAP preliminary analysis complete. Plots saved.")
# PART 9 Done

# ----- Worst prediction alanysis -----
def worst_predictions_analysis(model, X, y_true, feature_names, model_name, task, top_k=10):
    """
    Identify the top_k worst predictions and save SHAP explanations.
    Handles plain models, pipelines, and IFX_XGBoost wrappers.
    """
    # Keep the original model for predictions (handles numpy arrays)
    pred_model = model

    # Extract the underlying tree model for SHAP
    shap_model = None
    if hasattr(model, 'model') and isinstance(model.model, xgb.Booster):
        # IFX_XGBoost
        shap_model = model.model
    elif hasattr(model, 'steps'):
        # Pipeline: take the last estimator
        last_est = model.steps[-1][1]
        if hasattr(last_est, 'model') and isinstance(last_est.model, xgb.Booster):
            shap_model = last_est.model
        else:
            shap_model = last_est
    elif hasattr(model, 'named_steps'):
        # Imblearn pipeline
        last_est = model.named_steps[list(model.named_steps.keys())[-1]]
        if hasattr(last_est, 'model') and isinstance(last_est.model, xgb.Booster):
            shap_model = last_est.model
        else:
            shap_model = last_est
    else:
        shap_model = model

    # Determine worst indices using the prediction model
    if task == 'classification':
        probs = pred_model.predict_proba(X)
        per_sample_ll = [-np.log(probs[i, y_true[i]]) for i in range(len(y_true))]
        worst_idx = np.argsort(per_sample_ll)[-top_k:]
        worst_scores = [per_sample_ll[i] for i in worst_idx]
    else:  # regression
        preds = pred_model.predict(X)
        errors = np.abs(y_true - preds)
        worst_idx = np.argsort(errors)[-top_k:]
        worst_scores = [errors[i] for i in worst_idx]

    # Compute SHAP using the extracted tree model
    explainer = shap.TreeExplainer(shap_model)
    shap_values = explainer.shap_values(X[worst_idx])

    for j, idx in enumerate(worst_idx):
        sample = X[idx]
        true_label = y_true[idx]
        score = worst_scores[j]

        if task == 'classification':
            pred_class = np.argmax(pred_model.predict_proba([sample])[0])

            if isinstance(shap_values, list):
                # List of arrays (one per class)
                class_shap = shap_values[pred_class][j]
                base = explainer.expected_value[pred_class]
            elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
                # 3D array: (samples, features, classes)
                class_shap = shap_values[j, :, pred_class]
                base = explainer.expected_value[pred_class]
            else:
                # Binary / fallback
                class_shap = shap_values[j]
                base = explainer.expected_value

            explanation = shap.Explanation(
                values=class_shap,
                base_values=base,
                data=sample,
                feature_names=feature_names
            )
            shap.plots.waterfall(explanation, show=False)
            plt.title(f"{model_name} - Sample {idx} (true={true_label}, pred={pred_class}, log-loss={score:.4f})")

        else:  # regression
            if isinstance(shap_values, list):
                shap_vals = shap_values[0][j]
            elif hasattr(shap_values, 'ndim') and shap_values.ndim == 2:
                shap_vals = shap_values[j]
            else:
                shap_vals = shap_values[j]

            explanation = shap.Explanation(
                values=shap_vals,
                base_values=explainer.expected_value,
                data=sample,
                feature_names=feature_names
            )
            shap.plots.waterfall(explanation, show=False)
            plt.title(f"{model_name} - Sample {idx} (true={true_label}, pred={preds[idx]:.2f}, error={score:.2f})")

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'worst_{j+1}_{model_name}_{task}.png'))
        plt.close()

    print(f"Saved {top_k} worst prediction plots for {model_name} ({task}).")

# For the best in-play classifier (classification)
worst_predictions_analysis(best_inplay_clf, X_snap_test, y_snap_test_cls,
                           feature_names=snap_feat_cols,
                           model_name='Best_Inplay_Classifier', task='classification')

# For the best in-play regressor (regression)
worst_predictions_analysis(best_inplay_reg, X_snap_test, y_snap_test_reg,
                           feature_names=snap_feat_cols,
                           model_name='Best_Inplay_Regressor', task='regression')

# ======================================================================
# RESAMPLING COMPARISON
# ======================================================================
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN

resamplers = {
    'None': None,
    'PF-SMOTE': PF_SMOTE(random_state=42),
    'SMOTE': SMOTE(random_state=42),
    'BorderlineSMOTE': BorderlineSMOTE(random_state=42),
    'ADASYN': ADASYN(random_state=42),
}

base_model = XGBClassifier(random_state=42, n_estimators=100, learning_rate=0.1)
resample_results = []

for name, resampler in resamplers.items():
    print(f"Training XGBoost with {name} resampling (pre-match classification)...")
    steps = [('scaler', StandardScaler())]
    if resampler is not None:
        steps.append(('resampler', resampler))
    steps.append(('clf', base_model))
    pipe = ImbPipeline(steps)
    pipe.fit(X_pre_train, y_pre_train_cls)

    res = evaluate_classifier(pipe, X_pre_train, y_pre_train_cls,
                              X_pre_test, y_pre_test_cls,
                              model_name=name, task='resample_compare',
                              calibrate=True, X_cal=X_pre_val, y_cal=y_pre_val_cls)
    res['resampler'] = name
    resample_results.append(res)

df_resample = pd.DataFrame(resample_results)
print("Resampling comparison results:")
print(df_resample[['resampler', 'log_loss', 'rps', 'accuracy', 'ece']])
df_resample.to_csv('resampling_comparison.csv', index=False)

# ======================================================================
# SHAP TIMELINE FOR A SINGLE MATCH (Corrected)
# ======================================================================

def shap_timeline_for_match(match_id, clf_model, reg_model, test_snap_full, y_cls, y_reg, feature_names, fig_dir):
    """
    Generate SHAP timeline for a single match.
    Extracts the underlying tree model from pipelines and IFX wrappers.
    """
    # Filter snapshots for this match
    match_snap = test_snap_full[test_snap_full['match_id'] == match_id].copy()
    if match_snap.empty:
        print(f"No snapshots found for match {match_id}")
        return

    # Sort by time
    match_snap = match_snap.sort_values('snapshot_time')
    times = match_snap['snapshot_time'].values
    X_snap = match_snap[feature_names].values

    # ---------- Extract SHAP model from classifier ----------
    shap_clf = None
    if hasattr(clf_model, 'model') and isinstance(clf_model.model, xgb.Booster):
        shap_clf = clf_model.model
    elif hasattr(clf_model, 'steps'):
        last_est = clf_model.steps[-1][1]
        if hasattr(last_est, 'model') and isinstance(last_est.model, xgb.Booster):
            shap_clf = last_est.model
        else:
            shap_clf = last_est
    elif hasattr(clf_model, 'named_steps'):
        last_est = clf_model.named_steps[list(clf_model.named_steps.keys())[-1]]
        if hasattr(last_est, 'model') and isinstance(last_est.model, xgb.Booster):
            shap_clf = last_est.model
        else:
            shap_clf = last_est
    else:
        shap_clf = clf_model

    # ---------- Extract SHAP model from regressor ----------
    shap_reg = None
    if hasattr(reg_model, 'model') and isinstance(reg_model.model, xgb.Booster):
        shap_reg = reg_model.model
    elif hasattr(reg_model, 'steps'):
        last_est = reg_model.steps[-1][1]
        if hasattr(last_est, 'model') and isinstance(last_est.model, xgb.Booster):
            shap_reg = last_est.model
        else:
            shap_reg = last_est
    elif hasattr(reg_model, 'named_steps'):
        last_est = reg_model.named_steps[list(reg_model.named_steps.keys())[-1]]
        if hasattr(last_est, 'model') and isinstance(last_est.model, xgb.Booster):
            shap_reg = last_est.model
        else:
            shap_reg = last_est
    else:
        shap_reg = reg_model

    # ---------- Compute SHAP for classification (predicted class) ----------
    explainer_clf = shap.TreeExplainer(shap_clf)
    shap_values_clf = explainer_clf.shap_values(X_snap)

    # Predict classes using the original model (handles numpy arrays)
    pred_classes = clf_model.predict(X_snap)
    shap_class_vals = []
    for i, cls in enumerate(pred_classes):
        if isinstance(shap_values_clf, list):
            shap_class_vals.append(shap_values_clf[cls][i])
        elif hasattr(shap_values_clf, 'ndim') and shap_values_clf.ndim == 3:
            shap_class_vals.append(shap_values_clf[i, :, cls])
        else:
            shap_class_vals.append(shap_values_clf[i])
    shap_class_vals = np.array(shap_class_vals)  # (n_snapshots, n_features)

    # ---------- Compute SHAP for regression ----------
    explainer_reg = shap.TreeExplainer(shap_reg)
    shap_values_reg = explainer_reg.shap_values(X_snap)

    # ---------- Plot SHAP timeline (top 5 features) ----------
    mean_abs_shap = np.mean(np.abs(shap_class_vals), axis=0)
    top_n = 5
    top_idx = np.argsort(mean_abs_shap)[-top_n:]
    top_features = [feature_names[i] for i in top_idx]

    plt.figure(figsize=(12, 6))
    for i in top_idx:
        plt.plot(times, shap_class_vals[:, i], marker='o', label=feature_names[i])
    plt.xlabel('Match minute')
    plt.ylabel('SHAP value (contribution to predicted class)')
    plt.title(f'SHAP Timeline for Match {match_id} (Top {top_n} features)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, f'shap_timeline_{match_id}.png'))
    plt.close()

    # ---------- Waterfall plot at the middle snapshot ----------
    mid_snapshot = len(times) // 2
    snapshot_idx = mid_snapshot
    pred_class = pred_classes[snapshot_idx]

    if isinstance(shap_values_clf, list):
        class_shap = shap_values_clf[pred_class][snapshot_idx]
        base = explainer_clf.expected_value[pred_class]
    elif hasattr(shap_values_clf, 'ndim') and shap_values_clf.ndim == 3:
        class_shap = shap_values_clf[snapshot_idx, :, pred_class]
        base = explainer_clf.expected_value[pred_class]
    else:
        class_shap = shap_values_clf[snapshot_idx]
        base = explainer_clf.expected_value

    explanation = shap.Explanation(
        values=class_shap,
        base_values=base,
        data=X_snap[snapshot_idx],
        feature_names=feature_names
    )
    shap.plots.waterfall(explanation, show=False)
    plt.title(f'Match {match_id} at minute {times[snapshot_idx]} – Predicted class {pred_class}')
    plt.savefig(os.path.join(fig_dir, f'shap_waterfall_{match_id}_min{int(times[snapshot_idx])}.png'))
    plt.close()

    print(f"SHAP timeline for match {match_id} saved.")

# Example: pick a match from the test set (i.e. first one)
example_match_id = test_pre['match_id'].iloc[0]
shap_timeline_for_match(example_match_id,
                        best_inplay_clf,
                        best_inplay_reg,
                        test_snap,       # full test snapshots DataFrame
                        y_snap_test_cls,
                        y_snap_test_reg,
                        snap_feat_cols,
                        FIGURES_DIR)

# ======================================================================
# SAVE BEST MODELS AND RELATED DATA
# ======================================================================
import joblib

# Save the best in-play models
joblib.dump(best_inplay_clf, 'best_inplay_clf.pkl')
joblib.dump(best_inplay_reg, 'best_inplay_reg.pkl')

# Save feature names and other metadata (for app to use)
metadata = {
    'pre_feat_cols': pre_feat_cols,
    'snap_feat_cols': snap_feat_cols,
    'test_snap': test_snap,      # full test snapshots DataFrame
    'test_pre': test_pre,        # full test pre-match DataFrame
    'y_snap_test_cls': y_snap_test_cls,
    'y_snap_test_reg': y_snap_test_reg,
}
# We can't save large DataFrames in a pickle easily, so we save them as CSV
test_snap.to_csv('test_snapshots_for_app.csv', index=False)
test_pre.to_csv('test_prematch_for_app.csv', index=False)
# Save y arrays as numpy
np.save('y_snap_test_cls.npy', y_snap_test_cls)
np.save('y_snap_test_reg.npy', y_snap_test_reg)

print("Models and data saved for final defence.")