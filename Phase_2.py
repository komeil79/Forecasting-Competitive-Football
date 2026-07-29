import os
import time
import gc
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    Fit model (if not already fitted) and evaluate on test set.
    If calibrate=True and calibration data provided, fit a Platt calibrator
    on the model's predicted probabilities and apply to test.
    """
    # If model is not fitted, fit it (only for non-IFX models; IFX already fitted)
    if not hasattr(model, 'model') and not hasattr(model, 'predict_proba'):
        # fallback: maybe model is already fitted
        pass

    # Get raw probabilities on calibration and test sets
    if X_cal is not None:
        probs_cal = model.predict_proba(X_cal)
    else:
        probs_cal = None
    
    probs_test = model.predict_proba(X_test)
    
    if calibrate and probs_cal is not None and y_cal is not None:
        # Fit Platt scaling (LogisticRegression) on calibration probabilities
        # For multi-class, we need a one-vs-rest calibration (or use CalibratedClassifierCV with cv='prefit')
        # Since we have a separate calibration set, we can use CalibratedClassifierCV with cv='prefit' if sklearn version supports it.
        # Alternatively, we fit a LogisticRegression on the logits for each class (Platt scaling)
        # For simplicity, we'll use CalibratedClassifierCV with cv='prefit' if available, else we'll use a simple method.
        try:
            from sklearn.calibration import CalibratedClassifierCV
            # Use cv='prefit' (requires sklearn >= 0.24)
            calibrator = CalibratedClassifierCV(estimator=model, method='sigmoid', cv='prefit')
            calibrator.fit(X_cal, y_cal)
            probs_test = calibrator.predict_proba(X_test)
        except (ValueError, TypeError, AttributeError):
            # If cv='prefit' not supported, fallback to LogisticRegression on predicted probabilities
            # For each class, fit a logistic regression on the predicted probability of that class
            n_classes = probs_cal.shape[1]
            calibrated_probs = np.zeros_like(probs_test)
            for i in range(n_classes):
                # Fit a logistic regression on the logit of the probability? 
                # Simpler: use IsotonicRegression
                iso = IsotonicRegression(out_of_bounds='clip')
                iso.fit(probs_cal[:, i], (y_cal == i).astype(int))
                calibrated_probs[:, i] = iso.predict(probs_test[:, i])
            # Normalize to sum to 1
            calibrated_probs = calibrated_probs / calibrated_probs.sum(axis=1, keepdims=True)
            probs_test = calibrated_probs
    else:
        # No calibration
        pass

    # Compute metrics
    n_classes = probs_test.shape[1]
    ll = log_loss(y_test, probs_test)
    brier = np.mean([brier_score_loss((y_test == i).astype(int), probs_test[:, i]) for i in range(n_classes)])
    ece = np.mean([compute_ece((y_test == i).astype(int), probs_test[:, i]) for i in range(n_classes)])
    # RPS
    rps_list = []
    for i in range(len(y_test)):
        true_label = y_test[i]
        cum_pred = np.cumsum(probs_test[i, :])
        cum_true = np.zeros(n_classes)
        cum_true[true_label:] = 1
        rps_list.append(np.sum((cum_pred[:-1] - cum_true[:-1]) ** 2))
    rps = np.mean(rps_list)
    acc = accuracy_score(y_test, np.argmax(probs_test, axis=1))
    
    return {'log_loss': ll, 'brier': brier, 'ece': ece, 'rps': rps, 'accuracy': acc}

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

# In-play classification (model 3)
# We'll train on snapshot data, but we must respect match-level splitting.
# Already done by using separate train/val/test snapshots.
# In-play classification loop (replace the existing one)
for name, (model, param_grid) in clf_models.items():
    print(f"Training {name} (in-play classification)...")
    if name == 'IFX-XGBoost':
        model = IFX_XGBoost(random_state=42, n_iterations=3)
        model.fit(X_snap_train, y_snap_train_cls, X_snap_val, y_snap_val_cls)
        best_model = model
    elif name in ['KernelSVM', 'KernelRidge']:
        # Use approximate kernel for large dataset
        print(f"   ***Using Nystroem approximation for {name} (large dataset)***")
        if name == 'KernelSVM':
            approx_model = make_approx_kernel_pipeline('classification')
        else:
            # For KernelRidge we also use approximation
            approx_model = make_approx_kernel_pipeline('classification')
        # We'll use a simple grid for the approximate model (SGDClassifier)
        param_grid_adj = {
            'clf__alpha': [0.0001, 0.001, 0.01],
            'kernel_approx__n_components': [50, 100, 200]
        }
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('kernel_approx', Nystroem(kernel='rbf', random_state=42)),
            ('clf', SGDClassifier(loss='log_loss', random_state=42, max_iter=1000, tol=1e-3))
        ])
        gs = GridSearchCV(pipe, param_grid_adj, cv=3, scoring='neg_log_loss', n_jobs=-1, verbose=0)
        gs.fit(X_snap_train, y_snap_train_cls)
        best_model = gs.best_estimator_
    else:
        # Non‑kernel models: use PF‑SMOTE pipeline as before
        pipe = create_clf_pipeline(model, use_pf_smote=True, scaler=True)
        param_grid_adj = {}
        for k, v in param_grid.items():
            param_grid_adj['clf__' + k] = v
        gs = GridSearchCV(pipe, param_grid_adj, cv=3, scoring='neg_log_loss', n_jobs=-1, verbose=0)
        gs.fit(X_snap_train, y_snap_train_cls)
        best_model = gs.best_estimator_
    # Evaluate
    res = evaluate_classifier(best_model, X_snap_train, y_snap_train_cls,
                              X_snap_test, y_snap_test_cls,
                              model_name=name, task='inplay_cls',
                              calibrate=True, X_cal=X_snap_val, y_cal=y_snap_val_cls)
    res['model'] = name
    results_cls.append(res)

# In-play regression
# In-play regression loop
for name, (model, param_grid) in reg_models.items():
    print(f"Training {name} (in-play regression)...")
    if name == 'IFX-XGBoost':
        model = IFX_XGBoost(random_state=42, n_iterations=3, objective='reg:squarederror')
        model.fit(X_snap_train, y_snap_train_reg, X_snap_val, y_snap_val_reg)
        best_model = model
    elif name in ['KernelRidge', 'KernelSVR']:
        print(f"   ***Using Nystroem approximation for {name} (large dataset)***")
        # Build approximate pipeline for regression
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
    else:
        # Non‑kernel models: standard pipeline with scaling
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

# -------------------- 7. In-Play Evaluation: Metric vs Minute & Per-Phase Calibration --------------------
def eval_per_minute_cls(model, X, y, times):
    """Compute log-loss per 15-minute phase for classification."""
    bins = np.arange(0, 95, 15)  # 0-15, 15-30, ..., 75-90+
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
# Use the best models from the in-play training loops (if you stored them)
# or retrain simple ones (we'll retrain for clarity).
# ----------------------------------------------------------------------

# Retrain simple XGBoost models (you can also use the best models from earlier)
best_clf = XGBClassifier(random_state=42, n_estimators=100, learning_rate=0.1)
best_clf.fit(X_snap_train, y_snap_train_cls)

best_reg = XGBRegressor(random_state=42, n_estimators=100, learning_rate=0.1)
best_reg.fit(X_snap_train, y_snap_train_reg)

# Log-loss per phase (classification)
bins, ll_per_phase = eval_per_minute_cls(best_clf, X_snap_test, y_snap_test_cls, snap_times_test)
plt.figure()
plt.plot(bins + 7.5, ll_per_phase, marker='o')
plt.xlabel('Match minute (phase)')
plt.ylabel('Log-Loss')
plt.title('In-play classification log-loss per game phase')
plt.savefig(os.path.join(FIGURES_DIR, 'inplay_logloss_per_phase.png'))
plt.close()

# MAE per phase (regression)
bins, mae_per_phase = eval_per_minute_reg(best_reg, X_snap_test, y_snap_test_reg, snap_times_test)
plt.figure()
plt.plot(bins + 7.5, mae_per_phase, marker='o')
plt.xlabel('Match minute (phase)')
plt.ylabel('MAE')
plt.title('In-play regression MAE per game phase')
plt.savefig(os.path.join(FIGURES_DIR, 'inplay_mae_per_phase.png'))
plt.close()

print("Part 7 done: per‑phase evaluation plots saved.")

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