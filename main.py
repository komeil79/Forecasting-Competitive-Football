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

from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge, SGDClassifier, SGDRegressor
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from sklearn.isotonic import IsotonicRegression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.svm import SVC, SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, log_loss, brier_score_loss,
                             mean_absolute_error, mean_squared_error)
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import shap
import psutil

from PF_SMOTE import PF_SMOTE
from IFX_model import IFX_XGBoost

# ===================== CONFIGURATION =====================
DATA_DIR = "processed_data"
OUTPUT_DIR = "out"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, 'Forecasting-Competitive-Football\\figures')
SEED = 42
MIN_TRAIN_MATCHES = 50
MIN_TEST_MATCHES = 5
WINDOW_SIZE = None          # None = all previous seasons; set to 5 for last 5
CALIBRATE = True
USE_PF_SMOTE = True
HIGH_RISK_THRESHOLD = 0.3
ENSEMBLE_WINDOWS = [None, 5, 3]
# =========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ===================== DATA LOADING =====================
def load_original_data():
    """Load original pre-match and snapshot datasets (train/val/test) – kept for reference."""
    train_pre = pd.read_parquet(os.path.join(DATA_DIR, 'train_prematch.parquet'))
    val_pre = pd.read_parquet(os.path.join(DATA_DIR, 'val_prematch.parquet'))
    test_pre = pd.read_parquet(os.path.join(DATA_DIR, 'test_prematch.parquet'))
    train_snap = pd.read_parquet(os.path.join(DATA_DIR, 'train_snapshots.parquet'))
    val_snap = pd.read_parquet(os.path.join(DATA_DIR, 'val_snapshots.parquet'))
    test_snap = pd.read_parquet(os.path.join(DATA_DIR, 'test_snapshots.parquet'))
    return (train_pre, val_pre, test_pre), (train_snap, val_snap, test_snap)

def load_full_data():
    """Load full pre-match data with season info (for temporal folds)."""
    prematch = pd.read_csv(os.path.join(DATA_DIR, 'full_prematch.csv'))
    matches = pd.read_csv(os.path.join(DATA_DIR, 'matches_full.csv'))
    matches = matches[['match_id', 'competition_id', 'result']]
    df = prematch.merge(matches, on='match_id', how='left')
    df['match_date'] = pd.to_datetime(df['match_date'])
    df['season'] = df['match_date'].dt.year
    df.loc[df['match_date'].dt.month < 8, 'season'] = df['season'] - 1
    df['label'] = df['result'].map({'H':0, 'D':1, 'A':2})
    return df

def load_full_snapshots():
    """Load full snapshot data (for in-play temporal evaluation)."""
    snaps = pd.read_csv(os.path.join(DATA_DIR, 'full_snapshots.csv'))
    matches = pd.read_csv(os.path.join(DATA_DIR, 'matches_full.csv'))
    matches = matches[['match_id', 'match_date']]
    snaps = snaps.merge(matches, on='match_id', how='left')
    snaps['match_date'] = pd.to_datetime(snaps['match_date'])
    snaps['season'] = snaps['match_date'].dt.year
    snaps.loc[snaps['match_date'].dt.month < 8, 'season'] = snaps['season'] - 1
    snaps['label_cls'] = snaps['final_result'].map({'H':0, 'D':1, 'A':2})
    return snaps

def get_features_labels_prematch(df):
    """Extract pre-match features and labels."""
    feature_cols = [c for c in df.columns if c not in [
        'match_id', 'match_date', 'home_team', 'away_team',
        'label_goal_diff', 'label_result', 'result', 'season',
        'competition_id', 'label'
    ]]
    X = df[feature_cols].values
    y = df['label'].values
    return X, y

def get_features_labels_snapshot(snaps_df):
    """Extract snapshot features and labels."""
    feature_cols = [c for c in snaps_df.columns if c not in [
        'match_id', 'snapshot_time', 'final_goal_diff', 'final_result',
        'match_date', 'season', 'label_cls'
    ]]
    X = snaps_df[feature_cols].values
    y = snaps_df['label_cls'].values
    return X, y

# ===================== MODEL PIPELINES =====================
def create_clf_pipeline(model, resampler=None, scaler=True):
    """
    resampler: None, 'adasyn', 'smote', 'borderline', 'pf_smote'
    Default: 'adasyn' (best in our comparison)
    """
    steps = []
    if scaler:
        steps.append(('scaler', StandardScaler()))
    if resampler is not None:
        if resampler == 'adasyn':
            steps.append(('resampler', ADASYN(random_state=SEED)))
        elif resampler == 'smote':
            steps.append(('resampler', SMOTE(random_state=SEED)))
        elif resampler == 'borderline':
            steps.append(('resampler', BorderlineSMOTE(random_state=SEED)))
        elif resampler == 'pf_smote':
            steps.append(('resampler', PF_SMOTE(random_state=SEED)))
    steps.append(('clf', model))
    return ImbPipeline(steps)

def create_reg_pipeline(model, scaler=True):
    steps = []
    if scaler:
        steps.append(('scaler', StandardScaler()))
    steps.append(('reg', model))
    return Pipeline(steps)

# ===================== MODEL SUITE =====================
clf_models = {
    'Dummy': (DummyClassifier(strategy='most_frequent'), {}),
    'KernelSVM': (SVC(probability=True, random_state=SEED), 
                  {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}),
    'RandomForest': (RandomForestClassifier(random_state=SEED),
                     {'n_estimators': [100, 200], 'max_depth': [None, 5, 10]}),
    'GBM': (GradientBoostingClassifier(random_state=SEED),
            {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1, 0.2]}),
    'XGBoost': (XGBClassifier(random_state=SEED, eval_metric='mlogloss'),
                {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1, 0.2],
                 'max_depth': [3, 5, 7]}),
    'LightGBM': (LGBMClassifier(random_state=SEED, verbose=-1),
                 {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1, 0.2],
                  'num_leaves': [31, 63]}),
    'IFX-XGBoost': (IFX_XGBoost(random_state=SEED, n_iterations=3),
                    {'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]})
}

reg_models = {
    'Dummy': (DummyRegressor(strategy='mean'), {}),
    'KernelRidge': (KernelRidge(),
                    {'alpha': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}),
    'KernelSVR': (SVR(),
                  {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}),
    'RandomForest': (RandomForestRegressor(random_state=SEED),
                     {'n_estimators': [100, 200], 'max_depth': [None, 5, 10]}),
    'GBM': (GradientBoostingRegressor(random_state=SEED),
            {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1, 0.2]}),
    'XGBoost': (XGBRegressor(random_state=SEED),
                {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1, 0.2],
                 'max_depth': [3, 5, 7]}),
    'LightGBM': (LGBMRegressor(random_state=SEED, verbose=-1),
                 {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1, 0.2],
                  'num_leaves': [31, 63]}),
    'IFX-XGBoost': (IFX_XGBoost(random_state=SEED, n_iterations=3, objective='reg:squarederror'),
                    {'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]})
}

# ===================== TEMPORAL FOLDS =====================
def get_temporal_folds(df, min_train_seasons=3, window_size=None):
    seasons = sorted(df['season'].unique())
    folds = []
    for i in range(min_train_seasons, len(seasons)):
        test_season = seasons[i]
        if window_size is None:
            train_seasons = seasons[:i]
        else:
            train_seasons = seasons[max(0, i-window_size):i]
        train_mask = df['season'].isin(train_seasons)
        test_mask = df['season'] == test_season
        if train_mask.sum() >= MIN_TRAIN_MATCHES and test_mask.sum() >= MIN_TEST_MATCHES:
            folds.append((train_mask, test_mask))
    return folds

# ===================== CALIBRATION & METRICS =====================
def compute_ece(y_true, y_prob, n_bins=10):
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

def evaluate_classifier_temporal(model, X_train, y_train, X_test, y_test, calibrate=True,
                                 calib_method='auto', already_fitted=False):
    """
    Fit model (if not already_fitted), calibrate using inner validation, evaluate.
    calib_method: 'auto', 'platt', 'isotonic', 'none'
    already_fitted: True if model has been fitted externally (e.g., IFX)
    """
    # If model is not already fitted, fit it on the full training set
    if not already_fitted:
        model.fit(X_train, y_train)

    # Raw probabilities on test
    probs_uncal = model.predict_proba(X_test)

    if not calibrate or calib_method == 'none':
        ll = log_loss(y_test, probs_uncal) if len(set(y_test)) > 1 else np.nan
        acc = accuracy_score(y_test, np.argmax(probs_uncal, axis=1))
        rps = np.mean([np.sum((np.cumsum(probs_uncal[i,:])[:-1] - np.where(np.arange(3) < y_test[i], 1, 0)[:-1])**2) for i in range(len(y_test))])
        brier = np.mean([brier_score_loss((y_test==c).astype(int), probs_uncal[:,c]) for c in range(3)])
        return {'log_loss': ll, 'accuracy': acc, 'rps': rps, 'brier': brier}

    if already_fitted:
        ll = log_loss(y_test, probs_uncal) if len(set(y_test)) > 1 else np.nan
        acc = accuracy_score(y_test, np.argmax(probs_uncal, axis=1))
        rps = np.mean([np.sum((np.cumsum(probs_uncal[i,:])[:-1] - np.where(np.arange(3) < y_test[i], 1, 0)[:-1])**2) for i in range(len(y_test))])
        brier = np.mean([brier_score_loss((y_test==c).astype(int), probs_uncal[:,c]) for c in range(3)])
        return {'log_loss': ll, 'accuracy': acc, 'rps': rps, 'brier': brier}

    # Create inner validation split from training data (last 20%)
    n_train = len(X_train)
    cal_size = max(10, int(n_train * 0.2))
    X_inner_train = X_train[:n_train-cal_size]
    y_inner_train = y_train[:n_train-cal_size]
    X_cal = X_train[n_train-cal_size:]
    y_cal = y_train[n_train-cal_size:]

    # Fit model on inner_train
    model.fit(X_inner_train, y_inner_train)

    # Get probabilities on calibration set
    probs_cal = model.predict_proba(X_cal)

    # Calibrate
    try:
        if calib_method == 'isotonic':
            n_classes = probs_uncal.shape[1]
            probs_cal_final = np.zeros_like(probs_uncal)
            for c in range(n_classes):
                iso = IsotonicRegression(out_of_bounds='clip')
                iso.fit(probs_cal[:, c], (y_cal == c).astype(int))
                probs_cal_final[:, c] = iso.predict(probs_uncal[:, c])
            probs_cal_final = probs_cal_final / probs_cal_final.sum(axis=1, keepdims=True)
            probs_cal = probs_cal_final
        else:
            # Default: Platt scaling
            calibrator = CalibratedClassifierCV(estimator=model, method='sigmoid', cv='prefit')
            calibrator.fit(X_cal, y_cal)
            probs_cal = calibrator.predict_proba(X_test)
    except Exception as e:
        # Fallback: use uncalibrated if calibration fails
        probs_cal = probs_uncal

    # Metrics
    ll = log_loss(y_test, probs_cal) if len(set(y_test)) > 1 else np.nan
    acc = accuracy_score(y_test, np.argmax(probs_cal, axis=1))
    rps = np.mean([np.sum((np.cumsum(probs_cal[i,:])[:-1] - np.where(np.arange(3) < y_test[i], 1, 0)[:-1])**2) for i in range(len(y_test))])
    brier = np.mean([brier_score_loss((y_test==c).astype(int), probs_cal[:,c]) for c in range(3)])
    return {'log_loss': ll, 'accuracy': acc, 'rps': rps, 'brier': brier}

def evaluate_regressor_temporal(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    corr = np.corrcoef(y_test, y_pred)[0,1]
    return {'mae': mae, 'rmse': rmse, 'corr': corr}

# ===================== MAIN TEMPORAL LOOP (PRE-MATCH) =====================
def run_temporal_prematch_classification(df, model_name):
    folds = get_temporal_folds(df, min_train_seasons=3, window_size=WINDOW_SIZE)
    results = []
    for train_mask, test_mask in folds:
        train_df = df[train_mask]
        test_df = df[test_mask]
        X_train, y_train = get_features_labels_prematch(train_df)
        X_test, y_test = get_features_labels_prematch(test_df)
        model, param_grid = clf_models[model_name]
        already_fitted = False
        if model_name == 'IFX-XGBoost':
            model = IFX_XGBoost(random_state=SEED, n_iterations=3)
            model.fit(X_train, y_train)
            already_fitted = True
        else:
            model = create_clf_pipeline(model, resampler='adasyn', scaler=True)
        metrics = evaluate_classifier_temporal(model, X_train, y_train, X_test, y_test,
                                               calibrate=CALIBRATE, already_fitted=already_fitted)
        metrics['test_season'] = test_df['season'].iloc[0]
        metrics['model'] = model_name
        results.append(metrics)
    return pd.DataFrame(results)

def run_temporal_prematch_regression(df, model_name):
    folds = get_temporal_folds(df, min_train_seasons=3, window_size=WINDOW_SIZE)
    results = []
    for train_mask, test_mask in folds:
        train_df = df[train_mask]
        test_df = df[test_mask]
        X_train, y_train = get_features_labels_prematch(train_df)
        X_test, y_test = get_features_labels_prematch(test_df)
        model, param_grid = reg_models[model_name]
        if model_name == 'IFX-XGBoost':
            model = IFX_XGBoost(random_state=SEED, n_iterations=3, objective='reg:squarederror')
            model.fit(X_train, y_train)
        else:
            model = create_reg_pipeline(model, scaler=True)
        metrics = evaluate_regressor_temporal(model, X_train, y_train, X_test, y_test)
        metrics['test_season'] = test_df['season'].iloc[0]
        metrics['model'] = model_name
        results.append(metrics)
    return pd.DataFrame(results)

# ===================== MAIN TEMPORAL LOOP (IN-PLAY) =====================
def run_temporal_inplay_classification(snaps_df, df_pre, model_name):
    folds = get_temporal_folds(df_pre, min_train_seasons=3, window_size=WINDOW_SIZE)
    results = []
    snap_feature_cols = [c for c in snaps_df.columns if c not in [
        'match_id', 'snapshot_time', 'final_goal_diff', 'final_result',
        'match_date', 'season', 'label_cls'
    ]]
    for train_mask, test_mask in folds:
        train_matches = df_pre[train_mask]['match_id'].values
        test_matches = df_pre[test_mask]['match_id'].values
        train_snaps = snaps_df[snaps_df['match_id'].isin(train_matches)]
        test_snaps = snaps_df[snaps_df['match_id'].isin(test_matches)]
        if len(train_snaps) < MIN_TRAIN_MATCHES or len(test_snaps) < MIN_TEST_MATCHES:
            continue
        X_train = train_snaps[snap_feature_cols].values
        y_train = train_snaps['label_cls'].values
        X_test = test_snaps[snap_feature_cols].values
        y_test = test_snaps['label_cls'].values

        already_fitted = False
        if model_name in ['KernelSVM', 'KernelRidge']:
            # Use Nystroem approximation for large data
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('kernel_approx', Nystroem(kernel='rbf', n_components=100, random_state=SEED)),
                ('clf', SGDClassifier(loss='log_loss', random_state=SEED, max_iter=1000, tol=1e-3))
            ])
            model = pipe
        else:
            model, param_grid = clf_models[model_name]
            if model_name == 'IFX-XGBoost':
                model = IFX_XGBoost(random_state=SEED, n_iterations=3)
                model.fit(X_train, y_train)
                already_fitted = True
            else:
                model = create_clf_pipeline(model, resampler='adasyn', scaler=True)
        metrics = evaluate_classifier_temporal(model, X_train, y_train, X_test, y_test,
                                               calibrate=CALIBRATE, already_fitted=already_fitted)
        metrics['test_season'] = df_pre[test_mask]['season'].iloc[0]
        metrics['model'] = model_name
        results.append(metrics)
    return pd.DataFrame(results)

def run_temporal_inplay_regression(snaps_df, df_pre, model_name):
    folds = get_temporal_folds(df_pre, min_train_seasons=3, window_size=WINDOW_SIZE)
    results = []
    snap_feature_cols = [c for c in snaps_df.columns if c not in [
        'match_id', 'snapshot_time', 'final_goal_diff', 'final_result',
        'match_date', 'season', 'label_cls'
    ]]
    
    for train_mask, test_mask in folds:
        train_matches = df_pre[train_mask]['match_id'].values
        test_matches = df_pre[test_mask]['match_id'].values
        train_snaps = snaps_df[snaps_df['match_id'].isin(train_matches)]
        test_snaps = snaps_df[snaps_df['match_id'].isin(test_matches)]
        if len(train_snaps) < MIN_TRAIN_MATCHES or len(test_snaps) < MIN_TEST_MATCHES:
            continue
        
        X_train = train_snaps[snap_feature_cols].values
        y_train = train_snaps['final_goal_diff'].values
        X_test = test_snaps[snap_feature_cols].values
        y_test = test_snaps['final_goal_diff'].values
        
        # *** IMPORTANT: For kernel models, use approximation for large data ***
        if model_name in ['KernelRidge', 'KernelSVR']:
            # Use Nystroem approximation + SGD regressor for speed
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('kernel_approx', Nystroem(kernel='rbf', n_components=100, random_state=SEED)),
                ('reg', SGDRegressor(random_state=SEED, max_iter=1000, tol=1e-3))
            ])
            model = pipe
        else:
            model, param_grid = reg_models[model_name]
            if model_name == 'IFX-XGBoost':
                model = IFX_XGBoost(random_state=SEED, n_iterations=3, objective='reg:squarederror')
                # IFX needs validation set, but we'll use a small split
                if len(X_train) >= 100:
                    val_size = max(10, int(len(X_train) * 0.1))
                    model.fit(X_train[:-val_size], y_train[:-val_size], X_train[-val_size:], y_train[-val_size:])
                else:
                    model.fit(X_train, y_train)
            else:
                model = create_reg_pipeline(model, scaler=True)
        
        # Evaluate
        if model_name == 'IFX-XGBoost':
            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            corr = np.corrcoef(y_test, preds)[0,1]
            metrics = {'mae': mae, 'rmse': rmse, 'corr': corr}
        else:
            metrics = evaluate_regressor_temporal(model, X_train, y_train, X_test, y_test)
        
        metrics['test_season'] = df_pre[test_mask]['season'].iloc[0]
        metrics['model'] = model_name
        results.append(metrics)
    
    return pd.DataFrame(results)

# ===================== MAIN =====================
def main():
    print("Loading data...")
    (train_pre, val_pre, test_pre), (train_snap, val_snap, test_snap) = load_original_data()
    full_df = load_full_data()
    full_snaps = load_full_snapshots()
    
    print("Full data shape:", full_df.shape)
    print("Seasons:", sorted(full_df['season'].unique()))
    
    # ===================== TEMPORAL EVALUATION: PRE-MATCH CLASSIFICATION =====================
    print("\n========== TEMPORAL WALK-FORWARD: PRE-MATCH CLASSIFICATION ==========\n")
    clf_results = []
    for model_name in clf_models.keys():
        print(f"Evaluating {model_name} ...")
        res_df = run_temporal_prematch_classification(full_df, model_name)
        clf_results.append(res_df)
    clf_all = pd.concat(clf_results, ignore_index=True)
    clf_all.to_csv(os.path.join(OUTPUT_DIR, 'temporal_prematch_classification.csv'), index=False)
    print("\nPre-match classification results (per season):")
    print(clf_all.round(4).to_string(index=False))
    
    # ===================== TEMPORAL EVALUATION: PRE-MATCH REGRESSION =====================
    print("\n========== TEMPORAL WALK-FORWARD: PRE-MATCH REGRESSION ==========\n")
    reg_results = []
    for model_name in reg_models.keys():
        print(f"Evaluating {model_name} ...")
        res_df = run_temporal_prematch_regression(full_df, model_name)
        reg_results.append(res_df)
    reg_all = pd.concat(reg_results, ignore_index=True)
    reg_all.to_csv(os.path.join(OUTPUT_DIR, 'temporal_prematch_regression.csv'), index=False)
    print("\nPre-match regression results (per season):")
    print(reg_all.round(4).to_string(index=False))
    
    # ===================== TEMPORAL EVALUATION: IN-PLAY CLASSIFICATION =====================
    print("\n========== TEMPORAL WALK-FORWARD: IN-PLAY CLASSIFICATION ==========\n")
    inplay_clf_results = []
    for model_name in clf_models.keys():
        print(f"Evaluating {model_name} ...")
        res_df = run_temporal_inplay_classification(full_snaps, full_df, model_name)
        inplay_clf_results.append(res_df)
    inplay_clf_all = pd.concat(inplay_clf_results, ignore_index=True)
    inplay_clf_all.to_csv(os.path.join(OUTPUT_DIR, 'temporal_inplay_classification.csv'), index=False)
    print("\nIn-play classification results (per season):")
    print(inplay_clf_all.round(4).to_string(index=False))
    
    # ===================== TEMPORAL EVALUATION: IN-PLAY REGRESSION =====================
    print("\n========== TEMPORAL WALK-FORWARD: IN-PLAY REGRESSION ==========\n")
    inplay_reg_results = []
    for model_name in reg_models.keys():
        print(f"Evaluating {model_name} ...")
        res_df = run_temporal_inplay_regression(full_snaps, full_df, model_name)
        inplay_reg_results.append(res_df)
    inplay_reg_all = pd.concat(inplay_reg_results, ignore_index=True)
    inplay_reg_all.to_csv(os.path.join(OUTPUT_DIR, 'temporal_inplay_regression.csv'), index=False)
    print("\nIn-play regression results (per season):")
    print(inplay_reg_all.round(4).to_string(index=False))
    
    # ===================== PLOTS =====================
    # Pre-match classification log-loss per model
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=clf_all, x='test_season', y='log_loss', hue='model', marker='o')
    plt.xlabel('Test Season')
    plt.ylabel('Log-Loss')
    plt.title('Temporal Pre-Match Classification: Log-Loss per Season')
    plt.legend(title='Model')
    plt.grid(True)
    plt.ylim(0, 2)
    plt.savefig(os.path.join(FIGURES_DIR, 'temporal_prematch_classification_logloss.png'))
    plt.close()
    
    # Pre-match regression MAE
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=reg_all, x='test_season', y='mae', hue='model', marker='o')
    plt.xlabel('Test Season')
    plt.ylabel('MAE')
    plt.title('Temporal Pre-Match Regression: MAE per Season')
    plt.legend(title='Model')
    plt.grid(True)
    plt.savefig(os.path.join(FIGURES_DIR, 'temporal_prematch_regression_mae.png'))
    plt.close()
    
    # In-play classification log-loss
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=inplay_clf_all, x='test_season', y='log_loss', hue='model', marker='o')
    plt.xlabel('Test Season')
    plt.ylabel('Log-Loss')
    plt.title('Temporal In-Play Classification: Log-Loss per Season')
    plt.legend(title='Model')
    plt.grid(True)
    plt.ylim(0, 2)
    plt.savefig(os.path.join(FIGURES_DIR, 'temporal_inplay_classification_logloss.png'))
    plt.close()
    
    # In-play regression MAE
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=inplay_reg_all, x='test_season', y='mae', hue='model', marker='o')
    plt.xlabel('Test Season')
    plt.ylabel('MAE')
    plt.title('Temporal In-Play Regression: MAE per Season')
    plt.legend(title='Model')
    plt.grid(True)
    plt.savefig(os.path.join(FIGURES_DIR, 'temporal_inplay_regression_mae.png'))
    plt.close()
    
    print("Plots saved.")
    
    # ===================== KERNEL SCALING (using full pre-match features) =====================
    print("\n========== KERNEL SCALING ==========\n")
    # Use the first fold for training data
    folds = get_temporal_folds(full_df, min_train_seasons=3, window_size=WINDOW_SIZE)
    if folds:
        train_mask, _ = folds[0]
        X_temp, y_temp = get_features_labels_prematch(full_df[train_mask])
        subsample_sizes = [100, 500, 1000, 2000, 5000]
        kernel_times = []
        for n in subsample_sizes:
            if n > len(X_temp):
                break
            X_sub = X_temp[:n]
            y_sub = y_temp[:n]
            model = KernelRidge(alpha=1.0, kernel='rbf')
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024**2
            start = time.time()
            model.fit(X_sub, y_sub)
            end = time.time()
            mem_after = process.memory_info().rss / 1024**2
            kernel_times.append((end-start, mem_after - mem_before))
            gc.collect()
        plt.figure(figsize=(10, 6))
        plt.plot(subsample_sizes[:len(kernel_times)], [t for t,_ in kernel_times], marker='o')
        plt.xlabel('Training sample size')
        plt.ylabel('Time (seconds)')
        plt.title('Kernel Ridge scaling (O(n^2))')
        plt.savefig(os.path.join(FIGURES_DIR, 'kernel_scaling.png'))
        plt.close()
        print("Kernel scaling plot saved.")
    
    # ===================== SHAP ANALYSIS ON LAST FOLD =====================
    print("\n========== SHAP ANALYSIS ON LAST FOLD ==========\n")
    if folds:
        train_mask, test_mask = folds[-1]
        X_train, y_train = get_features_labels_prematch(full_df[train_mask])
        X_test, y_test = get_features_labels_prematch(full_df[test_mask])
        feature_cols = [c for c in full_df.columns if c not in [
            'match_id', 'match_date', 'home_team', 'away_team',
            'label_goal_diff', 'label_result', 'result', 'season',
            'competition_id', 'label'
        ]]
        shap_model = XGBClassifier(random_state=SEED, n_estimators=100, learning_rate=0.1,
                                   max_depth=5, eval_metric='mlogloss', verbosity=0)
        shap_model.fit(X_train, y_train)
        explainer = shap.TreeExplainer(shap_model)
        shap_values = explainer.shap_values(X_test[:100])
        shap.summary_plot(shap_values, X_test[:100], feature_names=feature_cols, show=False)
        plt.savefig(os.path.join(FIGURES_DIR, 'temporal_shap_summary.png'))
        plt.close()
        pred_class = np.argmax(shap_model.predict_proba(X_test[:1])[0])
        # Robust extraction for multi-class SHAP
        if isinstance(shap_values, list):
            class_shap = shap_values[pred_class][0]
            base = explainer.expected_value[pred_class]
        else:
            class_shap = shap_values[0, :, pred_class]
            base = explainer.expected_value[pred_class] if isinstance(explainer.expected_value, list) else explainer.expected_value
        explanation = shap.Explanation(
            values=class_shap,
            base_values=base,
            data=X_test[0],
            feature_names=feature_cols
        )
        shap.plots.waterfall(explanation, show=False)
        plt.savefig(os.path.join(FIGURES_DIR, 'temporal_shap_waterfall.png'))
        plt.close()
    
    # ===================== RESAMPLING COMPARISON ON LAST FOLD =====================
    print("\n========== RESAMPLING COMPARISON ON LAST FOLD ==========\n")
    if folds:
        train_mask, test_mask = folds[-1]
        X_train, y_train = get_features_labels_prematch(full_df[train_mask])
        X_test, y_test = get_features_labels_prematch(full_df[test_mask])
        from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
        resamplers = {
            'None': None,
            'PF-SMOTE': PF_SMOTE(random_state=SEED),
            'SMOTE': SMOTE(random_state=SEED),
            'BorderlineSMOTE': BorderlineSMOTE(random_state=SEED),
            'ADASYN': ADASYN(random_state=SEED),
        }
        resample_results = []
        base_model = XGBClassifier(random_state=SEED, n_estimators=100, learning_rate=0.1)
        for name, resampler in resamplers.items():
            steps = [('scaler', StandardScaler())]
            if resampler is not None:
                steps.append(('resampler', resampler))
            steps.append(('clf', base_model))
            pipe = ImbPipeline(steps)
            pipe.fit(X_train, y_train)
            probs = pipe.predict_proba(X_test)
            ll = log_loss(y_test, probs)
            rps = np.mean([np.sum((np.cumsum(probs[i,:])[:-1] - np.where(np.arange(3) < y_test[i], 1, 0)[:-1])**2) for i in range(len(y_test))])
            acc = accuracy_score(y_test, np.argmax(probs, axis=1))
            resample_results.append({'resampler': name, 'log_loss': ll, 'rps': rps, 'accuracy': acc})
        resample_df = pd.DataFrame(resample_results)
        resample_df.to_csv(os.path.join(OUTPUT_DIR, 'resampling_comparison.csv'), index=False)
        print(resample_df.round(4).to_string(index=False))
    
    # ===================== EXPECTED POINTS PER TEAM (TEMPORAL) =====================
    print("\n========== EXPECTED POINTS PER TEAM (ALL SEASONS) ==========\n")
    teams = pd.unique(full_df[['home_team','away_team']].values.ravel())
    ep_rows = []
    seasons = sorted(full_df['season'].unique())
    for team in teams:
        team_df = full_df[(full_df['home_team']==team) | (full_df['away_team']==team)].copy()
        total_pred = 0; total_actual = 0; n_seasons = 0
        for season in seasons:
            train_mask = full_df['season'] < season
            if train_mask.sum() < MIN_TRAIN_MATCHES:
                continue
            test_mask = (team_df['season']==season)
            if test_mask.sum() < MIN_TEST_MATCHES:
                continue
            X_train, y_train = get_features_labels_prematch(full_df[train_mask])
            model = XGBClassifier(random_state=SEED, n_estimators=100, learning_rate=0.1)
            model.fit(X_train, y_train)
            X_test, y_test = get_features_labels_prematch(team_df[test_mask])
            probs = model.predict_proba(X_test)
            home_flags = (team_df[test_mask]['home_team']==team).values
            exp_points = np.zeros(len(y_test))
            for i in range(len(y_test)):
                if home_flags[i]:
                    exp_points[i] = 3*probs[i,0] + 1*probs[i,1]
                else:
                    exp_points[i] = 3*probs[i,2] + 1*probs[i,1]
            actual_points = np.zeros(len(y_test))
            for i, row in enumerate(team_df[test_mask].itertuples()):
                if row.home_team == team:
                    if row.result == 'H': actual_points[i] = 3
                    elif row.result == 'D': actual_points[i] = 1
                else:
                    if row.result == 'A': actual_points[i] = 3
                    elif row.result == 'D': actual_points[i] = 1
            total_pred += exp_points.sum(); total_actual += actual_points.sum(); n_seasons += 1
        if n_seasons > 0:
            ep_rows.append({'team': team, 'n_seasons': n_seasons,
                            'predicted_total_points': total_pred, 'actual_total_points': total_actual,
                            'difference': total_pred - total_actual})
    ep_df = pd.DataFrame(ep_rows)
    if not ep_df.empty:
        ep_df = ep_df.sort_values('difference', ascending=False).head(10)
        ep_df.to_csv(os.path.join(OUTPUT_DIR, 'team_expected_points.csv'), index=False)
        plt.figure(figsize=(10,6))
        sns.barplot(x='difference', y='team', data=ep_df)
        plt.xlabel('Difference (Predicted - Actual Points)'); plt.ylabel('Team')
        plt.title('Expected Points Difference by Team (Top 10)')
        plt.savefig(os.path.join(FIGURES_DIR, 'expected_points_difference.png'))
        plt.close()
        print("Expected points saved.")
        print(ep_df.round(2).to_string(index=False))
    
    # ===================== HIGH-RISK BRIER (TEMPORAL) =====================
    print("\n========== HIGH-RISK BRIER (ALL SEASONS) ==========\n")
    hr_rows = []
    for season in seasons:
        train_mask = full_df['season'] < season
        test_mask = full_df['season'] == season
        if train_mask.sum() < MIN_TRAIN_MATCHES or test_mask.sum() < MIN_TEST_MATCHES:
            continue
        X_train, y_train = get_features_labels_prematch(full_df[train_mask])
        X_test, y_test = get_features_labels_prematch(full_df[test_mask])
        model = XGBClassifier(random_state=SEED, n_estimators=100, learning_rate=0.1)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)
        true_probs = probs[np.arange(len(y_test)), y_test]
        risk_mask = true_probs < HIGH_RISK_THRESHOLD
        if risk_mask.sum() > 0:
            brier = np.mean([brier_score_loss((y_test[risk_mask]==c).astype(int), probs[risk_mask, c]) for c in range(3)])
        else:
            brier = np.nan
        hr_rows.append({'season': season, 'high_risk_count': risk_mask.sum(), 'brier_high_risk': brier})
    hr_df = pd.DataFrame(hr_rows)
    hr_df.to_csv(os.path.join(OUTPUT_DIR, 'high_risk_brier.csv'), index=False)
    plt.figure(figsize=(10,6))
    plt.plot(hr_df['season'], hr_df['brier_high_risk'], marker='o')
    plt.xlabel('Season'); plt.ylabel('Brier (High-risk)')
    plt.title('High-Risk Brier over Seasons')
    plt.grid(True); plt.ylim(0,1)
    plt.savefig(os.path.join(FIGURES_DIR, 'high_risk_brier.png'))
    plt.close()
    print("High-risk Brier saved.")
    print(hr_df.round(4).to_string(index=False))
    
    # ===================== TRANSFER LEARNING (COMBINED BASE -> FINE-TUNE) =====================
    print("\n========== TRANSFER LEARNING ==========\n")
    # Train base on all data (first 80%), fine-tune on each league's last 20%
    combined = full_df.sort_values('match_date')
    n_combined = len(combined)
    combined_train = combined.iloc[:int(n_combined*0.8)]
    X_base, y_base = get_features_labels_prematch(combined_train)
    base_model = XGBClassifier(random_state=SEED, n_estimators=100, learning_rate=0.1)
    base_model.fit(X_base, y_base)
    transfer_results = []
    for comp_id, league_name in [(11,'La Liga'), (2,'Premier League')]:
        league_df = full_df[full_df['competition_id']==comp_id].copy()
        if len(league_df) < 100:
            continue
        league_df = league_df.sort_values('match_date')
        n = len(league_df)
        league_train = league_df.iloc[:int(n*0.8)]
        league_test = league_df.iloc[int(n*0.8):]
        if len(league_train) < MIN_TRAIN_MATCHES or len(league_test) < MIN_TEST_MATCHES:
            continue
        X_ft, y_ft = get_features_labels_prematch(league_train)
        fine_model = XGBClassifier(random_state=SEED, n_estimators=50, learning_rate=0.05, max_depth=5)
        fine_model.fit(X_ft, y_ft, xgb_model=base_model.get_booster())
        X_test, y_test = get_features_labels_prematch(league_test)
        probs_fine = fine_model.predict_proba(X_test)
        metrics_fine = evaluate_classifier_temporal(fine_model, X_ft, y_ft, X_test, y_test, calibrate=False)
        transfer_results.append({'league': league_name, 'method': 'Transfer (base combined + fine-tune)', **metrics_fine})
        probs_base = base_model.predict_proba(X_test)
        metrics_base = evaluate_classifier_temporal(base_model, X_base, y_base, X_test, y_test, calibrate=False)
        transfer_results.append({'league': league_name, 'method': 'Base combined (no fine-tune)', **metrics_base})
    transfer_df = pd.DataFrame(transfer_results)
    transfer_df.to_csv(os.path.join(OUTPUT_DIR, 'transfer_learning.csv'), index=False)
    plt.figure(figsize=(10,6))
    sns.barplot(x='league', y='accuracy', hue='method', data=transfer_df)
    plt.savefig(os.path.join(FIGURES_DIR, 'transfer_learning_accuracy.png'))
    plt.close()
    print("Transfer learning results saved.")
    print(transfer_df.round(4).to_string(index=False))
    
    # ===================== SEASONAL ENSEMBLE (AVERAGE OVER WINDOWS) =====================
    print("\n========== SEASONAL ENSEMBLE ==========\n")
    ensemble_results = []
    for season in seasons:
        prior = [s for s in seasons if s < season]
        if len(prior) < 3:
            continue
        models = []
        for w in ENSEMBLE_WINDOWS:
            if w is None:
                train_mask = full_df['season'] < season
            else:
                if len(prior) < w:
                    continue
                train_seasons = prior[-w:]
                train_mask = full_df['season'].isin(train_seasons)
            if train_mask.sum() < MIN_TRAIN_MATCHES:
                continue
            X_train, y_train = get_features_labels_prematch(full_df[train_mask])
            model = XGBClassifier(random_state=SEED, n_estimators=100, learning_rate=0.1)
            model.fit(X_train, y_train)
            models.append(model)
        if not models:
            continue
        test_mask = full_df['season'] == season
        if test_mask.sum() < MIN_TEST_MATCHES:
            continue
        X_test, y_test = get_features_labels_prematch(full_df[test_mask])
        prob_sum = None
        for model in models:
            probs = model.predict_proba(X_test)
            if prob_sum is None:
                prob_sum = probs
            else:
                prob_sum += probs
        probs = prob_sum / len(models)
        # Compute metrics manually (do NOT call evaluate_classifier_temporal with None)
        ll = log_loss(y_test, probs) if len(set(y_test)) > 1 else np.nan
        acc = accuracy_score(y_test, np.argmax(probs, axis=1))
        rps = np.mean([np.sum((np.cumsum(probs[i,:])[:-1] - np.where(np.arange(3) < y_test[i], 1, 0)[:-1])**2) for i in range(len(y_test))])
        brier = np.mean([brier_score_loss((y_test==c).astype(int), probs[:,c]) for c in range(3)])
        ensemble_results.append({'season': season, 'window': 'ensemble', 'log_loss': ll, 'accuracy': acc, 'rps': rps, 'brier': brier})
    ensemble_df = pd.DataFrame(ensemble_results)
    ensemble_df.to_csv(os.path.join(OUTPUT_DIR, 'seasonal_metrics_ensemble.csv'), index=False)
    all_df = pd.DataFrame()
    window_df = pd.DataFrame()
    for season in seasons:
        prior = [s for s in seasons if s < season]
        if not prior:
            continue
        # all past
        train_mask = full_df['season'] < season
        if train_mask.sum() >= MIN_TRAIN_MATCHES:
            X_train, y_train = get_features_labels_prematch(full_df[train_mask])
            X_test, y_test = get_features_labels_prematch(full_df[full_df['season']==season])
            if len(y_test) >= MIN_TEST_MATCHES:
                model = XGBClassifier(random_state=SEED, n_estimators=100, learning_rate=0.1)
                model.fit(X_train, y_train)
                probs = model.predict_proba(X_test)
                ll = log_loss(y_test, probs)
                acc = accuracy_score(y_test, np.argmax(probs, axis=1))
                rps = np.mean([np.sum((np.cumsum(probs[i,:])[:-1] - np.where(np.arange(3) < y_test[i], 1, 0)[:-1])**2) for i in range(len(y_test))])
                brier = np.mean([brier_score_loss((y_test==c).astype(int), probs[:,c]) for c in range(3)])
                all_df = pd.concat([all_df, pd.DataFrame([{'season':season, 'window':'all', 'log_loss':ll, 'accuracy':acc, 'rps':rps, 'brier':brier}])], ignore_index=True)
        # window 5
        if len(prior) >= 5:
            train_seasons = prior[-5:]
            train_mask = full_df['season'].isin(train_seasons)
            if train_mask.sum() >= MIN_TRAIN_MATCHES:
                X_train, y_train = get_features_labels_prematch(full_df[train_mask])
                X_test, y_test = get_features_labels_prematch(full_df[full_df['season']==season])
                if len(y_test) >= MIN_TEST_MATCHES:
                    model = XGBClassifier(random_state=SEED, n_estimators=100, learning_rate=0.1)
                    model.fit(X_train, y_train)
                    probs = model.predict_proba(X_test)
                    ll = log_loss(y_test, probs)
                    acc = accuracy_score(y_test, np.argmax(probs, axis=1))
                    rps = np.mean([np.sum((np.cumsum(probs[i,:])[:-1] - np.where(np.arange(3) < y_test[i], 1, 0)[:-1])**2) for i in range(len(y_test))])
                    brier = np.mean([brier_score_loss((y_test==c).astype(int), probs[:,c]) for c in range(3)])
                    window_df = pd.concat([window_df, pd.DataFrame([{'season':season, 'window':'window5', 'log_loss':ll, 'accuracy':acc, 'rps':rps, 'brier':brier}])], ignore_index=True)
    all_df.to_csv(os.path.join(OUTPUT_DIR, 'seasonal_metrics_all.csv'), index=False)
    window_df.to_csv(os.path.join(OUTPUT_DIR, 'seasonal_metrics_window.csv'), index=False)
    # Plot comparison
    plt.figure(figsize=(12,6))
    if not all_df.empty:
        all_df = all_df.sort_values('season')
        plt.plot(all_df['season'], all_df['log_loss'], marker='o', linestyle='--', label='All past')
    if not window_df.empty:
        window_df = window_df.sort_values('season')
        plt.plot(window_df['season'], window_df['log_loss'], marker='s', linestyle=':', label='Window=5')
    if not ensemble_df.empty:
        ensemble_df = ensemble_df.sort_values('season')
        plt.plot(ensemble_df['season'], ensemble_df['log_loss'], marker='^', label='Ensemble')
    plt.xlabel('Season'); plt.ylabel('Log-Loss')
    plt.title('Seasonal Metrics: All vs Window vs Ensemble')
    plt.legend(); plt.grid(True); plt.ylim(0,2)
    plt.savefig(os.path.join(FIGURES_DIR, 'seasonal_comparison_logloss.png'))
    plt.close()
    print("Seasonal ensemble results saved.")
    print(ensemble_df.round(4).to_string(index=False))
    
    # ===================== SAVE IN-PLAY MODELS FOR API (LAST FOLD) =====================
    print("\n========== SAVING IN-PLAY MODELS FOR API ==========\n")
    if folds:
        train_mask, test_mask = folds[-1]
        train_matches = full_df[train_mask]['match_id'].values
        test_matches = full_df[test_mask]['match_id'].values
        train_snaps = full_snaps[full_snaps['match_id'].isin(train_matches)]
        test_snaps = full_snaps[full_snaps['match_id'].isin(test_matches)]
        if len(train_snaps) > 0 and len(test_snaps) > 0:
            # In-play feature columns (exactly the ones used during training)
            snap_feature_cols = [c for c in train_snaps.columns if c not in [
                'match_id', 'snapshot_time', 'final_goal_diff', 'final_result',
                'match_date', 'season', 'label_cls'
            ]]
            X_train_snap = train_snaps[snap_feature_cols].values
            y_train_snap_cls = train_snaps['label_cls'].values
            y_train_snap_reg = train_snaps['final_goal_diff'].values

            # Classification pipeline with ADASYN + scaler
            clf_pipe = create_clf_pipeline(
                XGBClassifier(random_state=SEED, eval_metric='mlogloss'),
                resampler='adasyn', scaler=True
            )
            clf_pipe.fit(X_train_snap, y_train_snap_cls)

            # Regression pipeline with scaler
            reg_pipe = create_reg_pipeline(
                XGBRegressor(random_state=SEED), scaler=True
            )
            reg_pipe.fit(X_train_snap, y_train_snap_reg)

            # Save models
            import joblib
            joblib.dump(clf_pipe, os.path.join(OUTPUT_DIR, 'best_inplay_clf.pkl'))
            joblib.dump(reg_pipe, os.path.join(OUTPUT_DIR, 'best_inplay_reg.pkl'))

            # Save test snapshots for the app
            test_snaps.to_csv(os.path.join(OUTPUT_DIR, 'test_snapshots_for_app.csv'), index=False)
            full_df[test_mask].to_csv(os.path.join(OUTPUT_DIR, 'test_prematch_for_app.csv'), index=False)

            print("In-play models and test data saved.")
        else:
            print("Not enough in-play data for last fold.")
    else:
        print("No folds available.")
    
        # ===================== WORST PREDICTIONS ANALYSIS (LAST FOLD, PRE-MATCH) =====================
    print("\n========== WORST PREDICTIONS ANALYSIS (LAST FOLD, PRE-MATCH) ==========\n")
    if folds:
        train_mask, test_mask = folds[-1]
        X_train, y_train = get_features_labels_prematch(full_df[train_mask])
        X_test, y_test = get_features_labels_prematch(full_df[test_mask])
        feature_cols = [c for c in full_df.columns if c not in [
            'match_id', 'match_date', 'home_team', 'away_team',
            'label_goal_diff', 'label_result', 'result', 'season',
            'competition_id', 'label'
        ]]
        worst_model = XGBClassifier(random_state=SEED, n_estimators=100, learning_rate=0.1,
                                     max_depth=5, eval_metric='mlogloss', verbosity=0)
        worst_model.fit(X_train, y_train)
        probs = worst_model.predict_proba(X_test)
        per_sample_ll = -np.log(probs[np.arange(len(y_test)), y_test])
        worst_idx = np.argsort(per_sample_ll)[-10:]
        explainer_worst = shap.TreeExplainer(worst_model)
        shap_values_worst = explainer_worst.shap_values(X_test[worst_idx])
        for i, idx in enumerate(worst_idx):
            pred_class = np.argmax(probs[idx])
            true_label = y_test[idx]
            if isinstance(shap_values_worst, list):
                class_shap = shap_values_worst[pred_class][i]
                base = explainer_worst.expected_value[pred_class]
            else:
                class_shap = shap_values_worst[i, :, pred_class]
                base = explainer_worst.expected_value[pred_class] if isinstance(explainer_worst.expected_value, list) else explainer_worst.expected_value
            explanation = shap.Explanation(values=class_shap, base_values=base,
                                           data=X_test[idx], feature_names=feature_cols)
            plt.figure(figsize=(8,5))
            shap.plots.waterfall(explanation, show=False)
            plt.title(f"Worst Sample {idx} (true={true_label}, pred={pred_class}, LL={per_sample_ll[idx]:.4f})")
            plt.savefig(os.path.join(FIGURES_DIR, f'worst_{i+1}_prematch.png'))
            plt.close()
        print("Worst predictions analysis saved.")

    # ===================== SHAP TIMELINE (LAST FOLD, ONE MATCH) =====================
    print("\n========== SHAP TIMELINE (LAST FOLD, ONE MATCH) ==========\n")
    if folds:
        train_mask, test_mask = folds[-1]
        test_matches = full_df[test_mask]['match_id'].values
        if len(test_matches) > 0:
            chosen_match = test_matches[0]
            # Load snapshots for this match from full_snaps
            match_snaps = full_snaps[full_snaps['match_id'] == chosen_match].sort_values('snapshot_time')
            if len(match_snaps) >= 2:
                snap_feature_cols = [c for c in match_snaps.columns if c not in [
                    'match_id', 'snapshot_time', 'final_goal_diff', 'final_result',
                    'match_date', 'season', 'label_cls'
                ]]
                X_snap = match_snaps[snap_feature_cols].values
                y_snap = match_snaps['label_cls'].values
                # Train an in-play model on all training snapshots (before last fold)
                train_matches = full_df[train_mask]['match_id'].values
                train_snaps = full_snaps[full_snaps['match_id'].isin(train_matches)]
                if len(train_snaps) > 0:
                    X_train_snap = train_snaps[snap_feature_cols].values
                    y_train_snap = train_snaps['label_cls'].values
                    timeline_model = XGBClassifier(random_state=SEED, n_estimators=100, learning_rate=0.1,
                                                   max_depth=5, eval_metric='mlogloss', verbosity=0)
                    timeline_model.fit(X_train_snap, y_train_snap)
                    explainer_timeline = shap.TreeExplainer(timeline_model)
                    shap_values_timeline = explainer_timeline.shap_values(X_snap)
                    # Get predicted class for each snapshot
                    pred_classes = timeline_model.predict(X_snap)
                    # Extract SHAP for predicted class
                    shap_class_vals = []
                    for i, cls in enumerate(pred_classes):
                        if isinstance(shap_values_timeline, list):
                            shap_class_vals.append(shap_values_timeline[cls][i])
                        elif hasattr(shap_values_timeline, 'ndim') and shap_values_timeline.ndim == 3:
                            shap_class_vals.append(shap_values_timeline[i, :, cls])
                        else:
                            shap_class_vals.append(shap_values_timeline[i])
                    shap_class_vals = np.array(shap_class_vals)
                    # Plot top 5 features over time
                    mean_abs_shap = np.mean(np.abs(shap_class_vals), axis=0)
                    top_idx = np.argsort(mean_abs_shap)[-5:]
                    plt.figure(figsize=(12,6))
                    for idx in top_idx:
                        plt.plot(match_snaps['snapshot_time'], shap_class_vals[:, idx], marker='o', label=snap_feature_cols[idx])
                    plt.xlabel('Match minute')
                    plt.ylabel('SHAP value (contribution to predicted class)')
                    plt.title(f'SHAP Timeline for Match {chosen_match}')
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(os.path.join(FIGURES_DIR, 'temporal_shap_timeline.png'))
                    plt.close()
                    print("SHAP timeline saved.")

    # ===================== RELIABILITY DIAGRAMS WITH CALIBRATION COMPARISON =====================
    print("\n========== RELIABILITY DIAGRAMS WITH CALIBRATION COMPARISON ==========\n")
    if folds:
        train_mask, test_mask = folds[-1]
        X_train, y_train = get_features_labels_prematch(full_df[train_mask])
        X_test, y_test = get_features_labels_prematch(full_df[test_mask])
        feature_cols = [c for c in full_df.columns if c not in [
            'match_id', 'match_date', 'home_team', 'away_team',
            'label_goal_diff', 'label_result', 'result', 'season',
            'competition_id', 'label'
        ]]
        
        # Test calibration methods
        methods = ['none', 'platt', 'isotonic']
        cal_results = []
        for method in methods:
            model_tmp = XGBClassifier(random_state=SEED, n_estimators=100, learning_rate=0.1,
                                      max_depth=5, eval_metric='mlogloss', verbosity=0)
            metrics = evaluate_classifier_temporal(model_tmp, X_train, y_train, X_test, y_test,
                                                   calibrate=True, calib_method=method)
            metrics['calibration'] = method
            cal_results.append(metrics)
        cal_df = pd.DataFrame(cal_results)
        cal_df.to_csv(os.path.join(OUTPUT_DIR, 'calibration_comparison.csv'), index=False)
        print("Calibration comparison:")
        print(cal_df.round(4).to_string(index=False))
        
        # Choose best method based on Brier (or log-loss)
        best_method = cal_df.loc[cal_df['brier'].idxmin(), 'calibration']
        if best_method == 'none':
            best_method = 'platt'  # fallback
        
        # Use best method to plot reliability
        model_tmp = XGBClassifier(random_state=SEED, n_estimators=100, learning_rate=0.1,
                                  max_depth=5, eval_metric='mlogloss', verbosity=0)
        # Inner split
        n_train = len(X_train)
        cal_size = max(10, int(n_train * 0.2))
        X_inner_train = X_train[:n_train-cal_size]
        y_inner_train = y_train[:n_train-cal_size]
        X_cal = X_train[n_train-cal_size:]
        y_cal = y_train[n_train-cal_size:]
        model_tmp.fit(X_inner_train, y_inner_train)
        # Compute probabilities on calibration set (for isotonic fitting)
        probs_cal = model_tmp.predict_proba(X_cal)
        # Get test probabilities (raw)
        probs_uncal_test = model_tmp.predict_proba(X_test)
        
        if best_method == 'platt':
            calibrator = CalibratedClassifierCV(estimator=model_tmp, method='sigmoid', cv='prefit')
            calibrator.fit(X_cal, y_cal)
            probs = calibrator.predict_proba(X_test)
        elif best_method == 'isotonic':
            n_classes = probs_uncal_test.shape[1]
            probs = np.zeros_like(probs_uncal_test)
            for c in range(n_classes):
                iso = IsotonicRegression(out_of_bounds='clip')
                iso.fit(probs_cal[:, c], (y_cal == c).astype(int))
                probs[:, c] = iso.predict(probs_uncal_test[:, c])
            probs = probs / probs.sum(axis=1, keepdims=True)
        else:
            probs = probs_uncal_test
        
        # Plot reliability
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for c in range(3):
            from sklearn.calibration import calibration_curve
            frac_pos, mean_pred = calibration_curve((y_test == c).astype(int), probs[:, c], n_bins=10)
            axes[c].plot(mean_pred, frac_pos, 's-', label='Model')
            axes[c].plot([0,1], [0,1], 'k--', label='Perfect')
            axes[c].set_xlabel('Mean predicted probability')
            axes[c].set_ylabel('Fraction of positives')
            axes[c].set_title(f'Class {c} (Calib: {best_method})')
            axes[c].legend()
        plt.suptitle(f'Reliability Diagram – Pre-match Model (Last Fold, {best_method} calibration)')
        plt.savefig(os.path.join(FIGURES_DIR, 'temporal_reliability_prematch_' + best_method + '.png'))
        plt.close()
        print(f"Reliability diagram saved with {best_method} calibration.")

    # ===================== KERNEL SCALING (USING IN-PLAY DATA) =====================
    print("\n========== KERNEL SCALING (IN-PLAY DATA) ==========\n")
    if not full_snaps.empty:
        snap_feat_cols = [c for c in full_snaps.columns if c not in [
            'match_id', 'snapshot_time', 'final_goal_diff', 'final_result',
            'match_date', 'season', 'label_cls'
        ]]
        X_snap_temp = full_snaps[snap_feat_cols].values
        y_snap_temp = full_snaps['label_cls'].values
        max_n = len(X_snap_temp)
        
        subsample_sizes = [100, 500, 1000, 2000, 5000, 10000]
        subsample_sizes = [s for s in subsample_sizes if s < max_n]
        if len(subsample_sizes) < 3:
            subsample_sizes = [100, 500, 1000, 2000, 4000]
            subsample_sizes = [s for s in subsample_sizes if s < max_n]
        
        if len(subsample_sizes) >= 2:
            kernel_times = []
            approx_times = []
            for n in subsample_sizes:
                X_sub = X_snap_temp[:n]
                y_sub = y_snap_temp[:n]
                
                # Exact KernelRidge
                model = KernelRidge(alpha=1.0, kernel='rbf')
                start = time.time()
                model.fit(X_sub, y_sub)
                end = time.time()
                kernel_times.append((end-start, 0))
                
                # Approximate (Nystroem + Ridge)
                pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('kernel_approx', Nystroem(kernel='rbf', n_components=100, random_state=SEED)),
                    ('reg', Ridge(alpha=1.0))
                ])
                start = time.time()
                pipe.fit(X_sub, y_sub)
                end = time.time()
                approx_times.append((end-start, 0))
                gc.collect()
            
            # Plot
            plt.figure(figsize=(12, 7))
            plt.plot(subsample_sizes, [t for t,_ in kernel_times], marker='o', label='Exact KernelRidge')
            plt.plot(subsample_sizes, [t for t,_ in approx_times], marker='s', label='Approx (Nystroem+Ridge)')
            plt.xlabel('Training sample size')
            plt.ylabel('Time (seconds)')
            plt.title('Kernel Scaling: Exact vs Approximate (In-Play Data)')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(FIGURES_DIR, 'kernel_scaling_comparison_inplay.png'))
            plt.close()
            print("Kernel scaling plot (in-play) saved.")
        else:
            print("Not enough data for kernel scaling.")
    else:
        print("No in-play data available for kernel scaling.")

        # ===================== MULTI-STEP ROLLING EVALUATION (1, 2, 3 SEASONS AHEAD) =====================
    print("\n========== MULTI-STEP ROLLING EVALUATION (1,2,3 SEASONS AHEAD) ==========\n")
    # For each possible cutoff season, train on all previous seasons, then test on cutoff+1, cutoff+2, cutoff+3
    multi_results = []
    seasons = sorted(full_df['season'].unique())
    # Find minimum training season
    min_train_season = None
    for s in seasons:
        if (full_df['season'] < s).sum() >= MIN_TRAIN_MATCHES:
            min_train_season = s
            break
    if min_train_season is not None:
        for cutoff_season in seasons:
            if cutoff_season <= min_train_season:
                continue
            train_mask = full_df['season'] < cutoff_season
            if train_mask.sum() < MIN_TRAIN_MATCHES:
                continue
            X_train, y_train = get_features_labels_prematch(full_df[train_mask])
            for horizon in [1, 2, 3]:
                test_season = cutoff_season + horizon - 1
                if test_season not in seasons:
                    continue
                test_mask = full_df['season'] == test_season
                if test_mask.sum() < MIN_TEST_MATCHES:
                    continue
                X_test, y_test = get_features_labels_prematch(full_df[test_mask])
                # Train best model (XGBoost with ADASYN)
                model = create_clf_pipeline(XGBClassifier(random_state=SEED, eval_metric='mlogloss'),
                                             resampler='adasyn', scaler=True)
                model.fit(X_train, y_train)
                probs = model.predict_proba(X_test)
                # Compute metrics
                ll = log_loss(y_test, probs)
                acc = accuracy_score(y_test, np.argmax(probs, axis=1))
                rps = np.mean([np.sum((np.cumsum(probs[i,:])[:-1] - np.where(np.arange(3) < y_test[i], 1, 0)[:-1])**2) for i in range(len(y_test))])
                brier = np.mean([brier_score_loss((y_test==c).astype(int), probs[:,c]) for c in range(3)])
                multi_results.append({
                    'train_up_to': cutoff_season-1,
                    'test_season': test_season,
                    'horizon': horizon,
                    'log_loss': ll,
                    'accuracy': acc,
                    'rps': rps,
                    'brier': brier
                })
        multi_df = pd.DataFrame(multi_results)
        multi_df.to_csv(os.path.join(OUTPUT_DIR, 'multi_step_rolling.csv'), index=False)
        # Plot
        plt.figure(figsize=(12,6))
        for h in [1,2,3]:
            data = multi_df[multi_df['horizon']==h]
            plt.plot(data['test_season'], data['log_loss'], marker='o', label=f'Horizon {h}')
        plt.xlabel('Test Season')
        plt.ylabel('Log-Loss')
        plt.title('Multi-Step Rolling: Log-Loss (if model not updated)')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 2)
        plt.savefig(os.path.join(FIGURES_DIR, 'multi_step_rolling_logloss.png'))
        plt.close()
        print("Multi-step rolling results saved.")
        print(multi_df.round(4).to_string(index=False))

    # Final message
    print("\nALL DONE!")
    
if __name__ == '__main__':
    main()