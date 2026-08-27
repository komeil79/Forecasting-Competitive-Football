import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss, mean_absolute_error
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ===================== CONFIGURATION =====================
DATA_DIR = 'processed_data'
PREMATCH_FILE = os.path.join(DATA_DIR, 'full_prematch.csv')
MATCHES_FILE = os.path.join(DATA_DIR, 'matches_full.csv')
OUTPUT_DIR = 'out'
FIGURES_DIR = 'figures'
SEED = 42
MIN_TRAIN_MATCHES = 50
MIN_TEST_MATCHES = 5
HIGH_RISK_THRESHOLD = 0.3
ENSEMBLE_WINDOWS = [None, 5, 3]
EXPECTED_POINTS_MIN_SEASONS = 1
# =========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------- Data Loading ----------
def load_data():
    """Load and merge pre-match features with match labels, add football season."""
    prematch = pd.read_csv(PREMATCH_FILE)
    matches = pd.read_csv(MATCHES_FILE)
    matches = matches[['match_id', 'competition_id', 'result']]
    df = prematch.merge(matches, on='match_id', how='left')
    df['match_date'] = pd.to_datetime(df['match_date'])
    # Football season: if month >= 8, season = year; else season = year - 1
    df['season'] = df['match_date'].dt.year
    df.loc[df['match_date'].dt.month < 8, 'season'] = df['season'] - 1
    df['label'] = df['result'].map({'H':0, 'D':1, 'A':2})
    return df

def get_features_labels(df):
    """Extract feature matrix and labels."""
    feature_cols = [c for c in df.columns if c not in [
        'match_id', 'match_date', 'home_team', 'away_team',
        'label_goal_diff', 'label_result', 'result', 'season',
        'competition_id', 'label'
    ]]
    X = df[feature_cols].values
    y = df['label'].values
    return X, y

def train_model(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=5):
    """Train XGBoost classifier with fixed hyperparameters."""
    model = XGBClassifier(
        random_state=SEED,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        eval_metric='mlogloss',
        verbosity=0
    )
    model.fit(X_train, y_train)
    return model

def compute_metrics_from_probs(y_test, probs):
    """Compute all metrics from true labels and predicted probabilities."""
    try:
        ll = log_loss(y_test, probs)
    except ValueError:
        ll = np.nan
    acc = accuracy_score(y_test, np.argmax(probs, axis=1))
    rps_list = []
    for i in range(len(y_test)):
        cum_pred = np.cumsum(probs[i, :])
        cum_true = np.zeros(3)
        cum_true[y_test[i]:] = 1
        rps_list.append(np.sum((cum_pred[:-1] - cum_true[:-1]) ** 2))
    rps = np.mean(rps_list)
    brier = np.mean([brier_score_loss((y_test == c).astype(int), probs[:, c]) for c in range(3)])
    exp_points = 3 * probs[:, 0] + 1 * probs[:, 1]
    actual_points = np.where(y_test == 0, 3, np.where(y_test == 1, 1, 0))
    ep_mae = mean_absolute_error(actual_points, exp_points)
    return {
        'log_loss': ll,
        'accuracy': acc,
        'rps': rps,
        'brier': brier,
        'expected_points_mae': ep_mae
    }

# ---------- Evaluation Functions ----------
def rolling_multi_step_evaluation(df, horizons=(1,2,3)):
    """Multi-step rolling: train on all seasons before cutoff, test on next 1,2,3 seasons."""
    df = df.copy()
    results = []
    seasons = sorted(df['season'].unique())
    min_train_season = None
    for s in seasons:
        if (df['season'] < s).sum() >= MIN_TRAIN_MATCHES:
            min_train_season = s
            break
    if min_train_season is None:
        return pd.DataFrame()
    for cutoff_season in seasons:
        if cutoff_season <= min_train_season:
            continue
        train_mask = df['season'] < cutoff_season
        if train_mask.sum() < MIN_TRAIN_MATCHES:
            continue
        X_train, y_train = get_features_labels(df[train_mask])
        for horizon in horizons:
            test_season = cutoff_season + horizon - 1
            test_mask = df['season'] == test_season
            if test_mask.sum() < MIN_TEST_MATCHES:
                continue
            X_test, y_test = get_features_labels(df[test_mask])
            model = train_model(X_train, y_train)
            probs = model.predict_proba(X_test)
            metrics = compute_metrics_from_probs(y_test, probs)
            results.append({
                'train_up_to': cutoff_season - 1,
                'test_season': test_season,
                'horizon': horizon,
                **metrics
            })
    return pd.DataFrame(results)

def per_league_evaluation(df):
    """Train separate model for each league."""
    results = []
    for comp_id, league_name in [(11, 'La Liga'), (2, 'Premier League')]:
        league_df = df[df['competition_id'] == comp_id].copy()
        if len(league_df) < 100:
            continue
        league_df = league_df.sort_values('match_date')
        n = len(league_df)
        train_end = int(n * 0.7)
        test_start = int(n * 0.85)
        train = league_df.iloc[:train_end]
        test = league_df.iloc[test_start:]
        if len(train) < MIN_TRAIN_MATCHES or len(test) < MIN_TEST_MATCHES:
            continue
        X_train, y_train = get_features_labels(train)
        X_test, y_test = get_features_labels(test)
        model = train_model(X_train, y_train)
        probs = model.predict_proba(X_test)
        metrics = compute_metrics_from_probs(y_test, probs)
        results.append({
            'league': league_name,
            'train_size': len(train),
            'test_size': len(test),
            **metrics
        })
    return pd.DataFrame(results)

def transfer_learning_experiment(df):
    """
    Base model trained on La Liga + Premier League (combined), then fine-tuned separately on each league.
    """
    combined = df.copy()
    combined = combined.sort_values('match_date')
    n_combined = len(combined)
    combined_train = combined.iloc[:int(n_combined * 0.8)]
    X_base, y_base = get_features_labels(combined_train)
    base_model = train_model(X_base, y_base)
    
    results = []
    for comp_id, league_name in [(11, 'La Liga'), (2, 'Premier League')]:
        league_df = df[df['competition_id'] == comp_id].copy()
        if len(league_df) < 100:
            continue
        league_df = league_df.sort_values('match_date')
        n_league = len(league_df)
        league_train = league_df.iloc[:int(n_league * 0.8)]
        league_test = league_df.iloc[int(n_league * 0.8):]
        if len(league_train) < MIN_TRAIN_MATCHES or len(league_test) < MIN_TEST_MATCHES:
            continue
        
        # Fine-tune base model on league-specific training data
        X_ft, y_ft = get_features_labels(league_train)
        fine_model = XGBClassifier(
            random_state=SEED,
            n_estimators=50,
            learning_rate=0.05,
            max_depth=5,
            eval_metric='mlogloss',
            verbosity=0
        )
        fine_model.fit(X_ft, y_ft, xgb_model=base_model.get_booster())
        
        # Evaluate on league test
        X_test, y_test = get_features_labels(league_test)
        probs = fine_model.predict_proba(X_test)
        metrics = compute_metrics_from_probs(y_test, probs)
        results.append({
            'league': league_name,
            'method': 'Transfer (base combined + fine-tune)',
            **metrics
        })
        
        # Evaluate base model directly on this league's test
        probs_base = base_model.predict_proba(X_test)
        metrics_base = compute_metrics_from_probs(y_test, probs_base)
        results.append({
            'league': league_name,
            'method': 'Base combined (no fine-tune)',
            **metrics_base
        })
    
    return pd.DataFrame(results)

def seasonal_metrics_window(df, window=None):
    """
    Seasonal metrics: train on all past (window=None) or last `window` seasons.
    """
    df = df.copy()
    results = []
    seasons = sorted(df['season'].unique())
    for season in seasons:
        if window is None:
            train_mask = df['season'] < season
        else:
            prior_seasons = [s for s in seasons if s < season]
            if len(prior_seasons) < window:
                continue
            train_seasons = prior_seasons[-window:]
            train_mask = df['season'].isin(train_seasons)
        test_mask = df['season'] == season
        if train_mask.sum() < MIN_TRAIN_MATCHES or test_mask.sum() < MIN_TEST_MATCHES:
            continue
        X_train, y_train = get_features_labels(df[train_mask])
        X_test, y_test = get_features_labels(df[test_mask])
        model = train_model(X_train, y_train)
        probs = model.predict_proba(X_test)
        metrics = compute_metrics_from_probs(y_test, probs)
        results.append({'season': season, 'window': window if window is not None else 'all', **metrics})
    return pd.DataFrame(results)

def ensemble_seasonal_metrics(df, window_sizes=ENSEMBLE_WINDOWS):
    """For each season, train multiple models with different training windows, average probabilities."""
    results = []
    seasons = sorted(df['season'].unique())
    for season in seasons:
        prior_seasons = [s for s in seasons if s < season]
        if len(prior_seasons) < min(w for w in window_sizes if w is not None):
            continue
        
        models = []
        for w in window_sizes:
            if w is None:
                train_mask = df['season'] < season
            else:
                if len(prior_seasons) < w:
                    continue
                train_seasons = prior_seasons[-w:]
                train_mask = df['season'].isin(train_seasons)
            if train_mask.sum() < MIN_TRAIN_MATCHES:
                continue
            X_train, y_train = get_features_labels(df[train_mask])
            model = train_model(X_train, y_train)
            models.append(model)
        
        if not models:
            continue
        
        test_mask = df['season'] == season
        if test_mask.sum() < MIN_TEST_MATCHES:
            continue
        X_test, y_test = get_features_labels(df[test_mask])
        
        prob_sum = None
        for model in models:
            probs = model.predict_proba(X_test)
            if prob_sum is None:
                prob_sum = probs
            else:
                prob_sum += probs
        probs = prob_sum / len(models)
        
        metrics = compute_metrics_from_probs(y_test, probs)
        results.append({'season': season, 'window': 'ensemble', **metrics})
    
    return pd.DataFrame(results)

def expected_points_analysis(df, top_n=10):
    """Compute per-team expected points across seasons. Only include teams with >= EXPECTED_POINTS_MIN_SEASONS."""
    teams = pd.unique(df[['home_team', 'away_team']].values.ravel())
    rows = []
    seasons = sorted(df['season'].unique())

    for team in teams:
        team_df = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()
        total_pred = 0
        total_actual = 0
        n_seasons = 0
        seasons_used = []
        for season in seasons:
            train_mask = df['season'] < season
            if train_mask.sum() < MIN_TRAIN_MATCHES:
                continue
            test_mask = (team_df['season'] == season)
            if test_mask.sum() < MIN_TEST_MATCHES:
                continue
            X_train, y_train = get_features_labels(df[train_mask])
            model = train_model(X_train, y_train)
            X_test, y_test = get_features_labels(team_df[test_mask])
            probs = model.predict_proba(X_test)
            home_flags = (team_df[test_mask]['home_team'] == team).values
            exp_points = np.zeros(len(y_test))
            for i in range(len(y_test)):
                if home_flags[i]:
                    win_prob = probs[i, 0]
                else:
                    win_prob = probs[i, 2]
                draw_prob = probs[i, 1]
                exp_points[i] = 3 * win_prob + 1 * draw_prob
            actual_points = np.zeros(len(y_test))
            for i, (idx, row) in enumerate(team_df[test_mask].iterrows()):
                if row['home_team'] == team:
                    if row['result'] == 'H':
                        actual_points[i] = 3
                    elif row['result'] == 'D':
                        actual_points[i] = 1
                    else:
                        actual_points[i] = 0
                else:
                    if row['result'] == 'A':
                        actual_points[i] = 3
                    elif row['result'] == 'D':
                        actual_points[i] = 1
                    else:
                        actual_points[i] = 0
            total_pred += exp_points.sum()
            total_actual += actual_points.sum()
            n_seasons += 1
            seasons_used.append(season)
        
        if n_seasons < EXPECTED_POINTS_MIN_SEASONS:
            continue
        
        rows.append({
            'team': team,
            'n_seasons': n_seasons,
            'seasons_used': seasons_used,
            'predicted_total_points': total_pred,
            'actual_total_points': total_actual,
            'difference': total_pred - total_actual
        })
    
    results = pd.DataFrame(rows)
    if results.empty:
        return results
    results = results.sort_values('difference', ascending=False)
    if top_n is not None:
        results = results.head(top_n)
    return results

def high_risk_brier_analysis(df, threshold=HIGH_RISK_THRESHOLD):
    """High-risk matches defined as those where model's probability for true outcome < threshold."""
    df = df.copy()
    rows = []
    seasons = sorted(df['season'].unique())
    for season in seasons:
        train_mask = df['season'] < season
        test_mask = df['season'] == season
        if train_mask.sum() < MIN_TRAIN_MATCHES or test_mask.sum() < MIN_TEST_MATCHES:
            continue
        X_train, y_train = get_features_labels(df[train_mask])
        X_test, y_test = get_features_labels(df[test_mask])
        model = train_model(X_train, y_train)
        probs = model.predict_proba(X_test)
        true_probs = probs[np.arange(len(y_test)), y_test]
        risk_mask = true_probs < threshold
        if risk_mask.sum() > 0:
            brier = np.mean([brier_score_loss((y_test[risk_mask] == c).astype(int),
                                              probs[risk_mask, c]) for c in range(3)])
        else:
            brier = np.nan
        rows.append({'season': season, 'high_risk_count': risk_mask.sum(), 'brier_high_risk': brier})
    return pd.DataFrame(rows)

# ---------- Plotting Functions ----------
def plot_rolling_multi_step(rolling_df):
    if rolling_df.empty:
        return
    plt.figure(figsize=(12, 6))
    for h in [1, 2, 3]:
        data = rolling_df[rolling_df['horizon'] == h]
        plt.plot(data['test_season'], data['log_loss'], marker='o', label=f'Horizon {h}')
    plt.xlabel('Test Season')
    plt.ylabel('Log-Loss')
    plt.title('Rolling Multi-Step Evaluation (log-loss)')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 2)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'rolling_multi_step_logloss.png'))
    plt.close()

def plot_per_league(league_df):
    if league_df.empty:
        return
    plt.figure(figsize=(8, 5))
    sns.barplot(x='league', y='accuracy', data=league_df)
    plt.xlabel('League')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by League (Separate Training)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'per_league_accuracy.png'))
    plt.close()

def plot_transfer_learning(transfer_df):
    if transfer_df.empty:
        return
    plt.figure(figsize=(10, 6))
    sns.barplot(x='league', y='accuracy', hue='method', data=transfer_df)
    plt.xlabel('League')
    plt.ylabel('Accuracy')
    plt.title('Transfer Learning Comparison')
    plt.legend(title='Method')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'transfer_learning_accuracy.png'))
    plt.close()

def plot_ensemble_seasonal(ensemble_df, all_df, window_df):
    plt.figure(figsize=(12, 6))
    if not all_df.empty:
        all_df = all_df.sort_values('season')
        plt.plot(all_df['season'], all_df['log_loss'], marker='o', linestyle='--', label='All past')
    if not window_df.empty:
        window_df = window_df.sort_values('season')
        plt.plot(window_df['season'], window_df['log_loss'], marker='s', linestyle=':', label='Window=5')
    if not ensemble_df.empty:
        ensemble_df = ensemble_df.sort_values('season')
        plt.plot(ensemble_df['season'], ensemble_df['log_loss'], marker='^', label='Ensemble')
    plt.xlabel('Season')
    plt.ylabel('Log-Loss')
    plt.title('Seasonal Metrics: All vs Window vs Ensemble')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 2)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'seasonal_comparison_logloss.png'))
    plt.close()

def plot_expected_points(ep_df):
    if ep_df.empty:
        return
    plt.figure(figsize=(10, 6))
    sns.barplot(x='difference', y='team', data=ep_df)
    plt.xlabel('Difference (Predicted - Actual Points)')
    plt.ylabel('Team')
    plt.title('Expected Points Difference by Team (Top 10)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'expected_points_difference.png'))
    plt.close()

def plot_high_risk_brier(hr_df):
    if hr_df.empty:
        return
    plt.figure(figsize=(10, 6))
    plt.plot(hr_df['season'], hr_df['brier_high_risk'], marker='o')
    plt.xlabel('Season')
    plt.ylabel('Brier (High-risk)')
    plt.title('High-Risk Brier over Seasons')
    plt.grid(True)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'high_risk_brier.png'))
    plt.close()

# ---------- Main ----------
def main():
    print("Loading data...")
    df = load_data()
    print(f"Total matches: {len(df)}")
    print(f"Seasons: {sorted(df['season'].unique())}")

    print("\n=== 1. Multi-step rolling evaluation (1,2,3 seasons ahead) ===")
    rolling_df = rolling_multi_step_evaluation(df, horizons=(1,2,3))
    if rolling_df.empty:
        print("Not enough data for rolling evaluation.")
    else:
        print(rolling_df.round(4).to_string(index=False))
        rolling_df.to_csv(os.path.join(OUTPUT_DIR, 'rolling_multi_step.csv'), index=False)
        plot_rolling_multi_step(rolling_df)

    print("\n=== 2. Per-league evaluation ===")
    league_df = per_league_evaluation(df)
    print(league_df.round(4).to_string(index=False))
    league_df.to_csv(os.path.join(OUTPUT_DIR, 'per_league.csv'), index=False)
    plot_per_league(league_df)

    print("\n=== 2b. Transfer learning (base combined -> fine-tune each league) ===")
    transfer_df = transfer_learning_experiment(df)
    if transfer_df.empty:
        print("Not enough data for transfer learning.")
    else:
        print(transfer_df.round(4).to_string(index=False))
        transfer_df.to_csv(os.path.join(OUTPUT_DIR, 'transfer_learning.csv'), index=False)
        plot_transfer_learning(transfer_df)

    print("\n=== 3. Seasonal metrics ===")
    all_df = seasonal_metrics_window(df, window=None)
    window_df = seasonal_metrics_window(df, window=5)
    ensemble_df = ensemble_seasonal_metrics(df)
    print("Ensemble results:")
    print(ensemble_df.round(4).to_string(index=False))
    ensemble_df.to_csv(os.path.join(OUTPUT_DIR, 'seasonal_metrics_ensemble.csv'), index=False)
    all_df.to_csv(os.path.join(OUTPUT_DIR, 'seasonal_metrics_all.csv'), index=False)
    window_df.to_csv(os.path.join(OUTPUT_DIR, 'seasonal_metrics_window.csv'), index=False)
    plot_ensemble_seasonal(ensemble_df, all_df, window_df)

    print("\n=== 4. Expected Points per team (top 10) ===")
    ep_df = expected_points_analysis(df, top_n=10)
    if ep_df.empty:
        print("No team has enough seasons for this analysis.")
    else:
        print(ep_df.round(2).to_string(index=False))
        ep_df.to_csv(os.path.join(OUTPUT_DIR, 'team_expected_points.csv'), index=False)
        plot_expected_points(ep_df)

    print("\n=== 5. High-risk Brier analysis (threshold = {}) ===".format(HIGH_RISK_THRESHOLD))
    hr_df = high_risk_brier_analysis(df, threshold=HIGH_RISK_THRESHOLD)
    print(hr_df.round(4).to_string(index=False))
    hr_df.to_csv(os.path.join(OUTPUT_DIR, 'high_risk_brier.csv'), index=False)
    plot_high_risk_brier(hr_df)

    print("\nAll outputs and figures saved.")

if __name__ == '__main__':
    main()