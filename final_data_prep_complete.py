import json
import glob
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import gc
import warnings
warnings.filterwarnings('ignore')

# 0
DATA_ROOT = "data"
OUTPUT_DIR = "processed_data"
SNAPSHOT_INTERVAL = 5 
GOAL_DIFF_CLIP = 5
COMPETITION_IDS = [11, 2]  

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Starting full StatsBomb data processing (final version)...")
print("Includes Lineups, Rest Days, and advanced Missing Value handling.")
print("Both Parquet and CSV formats will be generated for all datasets.")

# 1
print("\n1) Loading and integrating files...")

# 1-1
comp_path = os.path.join(DATA_ROOT, "competitions.json")
with open(comp_path, 'r', encoding='utf-8-sig') as f:
    competitions = json.load(f)
df_competitions = pd.json_normalize(competitions)
if COMPETITION_IDS:
    df_competitions = df_competitions[df_competitions['competition_id'].isin(COMPETITION_IDS)]

# 1-2
all_matches = []
match_files = glob.glob(os.path.join(DATA_ROOT, "matches", "*", "*.json"))
print(f"   - Finding match files... ({len(match_files)} files)")

for file_path in tqdm(match_files, desc="   - Reading matches"):
    parts = file_path.split(os.sep)
    comp_id = int(parts[-2])
    if COMPETITION_IDS and comp_id not in COMPETITION_IDS:
        continue
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            matches = json.load(f)
        for m in matches:
            m['competition_id'] = comp_id
            if 'home_team' in m and isinstance(m['home_team'], dict):
                m['home_team_name'] = m['home_team'].get('home_team_name')

            if 'away_team' in m and isinstance(m['away_team'], dict):
                m['away_team_name'] = m['away_team'].get('away_team_name')

            if 'match_date' in m:
                m['match_date'] = pd.to_datetime(m['match_date']).date()
            all_matches.append(m)
    except Exception as e:
        print(f"   - Error reading {file_path}: {e}")

df_matches = pd.json_normalize(all_matches)
cols_keep = ['match_id', 'competition_id', 'season_id', 'match_date', 
             'home_team_name', 'away_team_name', 'home_score', 'away_score']
df_matches = df_matches[[c for c in cols_keep if c in df_matches.columns]]
df_matches = df_matches.dropna(subset=['match_id', 'home_team_name', 'away_team_name', 'match_date'])
df_matches['match_id'] = df_matches['match_id'].astype(int)

# labeling
df_matches['result'] = df_matches.apply(lambda r: 'H' if r['home_score'] > r['away_score'] 
                                        else ('A' if r['home_score'] < r['away_score'] else 'D'), axis=1)
df_matches['goal_diff'] = (df_matches['home_score'] - df_matches['away_score']).clip(-GOAL_DIFF_CLIP, GOAL_DIFF_CLIP)
print(f"   - Total matches loaded: {len(df_matches)}")

# df_matches
df_matches.to_parquet(os.path.join(OUTPUT_DIR, 'matches_full.parquet'), index=False)
df_matches.to_csv(os.path.join(OUTPUT_DIR, 'matches_full.csv'), index=False)
print("   - matches_full saved as both .parquet and .csv")

# 2
print("\n2) Building pre-match features (including Lineups and Rest Days)...")

def load_lineup_formation(match_id):
    path = os.path.join(DATA_ROOT, 'lineups', f'{match_id}.json')
    if not os.path.exists(path):
        return None, None
    try:
        with open(path, 'r', encoding='utf-8-sig') as f:
            lineups = json.load(f)
        home_formation = None
        away_formation = None
        for team in lineups:
            if 'tactics' in team and 'formation' in team['tactics']:
                formation = team['tactics']['formation']
                if team.get('team_name') in df_matches[df_matches['match_id'] == match_id]['home_team_name'].values:
                    home_formation = formation
                else:
                    away_formation = formation
        return home_formation, away_formation
    except:
        return None, None

def compute_advanced_team_stats(df_matches, team_name, current_date, n_prev=5):
    past = df_matches[
        ((df_matches['home_team_name'] == team_name) | (df_matches['away_team_name'] == team_name)) &
        (df_matches['match_date'] < current_date)
    ].sort_values('match_date', ascending=False)
    
    past_n = past.head(n_prev)
    if len(past_n) < 3:
        stats = {'avg_goals_scored': 0, 'avg_goals_conceded': 0, 'avg_points': 0, 'avg_goal_diff': 0}
    else:
        goals_scored = past_n.apply(lambda r: r['home_score'] if r['home_team_name'] == team_name else r['away_score'], axis=1)
        goals_conceded = past_n.apply(lambda r: r['away_score'] if r['home_team_name'] == team_name else r['home_score'], axis=1)
        points = past_n.apply(lambda r: 3 if r['result'] == ('H' if r['home_team_name'] == team_name else 'A') 
                              else (1 if r['result'] == 'D' else 0), axis=1)
        goal_diff = goals_scored - goals_conceded
        stats = {
            'avg_goals_scored': goals_scored.mean(),
            'avg_goals_conceded': goals_conceded.mean(),
            'avg_points': points.mean(),
            'avg_goal_diff': goal_diff.mean()
        }
    
    if len(past) >= 1:
        last_match_date = past.iloc[0]['match_date']
        rest_days = (current_date - last_match_date).days
    else:
        rest_days = 7
    
    stats['rest_days'] = rest_days
    return stats

prematch_rows = []
formation_cache = {} 

for idx, row in tqdm(df_matches.iterrows(), total=len(df_matches), desc="   - Computing features for each match"):
    mid = row['match_id']
    home = row['home_team_name']
    away = row['away_team_name']
    date = row['match_date']
    
    home_feats = compute_advanced_team_stats(df_matches, home, date)
    away_feats = compute_advanced_team_stats(df_matches, away, date)
    
    if mid not in formation_cache:
        home_form, away_form = load_lineup_formation(mid)
        formation_cache[mid] = (home_form, away_form)
    else:
        home_form, away_form = formation_cache[mid]

    home_form = home_form if home_form is not None else -1
    away_form = away_form if away_form is not None else -1
    
    prematch_rows.append({
        'match_id': mid,
        'match_date': date,
        'home_team': home,
        'away_team': away,

        'home_avg_goals_scored': home_feats['avg_goals_scored'],
        'home_avg_goals_conceded': home_feats['avg_goals_conceded'],
        'home_avg_points': home_feats['avg_points'],
        'home_avg_goal_diff': home_feats['avg_goal_diff'],
        'home_rest_days': home_feats['rest_days'],

        'away_avg_goals_scored': away_feats['avg_goals_scored'],
        'away_avg_goals_conceded': away_feats['avg_goals_conceded'],
        'away_avg_points': away_feats['avg_points'],
        'away_avg_goal_diff': away_feats['avg_goal_diff'],
        'away_rest_days': away_feats['rest_days'],

        'diff_avg_goals_scored': home_feats['avg_goals_scored'] - away_feats['avg_goals_scored'],
        'diff_avg_goals_conceded': home_feats['avg_goals_conceded'] - away_feats['avg_goals_conceded'],
        'diff_avg_points': home_feats['avg_points'] - away_feats['avg_points'],
        'diff_avg_goal_diff': home_feats['avg_goal_diff'] - away_feats['avg_goal_diff'],
        'diff_rest_days': home_feats['rest_days'] - away_feats['rest_days'],

        'home_formation': home_form,
        'away_formation': away_form,
        'formation_match': 1 if home_form == away_form else 0,
        # lebels
        'label_goal_diff': row['goal_diff'],
        'label_result': row['result']
    })

df_prematch = pd.DataFrame(prematch_rows)
print(f"   - Pre-match feature records: {len(df_prematch)}")

# save prematch
df_prematch.to_parquet(os.path.join(OUTPUT_DIR, 'full_prematch.parquet'), index=False)
df_prematch.to_csv(os.path.join(OUTPUT_DIR, 'full_prematch.csv'), index=False)
print("   - Full pre-match dataset saved as 'full_prematch.parquet' and 'full_prematch.csv'")

# 3
print("\n3) Building in-play snapshots (every {} minutes)...".format(SNAPSHOT_INTERVAL))
print("   - This may take time, but memory usage is optimized.")

def load_events_for_match(match_id):
    """بارگذاری رویدادهای یک مسابقه با مدیریت Missing Values"""
    path = os.path.join(DATA_ROOT, 'events', f'{match_id}.json')
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        with open(path, 'r', encoding='utf-8-sig') as f:
            events = json.load(f)

        df = pd.json_normalize(events, sep='.')
        df['match_id'] = match_id
        
        # Missing Values
        essential_cols = ['minute', 'second', 'period', 'type.name', 
                          'shot.outcome.name', 'possession_team.name', 'card.name']
        for col in essential_cols:
            if col not in df.columns:
                df[col] = np.nan  
        
        # sorting
        df = df.sort_values(['period', 'minute', 'second']).reset_index(drop=True)
        return df
    except Exception as e:
        print(f"   - Error reading events for match {match_id}: {e}")
        return pd.DataFrame()

def create_snapshots_for_match(match_id, df_matches, interval=5):
    match_events = load_events_for_match(match_id)
    if match_events.empty:
        return pd.DataFrame()
    
    match_info = df_matches[df_matches['match_id'] == match_id].iloc[0]
    max_minute = match_events['minute'].max()
    if pd.isna(max_minute) or max_minute < 10:
        return pd.DataFrame()
    
    snapshots = []
    for t in range(0, int(max_minute) + 1, interval):
        events_up_to_t = match_events[match_events['minute'] <= t]
        if len(events_up_to_t) < 5:
            continue
        
        home_goals = len(events_up_to_t[
            (events_up_to_t['type.name'].fillna('') == 'Shot') & 
            (events_up_to_t['shot.outcome.name'].fillna('') == 'Goal') &
            (events_up_to_t['possession_team.name'].fillna('') == match_info['home_team_name'])
        ])
        away_goals = len(events_up_to_t[
            (events_up_to_t['type.name'].fillna('') == 'Shot') & 
            (events_up_to_t['shot.outcome.name'].fillna('') == 'Goal') &
            (events_up_to_t['possession_team.name'].fillna('') == match_info['away_team_name'])
        ])

        red_home = len(events_up_to_t[
            (events_up_to_t['card.name'].fillna('') == 'Red Card') &
            (events_up_to_t['possession_team.name'].fillna('') == match_info['home_team_name'])
        ])
        red_away = len(events_up_to_t[
            (events_up_to_t['card.name'].fillna('') == 'Red Card') &
            (events_up_to_t['possession_team.name'].fillna('') == match_info['away_team_name'])
        ])

        window_start = max(0, t - 5)
        recent_events = events_up_to_t[(events_up_to_t['minute'] >= window_start) & (events_up_to_t['minute'] <= t)]
        shots_recent = len(recent_events[recent_events['type.name'].fillna('') == 'Shot'])
        passes_recent = len(recent_events[recent_events['type.name'].fillna('') == 'Pass'])
        pressures_recent = len(recent_events[recent_events['type.name'].fillna('') == 'Pressure'])

        home_shots_rec = len(recent_events[
            (recent_events['type.name'].fillna('') == 'Shot') & 
            (recent_events['possession_team.name'].fillna('') == match_info['home_team_name'])
        ])
        away_shots_rec = len(recent_events[
            (recent_events['type.name'].fillna('') == 'Shot') & 
            (recent_events['possession_team.name'].fillna('') == match_info['away_team_name'])
        ])
        total_shots_rec = home_shots_rec + away_shots_rec
        momentum = home_shots_rec / total_shots_rec if total_shots_rec > 0 else 0.5
        
        snap = {
            'match_id': match_id,
            'snapshot_time': t,
            'time_norm': t / max_minute if max_minute > 0 else 0,
            'current_home_score': home_goals,
            'current_away_score': away_goals,
            'red_card_diff': red_home - red_away,
            'shots_recent_5min': shots_recent,
            'passes_recent_5min': passes_recent,
            'pressures_recent_5min': pressures_recent,
            'momentum': momentum,
            'final_goal_diff': match_info['goal_diff'],
            'final_result': match_info['result']
        }
        snapshots.append(snap)
    
    del match_events
    gc.collect()
    return pd.DataFrame(snapshots)

# processing
match_ids = df_matches['match_id'].unique().tolist()
all_snapshots = []

for mid in tqdm(match_ids, desc="   - Creating snapshots for matches"):
    df_snap = create_snapshots_for_match(mid, df_matches, SNAPSHOT_INTERVAL)
    if not df_snap.empty:
        all_snapshots.append(df_snap)
    if len(all_snapshots) % 100 == 0:
        gc.collect()

df_snapshots = pd.concat(all_snapshots, ignore_index=True) if all_snapshots else pd.DataFrame()
print(f"   - Total in-play snapshots: {len(df_snapshots)}")

# save snapshots
df_snapshots.to_parquet(os.path.join(OUTPUT_DIR, 'full_snapshots.parquet'), index=False)
df_snapshots.to_csv(os.path.join(OUTPUT_DIR, 'full_snapshots.csv'), index=False)
print("   - Full snapshots dataset saved as 'full_snapshots.parquet' and 'full_snapshots.csv'")

# 4
print("\n4) Splitting data into train/validation/test...")

df_prematch_sorted = df_prematch.sort_values('match_date').reset_index(drop=True)
n = len(df_prematch_sorted)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

train_match_ids = df_prematch_sorted.iloc[:train_end]['match_id'].tolist()
val_match_ids = df_prematch_sorted.iloc[train_end:val_end]['match_id'].tolist()
test_match_ids = df_prematch_sorted.iloc[val_end:]['match_id'].tolist()

print(f"   - Match counts: train={len(train_match_ids)}, val={len(val_match_ids)}, test={len(test_match_ids)}")

train_pre = df_prematch[df_prematch['match_id'].isin(train_match_ids)]
val_pre = df_prematch[df_prematch['match_id'].isin(val_match_ids)]
test_pre = df_prematch[df_prematch['match_id'].isin(test_match_ids)]

train_snap = df_snapshots[df_snapshots['match_id'].isin(train_match_ids)] if not df_snapshots.empty else pd.DataFrame()
val_snap = df_snapshots[df_snapshots['match_id'].isin(val_match_ids)] if not df_snapshots.empty else pd.DataFrame()
test_snap = df_snapshots[df_snapshots['match_id'].isin(test_match_ids)] if not df_snapshots.empty else pd.DataFrame()

# 5
print("\n5) Saving final datasets (Parquet and CSV)...")

# Parquet
train_pre.to_parquet(os.path.join(OUTPUT_DIR, 'train_prematch.parquet'), index=False)
val_pre.to_parquet(os.path.join(OUTPUT_DIR, 'val_prematch.parquet'), index=False)
test_pre.to_parquet(os.path.join(OUTPUT_DIR, 'test_prematch.parquet'), index=False)
if not train_snap.empty:
    train_snap.to_parquet(os.path.join(OUTPUT_DIR, 'train_snapshots.parquet'), index=False)
    val_snap.to_parquet(os.path.join(OUTPUT_DIR, 'val_snapshots.parquet'), index=False)
    test_snap.to_parquet(os.path.join(OUTPUT_DIR, 'test_snapshots.parquet'), index=False)

# CSV
print("   - Saving CSV files (may take some time)...")
train_pre.to_csv(os.path.join(OUTPUT_DIR, 'train_prematch.csv'), index=False)
val_pre.to_csv(os.path.join(OUTPUT_DIR, 'val_prematch.csv'), index=False)
test_pre.to_csv(os.path.join(OUTPUT_DIR, 'test_prematch.csv'), index=False)
if not train_snap.empty:
    train_snap.to_csv(os.path.join(OUTPUT_DIR, 'train_snapshots.csv'), index=False)
    val_snap.to_csv(os.path.join(OUTPUT_DIR, 'val_snapshots.csv'), index=False)
    test_snap.to_csv(os.path.join(OUTPUT_DIR, 'test_snapshots.csv'), index=False)

# match_id list
pd.DataFrame({'match_id': train_match_ids}).to_csv(os.path.join(OUTPUT_DIR, 'train_match_ids.csv'), index=False)
pd.DataFrame({'match_id': val_match_ids}).to_csv(os.path.join(OUTPUT_DIR, 'val_match_ids.csv'), index=False)
pd.DataFrame({'match_id': test_match_ids}).to_csv(os.path.join(OUTPUT_DIR, 'test_match_ids.csv'), index=False)

# summary file
summary_data = {
    'Dataset': ['Train Prematch', 'Val Prematch', 'Test Prematch', 'Train Snapshots', 'Val Snapshots', 'Test Snapshots'],
    'Rows': [len(train_pre), len(val_pre), len(test_pre), len(train_snap), len(val_snap), len(test_snap)],
    'Columns': [len(train_pre.columns), len(val_pre.columns), len(test_pre.columns), 
                len(train_snap.columns) if not train_snap.empty else 0,
                len(val_snap.columns) if not val_snap.empty else 0,
                len(test_snap.columns) if not test_snap.empty else 0]
}
pd.DataFrame(summary_data).to_csv(os.path.join(OUTPUT_DIR, 'dataset_summary.csv'), index=False)

# Missing Values report
print("\nMissing values report in pre-match features:")
missing_report = df_prematch.isnull().sum()
missing_report = missing_report[missing_report > 0]
if len(missing_report) > 0:
    print(missing_report)
    missing_report.to_csv(os.path.join(OUTPUT_DIR, 'missing_values_report.csv'))
else:
    print("No missing values found in pre-match features.")
    pd.DataFrame({'Message': ['No missing values found']}).to_csv(os.path.join(OUTPUT_DIR, 'missing_values_report.csv'), index=False)

print("\nAll datasets successfully saved to '{}'.".format(OUTPUT_DIR))
print("Data preparation completed successfully!")