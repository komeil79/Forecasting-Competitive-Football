"""
Odds Integration - Fixed Version
--------------------------------
Downloads odds for La Liga and Premier League,
handles connection errors and messy CSV formats,
matches to StatsBomb matches, and evaluates market baseline.
"""

import os
import time
import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =========================== CONFIGURATION ==============================
MATCHES_FILE = 'processed_data//matches_full.csv'
TEST_IDS_FILE = 'processed_data//test_match_ids.csv'
OUTPUT_BASELINE_FILE = 'processed_data//market_baseline.csv'
DATA_DIR = 'processed_data//odds_data'

LEAGUE_CODES = {
    11: 'SP1',  # La Liga
    2:  'E0'    # Premier League
}

# Extended team name alias mapping (add more as needed)
TEAM_ALIASES = {
    'RC Deportivo La Coruña': 'Deportivo La Coruña',
    'Deportivo Alavés': 'Alaves',
    'Real Betis': 'Betis',
    'Real Sociedad': 'Sociedad',
    'Athletic Club': 'Athletic Bilbao',
    'AtlÃ©tico Madrid': 'Atletico Madrid',
    'MÃ¡laga': 'Malaga',
    'Levante UD': 'Levante',
    'Sporting GijÃ³n': 'Sporting Gijon',
    'GimnÃ stic Tarragona': 'Gimnastic',
    'Recreativo Huelva': 'Recreativo Huelva',
    'Real Valladolid': 'Valladolid',
    'Real Zaragoza': 'Zaragoza',
    'Racing Santander': 'Racing Santander',
    'CD Tenerife': 'Tenerife',
    'HÃ©rcules': 'Hercules',
    'Granada': 'Granada',
    'Osasuna': 'Osasuna',
    'Mallorca': 'Mallorca',
    'Villarreal': 'Villarreal',
    'Valencia': 'Valencia',
    'Sevilla': 'Sevilla',
    'Barcelona': 'Barcelona',
    'Real Madrid': 'Real Madrid',
    'Celta Vigo': 'Celta Vigo',
    'Getafe': 'Getafe',
    'Espanyol': 'Espanyol',
    'Eibar': 'Eibar',
    'Las Palmas': 'Las Palmas',
    'Girona': 'Girona',
    'LeganÃ©s': 'Leganes',
    'AlmerÃ­a': 'Almeria',
    'CÃ³rdoba CF': 'Cordoba',
    'Elche': 'Elche',
    'Huesca': 'Huesca',
    'Cádiz': 'Cadiz',
    'Manchester United': 'Man United',
    'Manchester City': 'Man City',
    'Tottenham Hotspur': 'Tottenham',
    'West Ham United': 'West Ham',
    'Newcastle United': 'Newcastle',
    'Aston Villa': 'Aston Villa',
    'Leicester City': 'Leicester',
    'Everton': 'Everton',
    'Liverpool': 'Liverpool',
    'Arsenal': 'Arsenal',
    'Chelsea': 'Chelsea',
    'Crystal Palace': 'Crystal Palace',
    'Southampton': 'Southampton',
    'Stoke City': 'Stoke City',
    'Swansea City': 'Swansea',
    'West Bromwich Albion': 'West Brom',
    'AFC Bournemouth': 'Bournemouth',
    'Norwich City': 'Norwich',
    'Watford': 'Watford',
    'Sunderland': 'Sunderland',
    'Fulham': 'Fulham',
    'Wolverhampton Wanderers': 'Wolves',
    'Brighton and Hove Albion': 'Brighton',
    'Burnley': 'Burnley',
    'Leeds United': 'Leeds',
    'Brentford': 'Brentford',
    'Nottingham Forest': 'Nott\'m Forest',
    'Nott\'m Forest': 'Nott\'m Forest',  # already covered
    'Man Utd': 'Man United',
    'Man City': 'Man City',
    'Nott\'m Forest': 'Nott\'m Forest',
    'Leeds': 'Leeds',
    'Wolves': 'Wolves',
    'Brighton': 'Brighton',
}

# =========================== HELPER FUNCTIONS ============================
def standardise_team_name(name):
    if not isinstance(name, str):
        return name
    name = name.strip()
    # Direct mapping
    if name in TEAM_ALIASES:
        return TEAM_ALIASES[name]
    # Remove common suffixes like " FC", " CF", etc.
    for suffix in [' FC', ' CF', ' United', ' City']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    # Return the cleaned name (still may need manual mapping)
    return name

def season_code_from_year(year):
    next_year = year + 1
    return f"{str(year)[-2:]}{str(next_year)[-2:]}"

def download_file_with_retries(url, local_path, max_retries=5, backoff_factor=1):
    """Download with retry on connection errors."""
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    for attempt in range(max_retries):
        try:
            response = session.get(url, timeout=30)
            if response.status_code == 200:
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                return True
            else:
                print(f"   Attempt {attempt+1}: status {response.status_code}")
        except Exception as e:
            print(f"   Attempt {attempt+1}: {e}")
        time.sleep(backoff_factor * (2 ** attempt))
    return False

def parse_odds_df(file_path):
    """Robust parser for Football-Data.co.uk CSV files."""
    try:
        # Try to read with flexible column handling
        df = pd.read_csv(
            file_path,
            encoding='latin1',
            on_bad_lines='skip',  # skip rows with too many fields
            low_memory=False
        )
    except Exception as e:
        print(f"   Could not parse {file_path}: {e}")
        return None

    # Rename columns to standard names (if they exist with slight variations)
    col_map = {
        'Date': 'Date',
        'HomeTeam': 'HomeTeam',
        'AwayTeam': 'AwayTeam',
        'B365H': 'B365H',
        'B365D': 'B365D',
        'B365A': 'B365A',
    }
    # Some files have 'Home' instead of 'HomeTeam', etc.
    for old, new in col_map.items():
        if old not in df.columns and old.lower() in [c.lower() for c in df.columns]:
            # find the actual column name
            for c in df.columns:
                if c.lower() == old.lower():
                    df = df.rename(columns={c: new})
                    break
    # Check if required columns exist
    required = ['Date', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A']
    missing = [col for col in required if col not in df.columns]
    if missing:
        # If missing Bet365 odds, we could try average odds, but we'll skip.
        print(f"   Missing columns: {missing}. Skipping file.")
        return None

    # Convert date
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    # Drop rows with invalid date or missing odds
    df = df.dropna(subset=['Date', 'B365H', 'B365D', 'B365A'])
    # Keep only relevant columns
    df = df[['Date', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A']]
    # Standardise team names
    df['HomeTeam'] = df['HomeTeam'].apply(standardise_team_name)
    df['AwayTeam'] = df['AwayTeam'].apply(standardise_team_name)
    return df

def join_odds_to_matches(matches_df, odds_df):
    matches = matches_df.copy()
    matches['match_date'] = pd.to_datetime(matches['match_date'])
    matches['home_team_name'] = matches['home_team_name'].apply(standardise_team_name)
    matches['away_team_name'] = matches['away_team_name'].apply(standardise_team_name)

    odds = odds_df.copy()
    odds['HomeTeam'] = odds['HomeTeam'].apply(standardise_team_name)
    odds['AwayTeam'] = odds['AwayTeam'].apply(standardise_team_name)

    merged = pd.merge(
        matches,
        odds,
        left_on=['match_date', 'home_team_name', 'away_team_name'],
        right_on=['Date', 'HomeTeam', 'AwayTeam'],
        how='inner'
    )
    if merged.empty:
        return pd.DataFrame()

    raw_probs = 1 / merged[['B365H', 'B365D', 'B365A']]
    sum_raw = raw_probs.sum(axis=1)
    de_vigged = raw_probs.div(sum_raw, axis=0)
    merged['prob_H'] = de_vigged['B365H']
    merged['prob_D'] = de_vigged['B365D']
    merged['prob_A'] = de_vigged['B365A']

    result = merged[['match_id', 'prob_H', 'prob_D', 'prob_A']]
    return result

# =========================== MAIN WORKFLOW ================================
def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Loading matches_full.csv...")
    matches = pd.read_csv(MATCHES_FILE)
    matches['match_date'] = pd.to_datetime(matches['match_date'])
    matches['season_year'] = matches['match_date'].dt.year

    unique_years = sorted(matches['season_year'].unique())
    unique_competitions = matches['competition_id'].unique()
    print(f"Seasons: {unique_years}")
    print(f"Competitions: {unique_competitions}")

    all_odds = []
    for year in unique_years:
        season_code = season_code_from_year(year)
        for comp_id in unique_competitions:
            if comp_id not in LEAGUE_CODES:
                continue
            league_code = LEAGUE_CODES[comp_id]
            print(f"Processing {league_code} season {season_code}...")
            file_path = os.path.join(DATA_DIR, f"{league_code}_{season_code}.csv")
            if os.path.exists(file_path):
                print(f"   Already exists: {file_path}")
            else:
                url = f"https://www.football-data.co.uk/mmz4281/{season_code}/{league_code}.csv"
                success = download_file_with_retries(url, file_path)
                if not success:
                    print(f"   Failed to download {url} after retries.")
                    continue
            # Parse
            odds_df = parse_odds_df(file_path)
            if odds_df is not None:
                all_odds.append(odds_df)

    if not all_odds:
        print("No odds data could be loaded.")
        return

    combined_odds = pd.concat(all_odds, ignore_index=True)
    print(f"Total odds rows: {len(combined_odds)}")

    print("Joining odds to matches...")
    baseline = join_odds_to_matches(matches, combined_odds)
    if baseline.empty:
        print("No matches could be matched. Check team name mapping.")
        return

    baseline.to_csv(OUTPUT_BASELINE_FILE, index=False)
    print(f"Baseline saved to {OUTPUT_BASELINE_FILE}")
    print(f"Number of matches with odds: {len(baseline)}")

    # Evaluate on test set
    if os.path.exists(TEST_IDS_FILE):
        test_ids = pd.read_csv(TEST_IDS_FILE)['match_id'].tolist()
        baseline_test = baseline[baseline['match_id'].isin(test_ids)]
        test_labels = matches[matches['match_id'].isin(test_ids)][['match_id', 'result']]
        result_map = {'H':0, 'D':1, 'A':2}
        test_labels['label'] = test_labels['result'].map(result_map)
        merged_test = pd.merge(test_labels, baseline_test, on='match_id')
        if not merged_test.empty:
            probs = merged_test[['prob_H', 'prob_D', 'prob_A']].values
            y_true = merged_test['label'].values
            from sklearn.metrics import log_loss
            ll = log_loss(y_true, probs)
            rps_list = []
            for i in range(len(y_true)):
                true_label = y_true[i]
                cum_pred = np.cumsum(probs[i, :])
                cum_true = np.zeros(3)
                cum_true[true_label:] = 1
                rps_list.append(np.sum((cum_pred[:-1] - cum_true[:-1]) ** 2))
            rps = np.mean(rps_list)
            print("\n===== MARKET BASELINE EVALUATION ON TEST SET =====")
            print(f"Test matches with odds: {len(merged_test)}")
            print(f"Log-Loss: {ll:.6f}")
            print(f"RPS: {rps:.6f}")
        else:
            print("No test matches found in odds baseline.")
    else:
        print("No test_match_ids.csv found; skipping baseline evaluation.")

    print("Done.")

if __name__ == "__main__":
    main()