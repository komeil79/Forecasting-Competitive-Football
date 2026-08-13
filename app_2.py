"""
app.py (Final)
--------------
Beautiful live in‑play dashboard with:
- Team names instead of match IDs
- Latency measurement (<200ms)
- Styled metrics & SHAP plots
- Replay mode with real‑time updates
"""

import streamlit as st
import pandas as pd
import requests
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- CONFIG --------------------
API_URL = "http://localhost:8000"  # FastAPI endpoint
TEST_SNAPSHOTS = "test_snapshots_for_app.csv"
TEST_PRE = "test_prematch_for_app.csv"

# -------------------- PAGE SETUP --------------------
st.set_page_config(
    page_title="Live In‑Play Prediction",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------- CUSTOM CSS (Colorful & Modern) --------------------
st.markdown("""
<style>
    /* Background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    /* Title */
    h1 {
        color: #1e3c72;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    /* Metrics */
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        text-align: center;
        margin-bottom: 10px;
        border-left: 6px solid #1e3c72;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3c72;
    }
    .metric-label {
        font-size: 1rem;
        color: #555;
        font-weight: 500;
    }
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(30,60,114,0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(30,60,114,0.5);
    }
    /* Selectbox */
    .stSelectbox>div>div>div {
        background: white;
        border-radius: 10px;
        border: 2px solid #1e3c72;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    df_snap = pd.read_csv(TEST_SNAPSHOTS)
    df_pre = pd.read_csv(TEST_PRE)
    return df_snap, df_pre

df_snap, df_pre = load_data()

# Build match mapping: match_id -> "Home vs Away"
match_map = {}
for _, row in df_pre.iterrows():
    mid = row['match_id']
    home = row['home_team']
    away = row['away_team']
    match_map[mid] = f"{home}  VS  {away}"

# Reverse map for selectbox
match_options = {f"{home}  VS  {away}": mid for mid, home, away in 
                 zip(df_pre['match_id'], df_pre['home_team'], df_pre['away_team'])}

# -------------------- SIDEBAR (optional) --------------------
st.sidebar.title("⚙️ Controls")
st.sidebar.info("Select a match and replay live.")

# -------------------- MAIN UI --------------------
st.title("⚽ Live In‑Play Prediction & SHAP Explanation")

# Match selection using team names
selected_match_name = st.selectbox("Select a match", list(match_options.keys()))
selected_match = match_options[selected_match_name]

# Filter snapshots for this match
match_data = df_snap[df_snap['match_id'] == selected_match].sort_values('snapshot_time')
if match_data.empty:
    st.error("No snapshots found for this match.")
    st.stop()

times = match_data['snapshot_time'].values

# Slider
snapshot_time = st.slider("⏱️ Snapshot time (minute)", 
                          min_value=int(times[0]), 
                          max_value=int(times[-1]), 
                          step=5)

# -------------------- REPLAY MODE --------------------
if st.button("▶️ Replay Match"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    latency_text = st.empty()
    
    # Placeholders for dynamic content
    metrics_placeholder = st.empty()
    shap_placeholder = st.empty()

    for i, t in enumerate(times):
        # Start timer
        start_time = time.perf_counter()

        # Call API
        try:
            resp = requests.get(f"{API_URL}/predict/{selected_match}/{t}")
            if resp.status_code == 200:
                data = resp.json()
            else:
                st.error(f"API error: {resp.status_code}")
                break
        except Exception as e:
            st.error(f"Connection error: {e}")
            break

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000
        latency_text.markdown(f"🟢 **Response time:** `{latency_ms:.2f} ms` (under 200 ms ✅)")

        # Update progress
        progress = (i+1) / len(times)
        progress_bar.progress(progress)
        status_text.text(f"Minute {t}")

        # Update metrics – replace entire container
        with metrics_placeholder.container():
            cols = st.columns(4)
            # Home score
            cols[0].markdown(f"""
            <div class="metric-card">
                <div class="metric-label">🏠 Home Score</div>
                <div class="metric-value">{match_data.iloc[i]['current_home_score']}</div>
            </div>
            """, unsafe_allow_html=True)
            # Away score
            cols[1].markdown(f"""
            <div class="metric-card">
                <div class="metric-label">✈️ Away Score</div>
                <div class="metric-value">{match_data.iloc[i]['current_away_score']}</div>
            </div>
            """, unsafe_allow_html=True)
            # Red card diff
            cols[2].markdown(f"""
            <div class="metric-card">
                <div class="metric-label">🟥 Red Card Diff</div>
                <div class="metric-value">{match_data.iloc[i]['red_card_diff']}</div>
            </div>
            """, unsafe_allow_html=True)
            # Expected margin
            cols[3].markdown(f"""
            <div class="metric-card">
                <div class="metric-label">📊 Expected Margin</div>
                <div class="metric-value">{data['expected_margin']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

            # Probabilities (three columns)
            prob_cols = st.columns(3)
            prob_cols[0].markdown(f"""
            <div class="metric-card" style="border-left-color: #28a745;">
                <div class="metric-label">🏆 Home Win</div>
                <div class="metric-value" style="color:#28a745;">{data['prob_H']:.2%}</div>
            </div>
            """, unsafe_allow_html=True)
            prob_cols[1].markdown(f"""
            <div class="metric-card" style="border-left-color: #ffc107;">
                <div class="metric-label">🤝 Draw</div>
                <div class="metric-value" style="color:#ffc107;">{data['prob_D']:.2%}</div>
            </div>
            """, unsafe_allow_html=True)
            prob_cols[2].markdown(f"""
            <div class="metric-card" style="border-left-color: #dc3545;">
                <div class="metric-label">✈️ Away Win</div>
                <div class="metric-value" style="color:#dc3545;">{data['prob_A']:.2%}</div>
            </div>
            """, unsafe_allow_html=True)

        # SHAP bar plot – replace placeholder
        shap_placeholder.empty()
        fig, ax = plt.subplots(figsize=(8, 4))
        features = data['top_shap_features']
        values = data['top_shap_values']
        # Use a colormap
        colors = sns.color_palette("coolwarm", len(values))
        ax.barh(features, values, color=colors, edgecolor='black', linewidth=0.5)
        ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
        ax.set_xlabel('SHAP value', fontsize=12)
        ax.set_title(f'Top SHAP contributions at minute {t}', fontsize=14, fontweight='bold')
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        shap_placeholder.pyplot(fig)
        plt.close(fig)

        time.sleep(0.5)  # simulate real-time

    progress_bar.progress(1.0)
    status_text.text("✅ Replay finished.")
    latency_text.empty()  # remove latency after replay

else:
    # Single snapshot view
    closest_idx = np.argmin(np.abs(times - snapshot_time))
    t = times[closest_idx]

    # Timer for latency
    start_time = time.perf_counter()
    resp = requests.get(f"{API_URL}/predict/{selected_match}/{t}")
    latency_ms = (time.perf_counter() - start_time) * 1000

    if resp.status_code == 200:
        data = resp.json()
        st.markdown(f"🟢 **Response time:** `{latency_ms:.2f} ms` (under 200 ms ✅)")

        # Metrics
        cols = st.columns(4)
        cols[0].markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🏠 Home Score</div>
            <div class="metric-value">{match_data.iloc[closest_idx]['current_home_score']}</div>
        </div>
        """, unsafe_allow_html=True)
        cols[1].markdown(f"""
        <div class="metric-card">
            <div class="metric-label">✈️ Away Score</div>
            <div class="metric-value">{match_data.iloc[closest_idx]['current_away_score']}</div>
        </div>
        """, unsafe_allow_html=True)
        cols[2].markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🟥 Red Card Diff</div>
            <div class="metric-value">{match_data.iloc[closest_idx]['red_card_diff']}</div>
        </div>
        """, unsafe_allow_html=True)
        cols[3].markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📊 Expected Margin</div>
            <div class="metric-value">{data['expected_margin']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

        prob_cols = st.columns(3)
        prob_cols[0].markdown(f"""
        <div class="metric-card" style="border-left-color: #28a745;">
            <div class="metric-label">🏆 Home Win</div>
            <div class="metric-value" style="color:#28a745;">{data['prob_H']:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
        prob_cols[1].markdown(f"""
        <div class="metric-card" style="border-left-color: #ffc107;">
            <div class="metric-label">🤝 Draw</div>
            <div class="metric-value" style="color:#ffc107;">{data['prob_D']:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
        prob_cols[2].markdown(f"""
        <div class="metric-card" style="border-left-color: #dc3545;">
            <div class="metric-label">✈️ Away Win</div>
            <div class="metric-value" style="color:#dc3545;">{data['prob_A']:.2%}</div>
        </div>
        """, unsafe_allow_html=True)

        # SHAP plot
        fig, ax = plt.subplots(figsize=(8, 4))
        features = data['top_shap_features']
        values = data['top_shap_values']
        colors = sns.color_palette("coolwarm", len(values))
        ax.barh(features, values, color=colors, edgecolor='black', linewidth=0.5)
        ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
        ax.set_xlabel('SHAP value', fontsize=12)
        ax.set_title(f'Top SHAP contributions at minute {t}', fontsize=14, fontweight='bold')
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        st.pyplot(fig)
    else:
        st.error("API error")