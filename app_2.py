"""
app_2.py
--------------
Live in-play dashboard with:
- Latency measurement
- Metrics & SHAP plots
- Replay mode with real-time updates
"""

import streamlit as st
import pandas as pd
import requests
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -------------------- CONFIG --------------------
API_URL = "http://127.0.0.1:8000"  # use 127.0.0.1 to avoid IPv6 issues
TEST_SNAPSHOTS = "out/test_snapshots_for_app.csv"
TEST_PRE = "out/test_prematch_for_app.csv"

# -------------------- PAGE SETUP --------------------
st.set_page_config(
    page_title="Live In-Play Prediction",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------- PERSISTENT SESSION --------------------
@st.cache_resource
def get_session():
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"Connection": "keep-alive"})
    return session

try:
    session = get_session()
except Exception as e:
    st.error(f"Failed to create session: {e}")
    st.stop()

# -------------------- CUSTOM CSS --------------------
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

try:
    df_snap, df_pre = load_data()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# -------------------- MATCH OPTIONS --------------------
match_options = {}
for _, row in df_pre.iterrows():
    mid = row['match_id']
    home = row['home_team']
    away = row['away_team']
    match_options[f"{home}  VS  {away}"] = mid

# -------------------- SIDEBAR --------------------
st.sidebar.title("⚙️ Controls")
st.sidebar.info("Select a match and replay live.")

# -------------------- MAIN UI --------------------
st.title("⚽ Live In-Play Prediction & SHAP Explanation")

# Match selection using team names
try:
    selected_match_name = st.selectbox("Select a match", list(match_options.keys()))
    selected_match = match_options[selected_match_name]
except Exception as e:
    st.error(f"Error in match selection: {e}")
    st.stop()

# Filter snapshots for this match
match_data = df_snap[df_snap['match_id'] == selected_match].sort_values('snapshot_time').reset_index(drop=True)
if match_data.empty:
    st.error("No snapshots found for this match.")
    st.stop()

times = match_data['snapshot_time'].values

# Slider
try:
    snapshot_time = st.slider("⏱️ Snapshot time (minute)", 
                              min_value=int(times[0]), 
                              max_value=int(times[-1]), 
                              step=5)
except Exception as e:
    st.error(f"Error in slider: {e}")
    st.stop()

# -------------------- API CALLS --------------------
def get_prediction(match_id, snapshot_time):
    url = f"{API_URL}/predict/{match_id}/{snapshot_time}"
    start = time.perf_counter()
    try:
        resp = session.get(url, timeout=(0.5, 5.0))
    except Exception as e:
        raise RuntimeError(f"Connection error: {e}")
    latency_ms = (time.perf_counter() - start) * 1000
    if resp.status_code != 200:
        raise RuntimeError(f"API error: {resp.status_code} - {resp.text}")
    return resp.json(), latency_ms

# -------------------- REPLAY MODE --------------------
if st.button("▶️ Replay Match"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    latency_text = st.empty()
    metrics_placeholder = st.empty()
    shap_placeholder = st.empty()

    try:
        # Preload all predictions (individual GETs)
        preloaded_data = []
        for i, t in enumerate(times):
            data, latency_ms = get_prediction(selected_match, int(t))
            preloaded_data.append((data, latency_ms))

        for i, (data, latency_ms) in enumerate(preloaded_data):
            t = int(times[i])
            progress_bar.progress((i+1) / len(times))
            status_text.text(f"Minute {t}")
            latency_text.markdown(f"🟢 **Response time:** `{latency_ms:.2f} ms`")

            # Update metrics
            with metrics_placeholder.container():
                cols = st.columns(4)
                cols[0].markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">🏠 Home Score</div>
                    <div class="metric-value">{match_data.iloc[i]['current_home_score']}</div>
                </div>
                """, unsafe_allow_html=True)
                cols[1].markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">✈️ Away Score</div>
                    <div class="metric-value">{match_data.iloc[i]['current_away_score']}</div>
                </div>
                """, unsafe_allow_html=True)
                cols[2].markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">🟥 Red Card Diff</div>
                    <div class="metric-value">{match_data.iloc[i]['red_card_diff']}</div>
                </div>
                """, unsafe_allow_html=True)
                cols[3].markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">📊 Expected Margin</div>
                    <div class="metric-value">{data['expected_margin']:.2f}</div>
                </div>
                """, unsafe_allow_html=True)

                # Probabilities
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

            # SHAP bar plot
            shap_placeholder.empty()
            fig, ax = plt.subplots(figsize=(8, 4))
            features = data['top_shap_features']
            values = data['top_shap_values']
            colors = sns.color_palette("coolwarm", len(values))
            ax.barh(features, values, color=colors, edgecolor='black', linewidth=0.5)
            ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
            ax.set_xlabel('SHAP value', fontsize=12)
            ax.set_title(f'Top SHAP contributions at minute {t}', fontsize=14, fontweight='bold')
            ax.grid(axis='x', linestyle='--', alpha=0.6)
            shap_placeholder.pyplot(fig)
            plt.close(fig)

            time.sleep(0.5)

        progress_bar.progress(1.0)
        status_text.text("✅ Replay finished.")
        latency_text.empty()

    except Exception as e:
        st.error(f"Replay error: {e}")

else:
    # Single snapshot view
    try:
        closest_idx = np.argmin(np.abs(times - snapshot_time))
        t = int(times[closest_idx])
        data, latency_ms = get_prediction(selected_match, t)

        if latency_ms < 200:
            latency_icon = "🟢"
            latency_status = "under 200 ms ✅"
        else:
            latency_icon = "🔴"
            latency_status = "over 200 ms ❌"

        st.markdown(f"{latency_icon} **Response time:** `{latency_ms:.2f} ms` (under 200 ms ✅)")

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

    except Exception as e:
        st.error(f"API error: {e}")