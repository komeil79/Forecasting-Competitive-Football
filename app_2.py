"""
app.py (updated)
----------------
Streamlit dashboard that calls the FastAPI service.
Replays a match from the test set as if live.
"""

import streamlit as st
import pandas as pd
import requests
import time
import numpy as np
import matplotlib.pyplot as plt

# -------------------- CONFIG --------------------
API_URL = "http://localhost:8000"  # adjust if deployed elsewhere
TEST_SNAPSHOTS = "test_snapshots_for_app.csv"
TEST_PRE = "test_prematch_for_app.csv"

st.set_page_config(page_title="Live In-Play Prediction", layout="wide")
st.title("⚽ Live In-Play Prediction & SHAP Explanation")

# Load data (only for match list and event markers)
df_snap = pd.read_csv(TEST_SNAPSHOTS)
df_pre = pd.read_csv(TEST_PRE)

match_ids = df_snap['match_id'].unique()
selected_match = st.selectbox("Select a match ID", match_ids)

# Filter snapshots for this match
match_data = df_snap[df_snap['match_id'] == selected_match].sort_values('snapshot_time')
if match_data.empty:
    st.error("No snapshots found for this match.")
    st.stop()

times = match_data['snapshot_time'].values
snapshot_time = st.slider("Snapshot time (minute)", 
                          min_value=int(times[0]), 
                          max_value=int(times[-1]), 
                          step=5)

# -------------------- REPLAY MODE --------------------
if st.button("▶️ Replay Match"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    # Create placeholders for dynamic content
    metrics_placeholder = st.empty()
    shap_placeholder = st.empty()

    for i, t in enumerate(times):
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

        # Update progress
        progress = (i+1) / len(times)
        progress_bar.progress(progress)
        status_text.text(f"Minute {t}")

        # Update metrics – replace entire container
        with metrics_placeholder.container():
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Home Score", match_data.iloc[i]['current_home_score'])
                st.metric("Away Score", match_data.iloc[i]['current_away_score'])
                st.metric("Red Card Diff", match_data.iloc[i]['red_card_diff'])
            with col2:
                st.metric("Predicted Home Win", f"{data['prob_H']:.2%}")
                st.metric("Predicted Draw", f"{data['prob_D']:.2%}")
                st.metric("Predicted Away Win", f"{data['prob_A']:.2%}")
                st.metric("Expected Margin", f"{data['expected_margin']:.2f}")

        # SHAP bar plot – replace placeholder
        shap_placeholder.empty()  # clear previous plot
        fig, ax = plt.subplots(figsize=(6, 3))
        features = data['top_shap_features']
        values = data['top_shap_values']
        y_pos = np.arange(len(features))
        ax.barh(y_pos, values, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('SHAP value')
        ax.set_title(f'Top SHAP contributions at minute {t}')
        shap_placeholder.pyplot(fig)
        plt.close(fig)

        time.sleep(0.5)  # simulate real-time (adjust as needed)

    progress_bar.progress(1.0)
    status_text.text("Replay finished.")
else:
    # Single snapshot view (same as before but calling API)
    # Find closest snapshot
    closest_idx = np.argmin(np.abs(times - snapshot_time))
    t = times[closest_idx]
    resp = requests.get(f"{API_URL}/predict/{selected_match}/{t}")
    if resp.status_code == 200:
        data = resp.json()
        # Display metrics and SHAP
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Home Score", match_data.iloc[closest_idx]['current_home_score'])
            st.metric("Away Score", match_data.iloc[closest_idx]['current_away_score'])
            st.metric("Red Card Diff", match_data.iloc[closest_idx]['red_card_diff'])
        with col2:
            st.metric("Predicted Home Win", f"{data['prob_H']:.2%}")
            st.metric("Predicted Draw", f"{data['prob_D']:.2%}")
            st.metric("Predicted Away Win", f"{data['prob_A']:.2%}")
            st.metric("Expected Margin", f"{data['expected_margin']:.2f}")

        # SHAP bar plot
        fig, ax = plt.subplots(figsize=(6, 3))
        features = data['top_shap_features']
        values = data['top_shap_values']
        ax.barh(np.arange(len(features)), values, align='center')
        ax.set_yticks(np.arange(len(features)))
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('SHAP value')
        ax.set_title(f'Top SHAP contributions at minute {t}')
        st.pyplot(fig)
    else:
        st.error("API error")