# Forecasting Competitive Football – Pre-Match & In-Play Prediction from Raw Event Data

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A complete machine learning pipeline that predicts football match outcomes and goal margins—both before kick‑off and live during the game—using StatsBomb event data. The project covers data integration, feature engineering, model training (including custom implementations of two research papers), calibration, interpretability (SHAP), and a bonus real‑time prediction service.

---

## Table of Contents

- [Overview](#overview)
- [Data](#data)
- [Features & Models](#features--models)
- [Reproducing the Results](#reproducing-the-results)
- [Results Summary](#results-summary)
- [Bonus: Real‑Time Prediction Service](#bonus-real‑time-prediction-service)
- [Requirements & Installation](#requirements--installation)
- [Project Structure](#project-structure)
- [License](#license)
- [Citation](#citation)

---

## Overview

This project implements three predictive models for football:

1. **Model 1 – Pre‑Match Outcome Classification**  
   Predicts the probability of Home Win / Draw / Away Win using only information available before kick‑off.

2. **Model 2 – Pre‑Match Goal‑Margin Regression**  
   Predicts the signed goal difference (home – away), clipped to [-5, +5].

3. **Model 3 – In‑Play (Live) Prediction**  
   Re‑estimates probabilities and margin at multiple snapshots during a match, using both pre‑match features and live event statistics (score, red cards, recent shots/passes/pressures, momentum).

The pipeline is built from scratch using the **StatsBomb Open Data** repository, with additional odds integration from **Football‑Data.co.uk** for market baseline comparison.

---

## Data

- **Source**: [StatsBomb Open Data](https://github.com/statsbomb/open-data)
- **Competitions**: La Liga (`competition_id = 11`) and Premier League (`competition_id = 2`) – over 800 matches spanning multiple seasons.
- **Event granularity**: ~4,000 events per match (passes, shots, pressures, cards, etc.).
- **Integration**: Five JSON file families (competitions, matches, events, lineups, 360) are parsed and joined into a relational store. The final datasets are saved as both Parquet and CSV.

The pre‑match dataset contains **899 matches** with 18 features.  
The in‑play snapshot dataset contains **~17,000 training snapshots** (every 5 minutes) with 8 live features combined with the pre‑match vector.

---

## Features & Models

### Pre‑Match Features
- Rolling averages (last 5 matches) of goals scored/conceded, points, goal difference per team.
- Rest days since last match.
- Difference variables (home – away).
- Formations (from lineups) and a binary indicator for same formation.

### In‑Play Features
- Current home / away scores.
- Red card difference (home – away).
- Recent event counts (last 5 min): shots, passes, pressures.
- Momentum: home shots / total shots in last 5 min.

### Model Suite
All models are trained and evaluated on both pre‑match and in‑play tasks.

- **Classification**: Dummy, Kernel SVM (exact & approximate), Random Forest, GBM, XGBoost, LightGBM, and our custom **IFX‑XGBoost** (see Papers).
- **Regression**: Dummy, Kernel Ridge (exact & approximate), Kernel SVR, Random Forest, GBM, XGBoost, LightGBM, and IFX‑XGBoost.

### Papers Reproduced
- **P1: PF‑SMOTE** (Parameter‑Free SMOTE) – A variant that avoids setting *k* and reduces noisy samples by separating minority points into *safe* (interpolation) and *boundary* (Gaussian noise) categories.
- **P2: IFX‑XGBoost** (Iterative Feature eXclusion) – A boosting algorithm that iteratively removes the most important feature (using SHAP) and continues training to prevent feature starvation.

Both methods are implemented from scratch and integrated into the pipeline.

---

## Reproducing the Results

1. **Data preparation** – Run `final_data_prep_complete.py` to download/parse StatsBomb JSONs and generate the pre‑match and snapshot datasets. Outputs are saved in `processed_data/`.

2. **Training & evaluation** – Run `main.py`. This script:
   - Loads the processed data.
   - Defines features and targets.
   - Trains all models (with hyperparameter tuning via GridSearchCV).
   - Calibrates classifiers (Platt scaling on validation set).
   - Computes all metrics and generates figures (saved in `figures/`).
   - Saves the best models (`best_inplay_clf.pkl`, `best_inplay_reg.pkl`) and exports test snapshots for the bonus API.

3. **Odds integration (market baseline)** – Run `odds_integration.py` to download Football‑Data.co.uk odds, join them to matches, compute de‑vigged probabilities, and evaluate the baseline on the test set.

4. **Bonus API & dashboard** – (Optional) Run `precompute_predictions.py` to precompute predictions and SHAP values for all test snapshots, then start the FastAPI service (`uvicorn api:app --host 0.0.0.0 --port 8000`) and the Streamlit dashboard (`streamlit run app_2.py`).

All code uses a fixed random seed (`42`) for reproducibility.

---

## Results Summary

| Task | Best Model | Metric | Value |
|------|------------|--------|-------|
| Pre‑Match Classification | KernelSVM / XGBoost | Accuracy | 69.4% |
| Pre‑Match Classification | KernelSVM | Calibrated Log‑Loss | 1.154 |
| Pre‑Match Regression | RandomForest | MAE | 1.484 |
| In‑Play Classification | XGBoost | Accuracy | 67.6% |
| In‑Play Classification | XGBoost | Calibrated Log‑Loss | 0.774 |
| In‑Play Regression | XGBoost | MAE | 1.832 |
| Market Baseline (odds) | – | Log‑Loss | 0.782 |

- The **in‑play classification model** outperforms the de‑vigged market baseline (log‑loss 0.774 vs 0.782) and shows consistent improvement as the match progresses.
- The **in‑play regression model** fails to beat its own pre‑match prior, indicating that coarse live features are insufficient for precise margin prediction.
- **Calibration** (Platt scaling) significantly improves probability estimates, as shown in reliability diagrams.
- **SHAP analysis** reveals that the model shifts from relying on historical averages to live score/event features as the match unfolds.

---

## Bonus: Real‑Time Prediction Service

A FastAPI service and Streamlit dashboard are included for demonstration purposes:

- **API**: Endpoint `/predict/{match_id}/{time}` returns precomputed probabilities, expected margin, and top‑5 SHAP contributions in under 200 ms.
- **Dashboard**: Replays a selected match snapshot‑by‑snapshot, showing evolving metrics, score, and SHAP bar plots.

Run `precompute_predictions.py` once, then start the API and the Streamlit app (see instructions above).

---

## Requirements & Installation

Install the required packages:

```bash
pip install -r requirements.txt

---

## Project Structure

├── final_data_prep_complete.py      # Data ingestion and feature engineering
├── main.py                           # Main training and evaluation script
├── PF_SMOTE.py                       # Implementation of Paper P1
├── IFX_model.py                      # Implementation of Paper P2
├── odds_integration.py               # Downloads odds and computes market baseline
├── precompute_predictions.py         # Precomputes predictions for bonus API
├── api.py                            # FastAPI service (bonus)
├── app_2.py                          # Streamlit dashboard (bonus)
├── processed_data/                   # Generated datasets (Parquet & CSV)
├── figures/                          # All output plots and figures
├── requirements.txt
└── README.md

---

## License

This project is licensed under the MIT License

---

## Citation

StatsBomb Open Data: https://github.com/statsbomb/open-data
Football‑Data.co.uk: https://www.football-data.co.uk

The paper reproductions are based on:

    P1: Parameter‑Free SMOTE (https://www.sciencedirect.com/science/article/abs/pii/S0925231222005495)

    P2: Iterative Feature eXclusion for Gradient Boosting (https://www.sciencedirect.com/science/article/abs/pii/S0950705124001813)

### For a detailed analysis, refer to the project report (report.pdf) included in the repository.
