"""
api.py
------
FastAPI service for in-play predictions.
Loads precomputed predictions and SHAP values.
Endpoints:
- GET /predict/{match_id}/{time}  → returns JSON with probabilities, margin, top SHAP
- GET /health → simple health check
"""

import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

# -------------------- CONFIG --------------------
PRECOMPUTED_PATH = 'precomputed_predictions.parquet'

# Load precomputed data once at startup
if not os.path.exists(PRECOMPUTED_PATH):
    raise FileNotFoundError(f"Precomputed file not found: {PRECOMPUTED_PATH}")

df = pd.read_parquet(PRECOMPUTED_PATH)

# Build a lookup dictionary with pure Python types
lookup = {}
for idx, row in df.iterrows():
    key = (row['match_id'], row['snapshot_time'])
    record = {
        "match_id": int(row['match_id']),
        "snapshot_time": int(row['snapshot_time']),
        "prob_H": float(row['prob_H']),
        "prob_D": float(row['prob_D']),
        "prob_A": float(row['prob_A']),
        "expected_margin": float(row['expected_margin']),
        "predicted_class": int(row['pred_class']),
        "top_shap_features": list(row['top_shap_features']),           # ensure list
        "top_shap_values": [float(v) for v in row['top_shap_values']]  # ensure list of floats
    }
    lookup[key] = record

app = FastAPI(title="In-Play Prediction API", description="Real-time football predictions")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/predict/{match_id}/{time}")
async def predict(match_id: int, time: int):
    key = (match_id, time)
    if key not in lookup:
        raise HTTPException(status_code=404, detail="Snapshot not found for this match/time")
    record = lookup[key]
    return JSONResponse(content=record)