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
import orjson
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

# -------------------- CONFIG --------------------
PRECOMPUTED_PATH = 'out/precomputed_predictions.parquet'

# Load precomputed data once at startup
if not os.path.exists(PRECOMPUTED_PATH):
    raise FileNotFoundError(f"Precomputed file not found: {PRECOMPUTED_PATH}")

df = pd.read_parquet(PRECOMPUTED_PATH, columns=[
    'match_id', 'snapshot_time', 'prob_H', 'prob_D', 'prob_A',
    'expected_margin', 'pred_class', 'top_shap_features', 'top_shap_values'
])

records = {}
serialized_records = {}

for _, row in df.iterrows():
    key = (int(row['match_id']), int(row['snapshot_time']))
    record = {
        "match_id": key[0],
        "snapshot_time": key[1],
        "prob_H": float(row['prob_H']),
        "prob_D": float(row['prob_D']),
        "prob_A": float(row['prob_A']),
        "expected_margin": float(row['expected_margin']),
        "predicted_class": int(row['pred_class']),
        "top_shap_features": list(row['top_shap_features']),
        "top_shap_values": [float(v) for v in row['top_shap_values']]
    }
    records[key] = record
    serialized_records[key] = orjson.dumps(record)

app = FastAPI(title="In-Play Prediction API", description="Real-time football predictions")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/predict/{match_id}/{time}")
async def predict(match_id: int, time: int):
    key = (match_id, time)
    if key not in serialized_records:
        raise HTTPException(status_code=404, detail="Snapshot not found for this match/time")
    
    return Response(content=serialized_records[key], media_type="application/json")