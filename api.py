"""
FastAPI service for in-play predictions.

Loads precomputed predictions and SHAP values into memory.

Endpoints:
- GET  /predict/{match_id}/{time}
- POST /predict/batch
- GET  /health

The API does NOT perform model inference at request time.
All prediction data is loaded and JSON-serialized at startup.
"""

import os
import time
from typing import List

import orjson
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel


# ============================================================
# CONFIG
# ============================================================

PRECOMPUTED_PATH = "out/precomputed_predictions.parquet"


# ============================================================
# LOAD PRECOMPUTED DATA
# ============================================================

if not os.path.exists(PRECOMPUTED_PATH):
    raise FileNotFoundError(
        f"Precomputed file not found: {PRECOMPUTED_PATH}"
    )

print(f"Loading precomputed predictions from: {PRECOMPUTED_PATH}")

df = pd.read_parquet(
    PRECOMPUTED_PATH,
    columns=[
        "match_id",
        "snapshot_time",
        "prob_H",
        "prob_D",
        "prob_A",
        "expected_margin",
        "pred_class",
        "top_shap_features",
        "top_shap_values",
    ],
)


# ============================================================
# PREPARE IN-MEMORY LOOKUP TABLE
# ============================================================

serialized_records = {}

for row in df.itertuples(index=False):
    key = (int(row.match_id), int(row.snapshot_time))

    record = {
        "match_id": key[0],
        "snapshot_time": key[1],
        "prob_H": float(row.prob_H),
        "prob_D": float(row.prob_D),
        "prob_A": float(row.prob_A),
        "expected_margin": float(row.expected_margin),
        "predicted_class": int(row.pred_class),
        "top_shap_features": list(row.top_shap_features),
        "top_shap_values": [float(v) for v in row.top_shap_values],
    }

    serialized_records[key] = orjson.dumps(record)

print(f"Loaded {len(serialized_records):,} prediction snapshots into memory.")


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="In-Play Prediction API",
    description="Real-time football predictions",
)


# ============================================================
# BATCH REQUEST MODEL
# ============================================================

class BatchPredictionRequest(BaseModel):
    match_id: int
    times: List[int]


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():
    return {"status": "ok"}


# ============================================================
# SINGLE PREDICTION (parameter renamed to avoid shadowing)
# ============================================================

@app.get("/predict/{match_id}/{snapshot_time}")
async def predict(match_id: int, snapshot_time: int):
    start = time.perf_counter()

    key = (match_id, snapshot_time)
    serialized = serialized_records.get(key)

    if serialized is None:
        raise HTTPException(
            status_code=404,
            detail="Snapshot not found for this match/time",
        )

    server_latency_ms = (time.perf_counter() - start) * 1000.0

    return Response(
        content=serialized,
        media_type="application/json",
        headers={
            "X-Server-Latency-ms": f"{server_latency_ms:.3f}",
        },
    )


# ============================================================
# BATCH PREDICTIONS
# ============================================================

@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    start = time.perf_counter()
    results = []

    for snapshot_time in request.times:
        key = (request.match_id, snapshot_time)
        serialized = serialized_records.get(key)

        if serialized is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Snapshot not found for match {request.match_id}, "
                    f"time {snapshot_time}"
                ),
            )

        results.append(serialized)

    content = b"[" + b",".join(results) + b"]"

    server_latency_ms = (time.perf_counter() - start) * 1000.0

    return Response(
        content=content,
        media_type="application/json",
        headers={
            "X-Server-Latency-ms": f"{server_latency_ms:.3f}",
        },
    )