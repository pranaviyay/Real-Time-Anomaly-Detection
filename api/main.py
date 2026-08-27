"""
FastAPI service for the anomaly-detection model.

Endpoints
---------
- GET  /         service info
- GET  /health   liveness probe
- POST /predict  score a single transaction (ad-hoc; logs alerts; updates state)
- POST /ingest   bulk endpoint used by the streaming consumer; same scoring,
                 returns the score so the consumer can log per-transaction
- GET  /alerts   recent fraud alerts (paginated)
- GET  /stats    counters + threshold + score distribution metrics
- GET  /score-history  recent scores for the dashboard's time-series chart

Single source of truth: this process owns the FeatureStore, the running
counters, and the alert log. The streaming consumer no longer scores locally —
it forwards each transaction to /ingest. That fixes the previous problem where
"Total Transactions" was always 0 (the API and consumer had separate state).
"""

from __future__ import annotations

import os
import csv
import json
import pickle
import sys
import threading
from collections import deque
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from streaming.feature_store import FeatureStore  # noqa: E402

CONFIG_PATH = os.environ.get("CONFIG_PATH", os.path.join(BASE_DIR, "configs/config.yaml"))
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

MODEL_PATH = os.path.join(BASE_DIR, CFG["paths"]["model_pkl"])
FEATURES_PATH = os.path.join(BASE_DIR, CFG["paths"]["feature_columns"])
THRESHOLD_PATH = os.path.join(BASE_DIR, CFG["paths"]["threshold"])
CAT_PATH = os.path.join(BASE_DIR, CFG["paths"]["label_encoder"])
LOC_PATH = os.path.join(BASE_DIR, CFG["paths"]["location_encoder"])
LOG_PATH = os.path.join(BASE_DIR, CFG["paths"]["fraud_log"])

with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)
with open(FEATURES_PATH) as f:
    FEATURES = json.load(f)
with open(THRESHOLD_PATH) as f:
    THRESHOLD = float(json.load(f)["threshold"])
with open(CAT_PATH) as f:
    CAT_CLASSES = json.load(f)
with open(LOC_PATH) as f:
    LOC_CLASSES = json.load(f)

CAT_LOOKUP = {c: i for i, c in enumerate(CAT_CLASSES)}
LOC_LOOKUP = {c: i for i, c in enumerate(LOC_CLASSES)}

ALERT_FIELDS = [
    "transaction_id", "user_id", "amount", "location",
    "fraud_score", "is_fraud", "flag", "timestamp",
]

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# ── runtime state (protected by a lock; FastAPI workers are single-process by default with uvicorn) ──
_lock = threading.Lock()
feature_store = FeatureStore()
score_history: deque = deque(maxlen=500)   # recent (timestamp, score, is_fraud) for the chart
fraud_count = 0
score_sum = 0.0  # sum over ALL transactions, not just fraud — for honest avg score


def _bootstrap_fraud_count_from_csv() -> None:
    """If the alerts CSV already has rows from a previous run, pick up the count."""
    global fraud_count
    if os.path.isfile(LOG_PATH):
        try:
            with open(LOG_PATH) as f:
                fraud_count = sum(1 for _ in csv.DictReader(f))
        except Exception:
            fraud_count = 0


_bootstrap_fraud_count_from_csv()

app = FastAPI(title="Anomaly Detection API", version="2.1")


# ──────────────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────────────

class Transaction(BaseModel):
    transaction_id: str
    user_id: int
    amount: float = Field(..., gt=0)
    location: str
    merchant_category: Optional[str] = "Groceries"
    timestamp: str
    is_fraud: Optional[int] = 0


class PredictionResponse(BaseModel):
    transaction_id: str
    user_id: int
    amount: float
    location: str
    fraud_score: float
    is_fraud: int
    flag: str
    threshold: float


# ──────────────────────────────────────────────────────────────────────────
# Internal: scoring + state update
# ──────────────────────────────────────────────────────────────────────────

def _score_and_update(txn_dict: dict) -> tuple[float, int, str]:
    """Compute features, score, update counters/history/log under the lock."""
    global fraud_count, score_sum

    features = feature_store.compute_features(txn_dict)
    features["merchant_category_encoded"] = CAT_LOOKUP.get(
        txn_dict.get("merchant_category", "Groceries"), 0,
    )
    features["location_encoded"] = LOC_LOOKUP.get(txn_dict["location"], 0)

    row = {col: features.get(col, np.nan) for col in FEATURES}
    X = pd.DataFrame([row])

    score = float(MODEL.predict_proba(X)[0][1])
    is_fraud = int(score >= THRESHOLD)
    flag = "FRAUD" if is_fraud else "OK"

    score_sum += score
    score_history.append({
        "timestamp": txn_dict["timestamp"],
        "score": round(score, 4),
        "is_fraud": is_fraud,
        "user_id": txn_dict["user_id"],
        "amount": txn_dict["amount"],
        "location": txn_dict["location"],
    })

    if is_fraud:
        fraud_count += 1
        file_exists = os.path.isfile(LOG_PATH)
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ALERT_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "transaction_id": txn_dict["transaction_id"],
                "user_id": txn_dict["user_id"],
                "amount": txn_dict["amount"],
                "location": txn_dict["location"],
                "fraud_score": round(score, 4),
                "is_fraud": is_fraud,
                "flag": flag,
                "timestamp": txn_dict["timestamp"],
            })

    return score, is_fraud, flag


# ──────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "anomaly-detection-api",
        "version": "2.1",
        "model_loaded": True,
        "threshold": round(THRESHOLD, 4),
        "n_features": len(FEATURES),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(txn: Transaction) -> PredictionResponse:
    txn_dict = txn.model_dump()
    with _lock:
        score, is_fraud, flag = _score_and_update(txn_dict)
    return PredictionResponse(
        transaction_id=txn_dict["transaction_id"],
        user_id=txn_dict["user_id"],
        amount=txn_dict["amount"],
        location=txn_dict["location"],
        fraud_score=round(score, 4),
        is_fraud=is_fraud,
        flag=flag,
        threshold=round(THRESHOLD, 4),
    )


@app.post("/ingest", response_model=PredictionResponse)
def ingest(txn: Transaction) -> PredictionResponse:
    """Same as /predict; named distinctly so streaming traffic is observable
    in metrics/logs separately from ad-hoc API calls."""
    return predict(txn)


@app.get("/alerts")
def get_alerts(limit: int = Query(500, ge=1, le=10000)):
    if not os.path.isfile(LOG_PATH):
        return {"alerts": [], "count": 0}
    rows: list[dict] = []
    with open(LOG_PATH) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    rows.reverse()  # most recent first
    return {"alerts": rows[:limit], "count": len(rows)}


@app.get("/stats")
def get_stats():
    with _lock:
        total = feature_store.get_total_transactions()
        avg_score = round(score_sum / total, 4) if total else None
        rate = round(fraud_count / total, 4) if total else None
        f_count = fraud_count

    return {
        "total_transactions": total,
        "fraud_detected": f_count,
        "fraud_rate": rate,
        "fraud_score_avg": avg_score,
        "threshold": round(THRESHOLD, 4),
    }


@app.get("/score-history")
def get_score_history(limit: int = Query(200, ge=1, le=500)):
    """Recent scores (fraud + non-fraud) for the dashboard time-series chart.
    Snappier than /alerts because it's already in memory and includes non-fraud."""
    with _lock:
        items = list(score_history)
    return {"history": items[-limit:]}
