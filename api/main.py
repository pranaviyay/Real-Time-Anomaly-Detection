import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pickle
import yaml
import csv
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from streaming.feature_store import FeatureStore

with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/feature_columns.json") as f:
    FEATURES = json.load(f)

with open("model/threshold.json") as f:
    THRESHOLD = json.load(f)["threshold"]

with open("model/label_encoder_classes.json") as f:
    le_classes = json.load(f)

feature_store = FeatureStore()
app = FastAPI(title="Fraud Detection API", version="1.0")

class Transaction(BaseModel):
    transaction_id: str
    user_id: int
    amount: float
    location: str
    merchant_category: Optional[str] = "Groceries"
    timestamp: str
    is_fraud: Optional[int] = 0

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def predict(txn: Transaction):
    txn_dict = txn.dict()

    category = txn_dict.get("merchant_category", "Groceries")
    encoded = le_classes.index(category) if category in le_classes else 0
    txn_dict["merchant_category_encoded"] = encoded

    features = feature_store.compute_features(txn_dict)
    features["merchant_category_encoded"] = encoded

    X = pd.DataFrame([{col: features.get(col, 0) for col in FEATURES}])

    fraud_prob = float(model.predict_proba(X)[0][1])
    is_fraud = int(fraud_prob >= THRESHOLD)

    if is_fraud:
        log_path = "logs/fraud_alerts.csv"
        os.makedirs("logs", exist_ok=True)
        file_exists = os.path.isfile(log_path)

        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "transaction_id", "user_id", "amount", "location",
                "fraud_score", "is_fraud", "flag", "timestamp"
            ])

            if not file_exists:
                writer.writeheader()

            writer.writerow({
                "transaction_id": txn_dict["transaction_id"],
                "user_id": txn_dict["user_id"],
                "amount": txn_dict["amount"],
                "location": txn_dict["location"],
                "fraud_score": round(fraud_prob, 3),
                "is_fraud": is_fraud,
                "flag": "FRAUD",
                "timestamp": txn_dict["timestamp"]
            })

    return {
        "transaction_id": txn_dict["transaction_id"],
        "user_id": txn_dict["user_id"],
        "amount": txn_dict["amount"],
        "location": txn_dict["location"],
        "fraud_score": round(fraud_prob, 3),
        "is_fraud": is_fraud,
        "flag": "FRAUD" if is_fraud else "OK"
    }

@app.get("/alerts")
def get_alerts():
    log_path = "logs/fraud_alerts.csv"
    if not os.path.isfile(log_path):
        return {"alerts": [], "count": 0}

    alerts = []
    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            alerts.append(row)

    return {"alerts": alerts, "count": len(alerts)}

@app.get("/stats")
def get_stats():
    log_path = "logs/fraud_alerts.csv"
    fraud_count = 0

    if os.path.isfile(log_path):
        with open(log_path, "r") as f:
            reader = csv.DictReader(f)
            fraud_count = sum(1 for _ in reader)

    total = feature_store.get_total_transactions() if hasattr(feature_store, "get_total_transactions") else "N/A"

    return {
        "total_transactions": total,
        "fraud_detected": fraud_count,
        "fraud_rate": f"{(fraud_count / total * 100):.2f}%" if isinstance(total, int) and total > 0 else "N/A"
    }