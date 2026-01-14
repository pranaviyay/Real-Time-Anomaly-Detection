import os
import csv
import json
import pickle
import yaml
import pandas as pd
import random
from kafka import KafkaConsumer
from feature_store import FeatureStore

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE_DIR, "configs", "config.yaml")) as f:
    config = yaml.safe_load(f)

with open(os.path.join(BASE_DIR, "model", "model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "model", "feature_columns.json")) as f:
    FEATURES = json.load(f)

with open(os.path.join(BASE_DIR, "model", "threshold.json")) as f:
    THRESHOLD = json.load(f)["threshold"]

with open(os.path.join(BASE_DIR, "model", "label_encoder_classes.json")) as f:
    cat_classes = json.load(f)

with open(os.path.join(BASE_DIR, "model", "location_encoder_classes.json")) as f:
    loc_classes = json.load(f)

print(f"Model loaded | Threshold: {THRESHOLD:.2f}")
print("Fraud consumer started...\n")

feature_store = FeatureStore()

consumer = KafkaConsumer(
    config["kafka"]["topic"],
    bootstrap_servers=config["kafka"]["bootstrap_servers"],
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    auto_offset_reset="latest"
)

for message in consumer:
    txn = message.value

    category = txn.get("merchant_category", "Groceries")
    cat_encoded = cat_classes.index(category) if category in cat_classes else 0

    location = txn.get("location", "Unknown")
    loc_encoded = loc_classes.index(location) if location in loc_classes else 0

    features = feature_store.compute_features(txn)
    features["merchant_category_encoded"] = cat_encoded
    features["location_encoded"] = loc_encoded

    X = pd.DataFrame([{col: features.get(col, 0) for col in FEATURES}])

    raw_score = float(model.predict_proba(X)[0][1])
    score = max(0.0, min(raw_score + random.uniform(-0.05, 0.05), 1.0))
    is_fraud = int(score >= THRESHOLD)

    flag = "🚨 FRAUD" if is_fraud else "✅ OK"

    print(
        f"{flag} | User {txn['user_id']} | ₹{txn['amount']:>10.2f} | "
        f"{txn['location']:<12} | Score: {score:.3f}"
    )

    print(f"Decision | score={score:.3f} | threshold={THRESHOLD:.3f} | fraud={is_fraud}")

    if is_fraud:
        log_path = os.path.join(BASE_DIR, "logs", "fraud_alerts.csv")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_exists = os.path.isfile(log_path)

        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["transaction_id", "user_id", "amount", "location", "fraud_score", "timestamp"]
            )
            if not file_exists:
                writer.writeheader()

            writer.writerow({
                "transaction_id": txn["transaction_id"],
                "user_id": txn["user_id"],
                "amount": txn["amount"],
                "location": txn["location"],
                "fraud_score": round(score, 3),
                "timestamp": txn["timestamp"]
            })