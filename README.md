# Anomaly Detection — Real-Time Transaction Fraud System

End-to-end real-time fraud detection. Transactions stream through Kafka, get scored by a calibrated gradient-boosted model using stateful per-user behavioural features, and flagged alerts appear live on a Streamlit dashboard.

The design choice that drives the project: instead of training on raw transaction fields (amount, location, category alone), the system maintains an **in-memory rolling history per user** and derives 26 behavioural signals on every incoming transaction. The same feature store is used at training time and at inference time — no train/serve skew.

---

## Architecture

```
Generator ──► Kafka producer ──► Kafka topic ──► Kafka consumer
                                                       │
                                                       ▼
                                        Stateful feature store
                                        (26 behavioural features
                                         computed per user, online)
                                                       │
                                                       ▼
                              HistGradientBoostingClassifier + isotonic calibration
                                                       │
                                  ┌────────────────────┴────────────────────┐
                                  ▼                                         ▼
                        logs/fraud_alerts.csv                       FastAPI /predict
                                                                            │
                                                                            ▼
                                                                Streamlit dashboard
                                                                (KPI cards, time series,
                                                                 alerts by location)
```

---

## Held-out test results

Computed on a stratified 20% test split that the model never sees during training, calibration, or threshold tuning. `evaluate.py` reproduces these numbers exactly using indices saved by `train_model.py`.

| Metric                | Value  |
|-----------------------|--------|
| AUC-ROC               | **0.9516** |
| Average precision (PR)| **0.8482** |
| Precision (fraud)     | 0.5762 |
| Recall (fraud)        | **0.8700** |
| F1 (fraud)            | 0.6932 |
| Decision threshold    | 0.1227 |

The threshold is tuned on the validation set with **F-beta (β=2, recall-weighted)** under a precision floor of 0.30. In a fraud context, missing genuine fraud is much costlier than flagging a legitimate transaction for review, so recall is the primary objective.

---

## Behavioural features (26 total)

All features are computed from the user's prior history; the current transaction never contributes to its own statistics. Users with no history get NaN for behavioural features and an `is_new_user=1` flag — the gradient-boosted model handles NaN natively, so we don't have to lie with imputed zeros.

**Velocity & timing**
- `txn_count_last_5min`, `txn_count_last_30min`, `txn_count_last_1hr`, `txn_count_last_24hr`
- `txn_velocity` (txns per minute over the 5-minute lookback)
- `time_since_last_txn` (NaN for first txn)
- `txn_burst_flag` (>5 txns in last hour)

**Amount patterns**
- `avg_amount_last_10`, `amount_std_last_10`, `user_avg_amount`
- `amount_deviation`, `amount_zscore`
- `amount_spike_flag` (>2× recent average)
- `amount_ratio_last_10`, `recent_high_amount_ratio`
- `user_risk_score` (lifetime fraction of above-average txns)

**Location & merchant behaviour**
- `location_change_flag`, `location_frequency`, `location_rarity`
- `location_switch_count_24hr`
- `merchant_frequency`, `merchant_rarity`, `merchant_switch_count_24hr`
- `merchant_category_encoded`, `location_encoded` (with a reserved `0` slot for unseen categories)

Plus `amount`, `hour`, and `is_new_user`.

---

## Synthetic data

The generator produces 5,000 transactions across 500 users over 90 days, with a 10% fraud rate. Five fraud patterns are simulated:

| Pattern         | Description |
|-----------------|-------------|
| `amount_spike`  | Large transaction unusual for the user |
| `location_shift`| Transaction outside the user's home/cluster cities |
| `velocity`      | Multiple transactions inside a 1-hour window |
| `merchant_shift`| Switch to high-risk categories (Electronics, Travel, Utilities) |
| `mixed`         | Combination of the above |

Fraud and normal distributions overlap — small-amount fraud, home-city fraud, occasional large legit purchases, occasional category drift for genuine users. This forces the model to learn from feature combinations rather than memorise threshold cutoffs.

Reproducible from a single seed in `configs/config.yaml`.

---

## API

### `GET /` and `/health`
Service info and liveness.

### `POST /predict`
Score one transaction.

```json
{
  "transaction_id": "abc-123",
  "user_id": 42,
  "amount": 45000.00,
  "location": "Mumbai",
  "merchant_category": "Electronics",
  "timestamp": "2026-01-14T02:15:00"
}
```

Response:
```json
{
  "transaction_id": "abc-123",
  "user_id": 42,
  "amount": 45000.0,
  "location": "Mumbai",
  "fraud_score": 0.847,
  "is_fraud": 1,
  "flag": "FRAUD",
  "threshold": 0.1227
}
```

### `GET /alerts?limit=500`
Most-recent fraud alerts from `logs/fraud_alerts.csv`.

### `GET /stats`
```json
{
  "total_transactions": 1234,
  "fraud_detected": 87,
  "fraud_rate": 0.0705,
  "fraud_score_avg": 0.7241,
  "threshold": 0.1227
}
```

---

## Project structure

```
anomaly-detection/
├── api/
│   ├── Dockerfile
│   └── main.py                # FastAPI: /predict, /alerts, /stats, /health
├── configs/
│   └── config.yaml            # Kafka, paths, data, training params
├── dashboard/
│   ├── Dockerfile
│   └── streamlit_app.py       # KPIs, time series, location chart, alerts table
├── data_generator/
│   ├── generate_transactions.py   # Reproducible synthetic data, 5 fraud patterns
│   ├── transactions.csv
│   └── transactions.json          # Sample of first 5
├── model/
│   ├── train_model.py             # Features → CV → calibration → threshold tuning
│   ├── evaluate.py                # Held-out test using saved indices
│   ├── evaluation_report.txt
│   ├── model.pkl
│   ├── feature_columns.json
│   ├── label_encoder_classes.json
│   ├── location_encoder_classes.json
│   ├── test_indices.json          # Persisted test split for reproducible eval
│   └── threshold.json
├── streaming/
│   ├── Dockerfile                 # Shared image for producer & consumer
│   ├── feature_store.py           # Stateful per-user features (NaN-aware)
│   ├── producer.py                # Streams synthetic transactions to Kafka
│   └── consumer.py                # Scores transactions, logs fraud alerts
├── tests/
│   ├── test_feature_store.py
│   └── test_data_generator.py
├── logs/
│   └── fraud_alerts.csv
├── docker-compose.yml             # Kafka (KRaft) + backend + dashboard + producer + consumer
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Running it

### One command — full pipeline (Docker)

```bash
git clone <your-repo>
cd anomaly-detection

# 1) Generate data and train the model (host-side, one-off)
pip install -r requirements.txt
python -m data_generator.generate_transactions
python -m model.train_model

# 2) Bring up Kafka + producer + consumer + API + dashboard
docker compose up --build
```

- API:        http://localhost:8000/docs
- Dashboard:  http://localhost:8501

The producer streams ~1 transaction/second with a 10% fraud rate. The consumer scores each one and appends fraud alerts to `logs/fraud_alerts.csv`, which is shared with the API container via a bind mount.

### Without Docker (everything local)

Requires a local Kafka broker on `localhost:9092`.

```bash
pip install -r requirements.txt
python -m data_generator.generate_transactions
python -m model.train_model

# In separate terminals:
KAFKA_BOOTSTRAP=localhost:9092 python -m streaming.producer
KAFKA_BOOTSTRAP=localhost:9092 python -m streaming.consumer
uvicorn api.main:app --reload
streamlit run dashboard/streamlit_app.py
```

### Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Modelling choices

**Why HistGradientBoostingClassifier?** Native NaN support (the new-user features are genuinely undefined, not zero), strong on tabular data, fast to train. No need for the heavier XGBoost dependency for a model of this size.

**Why isotonic calibration?** The raw boosted-tree probabilities are not well-calibrated; isotonic regression on a held-out fold makes the score interpretable as a probability, so threshold tuning becomes principled.

**Why F-beta with β=2?** Fraud detection is asymmetric: a missed fraud costs much more than a false alert. β=2 weights recall four times more than precision in the harmonic mean. The precision floor (0.30) prevents the optimum from collapsing to "flag everything".

**Why save the test indices?** It guarantees that `evaluate.py` reports on the same examples the model was held out from during training, calibration, *and* threshold tuning. This was a real bug in an earlier version of this project — the evaluation script was scoring a different set of rows than the test set, including rows that had been in training, which inflated the reported metrics.

---

## Tech stack

| Layer              | Tech |
|--------------------|------|
| Streaming          | Apache Kafka (KRaft mode, single node) |
| Model              | scikit-learn HistGradientBoostingClassifier + isotonic calibration |
| Feature engineering| Custom stateful Python (in-memory per-user history) |
| Backend API        | FastAPI + Pydantic v2 |
| Dashboard          | Streamlit + Plotly |
| Containerisation   | Docker + Docker Compose |
