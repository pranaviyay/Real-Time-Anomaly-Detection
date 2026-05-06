# Real-Time Anomaly Detection System

An end-to-end real-time Anomaly detection system built on Apache Kafka, FastAPI, and XGBoost. Transactions stream through a Kafka pipeline, get scored by a trained ML model using stateful behavioural features, and flagged alerts appear live on a Streamlit dashboard.

The core idea behind this project was to move beyond static anomaly detection — instead of training on fixed features like amount and location alone, the system maintains a rolling in-memory history per user and derives behavioural signals on every incoming transaction in real time.

---

## Architecture

```
Data Generator
      │
      ▼
Kafka Producer ──────────────► Kafka Topic (transactions)
                                        │
                                        ▼
                                Kafka Consumer
                                        │
                                Stateful Feature Store
                                (27 behavioural features,
                                 computed per user in real time)
                                        │
                                XGBoost Model (threshold: 0.336)
                                        │
                          ┌─────────────┴─────────────┐
                          ▼                           ▼
                  logs/fraud_alerts.csv          FastAPI /predict
                                                      │
                                             Streamlit Dashboard
```

---

## Behavioural Features (27 total)

All features are derived in real time from the user's rolling transaction history — no batch jobs, no pre-aggregation:

**Velocity & timing**
- Transaction count in last 5 min, 30 min, 1 hr, 24 hr
- Transaction velocity (count / time since last txn)
- Time since last transaction
- Burst flag (>5 transactions in last hour)

**Amount patterns**
- Rolling average and std dev of last 10 transactions
- Amount deviation from personal baseline
- Amount z-score
- Amount spike flag (>2x personal average)
- Amount ratio relative to recent history
- Recent high-amount ratio
- User-level risk score (lifetime high-amount frequency)

**Location & merchant behaviour**
- Location change flag
- Location frequency and rarity for this user
- Location switch count in 24 hrs
- Merchant category frequency and rarity
- Merchant switch count in 24 hrs
- Encoded location and merchant category

---

## Model

**Algorithm:** `HistGradientBoostingClassifier` with isotonic probability calibration (`CalibratedClassifierCV`)

**Training pipeline:**
- 5,000 synthetic transactions with a 10% fraud rate
- Stratified 60/20/20 train/val/test split
- `GridSearchCV` over depth, learning rate, iterations, min samples leaf, and L2 regularisation
- Threshold tuned on validation set using F2-score (recall-weighted) with a minimum precision floor of 0.30

**Results on held-out test set:**

| Metric | Value |
|--------|-------|
| AUC-ROC | **0.88** |
| Precision (fraud class) | 0.16 |
| Recall (fraud class) | **0.91** |
| F1 (fraud class) | 0.28 |
| Decision threshold | 0.336 |

The low precision is a deliberate tradeoff — the threshold is tuned to maximise recall (catching 91% of actual fraud) at the cost of more false positives. In a real fraud detection context, missing genuine fraud is far more costly than flagging a legitimate transaction for manual review.

---

## Fraud Patterns in Synthetic Data

The data generator simulates five distinct fraud patterns to train on realistic signals:

| Pattern | Description |
|---------|-------------|
| Amount spike | Single very large transaction, unusual for that user |
| Location shift | Transaction from a city outside the user's home cluster |
| Velocity burst | Multiple transactions in a short window, often late at night |
| Merchant shift | Sudden switch to high-risk categories (Electronics, Travel, Utilities) |
| Mixed | Combination of the above signals |

Each pattern includes 15% noise (reverting to normal behaviour) to prevent the model from learning overly clean boundaries.

---

## API Endpoints

### `POST /predict`
Score an incoming transaction in real time.

**Request:**
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

**Response:**
```json
{
  "transaction_id": "abc-123",
  "user_id": 42,
  "amount": 45000.0,
  "location": "Mumbai",
  "fraud_score": 0.847,
  "is_fraud": 1,
  "flag": "FRAUD"
}
```

### `GET /alerts`
Returns all logged fraud alerts from `logs/fraud_alerts.csv`.

### `GET /stats`
Returns total transactions processed, fraud count, and fraud rate.

---

## Project Structure

```
anomaly-detection-system/
│
├── api/
│   ├── Dockerfile
│   └── main.py               # FastAPI: /predict, /alerts, /stats
│
├── configs/
│   └── config.yaml           # Kafka bootstrap server, topic, group ID
│
├── dashboard/
│   ├── Dockerfile
│   └── streamlit_app.py      # Live monitoring: KPI cards, time series, location chart
│
├── data_generator/
│   ├── __init__.py
│   ├── generate_transactions.py   # Synthetic data with 5 fraud patterns
│   ├── transactions.csv           # 5,000 generated transactions
│   └── transactions.json          # Sample records (first 5)
│
├── model/
│   ├── train_model.py             # GridSearchCV + calibration + threshold tuning
│   ├── evaluate.py                # Evaluation script on held-out test set
│   ├── evaluation_report.txt      # Classification report + AUC
│   ├── model.pkl                  # Trained calibrated model
│   ├── feature_columns.json       # Feature order for inference
│   ├── label_encoder_classes.json # Merchant category classes
│   ├── location_encoder_classes.json
│   └── threshold.json             # Tuned decision threshold (0.336)
│
├── streaming/
│   ├── feature_store.py      # Stateful per-user feature computation
│   ├── producer.py           # Kafka producer: streams synthetic transactions
│   └── consumer.py           # Kafka consumer: features → model → fraud logging
│
├── logs/
│   └── fraud_alerts.csv      # Appended by consumer and API on fraud detection
│
├── docker-compose.yml        # Spins up API + dashboard together
├── .dockerignore
├── requirements.txt
└── README.md
```

---

## Running the Project

### Option 1 — Docker (API + Dashboard only)

```bash
git clone https://github.com/pranaviyay/Real-Time-Anomaly-Detection
cd Real-Time-Anomaly-Detection
docker-compose up --build
```

- API docs: `http://localhost:8000/docs`
- Dashboard: `http://localhost:8501`

### Option 2 — Full Kafka Pipeline

**1. Generate data and train the model:**
```bash
python data_generator/generate_transactions.py
python model/train_model.py
```

**2. Start Kafka** (must be running locally on `localhost:9092`)

**3. Start the producer and consumer in separate terminals:**
```bash
python streaming/producer.py
python streaming/consumer.py
```

**4. Start the API:**
```bash
uvicorn api.main:app --reload
```

**5. Launch the dashboard:**
```bash
streamlit run dashboard/streamlit_app.py
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Streaming | Apache Kafka |
| ML Model | XGBoost / HistGradientBoosting + Calibration |
| Feature Engineering | Custom stateful Python (in-memory per-user history) |
| Backend API | FastAPI |
| Dashboard | Streamlit + Plotly |
| Containerisation | Docker + Docker Compose |
| Data Generation | Faker (en_IN locale) |
