# Real-Time Behavioural Anomaly Detection System

> Real-time ML system combining streaming pipelines + behavioural feature engineering for low-latency anomaly detection.

Most fraud detection projects stop at training a model on static data.  
This project goes further — it simulates a **production-grade, real-time system** that detects anomalies from live transaction streams using **stateful behavioural intelligence**.

Built with Apache Kafka, FastAPI, and XGBoost, the system processes 5,000+ transactions with **<100 ms latency**, capturing user behavior patterns (velocity, time gaps, spending deviation) to detect subtle fraud signals in real time.

---

## Problem Statement

Traditional fraud detection systems rely on static, batch-processed data and fail to capture dynamic user behavior.

This project addresses that gap by building a **real-time anomaly detection pipeline** that:
- Processes streaming transaction data  
- Captures temporal and behavioural patterns  
- Performs low-latency fraud prediction  

---

## Tech Stack

- **Programming:** Python  
- **Machine Learning:** XGBoost, Scikit-learn  
- **Data Processing:** Pandas, NumPy  
- **Streaming:** Apache Kafka  
- **Backend API:** FastAPI  
- **Dashboard:** Streamlit  

---

## System Architecture
Transaction Generator → Kafka Producer → Kafka Topic
→ Kafka Consumer → Stateful Feature Engineering
→ XGBoost Model → FastAPI → Streamlit Dashboard


Designed with production scalability and real-time constraints in mind.

---

## Key Features

### 1. Real-Time Streaming Pipeline
- Kafka-based producer-consumer architecture  
- Simulates live financial transactions  

### 2. Stateful Feature Engineering
Captures user behaviour over time:
- Transaction velocity  
- Time gaps between transactions  
- Deviation from usual spending  

### 3. Low-Latency Inference
- Real-time predictions using XGBoost  
- <100 ms response time  

### 4. End-to-End ML System
- Data generation → streaming → feature engineering → prediction → visualization  

---

##  Results

- **AUC:** 0.75 – 0.80 (imbalanced dataset)  
- **Precision:** 0.87  
- **F1-score:** ~0.50+  
- **Latency:** <100 ms per prediction  
- **Transactions processed:** 5,000+  

---

##  Key Learnings

- Designing real-time ML systems vs offline models  
- Handling data imbalance using SMOTE + threshold tuning  
- Building stateful features for behavioural analysis  
- Managing streaming pipelines with Kafka  
- Ensuring low-latency inference for production systems  

---

##  How to Run

### 1. Clone the repository
```powershell
git clone https://github.com/your-username/real-time-anomaly-detection-system.git
cd real-time-anomaly-detection-system
```
### 2. Install dependencies
```powershell
pip install -r requirements.txt
```
### 3. Start Kafka
```powershell
Make sure Kafka is running.
```
### 4. Run Producer
```powershell
python src/producer/kafka_producer.py
```
### 5. Run Consumer
```powershell
python src/consumer/kafka_consumer.py
```
### 6. Start FastAPI
```powershell
uvicorn src.api.app:app --reload
```
### 7. Launch Dashboard
```powershell
streamlit run dashboard/streamlit_app.py
```
