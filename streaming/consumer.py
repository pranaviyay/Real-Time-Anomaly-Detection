"""
Kafka consumer for streaming transactions.

Forwards every transaction to the API's /ingest endpoint. The API owns the
FeatureStore, the model, the threshold, the alert log, and the running
counters — single source of truth. The consumer's job is now just bridging
Kafka to HTTP and surfacing per-transaction decisions to stdout.

This replaces the previous design where the consumer scored locally with its
own FeatureStore, which produced a duplicate state machine that disagreed
with the API's view (most visibly: the dashboard's "Total Transactions"
counter was always 0 because the API never saw streaming traffic).
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse

import requests
import yaml
from kafka import KafkaConsumer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(BASE_DIR, "configs/config.yaml"))
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    bootstrap = os.getenv("KAFKA_BOOTSTRAP", cfg["kafka"]["bootstrap_servers"])
    topic = cfg["kafka"]["topic"]
    group_id = cfg["kafka"]["group_id"]
    api_base = os.getenv("API_BASE", "http://backend:8000")

    print(f"Consumer → kafka={bootstrap}, topic={topic}, api={api_base}")

    # Wait for the API to come up. /health is cheap and idempotent.
    for attempt in range(30):
        try:
            r = requests.get(f"{api_base}/health", timeout=2)
            if r.ok:
                print(f"API healthy after {attempt} attempts.")
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        print("API never came up; consumer exiting.", file=sys.stderr)
        sys.exit(1)

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap,
        group_id=group_id,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="latest",
    )

    session = requests.Session()
    print("Forwarding transactions to /ingest...\n")

    for message in consumer:
        txn = message.value
        try:
            resp = session.post(f"{api_base}/ingest", json=txn, timeout=5)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"!! ingest failed for {txn.get('transaction_id', '?')}: {e}", file=sys.stderr)
            continue

        is_fraud = data["is_fraud"]
        marker = "🚨" if is_fraud else "✓ "
        print(f"{marker} user={txn['user_id']:>3}  ₹{txn['amount']:>10,.2f}  "
              f"{txn['location']:<12}  score={data['fraud_score']:.3f}  "
              f"decision={data['flag']}")


if __name__ == "__main__":
    main()
