"""
Kafka producer that emits a stream of synthetic transactions.

Honest 10% fraud rate (matches training distribution). The previous version
forced 80% fraud to make the demo dashboard look busy — that produces a
dashboard that has nothing to do with the model's real-world behaviour.
"""

from __future__ import annotations

import os
import sys
import json
import time
import random
import argparse
from datetime import datetime

import yaml
from kafka import KafkaProducer

# Make sibling packages importable when run as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_generator.generate_transactions import (  # noqa: E402
    LOCATIONS, NORMAL_BIASED_CATEGORIES, _normal_txn, _fraud_txn,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--rate", type=float, default=1.0,
                        help="Seconds between transactions")
    parser.add_argument("--fraud-rate", type=float, default=None,
                        help="Override fraud rate (default: from config)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Allow KAFKA_BOOTSTRAP env var to override config (handy in docker-compose)
    bootstrap = os.getenv("KAFKA_BOOTSTRAP", cfg["kafka"]["bootstrap_servers"])
    topic = cfg["kafka"]["topic"]
    fraud_rate = args.fraud_rate if args.fraud_rate is not None else cfg["data"]["fraud_rate"]
    seed = cfg["data"]["random_seed"]

    print(f"Producer connecting to {bootstrap}, topic={topic}, fraud_rate={fraud_rate}")

    producer = KafkaProducer(
        bootstrap_servers=bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        retries=5,
    )

    rng = random.Random(seed)
    n_users = cfg["data"]["n_users"]
    user_home_city = {u: rng.choice(LOCATIONS) for u in range(1, n_users + 1)}
    user_home_category = {u: rng.choice(NORMAL_BIASED_CATEGORIES) for u in range(1, n_users + 1)}

    print("Streaming transactions... Ctrl-C to stop\n")
    try:
        while True:
            user = rng.randint(1, n_users)
            home_city = user_home_city[user]
            home_category = user_home_category[user]
            now = datetime.now()

            if rng.random() < fraud_rate:
                txn = _fraud_txn(rng, user, home_city, home_category, now, 0)
            else:
                txn = _normal_txn(rng, user, home_city, home_category, now, 0)

            # Use real wall-clock timestamp for the streaming demo
            txn["timestamp"] = now.isoformat()
            txn.pop("_pattern", None)

            producer.send(topic, value=txn)
            tag = "FRAUD" if txn["is_fraud"] else "OK   "
            print(f"[{tag}] user={txn['user_id']:>3}  ₹{txn['amount']:>10,.2f}  "
                  f"{txn['location']:<12}  {txn['merchant_category']}")
            time.sleep(args.rate)

    except KeyboardInterrupt:
        print("\nProducer stopped.")
    finally:
        producer.flush()
        producer.close()


if __name__ == "__main__":
    main()
