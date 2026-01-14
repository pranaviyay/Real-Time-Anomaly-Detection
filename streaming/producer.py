import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import random
import yaml
from kafka import KafkaProducer
from data_generator.generate_transactions import generate_transaction

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

producer = KafkaProducer(
    bootstrap_servers=config["kafka"]["bootstrap_servers"],
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

print("Producer started... sending transactions\n")

try:
    while True:
        # is_fraud = random.random() < 0.10
        is_fraud = random.random() < 0.8
        transaction = generate_transaction(fraud=is_fraud)

        producer.send(config["kafka"]["topic"], value=transaction)
        print(
            f"Sent → User {transaction['user_id']} | ₹{transaction['amount']} | "
            f"{transaction['location']} | fraud={transaction['is_fraud']}"
        )

        time.sleep(1)

except KeyboardInterrupt:
    print("\nProducer stopped.")
    producer.close()