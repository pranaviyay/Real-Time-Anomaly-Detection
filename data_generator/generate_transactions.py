import uuid
import random
from datetime import datetime, timedelta
from faker import Faker
import pandas as pd
import json
import os

fake = Faker("en_IN")

LOCATIONS = ["Bangalore", "Mumbai", "Delhi", "Chennai", "Hyderabad", "Kolkata", "Pune", "Jaipur"]
CATEGORIES = ["Electronics", "Groceries", "Food", "Travel", "Clothing", "Utilities", "Entertainment"]

CITY_CLUSTER = {
    "Bangalore": ["Hyderabad", "Chennai", "Pune"],
    "Mumbai": ["Pune", "Delhi", "Bangalore"],
    "Delhi": ["Jaipur", "Mumbai", "Chennai"],
    "Chennai": ["Bangalore", "Hyderabad", "Kolkata"],
    "Hyderabad": ["Bangalore", "Chennai", "Pune"],
    "Kolkata": ["Delhi", "Chennai", "Mumbai"],
    "Pune": ["Mumbai", "Bangalore", "Hyderabad"],
    "Jaipur": ["Delhi", "Mumbai", "Pune"],
}

FRAUD_BIASED_CATEGORIES = ["Electronics", "Travel", "Utilities"]
NORMAL_BIASED_CATEGORIES = ["Groceries", "Food", "Clothing"]

USER_HOME_CITY = {i: random.choice(LOCATIONS) for i in range(1, 501)}
USER_HOME_CATEGORY = {i: random.choice(NORMAL_BIASED_CATEGORIES) for i in range(1, 501)}

def random_timestamp(hour_min=8, hour_max=22):
    hour = random.randint(hour_min, hour_max)
    return datetime.now().replace(
        hour=hour,
        minute=random.randint(0, 59),
        second=random.randint(0, 59),
        microsecond=0
    )

def generate_transaction(fraud=False):
    user_id = random.randint(1, 500)
    home_city = USER_HOME_CITY[user_id]
    home_category = USER_HOME_CATEGORY[user_id]

    if fraud:
        fraud_type = random.choice(["amount_spike", "location_shift", "velocity", "merchant_shift", "mixed"])

        if fraud_type == "amount_spike":
            amount = round(random.uniform(25000, 80000), 2)
            location = home_city
            category = random.choice(FRAUD_BIASED_CATEGORIES)
            ts = random_timestamp(0, 23)

        elif fraud_type == "location_shift":
            amount = round(random.uniform(8000, 35000), 2)
            location = random.choice(CITY_CLUSTER[home_city] + [c for c in LOCATIONS if c != home_city])
            category = home_category if random.random() < 0.3 else random.choice(FRAUD_BIASED_CATEGORIES)
            ts = random_timestamp(1, 23)

        elif fraud_type == "velocity":
            amount = round(random.uniform(1000, 12000), 2)
            location = home_city
            category = random.choice(FRAUD_BIASED_CATEGORIES)
            ts = random_timestamp(0, 5) if random.random() < 0.7 else random_timestamp(18, 23)

        elif fraud_type == "merchant_shift":
            amount = round(random.uniform(5000, 40000), 2)
            location = home_city
            category = random.choice([c for c in CATEGORIES if c not in NORMAL_BIASED_CATEGORIES])
            ts = random_timestamp(0, 23)

        else:
            amount = round(random.uniform(12000, 70000), 2)
            location = random.choice(LOCATIONS)
            category = random.choice(FRAUD_BIASED_CATEGORIES)
            ts = random_timestamp(0, 23)

        if random.random() < 0.15:
            location = home_city
        if random.random() < 0.15:
            category = home_category

        is_fraud = 1

    else:
        amount = round(random.uniform(100, 22000), 2)
        location = home_city if random.random() < 0.9 else random.choice(CITY_CLUSTER[home_city])
        category = home_category if random.random() < 0.7 else random.choice(NORMAL_BIASED_CATEGORIES)
        ts = random_timestamp(8, 22)
        if random.random() < 0.05:
            location = random.choice(LOCATIONS)
        is_fraud = 0

    return {
        "transaction_id": str(uuid.uuid4()),
        "user_id": user_id,
        "amount": amount,
        "location": location,
        "merchant_category": category,
        "timestamp": ts.isoformat(),
        "is_fraud": is_fraud,
    }

def generate_dataset(n=5000):
    transactions = []
    fraud_count = int(n * 0.10)
    normal_count = n - fraud_count

    for _ in range(normal_count):
        transactions.append(generate_transaction(fraud=False))
    for _ in range(fraud_count):
        transactions.append(generate_transaction(fraud=True))

    random.shuffle(transactions)
    return transactions

if __name__ == "__main__":
    data = generate_dataset(5000)

    df = pd.DataFrame(data)
    os.makedirs("data_generator", exist_ok=True)
    df.to_csv("data_generator/transactions.csv", index=False)
    print(f"CSV saved: {len(df)} transactions, {df['is_fraud'].sum()} fraudulent")

    with open("data_generator/transactions.json", "w") as f:
        json.dump(data[:5], f, indent=2)
    print("Sample JSON saved (first 5 records)")
    print("\nSample record:")
    print(json.dumps(data[0], indent=2))