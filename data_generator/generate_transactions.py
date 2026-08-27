import os
import json
import uuid
import random
import argparse
from datetime import datetime, timedelta

import yaml
import pandas as pd

LOCATIONS = ["Bangalore", "Mumbai", "Delhi", "Chennai", "Hyderabad", "Kolkata", "Pune", "Jaipur"]
CATEGORIES = ["Electronics", "Groceries", "Food", "Travel", "Clothing", "Utilities", "Entertainment"]

# Cities that are "nearby" — a normal user occasionally transacts in these
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


def _ts(base: datetime, day_offset: int, hour: int, minute: int, second: int) -> datetime:
    return base + timedelta(days=day_offset, hours=hour, minutes=minute, seconds=second)


def _normal_txn(rng: random.Random, user_id: int, home_city: str, home_category: str,
                base_date: datetime, day_offset: int) -> dict:
    
    # 5% of normal users occasionally make a larger purchase
    if rng.random() < 0.05:
        amount = round(rng.uniform(15000, 35000), 2)
    else:
        amount = round(rng.uniform(100, 12000), 2)

    # Hours: mostly 8-22, but ~10% drift into evening/early morning
    if rng.random() < 0.10:
        hour = rng.choice(list(range(0, 8)) + [22, 23])
    else:
        hour = rng.randint(8, 22)
    minute = rng.randint(0, 59)
    second = rng.randint(0, 59)
    ts = _ts(base_date, day_offset, hour, minute, second)

    if rng.random() < 0.85:
        location = home_city
    else:
        location = rng.choice(CITY_CLUSTER[home_city])
    if rng.random() < 0.05:
        location = rng.choice(LOCATIONS)

    if rng.random() < 0.65:
        category = home_category
    elif rng.random() < 0.85:
        category = rng.choice(NORMAL_BIASED_CATEGORIES)
    else:
        # Occasional "fraud-biased" category for genuine users (e.g. legit Electronics purchase)
        category = rng.choice(FRAUD_BIASED_CATEGORIES)

    return {
        "transaction_id": str(uuid.UUID(int=rng.getrandbits(128), version=4)),
        "user_id": user_id,
        "amount": amount,
        "location": location,
        "merchant_category": category,
        "timestamp": ts.isoformat(),
        "is_fraud": 0,
    }


def _fraud_txn(rng: random.Random, user_id: int, home_city: str, home_category: str,
               base_date: datetime, day_offset: int) -> dict:
    """A fraudulent transaction matching one of five patterns.

    Fraud signals overlap meaningfully with the normal distribution — small
    amount spikes, occasional home-city fraud, etc. This makes the model
    learn from features rather than memorise threshold cutoffs.
    """
    pattern = rng.choice(["amount_spike", "location_shift", "velocity",
                          "merchant_shift", "mixed"])

    minute = rng.randint(0, 59)
    second = rng.randint(0, 59)

    if pattern == "amount_spike":
        # Amounts overlap with the high tail of normal txns (15-35k)
        amount = round(rng.uniform(18000, 60000), 2)
        location = home_city
        category = rng.choice(FRAUD_BIASED_CATEGORIES + [home_category])
        hour = rng.randint(0, 23)

    elif pattern == "location_shift":
        amount = round(rng.uniform(3000, 25000), 2)
        # Sometimes a "nearby" cluster city, not just any far city
        if rng.random() < 0.4:
            location = rng.choice(CITY_CLUSTER[home_city])
        else:
            location = rng.choice([c for c in LOCATIONS if c != home_city])
        category = rng.choice(FRAUD_BIASED_CATEGORIES + NORMAL_BIASED_CATEGORIES)
        hour = rng.randint(1, 23)

    elif pattern == "velocity":
        amount = round(rng.uniform(500, 8000), 2)
        location = home_city
        category = rng.choice(FRAUD_BIASED_CATEGORIES + NORMAL_BIASED_CATEGORIES)
        hour = rng.randint(0, 5) if rng.random() < 0.6 else rng.randint(18, 23)

    elif pattern == "merchant_shift":
        amount = round(rng.uniform(2000, 30000), 2)
        location = home_city
        category = rng.choice([c for c in CATEGORIES if c not in NORMAL_BIASED_CATEGORIES])
        hour = rng.randint(0, 23)

    else:  # mixed
        amount = round(rng.uniform(8000, 50000), 2)
        location = rng.choice([c for c in LOCATIONS if c != home_city])
        category = rng.choice(FRAUD_BIASED_CATEGORIES)
        hour = rng.randint(0, 23)

    ts = _ts(base_date, day_offset, hour, minute, second)
    return {
        "transaction_id": str(uuid.UUID(int=rng.getrandbits(128), version=4)),
        "user_id": user_id,
        "amount": amount,
        "location": location,
        "merchant_category": category,
        "timestamp": ts.isoformat(),
        "is_fraud": 1,
        "_pattern": pattern,
    }


def generate_dataset(n_transactions: int, n_users: int, fraud_rate: float,
                     seed: int, span_days: int = 90) -> pd.DataFrame:
    """
    Generate a dataset spread over `span_days`. Each user gets roughly
    n_transactions / n_users transactions distributed across the timeline.
    """
    rng = random.Random(seed)
    base_date = datetime(2026, 1, 1, 0, 0, 0)

    user_home_city = {u: rng.choice(LOCATIONS) for u in range(1, n_users + 1)}
    user_home_category = {u: rng.choice(NORMAL_BIASED_CATEGORIES) for u in range(1, n_users + 1)}

    n_fraud = int(n_transactions * fraud_rate)
    n_normal = n_transactions - n_fraud

    txns = []

    # Distribute normal txns across users and days.
    for _ in range(n_normal):
        user = rng.randint(1, n_users)
        day_offset = rng.randint(0, span_days - 1)
        txns.append(_normal_txn(
            rng, user, user_home_city[user], user_home_category[user],
            base_date, day_offset,
        ))

    # Velocity-pattern fraud needs bursts: emit 3-6 txns within an hour for one user
    velocity_target = max(1, n_fraud // 5)  # roughly 1/5 of fraud is velocity
    velocity_emitted = 0
    burst_starts = []
    while velocity_emitted < velocity_target:
        user = rng.randint(1, n_users)
        day_offset = rng.randint(0, span_days - 1)
        burst_size = rng.randint(3, 6)
        burst_starts.append((user, day_offset, burst_size))
        velocity_emitted += burst_size

    for user, day_offset, burst_size in burst_starts:
        if len(txns) - n_normal >= n_fraud:
            break
        # All in a 1-hour window during late night
        start_hour = rng.randint(0, 4)
        for i in range(burst_size):
            if len(txns) - n_normal >= n_fraud:
                break
            txn = _fraud_txn(
                rng, user, user_home_city[user], user_home_category[user],
                base_date, day_offset,
            )
            # Override timestamp to keep them in the same hour
            t = base_date + timedelta(
                days=day_offset,
                hours=start_hour,
                minutes=i * rng.randint(2, 8),
                seconds=rng.randint(0, 59),
            )
            txn["timestamp"] = t.isoformat()
            txn["_pattern"] = "velocity"
            txn["is_fraud"] = 1
            txns.append(txn)

    # Fill remaining fraud slots with non-velocity patterns
    while len([t for t in txns if t["is_fraud"] == 1]) < n_fraud:
        user = rng.randint(1, n_users)
        day_offset = rng.randint(0, span_days - 1)
        txn = _fraud_txn(
            rng, user, user_home_city[user], user_home_category[user],
            base_date, day_offset,
        )
        # Ensure non-velocity for these fillers
        if txn.get("_pattern") == "velocity":
            txn["_pattern"] = "amount_spike"
        txns.append(txn)

    df = pd.DataFrame(txns)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    if "_pattern" in df.columns:
        df = df.drop(columns=["_pattern"])
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    df = generate_dataset(
        n_transactions=cfg["data"]["n_transactions"],
        n_users=cfg["data"]["n_users"],
        fraud_rate=cfg["data"]["fraud_rate"],
        seed=cfg["data"]["random_seed"],
    )

    out_csv = cfg["paths"]["data_csv"]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df):,} transactions to {out_csv}")
    print(f"  Fraud: {int(df['is_fraud'].sum()):,} ({df['is_fraud'].mean():.1%})")
    print(f"  Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  Unique users: {df['user_id'].nunique()}")

    # Sample JSON for the README and quick inspection
    sample = df.head(5).copy()
    sample["timestamp"] = sample["timestamp"].astype(str)
    with open(out_csv.replace(".csv", ".json"), "w") as f:
        json.dump(sample.to_dict(orient="records"), f, indent=2)


if __name__ == "__main__":
    main()
