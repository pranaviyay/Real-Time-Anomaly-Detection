"""
Stateful per-user feature computation for the anomaly-detection pipeline.

Design notes
------------
- All features are computed *before* the current transaction is appended to
  the user's history. The current transaction never contributes to its own
  rolling/aggregate statistics.
- For users with no prior history, "undefined" behavioural features are NaN.
  HistGradientBoostingClassifier handles NaN natively; this is more honest
  than imputing zero or making the average equal the current amount, both
  of which silently lie to the model about the baseline.
- An is_new_user flag tells the model whether the rolling features are
  meaningful for this transaction.
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from datetime import datetime
from typing import Any


class FeatureStore:
    def __init__(self) -> None:
        self.user_txns: dict[Any, list[datetime]] = defaultdict(list)
        self.user_amounts: dict[Any, list[float]] = defaultdict(list)
        self.user_locations: dict[Any, list[str]] = defaultdict(list)
        self.user_merchants: dict[Any, list[str]] = defaultdict(list)
        self._total_transactions = 0

    def get_total_transactions(self) -> int:
        return self._total_transactions

    def reset(self) -> None:
        self.user_txns.clear()
        self.user_amounts.clear()
        self.user_locations.clear()
        self.user_merchants.clear()
        self._total_transactions = 0

    def compute_features(self, txn: dict) -> dict:
        user = txn["user_id"]
        amount = float(txn["amount"])
        ts_raw = txn["timestamp"]
        timestamp = ts_raw if isinstance(ts_raw, datetime) else datetime.fromisoformat(ts_raw)
        current_loc = txn["location"]
        current_merchant = txn["merchant_category"]

        history = self.user_txns[user]
        amounts = self.user_amounts[user]
        locations = self.user_locations[user]
        merchants = self.user_merchants[user]

        is_new_user = len(history) == 0
        n_amounts = len(amounts)
        n_locations = len(locations)
        n_merchants = len(merchants)

        hour = timestamp.hour

        # ---- velocity & timing ----
        txn_count_last_5min = sum(1 for t in history if (timestamp - t).total_seconds() <= 300)
        txn_count_last_30min = sum(1 for t in history if (timestamp - t).total_seconds() <= 1800)
        txn_count_last_1hr = sum(1 for t in history if (timestamp - t).total_seconds() <= 3600)
        txn_count_last_24hr = sum(1 for t in history if (timestamp - t).total_seconds() <= 86400)
        txn_velocity = txn_count_last_5min / 5.0  # txns per minute over a 5-min window

        if history:
            time_since_last_txn = (timestamp - history[-1]).total_seconds()
        else:
            time_since_last_txn = np.nan

        txn_burst_flag = int(txn_count_last_1hr > 5)

        # ---- amount patterns ----
        if n_amounts == 0:
            avg_amount_last_10 = np.nan
            amount_std_last_10 = np.nan
            user_avg_amount = np.nan
            amount_deviation = np.nan
            amount_zscore = np.nan
            amount_spike_flag = 0
            amount_ratio_last_10 = np.nan
            recent_high_amount_ratio = np.nan
            user_risk_score = np.nan
        else:
            recent = amounts[-10:]
            avg_amount_last_10 = float(np.mean(recent))
            amount_std_last_10 = float(np.std(recent)) if len(recent) >= 2 else 0.0
            user_avg_amount = float(np.mean(amounts))
            amount_deviation = abs(amount - avg_amount_last_10)
            amount_zscore = (
                (amount - avg_amount_last_10) / amount_std_last_10
                if amount_std_last_10 > 0 else 0.0
            )
            amount_spike_flag = int(amount > 2 * avg_amount_last_10)
            amount_ratio_last_10 = amount / (avg_amount_last_10 + 1e-9)
            recent_high_amount_ratio = sum(1 for a in recent if a > avg_amount_last_10) / len(recent)
            high_amount_count = sum(1 for a in amounts if a > user_avg_amount)
            user_risk_score = high_amount_count / n_amounts

        # ---- location behaviour ----
        if n_locations == 0:
            location_change_flag = 0
            location_frequency = np.nan
            location_rarity = np.nan
            location_switch_count_24hr = 0
        else:
            location_change_flag = int(locations[-1] != current_loc)
            location_frequency = locations.count(current_loc) / n_locations
            location_rarity = 1.0 - location_frequency
            location_switch_count_24hr = sum(1 for loc in locations if loc != current_loc)

        # ---- merchant behaviour ----
        if n_merchants == 0:
            merchant_frequency = np.nan
            merchant_rarity = np.nan
            merchant_switch_count_24hr = 0
        else:
            merchant_frequency = merchants.count(current_merchant) / n_merchants
            merchant_rarity = 1.0 - merchant_frequency
            merchant_switch_count_24hr = sum(1 for m in merchants if m != current_merchant)

        # ---- update state AFTER feature computation ----
        history.append(timestamp)
        amounts.append(amount)
        locations.append(current_loc)
        merchants.append(current_merchant)
        self._total_transactions += 1

        return {
            "amount": amount,
            "hour": hour,
            "is_new_user": int(is_new_user),
            "txn_count_last_5min": txn_count_last_5min,
            "txn_count_last_30min": txn_count_last_30min,
            "txn_count_last_1hr": txn_count_last_1hr,
            "txn_count_last_24hr": txn_count_last_24hr,
            "txn_velocity": txn_velocity,
            "time_since_last_txn": time_since_last_txn,
            "txn_burst_flag": txn_burst_flag,
            "avg_amount_last_10": avg_amount_last_10,
            "amount_std_last_10": amount_std_last_10,
            "user_avg_amount": user_avg_amount,
            "amount_deviation": amount_deviation,
            "amount_zscore": amount_zscore,
            "amount_spike_flag": amount_spike_flag,
            "amount_ratio_last_10": amount_ratio_last_10,
            "recent_high_amount_ratio": recent_high_amount_ratio,
            "user_risk_score": user_risk_score,
            "location_change_flag": location_change_flag,
            "location_frequency": location_frequency,
            "location_rarity": location_rarity,
            "location_switch_count_24hr": location_switch_count_24hr,
            "merchant_frequency": merchant_frequency,
            "merchant_rarity": merchant_rarity,
            "merchant_switch_count_24hr": merchant_switch_count_24hr,
        }
