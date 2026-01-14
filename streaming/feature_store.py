import numpy as np
from collections import defaultdict
from datetime import datetime

class FeatureStore:
    def __init__(self):
        self.user_txns = defaultdict(list)
        self.user_amounts = defaultdict(list)
        self.user_locations = defaultdict(list)
        self.user_merchants = defaultdict(list)

    def compute_features(self, txn):
        user = txn["user_id"]
        amount = float(txn["amount"])
        timestamp = datetime.fromisoformat(txn["timestamp"])

        history = self.user_txns[user]
        amounts = self.user_amounts[user]
        locations = self.user_locations[user]
        merchants = self.user_merchants[user]

        hour = timestamp.hour

        txn_count_last_5min = sum(1 for t in history if (timestamp - t).total_seconds() <= 300)
        txn_count_last_1hr = sum(1 for t in history if (timestamp - t).total_seconds() <= 3600)
        txn_count_last_24hr = sum(1 for t in history if (timestamp - t).total_seconds() <= 86400)
        txn_count_last_30min = sum(1 for t in history if (timestamp - t).total_seconds() <= 1800)

        avg_amount_last_10 = np.mean(amounts[-10:]) if amounts else amount
        amount_std_last_10 = np.std(amounts[-10:]) if len(amounts) >= 2 else 0
        user_avg_amount = np.mean(amounts) if amounts else amount

        amount_deviation = abs(amount - avg_amount_last_10)
        amount_zscore = (
            (amount - avg_amount_last_10) / (amount_std_last_10 + 1e-5)
            if amount_std_last_10 > 0 else 0
        )
        amount_spike_flag = int(amount > 2 * avg_amount_last_10)
        amount_ratio_last_10 = amount / (avg_amount_last_10 + 1e-5)

        recent_high_amount_ratio = sum(
            1 for a in amounts[-10:] if a > avg_amount_last_10
        ) / (len(amounts[-10:]) + 1)

        high_amount_count = sum(1 for a in amounts if a > user_avg_amount)
        user_risk_score = high_amount_count / (len(amounts) + 1)

        if history:
            time_since_last_txn = (timestamp - history[-1]).total_seconds()
        else:
            time_since_last_txn = 0

        txn_velocity = txn_count_last_5min / (time_since_last_txn + 1)
        txn_burst_flag = int(txn_count_last_1hr > 5)

        current_loc = txn["location"]
        location_change_flag = int(len(locations) > 0 and locations[-1] != current_loc)
        location_frequency = locations.count(current_loc) / (len(locations) + 1)
        location_rarity = 1 - location_frequency
        location_switch_count_24hr = sum(1 for loc in locations if loc != current_loc)

        current_merchant = txn["merchant_category"]
        merchant_frequency = merchants.count(current_merchant) / (len(merchants) + 1)
        merchant_rarity = 1 - merchant_frequency
        merchant_switch_count_24hr = sum(1 for m in merchants if m != current_merchant)

        history.append(timestamp)
        amounts.append(amount)
        locations.append(current_loc)
        merchants.append(current_merchant)

        return {
            "amount": amount,
            "hour": hour,
            "txn_count_last_5min": txn_count_last_5min,
            "txn_count_last_1hr": txn_count_last_1hr,
            "txn_count_last_24hr": txn_count_last_24hr,
            "txn_count_last_30min": txn_count_last_30min,
            "txn_velocity": txn_velocity,
            "avg_amount_last_10": avg_amount_last_10,
            "amount_std_last_10": amount_std_last_10,
            "user_avg_amount": user_avg_amount,
            "amount_deviation": amount_deviation,
            "amount_zscore": amount_zscore,
            "amount_spike_flag": amount_spike_flag,
            "amount_ratio_last_10": amount_ratio_last_10,
            "recent_high_amount_ratio": recent_high_amount_ratio,
            "user_risk_score": user_risk_score,
            "time_since_last_txn": time_since_last_txn,
            "txn_burst_flag": txn_burst_flag,
            "location_change_flag": location_change_flag,
            "location_frequency": location_frequency,
            "location_rarity": location_rarity,
            "location_switch_count_24hr": location_switch_count_24hr,
            "merchant_frequency": merchant_frequency,
            "merchant_rarity": merchant_rarity,
            "merchant_switch_count_24hr": merchant_switch_count_24hr,
        }