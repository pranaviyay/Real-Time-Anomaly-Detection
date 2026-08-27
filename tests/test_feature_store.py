"""Sanity tests for the feature store."""

import math
import numpy as np
import pytest

from streaming.feature_store import FeatureStore


def _txn(user, amount, location, category, ts):
    return {
        "user_id": user, "amount": amount, "location": location,
        "merchant_category": category, "timestamp": ts,
    }


def test_first_transaction_is_marked_new_user():
    fs = FeatureStore()
    f = fs.compute_features(_txn(1, 500.0, "Mumbai", "Groceries", "2026-01-01T10:00:00"))
    assert f["is_new_user"] == 1
    # All "rolling" baselines should be NaN, not silently zeroed
    assert math.isnan(f["user_avg_amount"])
    assert math.isnan(f["amount_zscore"])
    assert math.isnan(f["time_since_last_txn"])
    # Counters remain 0 (deterministic)
    assert f["txn_count_last_5min"] == 0
    assert f["amount_spike_flag"] == 0


def test_amount_spike_flag_after_baseline():
    fs = FeatureStore()
    # 5 small transactions to establish baseline
    for i in range(5):
        fs.compute_features(_txn(1, 500.0, "Mumbai", "Groceries", f"2026-01-0{i+1}T10:00:00"))
    f = fs.compute_features(_txn(1, 50000.0, "Mumbai", "Groceries", "2026-01-06T10:00:00"))
    assert f["is_new_user"] == 0
    assert f["amount_spike_flag"] == 1
    assert f["amount_ratio_last_10"] > 50


def test_velocity_burst_detection():
    fs = FeatureStore()
    base = "2026-01-01T10:00:"
    for i in range(7):
        fs.compute_features(_txn(1, 1000.0, "Mumbai", "Groceries", f"{base}{i:02d}"))
    f = fs.compute_features(_txn(1, 1000.0, "Mumbai", "Groceries", "2026-01-01T10:00:08"))
    assert f["txn_count_last_5min"] >= 6
    assert f["txn_burst_flag"] == 1
    assert f["txn_velocity"] > 1.0


def test_location_change_flag():
    fs = FeatureStore()
    fs.compute_features(_txn(1, 1000.0, "Mumbai", "Groceries", "2026-01-01T10:00:00"))
    f = fs.compute_features(_txn(1, 1000.0, "Delhi", "Groceries", "2026-01-02T10:00:00"))
    assert f["location_change_flag"] == 1
    assert f["location_rarity"] == 1.0  # Delhi never seen before


def test_current_txn_does_not_pollute_its_own_features():
    """avg_amount_last_10 must be the avg of PRIOR txns, not include the current one."""
    fs = FeatureStore()
    fs.compute_features(_txn(1, 1000.0, "Mumbai", "Groceries", "2026-01-01T10:00:00"))
    fs.compute_features(_txn(1, 1000.0, "Mumbai", "Groceries", "2026-01-02T10:00:00"))
    # Now: existing avg = 1000. Current txn = 9000. avg_last_10 should be 1000, not (1000+1000+9000)/3.
    f = fs.compute_features(_txn(1, 9000.0, "Mumbai", "Groceries", "2026-01-03T10:00:00"))
    assert f["avg_amount_last_10"] == pytest.approx(1000.0)


def test_total_transactions_counter():
    fs = FeatureStore()
    assert fs.get_total_transactions() == 0
    for i in range(5):
        fs.compute_features(_txn(1, 1000.0, "Mumbai", "Groceries", f"2026-01-0{i+1}T10:00:00"))
    assert fs.get_total_transactions() == 5


def test_per_user_isolation():
    fs = FeatureStore()
    fs.compute_features(_txn(1, 1000.0, "Mumbai", "Groceries", "2026-01-01T10:00:00"))
    f = fs.compute_features(_txn(2, 50000.0, "Delhi", "Electronics", "2026-01-01T10:00:01"))
    # User 2's first txn → still flagged as new user
    assert f["is_new_user"] == 1
    assert math.isnan(f["user_avg_amount"])
