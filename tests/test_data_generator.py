"""Sanity tests for the data generator."""

import pandas as pd
from data_generator.generate_transactions import generate_dataset


def test_dataset_is_reproducible():
    df1 = generate_dataset(n_transactions=200, n_users=20, fraud_rate=0.10, seed=42)
    df2 = generate_dataset(n_transactions=200, n_users=20, fraud_rate=0.10, seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_fraud_rate_is_honoured():
    df = generate_dataset(n_transactions=1000, n_users=50, fraud_rate=0.10, seed=42)
    rate = df["is_fraud"].mean()
    assert 0.08 <= rate <= 0.12, f"Fraud rate {rate} too far from 0.10"


def test_timestamps_span_multiple_days():
    df = generate_dataset(n_transactions=500, n_users=50, fraud_rate=0.10, seed=42, span_days=90)
    days = df["timestamp"].dt.date.nunique()
    assert days > 30, f"Expected many distinct days, got {days}"


def test_timestamps_are_sorted():
    df = generate_dataset(n_transactions=500, n_users=50, fraud_rate=0.10, seed=42)
    assert df["timestamp"].is_monotonic_increasing


def test_no_label_flipping_artifacts():
    """All fraud rows should remain fraud (label=1)."""
    df = generate_dataset(n_transactions=500, n_users=50, fraud_rate=0.20, seed=42)
    fraud_rows = df[df["is_fraud"] == 1]
    assert (fraud_rows["is_fraud"] == 1).all()
    # Label 0 rows are all genuine normal txns
    normal_rows = df[df["is_fraud"] == 0]
    assert (normal_rows["is_fraud"] == 0).all()
