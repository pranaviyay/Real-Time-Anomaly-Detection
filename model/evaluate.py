from __future__ import annotations

import json
import pickle
import argparse

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)

from model.train_model import build_feature_matrix, encode_categorical


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv(cfg["paths"]["data_csv"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print("Computing features over full sorted timeline...")
    feat_df = build_feature_matrix(df)

    y = feat_df["is_fraud"].astype(int).values
    X = feat_df.drop(columns=["is_fraud"])

    with open(cfg["paths"]["label_encoder"]) as f:
        cat_classes = json.load(f)
    with open(cfg["paths"]["location_encoder"]) as f:
        loc_classes = json.load(f)

    X["merchant_category_encoded"] = encode_categorical(X["merchant_category"], cat_classes)
    X["location_encoded"] = encode_categorical(X["location"], loc_classes)
    X = X.drop(columns=["merchant_category", "location"])

    with open(cfg["paths"]["feature_columns"]) as f:
        FEATURES = json.load(f)
    X = X.reindex(columns=FEATURES, fill_value=np.nan)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    with open(cfg["paths"]["test_indices"]) as f:
        idx_test = np.array(json.load(f)["test_indices"])

    with open(cfg["paths"]["model_pkl"], "rb") as f:
        model = pickle.load(f)
    with open(cfg["paths"]["threshold"]) as f:
        threshold = json.load(f)["threshold"]

    X_test, y_test = X.iloc[idx_test], y[idx_test]
    probs = model.predict_proba(X_test)[:, 1]
    pred = (probs >= threshold).astype(int)

    print("\n" + "=" * 60)
    print("EVALUATION (held-out test set)")
    print("=" * 60)
    print(f"Test rows: {len(X_test):,}  (fraud rate: {y_test.mean():.3f})")
    print(f"Threshold: {threshold:.4f}")
    print(f"AUC-ROC:   {roc_auc_score(y_test, probs):.4f}")
    print(f"AP (PR):   {average_precision_score(y_test, probs):.4f}\n")
    print(classification_report(y_test, pred, target_names=["Normal", "Fraud"], digits=4))


if __name__ == "__main__":
    main()
