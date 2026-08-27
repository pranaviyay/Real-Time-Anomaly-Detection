"""
Train the anomaly-detection model.

Key correctness properties:

1. Features are computed on the FULL sorted timeline once, then the resulting
   feature matrix is split into train / val / test. The naive version split
   the raw transactions first and then ran the FeatureStore separately on
   each split, which produced different rolling histories than the streaming
   pipeline ever sees — classic train/serve skew.

2. The split is stratified-random over the feature matrix. Because features
   only depend on each user's *past*, this does not leak future information.
   (We are not splitting time; we are splitting samples whose features were
   already locked in by the streaming-equivalent traversal.)

3. The categorical encoders reserve index 0 for "unknown". Unseen categories
   at inference go to 0 instead of colliding with whichever class happened
   to be alphabetically first.

4. The threshold is tuned on the validation set with F-beta (beta=2, recall-
   weighted) under a precision floor. Final metrics are reported on the
   genuinely held-out test set, and the test indices are persisted so that
   evaluate.py can reproduce them exactly.
"""

from __future__ import annotations

import os
import json
import pickle
import argparse

import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from streaming.feature_store import FeatureStore


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the FeatureStore over the full sorted dataset and return a feature
    DataFrame indexed in the same order as df. Mirrors exactly what the
    streaming consumer does at inference.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    fs = FeatureStore()
    rows = []
    for _, txn in df.iterrows():
        txn_dict = txn.to_dict()
        # Normalise timestamp to ISO string for the feature store
        if isinstance(txn_dict["timestamp"], pd.Timestamp):
            txn_dict["timestamp"] = txn_dict["timestamp"].isoformat()
        feats = fs.compute_features(txn_dict)
        feats["is_fraud"] = int(txn_dict["is_fraud"])
        feats["merchant_category"] = txn_dict["merchant_category"]
        feats["location"] = txn_dict["location"]
        rows.append(feats)
    return pd.DataFrame(rows)


def encode_categorical(values: pd.Series, classes: list[str]) -> pd.Series:
    """Map values to indices using `classes`. Unseen values → 0 (reserved for unknown)."""
    lookup = {c: i for i, c in enumerate(classes)}
    return values.map(lookup).fillna(0).astype(int)


def fbeta(p: np.ndarray, r: np.ndarray, beta: float = 2.0) -> np.ndarray:
    return (1 + beta * beta) * (p * r) / (beta * beta * p + r + 1e-9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs("model", exist_ok=True)

    # ------ load and sort raw data ------
    df = pd.read_csv(cfg["paths"]["data_csv"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ------ compute features over the full timeline once ------
    print("Computing features over full sorted timeline...")
    feat_df = build_feature_matrix(df)

    y = feat_df["is_fraud"].astype(int).values
    X_raw = feat_df.drop(columns=["is_fraud"])

    # Build categorical encoders. Reserve 0 for "<unknown>".
    cat_classes = ["<unknown>"] + sorted(X_raw["merchant_category"].unique().tolist())
    loc_classes = ["<unknown>"] + sorted(X_raw["location"].unique().tolist())

    X_raw["merchant_category_encoded"] = encode_categorical(X_raw["merchant_category"], cat_classes)
    X_raw["location_encoded"] = encode_categorical(X_raw["location"], loc_classes)
    X = X_raw.drop(columns=["merchant_category", "location"])

    # Make all numeric, preserving NaN (HGB handles NaN natively)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # ------ stratified 60/20/20 split on the FEATURE matrix ------
    val_size = cfg["training"]["validation_size"]
    test_size = cfg["training"]["test_size"]
    seed = cfg["training"]["random_seed"]

    idx = np.arange(len(X))
    idx_train_val, idx_test = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=y,
    )
    rel_val = val_size / (1.0 - test_size)
    idx_train, idx_val = train_test_split(
        idx_train_val, test_size=rel_val, random_state=seed, stratify=y[idx_train_val],
    )

    X_train, y_train = X.iloc[idx_train], y[idx_train]
    X_val, y_val = X.iloc[idx_val], y[idx_val]
    X_test, y_test = X.iloc[idx_test], y[idx_test]

    print(f"Splits — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
    print(f"Fraud rate — train: {y_train.mean():.3f}, val: {y_val.mean():.3f}, test: {y_test.mean():.3f}")

    # ------ grid search ------
    print("\nGrid-searching HistGradientBoostingClassifier...")
    base = HistGradientBoostingClassifier(random_state=seed)
    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_iter": [300, 500],
        "min_samples_leaf": [20, 50],
        "l2_regularization": [0.0, 0.1],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    search = GridSearchCV(
        base, param_grid=param_grid, scoring="average_precision",
        cv=cv, n_jobs=-1, verbose=0,
    )
    search.fit(X_train, y_train)
    print(f"Best CV AP: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    # ------ calibrate ------
    print("\nCalibrating probabilities (isotonic, 3-fold)...")
    calibrated = CalibratedClassifierCV(
        HistGradientBoostingClassifier(random_state=seed, **search.best_params_),
        method="isotonic", cv=3,
    )
    calibrated.fit(X_train, y_train)

    # ------ tune threshold on validation, with precision floor ------
    val_probs = calibrated.predict_proba(X_val)[:, 1]
    p_curve, r_curve, t_curve = precision_recall_curve(y_val, val_probs)

    floor = cfg["training"]["min_precision_floor"]
    beta = cfg["training"]["fbeta"]

    best_t = 0.5
    best_score = -1.0
    for p, r, t in zip(p_curve[:-1], r_curve[:-1], t_curve):
        if p >= floor:
            s = fbeta(p, r, beta=beta)
            if s > best_score:
                best_score = s
                best_t = float(t)
    if best_score < 0:
        # Floor was unattainable on validation; fall back to unconstrained F-beta
        for p, r, t in zip(p_curve[:-1], r_curve[:-1], t_curve):
            s = fbeta(p, r, beta=beta)
            if s > best_score:
                best_score = s
                best_t = float(t)

    # ------ test-set evaluation ------
    test_probs = calibrated.predict_proba(X_test)[:, 1]
    test_pred = (test_probs >= best_t).astype(int)

    print("\n" + "=" * 60)
    print("TEST RESULTS (held-out)")
    print("=" * 60)
    print(f"Threshold: {best_t:.4f}")
    print(f"AUC-ROC:   {roc_auc_score(y_test, test_probs):.4f}")
    print(f"AP (PR):   {average_precision_score(y_test, test_probs):.4f}")
    print()
    print(classification_report(y_test, test_pred, target_names=["Normal", "Fraud"], digits=4))

    # ------ persist artefacts ------
    feature_columns = list(X.columns)

    with open(cfg["paths"]["model_pkl"], "wb") as f:
        pickle.dump(calibrated, f)
    with open(cfg["paths"]["feature_columns"], "w") as f:
        json.dump(feature_columns, f, indent=2)
    with open(cfg["paths"]["threshold"], "w") as f:
        json.dump({"threshold": best_t}, f, indent=2)
    with open(cfg["paths"]["label_encoder"], "w") as f:
        json.dump(cat_classes, f, indent=2)
    with open(cfg["paths"]["location_encoder"], "w") as f:
        json.dump(loc_classes, f, indent=2)
    with open(cfg["paths"]["test_indices"], "w") as f:
        json.dump({"test_indices": idx_test.tolist()}, f)

    # Human-readable evaluation report
    report_path = "model/evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write("Held-out test set\n")
        f.write("=" * 60 + "\n")
        f.write(f"AUC-ROC: {roc_auc_score(y_test, test_probs):.4f}\n")
        f.write(f"AP:      {average_precision_score(y_test, test_probs):.4f}\n")
        f.write(f"Threshold: {best_t:.4f}\n\n")
        f.write(classification_report(y_test, test_pred, target_names=["Normal", "Fraud"], digits=4))

    print(f"\n✓ Saved model + artefacts under model/")
    print(f"✓ Test indices saved to {cfg['paths']['test_indices']}")


if __name__ == "__main__":
    main()
