import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from streaming.feature_store import FeatureStore

os.makedirs("model", exist_ok=True)

df = pd.read_csv("data_generator/transactions.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

train_df, temp_df = train_test_split(
    df, test_size=0.4, random_state=42, stratify=df["is_fraud"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, stratify=temp_df["is_fraud"]
)

def process(dataframe):
    fs = FeatureStore()
    rows = []
    for _, txn in dataframe.iterrows():
        txn_dict = txn.to_dict()
        txn_dict["timestamp"] = pd.to_datetime(txn_dict["timestamp"]).isoformat()
        features = fs.compute_features(txn_dict)
        features["is_fraud"] = int(txn_dict["is_fraud"])
        features["merchant_category"] = txn_dict["merchant_category"]
        features["location"] = txn_dict["location"]
        rows.append(features)
    return pd.DataFrame(rows)

print("Generating features...")
train_feat = process(train_df)
val_feat = process(val_df)
test_feat = process(test_df)

y_train = train_feat["is_fraud"].astype(int)
X_train = train_feat.drop(columns=["is_fraud"])

y_val = val_feat["is_fraud"].astype(int)
X_val = val_feat.drop(columns=["is_fraud"])

y_test = test_feat["is_fraud"].astype(int)
X_test = test_feat.drop(columns=["is_fraud"])

le_cat = LabelEncoder()
X_train["merchant_category_encoded"] = le_cat.fit_transform(X_train["merchant_category"])
X_val["merchant_category_encoded"] = X_val["merchant_category"].apply(
    lambda x: le_cat.transform([x])[0] if x in le_cat.classes_ else 0
)
X_test["merchant_category_encoded"] = X_test["merchant_category"].apply(
    lambda x: le_cat.transform([x])[0] if x in le_cat.classes_ else 0
)

with open("model/label_encoder_classes.json", "w") as f:
    json.dump(list(le_cat.classes_), f)

le_loc = LabelEncoder()
X_train["location_encoded"] = le_loc.fit_transform(X_train["location"])
X_val["location_encoded"] = X_val["location"].apply(
    lambda x: le_loc.transform([x])[0] if x in le_loc.classes_ else 0
)
X_test["location_encoded"] = X_test["location"].apply(
    lambda x: le_loc.transform([x])[0] if x in le_loc.classes_ else 0
)

with open("model/location_encoder_classes.json", "w") as f:
    json.dump(list(le_loc.classes_), f)

X_train = X_train.drop(columns=["merchant_category", "location"])
X_val = X_val.drop(columns=["merchant_category", "location"])
X_test = X_test.drop(columns=["merchant_category", "location"])

for col in X_train.columns:
    X_train[col] = pd.to_numeric(X_train[col], errors="coerce").fillna(0)
    X_val[col] = pd.to_numeric(X_val[col], errors="coerce").fillna(0)
    X_test[col] = pd.to_numeric(X_test[col], errors="coerce").fillna(0)

print("Tuning model...")
base = HistGradientBoostingClassifier(random_state=42)

param_grid = {
    "max_depth": [3, 4, 5],
    "learning_rate": [0.03, 0.05],
    "max_iter": [300, 500],
    "min_samples_leaf": [20, 30, 50],
    "l2_regularization": [0.0, 0.1, 0.2],
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
search = GridSearchCV(
    base,
    param_grid=param_grid,
    scoring="average_precision",
    cv=cv,
    n_jobs=-1,
    verbose=0
)
search.fit(X_train, y_train)
best_model = search.best_estimator_

print("Calibrating probabilities...")
calibrated = CalibratedClassifierCV(best_model, method="isotonic", cv=3)
calibrated.fit(X_train, y_train)

val_probs = calibrated.predict_proba(X_val)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)

def fbeta(p, r, beta=2.0):
    return (1 + beta * beta) * (p * r) / (beta * beta * p + r + 1e-9)

best_threshold = 0.5
best_score = -1

for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
    if p >= 0.30:
        score = fbeta(p, r, beta=2.0)
        if score > best_score:
            best_score = score
            best_threshold = float(t)

if best_score < 0:
    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        score = fbeta(p, r, beta=2.0)
        if score > best_score:
            best_score = score
            best_threshold = float(t)

test_probs = calibrated.predict_proba(X_test)[:, 1]
test_pred = (test_probs >= best_threshold).astype(int)

print("\nBEST MODEL")
print("==========")
print("Best params:", search.best_params_)
print("Validation AP:", average_precision_score(y_val, val_probs))
print("Test AP:", average_precision_score(y_test, test_probs))
print("Threshold:", best_threshold)

print("\nFINAL REPORT")
print("=" * 50)
print(classification_report(y_test, test_pred, digits=4))
print("AUC:", roc_auc_score(y_test, test_probs))

with open("model/model.pkl", "wb") as f:
    pickle.dump(calibrated, f)

with open("model/feature_columns.json", "w") as f:
    json.dump(list(X_train.columns), f)

with open("model/threshold.json", "w") as f:
    json.dump({"threshold": float(best_threshold)}, f)

print("\n✅ DONE")