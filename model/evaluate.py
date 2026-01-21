import json
import pickle
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from streaming.feature_store import FeatureStore

df = pd.read_csv("data_generator/transactions.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

test_start = int(len(df) * 0.8)
test_df = df.iloc[test_start:].copy()

fs = FeatureStore()
rows = []

print("Generating evaluation features...")

for _, txn in test_df.iterrows():
    txn_dict = txn.to_dict()
    txn_dict["timestamp"] = pd.to_datetime(txn_dict["timestamp"]).isoformat()
    features = fs.compute_features(txn_dict)
    features["is_fraud"] = txn_dict["is_fraud"]
    features["merchant_category"] = txn_dict["merchant_category"]
    features["location"] = txn_dict["location"]
    rows.append(features)

df_feat = pd.DataFrame(rows)

y = df_feat["is_fraud"].astype(int)
X = df_feat.drop(columns=["is_fraud"])

with open("model/label_encoder_classes.json") as f:
    cat_classes = json.load(f)

with open("model/location_encoder_classes.json") as f:
    loc_classes = json.load(f)

X["merchant_category_encoded"] = X["merchant_category"].apply(
    lambda x: cat_classes.index(x) if x in cat_classes else 0
)
X["location_encoded"] = X["location"].apply(
    lambda x: loc_classes.index(x) if x in loc_classes else 0
)

X = X.drop(columns=["merchant_category", "location"])

with open("model/feature_columns.json") as f:
    FEATURES = json.load(f)

X = X.reindex(columns=FEATURES, fill_value=0)

with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/threshold.json") as f:
    THRESHOLD = json.load(f)["threshold"]

y_prob = model.predict_proba(X)[:, 1]
y_pred = (y_prob >= THRESHOLD).astype(int)

print("\nEVALUATION REPORT")
print("=" * 50)
print(classification_report(y, y_pred, digits=4))
print("AUC:", roc_auc_score(y, y_prob))
print("Threshold:", THRESHOLD)
print("F1:", f1_score(y, y_pred))