import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score
)

# ------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------

df = pd.read_csv(
    "data/PS26170_synthetic_burnin_dataset.csv"
)

print("Dataset loaded:", df.shape)


# ------------------------------------------------
# 2. CREATE FEATURES
# ------------------------------------------------

# Changes between measurements

df["change_0_24"] = (
    df["value_24h_uA"] -
    df["value_0h_uA"]
)

df["change_24_96"] = (
    df["value_96h_uA"] -
    df["value_24h_uA"]
)

df["change_96_168"] = (
    df["value_168h_uA"] -
    df["value_96h_uA"]
)


# Overall change

df["total_change"] = (
    df["value_168h_uA"] -
    df["value_0h_uA"]
)


# ------------------------------------------------
# 3. DRIFT / SLOPE
# ------------------------------------------------

df["slope_0_24"] = (
    df["change_0_24"] / 24
)

df["slope_24_96"] = (
    df["change_24_96"] / 72
)

df["slope_96_168"] = (
    df["change_96_168"] / 72
)

df["slope_overall"] = (
    df["total_change"] / 168
)


# ------------------------------------------------
# 4. DRIFT ACCELERATION
# ------------------------------------------------

df["slope_acceleration"] = (
    df["slope_96_168"] -
    df["slope_0_24"]
)


# ------------------------------------------------
# 5. LOT STATISTICS
# ------------------------------------------------

group = df.groupby("lot_id")["value_24h_uA"]

lot_mean = group.transform("mean")
lot_std = group.transform("std")

df["lot_mean_24h"] = lot_mean
df["lot_std_24h"] = lot_std


# Z-score

df["z_score_24h"] = (
    df["value_24h_uA"] -
    df["lot_mean_24h"]
) / df["lot_std_24h"]


# ------------------------------------------------
# 6. SELECT FEATURES
# ------------------------------------------------

features = [
    "value_0h_uA",
    "value_24h_uA",
    "value_96h_uA",
    "value_168h_uA",

    "change_0_24",
    "change_24_96",
    "change_96_168",

    "total_change",

    "slope_0_24",
    "slope_24_96",
    "slope_96_168",
    "slope_overall",

    "slope_acceleration",

    "z_score_24h"
]

X = df[features].replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(0)


# ------------------------------------------------
# 7. TRAIN ISOLATION FOREST
# ------------------------------------------------

model = IsolationForest(
    n_estimators=300,
    contamination=0.25,
    random_state=42
)

model.fit(X)
# Save trained drift model
joblib.dump(
    model,
    "models/anomaly_model.pkl"
)

print("\nAnomaly model saved to models/anomaly_model.pkl")

# ------------------------------------------------
# 8. GET MODEL PREDICTIONS
# ------------------------------------------------

prediction = model.predict(X)

# Isolation Forest:
#
# +1 = normal
# -1 = anomaly

df["anomaly_prediction"] = prediction

df["predicted_anomaly"] = (
    df["anomaly_prediction"] == -1
).astype(int)


# ------------------------------------------------
# 9. ANOMALY SCORE
# ------------------------------------------------

raw_score = -model.decision_function(X)

# Convert to 0–1 for easier interpretation

low = np.percentile(raw_score, 1)
high = np.percentile(raw_score, 99)

df["anomaly_score"] = np.clip(
    (raw_score - low) /
    (high - low),
    0,
    1
)


# ------------------------------------------------
# 10. GROUND TRUTH
# ------------------------------------------------

df["actual_defect"] = (
    df["ground_truth_class"] != "Healthy"
).astype(int)


# ------------------------------------------------
# 11. EVALUATION
# ------------------------------------------------

y_true = df["actual_defect"]
y_pred = df["predicted_anomaly"]

print("\n==============================")
print("ANOMALY DETECTION PERFORMANCE")
print("==============================")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "Healthy",
            "Defective"
        ]
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_true,
        y_pred
    )
)

print(
    "\nRecall:",
    recall_score(y_true, y_pred)
)

print(
    "Precision:",
    precision_score(y_true, y_pred)
)

print(
    "F1 Score:",
    f1_score(y_true, y_pred)
)


# ------------------------------------------------
# 12. SHOW MOST ANOMALOUS COMPONENTS
# ------------------------------------------------

top = df.sort_values(
    "anomaly_score",
    ascending=False
).head(20)

print("\nMOST ANOMALOUS COMPONENTS:")

print(
    top[
        [
            "component_id",
            "lot_id",
            "value_0h_uA",
            "value_24h_uA",
            "value_96h_uA",
            "value_168h_uA",
            "anomaly_score",
            "ground_truth_class"
        ]
    ].to_string(index=False)
)
# ------------------------------------------------
# 13. CHECK LATENT DEFECT DETECTION
# ------------------------------------------------

latent = df[
    df["ground_truth_class"] == "Latent_Defect"
].sort_values(
    "anomaly_score",
    ascending=False
).head(20)

print("\n================================")
print("TOP LATENT-DEFECT COMPONENTS")
print("================================")

print(
    latent[
        [
            "component_id",
            "lot_id",
            "value_0h_uA",
            "value_24h_uA",
            "value_96h_uA",
            "value_168h_uA",
            "anomaly_score",
            "ground_truth_class"
        ]
    ].to_string(index=False)
)