import os
import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "PS26170_synthetic_burnin_dataset.csv"
)

ANOMALY_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "anomaly_model.pkl"
)

DRIFT_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "drift_model.pkl"
)


# ============================================================
# LOAD DATA AND MODELS
# ============================================================

df = pd.read_csv(DATA_PATH)

anomaly_model = joblib.load(ANOMALY_MODEL_PATH)
drift_model = joblib.load(DRIFT_MODEL_PATH)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(row):

    v0 = float(row["value_0h_uA"])
    v24 = float(row["value_24h_uA"])
    v96 = float(row["value_96h_uA"])
    v168 = float(row["value_168h_uA"])

    # Changes
    change_0_24 = v24 - v0
    change_24_96 = v96 - v24
    change_96_168 = v168 - v96

    total_change = v168 - v0

    # Slopes
    slope_0_24 = (v24 - v0) / 24
    slope_24_96 = (v96 - v24) / 72
    slope_96_168 = (v168 - v96) / 72
    slope_overall = (v168 - v0) / 168

    # Acceleration
    slope_acceleration = slope_96_168 - slope_24_96

    # Z-score relative to lot
    lot_id = row["lot_id"]

    lot_data = df[df["lot_id"] == lot_id]

    lot_mean_24 = lot_data["value_24h_uA"].mean()
    lot_std_24 = lot_data["value_24h_uA"].std()

    if lot_std_24 == 0 or pd.isna(lot_std_24):
        z_score_24h = 0
    else:
        z_score_24h = (v24 - lot_mean_24) / lot_std_24

    features = pd.DataFrame([{
        "value_0h_uA": v0,
        "value_24h_uA": v24,
        "value_96h_uA": v96,
        "value_168h_uA": v168,

        "change_0_24": change_0_24,
        "change_24_96": change_24_96,
        "change_96_168": change_96_168,

        "total_change": total_change,

        "slope_0_24": slope_0_24,
        "slope_24_96": slope_24_96,
        "slope_96_168": slope_96_168,
        "slope_overall": slope_overall,

        "slope_acceleration": slope_acceleration,
        "z_score_24h": z_score_24h
    }])

    features = features.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    return features


# ============================================================
# ANALYZE COMPONENT
# ============================================================

def analyze_component(component_id):

    matches = df[
        df["component_id"] == component_id
    ]

    if len(matches) == 0:
        raise ValueError(
            f"Component {component_id} not found."
        )

    row = matches.iloc[0]

    # --------------------------------------------------------
    # MODULE A: ANOMALY DETECTION
    # --------------------------------------------------------

    X_anomaly = create_features(row)

    anomaly_prediction = anomaly_model.predict(
        X_anomaly
    )[0]

    anomaly_score = anomaly_model.decision_function(
        X_anomaly
    )[0]

    is_anomaly = anomaly_prediction == -1


    # --------------------------------------------------------
    # MODULE B: DRIFT PREDICTION
    # --------------------------------------------------------

    X_drift = pd.DataFrame([{
        "value_0h_uA": row["value_0h_uA"],
        "value_24h_uA": row["value_24h_uA"]
    }])

    predicted_168h = drift_model.predict(
        X_drift
    )[0]


    # --------------------------------------------------------
    # SAFETY SLOPE
    # --------------------------------------------------------

    current_24h = float(
        row["value_24h_uA"]
    )

    datasheet_limit = float(
        row["datasheet_limit_uA"]
    )

    remaining_hours = 168 - 24

    safety_slope = (
        datasheet_limit - current_24h
    ) / remaining_hours

    predicted_slope = (
        predicted_168h - current_24h
    ) / remaining_hours

    early_warning = (
        predicted_slope > safety_slope
    )


    # --------------------------------------------------------
    # FINAL RISK DECISION
    # --------------------------------------------------------

    if predicted_168h >= datasheet_limit:

        risk_level = "CRITICAL"

        reason = (
            "Predicted 168h value exceeds "
            "the datasheet safety limit."
        )

    elif is_anomaly and early_warning:

        risk_level = "CRITICAL"

        reason = (
            "Dynamic anomaly detected and "
            "predicted drift exceeds the "
            "allowable safety slope."
        )

    elif is_anomaly:

        risk_level = "WARNING"

        reason = (
            "Component behaviour is anomalous "
            "relative to its lot."
        )

    elif early_warning:

        risk_level = "WARNING"

        reason = (
            "Predicted drift exceeds the "
            "allowable safety slope."
        )

    else:

        risk_level = "SAFE"

        reason = (
            "Component behaviour is within "
            "expected limits."
        )


    # --------------------------------------------------------
    # RETURN RESULT
    # --------------------------------------------------------

    return {

        "component_id": component_id,

        "value_0h_uA":
            float(row["value_0h_uA"]),

        "value_24h_uA":
            float(row["value_24h_uA"]),

        "value_96h_uA":
            float(row["value_96h_uA"]),

        "value_168h_uA":
            float(row["value_168h_uA"]),

        "predicted_168h":
            float(predicted_168h),

        "datasheet_limit_uA":
            datasheet_limit,

        "anomaly_score":
            float(anomaly_score),

        "is_anomaly":
            bool(is_anomaly),

        "safety_slope":
            float(safety_slope),

        "predicted_slope":
            float(predicted_slope),

        "early_warning":
            bool(early_warning),

        "risk_level":
            risk_level,

        "reason":
            reason
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_component = df.iloc[0]["component_id"]

    result = analyze_component(
        test_component
    )

    print("\n================================")
    print("PREDICTIVE COMPONENT ANALYSIS")
    print("================================")

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )