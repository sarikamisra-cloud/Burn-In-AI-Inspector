import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib


# ============================================
# 1. LOAD DATA
# ============================================

df = pd.read_csv(
    "data/PS26170_synthetic_burnin_dataset.csv"
)

print("Dataset loaded:", df.shape)


# ============================================
# 2. INPUTS AND TARGET
# ============================================

X = df[
    [
        "value_0h_uA",
        "value_24h_uA"
    ]
]

y = df["value_168h_uA"]


# ============================================
# 3. TRAIN / TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================
# 4. RANDOM FOREST REGRESSOR
# ============================================

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    min_samples_leaf=3
)


# ============================================
# 5. TRAIN
# ============================================

model.fit(
    X_train,
    y_train
)
# Save trained drift model
joblib.dump(
    model,
    "models/drift_model.pkl"
)

print("\nDrift model saved to models/drift_model.pkl")


# ============================================
# 6. PREDICT 168h
# ============================================

y_pred = model.predict(X_test)


# ============================================
# 7. EVALUATE
# ============================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n==============================")
print("DRIFT PREDICTION PERFORMANCE")
print("==============================")

print(
    f"Mean Absolute Error: {mae:.3f} µA"
)

print(
    f"R² Score: {r2:.3f}"
)


# ============================================
# 8. SHOW EXAMPLES
# ============================================

results = X_test.copy()

results["actual_168h"] = y_test

results["predicted_168h"] = y_pred

results["prediction_error"] = (
    results["predicted_168h"] -
    results["actual_168h"]
)


print("\nSample predictions:\n")

print(
    results.head(20).to_string()
)


# ============================================
# 9. FEATURE IMPORTANCE
# ============================================

print("\nFeature Importance:")

for feature, importance in zip(
    X.columns,
    model.feature_importances_
):

    print(
        f"{feature}: {importance:.3f}"
    )
# ============================================
# 10. PREDICTIVE EARLY-WARNING ANALYSIS
# ============================================

results["predicted_drift"] = (
    results["predicted_168h"] -
    results["value_0h_uA"]
) / 168


# Safety slope based on datasheet limit
datasheet_limit = 50.0

results["safety_slope"] = (
    datasheet_limit -
    results["value_0h_uA"]
) / 168


# Flag if predicted 168h exceeds limit
results["predicted_failure"] = (
    results["predicted_168h"] >= datasheet_limit
)


print("\n================================")
print("EARLY WARNING ANALYSIS")
print("================================")

print(
    results[
        [
            "value_0h_uA",
            "value_24h_uA",
            "actual_168h",
            "predicted_168h",
            "predicted_drift",
            "safety_slope",
            "predicted_failure"
        ]
    ].head(20).to_string()
)


# ============================================
# 11. LATENT DEFECT PREDICTION
# ============================================

test_indices = X_test.index

test_results = df.loc[
    test_indices
].copy()

test_results["predicted_168h"] = y_pred

test_results["prediction_error"] = abs(
    test_results["value_168h_uA"] -
    test_results["predicted_168h"]
)

latent_test = test_results[
    test_results["ground_truth_class"] ==
    "Latent_Defect"
].copy()


print("\n================================")
print("LATENT DEFECT PREDICTIONS")
print("================================")

print(
    latent_test[
        [
            "component_id",
            "value_0h_uA",
            "value_24h_uA",
            "value_168h_uA",
            "predicted_168h",
            "prediction_error"
        ]
    ].head(20).to_string(index=False)
)
# ============================================
# 12. SAFETY SLOPE ANALYSIS
# ============================================

SAFETY_LIMIT = 50.0

# Hours remaining after the 24h measurement
REMAINING_HOURS = 168 - 24


# Safety slope:
# How fast can the component increase from
# its current 24h value before reaching 50 µA?

latent_test["safety_slope"] = (
    SAFETY_LIMIT -
    latent_test["value_24h_uA"]
) / REMAINING_HOURS


# Predicted slope from 24h to predicted 168h

latent_test["predicted_slope"] = (
    latent_test["predicted_168h"] -
    latent_test["value_24h_uA"]
) / REMAINING_HOURS


# Early warning condition

latent_test["early_warning"] = (
    latent_test["predicted_slope"] >
    latent_test["safety_slope"]
)


# ============================================
# 13. PRINT SAFETY ANALYSIS
# ============================================

print("\n================================")
print("SAFETY SLOPE ANALYSIS")
print("================================")

print(
    latent_test[
        [
            "component_id",
            "value_24h_uA",
            "predicted_168h",
            "safety_slope",
            "predicted_slope",
            "early_warning"
        ]
    ].head(20).to_string(index=False)
)