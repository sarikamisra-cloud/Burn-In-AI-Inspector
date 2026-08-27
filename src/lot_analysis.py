import pandas as pd

df = pd.read_csv(
    "data/PS26170_synthetic_burnin_dataset.csv"
)

# We will examine the 24h measurement
grouped = df.groupby("lot_id")["value_24h_uA"]

lot_mean = grouped.transform("mean")
lot_std = grouped.transform("std")

df["lot_mean_24h"] = lot_mean
df["lot_std_24h"] = lot_std

df["z_score_24h"] = (
    df["value_24h_uA"] - df["lot_mean_24h"]
) / df["lot_std_24h"]

# Show the most unusual components
top = df.sort_values(
    "z_score_24h",
    ascending=False
).head(10)

print("\nMost unusual components at 24h:\n")

print(
    top[
        [
            "component_id",
            "lot_id",
            "value_24h_uA",
            "lot_mean_24h",
            "lot_std_24h",
            "z_score_24h",
            "ground_truth_class"
        ]
    ].to_string(index=False)
)