import pandas as pd

df = pd.read_csv(
    "data/PS26170_synthetic_burnin_dataset.csv"
)

# Select only latent defects
latent = df[
    df["ground_truth_class"] == "Latent_Defect"
].copy()

# Calculate drift rates
latent["slope_0_24"] = (
    latent["value_24h_uA"] -
    latent["value_0h_uA"]
) / 24

latent["slope_24_96"] = (
    latent["value_96h_uA"] -
    latent["value_24h_uA"]
) / 72

latent["slope_96_168"] = (
    latent["value_168h_uA"] -
    latent["value_96h_uA"]
) / 72

# How much did the component change overall?
latent["total_change"] = (
    latent["value_168h_uA"] -
    latent["value_0h_uA"]
)

# Show 10 examples
print("\nExamples of LATENT DEFECT components:\n")

print(
    latent[
        [
            "component_id",
            "lot_id",
            "value_0h_uA",
            "value_24h_uA",
            "value_96h_uA",
            "value_168h_uA",
            "slope_0_24",
            "slope_24_96",
            "slope_96_168",
            "total_change"
        ]
    ].head(10).to_string(index=False)
)

print("\nAverage latent-defect behaviour:\n")

print(
    latent[
        [
            "value_0h_uA",
            "value_24h_uA",
            "value_96h_uA",
            "value_168h_uA",
            "slope_0_24",
            "slope_24_96",
            "slope_96_168"
        ]
    ].mean()
)