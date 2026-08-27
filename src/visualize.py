import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/PS26170_synthetic_burnin_dataset.csv")

# Choose one component from each class
classes = [
    "Healthy",
    "Latent_Defect",
    "Rapid_Drift",
    "Obvious_Failure"
]

plt.figure(figsize=(10, 6))

for cls in classes:

    component = df[df["ground_truth_class"] == cls].iloc[0]

    time = [0, 24, 96, 168]

    values = [
        component["value_0h_uA"],
        component["value_24h_uA"],
        component["value_96h_uA"],
        component["value_168h_uA"]
    ]

    plt.plot(
        time,
        values,
        marker="o",
        label=cls
    )

# Datasheet limit
plt.axhline(
    y=50,
    linestyle="--",
    label="Datasheet Limit"
)

plt.xlabel("Burn-in Time (hours)")
plt.ylabel("Leakage Current (µA)")
plt.title("Component Behaviour During Burn-In")

plt.legend()
plt.grid(True)

plt.show()