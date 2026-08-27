import pandas as pd

# Load dataset
df = pd.read_csv("data/PS26170_synthetic_burnin_dataset.csv")

# 1. Dataset size
print("Dataset shape:")
print(df.shape)

# 2. Column names
print("\nColumns:")
print(df.columns.tolist())

# 3. First 10 components
print("\nFirst 10 rows:")
print(df.head(10))

# 4. Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# 5. Component classes
print("\nComponent classes:")
print(df["ground_truth_class"].value_counts())

# 6. Basic statistics
print("\nStatistics:")
print(df.describe()) 