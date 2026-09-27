import pandas as pd
import numpy as np
import os

# ============================================================
# AEROVISTA - CREATE MASTER DATASET
# ============================================================

input_file = "data/raw/source2_features.csv"
output_file = "data/processed/aerovista_master_dataset.csv"

print("Loading Source 2 feature dataset...")

# Load Source 2
df = pd.read_csv(input_file)

print(f"Original records: {len(df)}")
print(f"Original columns: {len(df.columns)}")

# ------------------------------------------------------------
# Remove intermediate calculation columns
# ------------------------------------------------------------

for col in ["dx", "dy", "dz"]:
    if col in df.columns:
        df = df.drop(columns=col)

# ------------------------------------------------------------
# Create additional features
# ------------------------------------------------------------

df["absolute_vertical_speed"] = df["vertical_speed"].abs()

df["absolute_acceleration"] = df["acceleration"].abs()

df["horizontal_movement_ratio"] = np.where(
    df["distance_3d"] > 0,
    df["horizontal_distance"] / df["distance_3d"],
    0
)

# ------------------------------------------------------------
# Clean infinite values
# ------------------------------------------------------------

df = df.replace([np.inf, -np.inf], np.nan)

# Remove missing values
before = len(df)

df = df.dropna()

after = len(df)

print(f"Rows removed during cleaning: {before - after}")

# ------------------------------------------------------------
# Ensure distance and speed values are non-negative
# ------------------------------------------------------------

for col in [
    "horizontal_distance",
    "distance_3d",
    "speed",
    "horizontal_speed"
]:
    if col in df.columns:
        df[col] = df[col].clip(lower=0)

# ------------------------------------------------------------
# Create output folder
# ------------------------------------------------------------

os.makedirs("data/processed", exist_ok=True)

# ------------------------------------------------------------
# Save final dataset
# ------------------------------------------------------------

df.to_csv(output_file, index=False)

# ------------------------------------------------------------
# Final report
# ------------------------------------------------------------

print("\n========== AEROVISTA MASTER DATASET ==========")

print(f"Final records: {len(df)}")
print(f"Final columns: {len(df.columns)}")

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nFirst 5 records:")
print(df.head())

print("\nSaved to:")
print(output_file)

print("\n========== MASTER DATASET READY ==========")