import pandas as pd
import numpy as np
import os

# ============================================================
# AEROVISTA — CREATE MASTER DATASET
# ============================================================

input_file = "data/raw/source2_features.csv"
output_file = "data/processed/aerovista_master_dataset.csv"

print("Loading Source 2 feature dataset...")

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv(input_file)

print(f"Original records: {len(df)}")
print(f"Original columns: {len(df.columns)}")

# ------------------------------------------------------------
# 2. REMOVE INTERMEDIATE CALCULATION COLUMNS
# ------------------------------------------------------------

columns_to_remove = [
    "dx",
    "dy",
    "dz"
]

existing_columns = [
    col for col in columns_to_remove
    if col in df.columns
]

df = df.drop(columns=existing_columns)

# ------------------------------------------------------------
# 3. CREATE ADDITIONAL BEHAVIOURAL FEATURES
# ------------------------------------------------------------

df["absolute_vertical_speed"] = df["vertical_speed"].abs()

df["absolute_acceleration"] = df["acceleration"].abs()

df["horizontal_movement_ratio"] = np.where(
    df["distance_3d"] > 0,
    df["horizontal_distance"] / df["distance_3d"],
    0
)

# ------------------------------------------------------------
# 4. HANDLE INFINITE VALUES
# ------------------------------------------------------------

numeric_columns = [
    "x",
    "y",
    "z",
    "time_diff",
    "horizontal_distance",
    "distance_3d",
    "horizontal_speed",
    "speed",
    "vertical_speed",
    "acceleration",
    "absolute_vertical_speed",
    "absolute_acceleration",
    "horizontal_movement_ratio"
]

existing_numeric_columns = [
    col for col in numeric_columns
    if col in df.columns
]

df[existing_numeric_columns] = df[existing_numeric_columns].replace(
    [np.inf, -np.inf],
    np.nan
)

# ------------------------------------------------------------
# 5. REMOVE MISSING VALUES
# ------------------------------------------------------------

before_cleaning = len(df)

df = df.dropna(subset=existing_numeric_columns)

after_cleaning = len(df)

print(f"Rows removed during cleaning: {before_cleaning - after_cleaning}")

# ------------------------------------------------------------
# 6. ENSURE MOVEMENT VALUES ARE NON-NEGATIVE
# ------------------------------------------------------------

for column in [
    "horizontal_distance",
    "distance_3d",
    "speed",
    "horizontal_speed"
]:
    if column in df.columns:
        df[column] = df[column].clip(lower=0)

# ------------------------------------------------------------
# 7. CREATE OUTPUT FOLDER
# ------------------------------------------------------------

os.makedirs("data/processed", exist_ok=True)

# ------------------------------------------------------------
# 8. SAVE MASTER DATASET
# ------------------------------------------------------------

df.to_csv(output_file, index=False)

# ------------------------------------------------------------
# 9. FINAL CHECK
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