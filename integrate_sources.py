import pandas as pd
import os

# ============================================================
# AEROVISTA - INTEGRATE UAV DATA SOURCES
# ============================================================

source2_file = "data/processed/aerovista_master_dataset.csv"
source1_file = "data/raw/source1_real_telemetry.csv"

output_file = "data/processed/aerovista_final_dataset.csv"

print("Loading Source 2 trajectory dataset...")
source2 = pd.read_csv(source2_file)

print("Loading Source 1 real UAV telemetry...")
source1 = pd.read_csv(source1_file)

print("\n========== SOURCE DATASETS ==========")

print(f"Source 2 records: {len(source2)}")
print(f"Source 1 records: {len(source1)}")

# ------------------------------------------------------------
# Add source identification
# ------------------------------------------------------------

source2["data_source"] = "UAV_Trajectory_Dataset"
source1["data_source"] = "Real_UAV_Telemetry"

# ------------------------------------------------------------
# Standardize important location fields
#
# Source 2:
# x, y, z
#
# Source 1:
# aircraftLongitude, aircraftLatitude, aircraftAltitude
# ------------------------------------------------------------

source1["x"] = source1["aircraftLongitude"]
source1["y"] = source1["aircraftLatitude"]
source1["z"] = source1["aircraftAltitude"]

# ------------------------------------------------------------
# Add coordinate type so we don't confuse the two systems
# ------------------------------------------------------------

source2["coordinate_type"] = "trajectory_xyz"
source1["coordinate_type"] = "geographic_longitude_latitude_altitude"

# ------------------------------------------------------------
# Combine both datasets
# ------------------------------------------------------------

print("\nCombining datasets...")

final_dataset = pd.concat(
    [source2, source1],
    ignore_index=True,
    sort=False
)

# ------------------------------------------------------------
# Basic cleaning
# ------------------------------------------------------------

final_dataset = final_dataset.drop_duplicates()

# ------------------------------------------------------------
# Create processed folder if necessary
# ------------------------------------------------------------

os.makedirs("data/processed", exist_ok=True)

# ------------------------------------------------------------
# Save final dataset
# ------------------------------------------------------------

final_dataset.to_csv(output_file, index=False)

# ------------------------------------------------------------
# Final report
# ------------------------------------------------------------

print("\n========== AEROVISTA FINAL DATASET ==========")

print(f"Final records: {len(final_dataset)}")
print(f"Final columns: {len(final_dataset.columns)}")

print("\nData sources:")
print(final_dataset["data_source"].value_counts())

print("\nCoordinate types:")
print(final_dataset["coordinate_type"].value_counts())

print("\nMissing values:")
print(final_dataset.isnull().sum().sum())

print("\nDuplicate rows:")
print(final_dataset.duplicated().sum())

print("\nFirst 5 records:")
print(final_dataset.head())

print("\nSaved to:")
print(output_file)

print("\n========== FINAL DATASET CREATED ==========")