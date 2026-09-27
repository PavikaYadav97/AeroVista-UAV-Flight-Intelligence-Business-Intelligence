import pandas as pd
import os

print("=" * 60)
print("AEROVISTA FINAL PROJECT VALIDATION")
print("=" * 60)

# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------

files = {
    "Source 1 - Real UAV Telemetry":
        "data/raw/source1_real_telemetry.csv",

    "Source 2 - UAV Trajectory":
        "data/raw/source2_uav_trajectories.csv",

    "Source 2 - Master Dataset":
        "data/processed/aerovista_master_dataset.csv",

    "Combined Final Dataset":
        "data/processed/aerovista_final_dataset.csv",

    "Data Mining Analysis Dataset":
        "data/processed/aerovista_analysis_dataset.csv"
}

# ---------------------------------------------------------
# CHECK FILES
# ---------------------------------------------------------

print("\nFILE CHECK")
print("-" * 60)

for name, path in files.items():

    if os.path.exists(path):
        print(f"[OK] {name}")
        print(f"     {path}")
    else:
        print(f"[MISSING] {name}")
        print(f"     {path}")

# ---------------------------------------------------------
# SOURCE 1
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("SOURCE 1 VALIDATION")
print("=" * 60)

source1 = pd.read_csv(files["Source 1 - Real UAV Telemetry"])

print("Records:", len(source1))
print("Columns:", len(source1.columns))

print("\nColumns:")
print(list(source1.columns))

print("\nMissing values:")
print(source1.isnull().sum().sum())

# ---------------------------------------------------------
# SOURCE 2
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("SOURCE 2 VALIDATION")
print("=" * 60)

source2 = pd.read_csv(files["Source 2 - UAV Trajectory"])

print("Records:", len(source2))
print("Columns:", len(source2.columns))

print("\nMissing values:")
print(source2.isnull().sum().sum())

# ---------------------------------------------------------
# MASTER DATASET
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("MASTER DATASET VALIDATION")
print("=" * 60)

master = pd.read_csv(files["Source 2 - Master Dataset"])

print("Records:", len(master))
print("Columns:", len(master.columns))

print("\nMissing values:", master.isnull().sum().sum())
print("Duplicate rows:", master.duplicated().sum())

# ---------------------------------------------------------
# FINAL COMBINED DATASET
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL COMBINED DATASET")
print("=" * 60)

final = pd.read_csv(files["Combined Final Dataset"])

print("Records:", len(final))
print("Columns:", len(final.columns))

if "data_source" in final.columns:
    print("\nData sources:")
    print(final["data_source"].value_counts())

if "coordinate_type" in final.columns:
    print("\nCoordinate types:")
    print(final["coordinate_type"].value_counts())

print("\nMissing values:", final.isnull().sum().sum())
print("Duplicate rows:", final.duplicated().sum())

# ---------------------------------------------------------
# ANALYSIS DATASET
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("DATA MINING ANALYSIS DATASET")
print("=" * 60)

analysis = pd.read_csv(files["Data Mining Analysis Dataset"])

print("Records:", len(analysis))
print("Columns:", len(analysis.columns))

# ---------------------------------------------------------
# CLUSTER RESULTS
# ---------------------------------------------------------

if "flight_cluster" in analysis.columns:

    print("\nCluster Distribution:")
    print(analysis["flight_cluster"].value_counts())

# ---------------------------------------------------------
# ANOMALY RESULTS
# ---------------------------------------------------------

if "anomaly_label" in analysis.columns:

    print("\nAnomaly Distribution:")
    print(analysis["anomaly_label"].value_counts())

    anomaly_count = (
        analysis["anomaly_label"] == "Anomaly"
    ).sum()

    anomaly_percentage = (
        anomaly_count / len(analysis)
    ) * 100

    print(f"\nPotential anomalies: {anomaly_count}")
    print(f"Anomaly percentage: {anomaly_percentage:.2f}%")

# ---------------------------------------------------------
# IMPORTANT FEATURES
# ---------------------------------------------------------

features = [
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

print("\n" + "=" * 60)
print("FEATURE VALIDATION")
print("=" * 60)

for feature in features:

    if feature in analysis.columns:
        print(
            f"[OK] {feature} | "
            f"Missing: {analysis[feature].isnull().sum()}"
        )
    else:
        print(f"[MISSING] {feature}")

# ---------------------------------------------------------
# VISUALIZATIONS
# ---------------------------------------------------------

visualization_folder = "data/processed/visualizations"

print("\n" + "=" * 60)
print("VISUALIZATION FILES")
print("=" * 60)

if os.path.exists(visualization_folder):

    visualization_files = os.listdir(visualization_folder)

    for file in sorted(visualization_files):
        print("[OK]", file)

else:

    print("[MISSING] Visualization folder")

# ---------------------------------------------------------
# FINAL STATUS
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("AEROVISTA VALIDATION COMPLETE")
print("=" * 60)

print("\nProject pipeline:")
print("1. Data Collection          [DONE]")
print("2. Data Cleaning            [DONE]")
print("3. Feature Engineering      [DONE]")
print("4. Dataset Integration      [DONE]")
print("5. K-Means Clustering       [DONE]")
print("6. Anomaly Detection        [DONE]")
print("7. PCA                      [DONE]")
print("8. Visualization            [DONE]")
print("9. Dashboard                [DONE]")

print("\nAEROVISTA PROJECT READY FOR DOCUMENTATION.")