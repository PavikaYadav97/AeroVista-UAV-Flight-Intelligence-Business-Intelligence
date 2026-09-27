import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================================
# AEROVISTA - DATA MINING VISUALIZATION
# ============================================================

INPUT_FILE = "data/processed/aerovista_analysis_dataset.csv"
OUTPUT_DIR = "data/processed/visualizations"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("AEROVISTA DATA MINING VISUALIZATION")
print("=" * 60)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\nLoading analysis dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Records loaded: {len(df)}")
print(f"Columns loaded: {len(df.columns)}")

# ------------------------------------------------------------
# 1. K-MEANS CLUSTER DISTRIBUTION
# ------------------------------------------------------------

print("\nCreating Cluster Distribution plot...")

cluster_counts = df["flight_cluster"].value_counts().sort_index()

plt.figure(figsize=(8, 5))
plt.bar(
    cluster_counts.index.astype(str),
    cluster_counts.values
)

plt.title("Flight Behavior Cluster Distribution")
plt.xlabel("Flight Cluster")
plt.ylabel("Number of Records")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/01_cluster_distribution.png",
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# 2. PCA CLUSTER VISUALIZATION
# ------------------------------------------------------------

print("Creating PCA Cluster plot...")

plt.figure(figsize=(10, 7))

for cluster in sorted(df["flight_cluster"].unique()):
    subset = df[df["flight_cluster"] == cluster]

    # Sample large clusters so plotting remains fast
    if len(subset) > 10000:
        subset = subset.sample(10000, random_state=42)

    plt.scatter(
        subset["PCA1"],
        subset["PCA2"],
        s=5,
        alpha=0.5,
        label=f"Cluster {cluster}"
    )

plt.title("PCA Visualization of Flight Behavior Clusters")
plt.xlabel("PCA1 (47.78% variance)")
plt.ylabel("PCA2 (15.37% variance)")
plt.legend()
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/02_pca_clusters.png",
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# 3. ANOMALY DISTRIBUTION
# ------------------------------------------------------------

print("Creating Anomaly Distribution plot...")

anomaly_counts = df["anomaly_label"].value_counts()

plt.figure(figsize=(8, 5))
plt.bar(
    anomaly_counts.index,
    anomaly_counts.values
)

plt.title("Normal vs Potential Anomalous Flight Behavior")
plt.xlabel("Behavior Type")
plt.ylabel("Number of Records")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/03_anomaly_distribution.png",
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# 4. SPEED DISTRIBUTION
# ------------------------------------------------------------

print("Creating Speed Distribution plot...")

plt.figure(figsize=(9, 5))

plt.hist(
    df["speed"],
    bins=60
)

plt.title("Drone Speed Distribution")
plt.xlabel("Speed")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/04_speed_distribution.png",
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# 5. ACCELERATION DISTRIBUTION
# ------------------------------------------------------------

print("Creating Acceleration Distribution plot...")

plt.figure(figsize=(9, 5))

plt.hist(
    df["acceleration"],
    bins=60
)

plt.title("Drone Acceleration Distribution")
plt.xlabel("Acceleration")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/05_acceleration_distribution.png",
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# 6. ALTITUDE DISTRIBUTION
# ------------------------------------------------------------

print("Creating Altitude Distribution plot...")

plt.figure(figsize=(9, 5))

plt.hist(
    df["z"],
    bins=60
)

plt.title("Drone Altitude Distribution")
plt.xlabel("Altitude (Z)")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/06_altitude_distribution.png",
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# 7. HORIZONTAL SPEED VS VERTICAL SPEED
# ------------------------------------------------------------

print("Creating Movement Behavior plot...")

sample = df.sample(
    min(15000, len(df)),
    random_state=42
)

plt.figure(figsize=(9, 6))

plt.scatter(
    sample["horizontal_speed"],
    sample["absolute_vertical_speed"],
    s=5,
    alpha=0.4
)

plt.title("Horizontal Speed vs Vertical Movement")
plt.xlabel("Horizontal Speed")
plt.ylabel("Absolute Vertical Speed")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/07_speed_movement.png",
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# 8. TRAJECTORY SEGMENT DISTRIBUTION
# ------------------------------------------------------------

print("Creating Trajectory Segment plot...")

segment_counts = df["trajectory_segment_id"].value_counts()

plt.figure(figsize=(10, 5))

plt.hist(
    segment_counts.values,
    bins=40
)

plt.title("Distribution of Records Across Trajectory Segments")
plt.xlabel("Records per Trajectory Segment")
plt.ylabel("Number of Segments")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/08_trajectory_segments.png",
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# FINAL MESSAGE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("VISUALIZATION COMPLETE")
print("=" * 60)

print(f"\nPlots saved inside:")
print(OUTPUT_DIR)

print("\nGenerated files:")

for file in sorted(os.listdir(OUTPUT_DIR)):
    print("-", file)

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)