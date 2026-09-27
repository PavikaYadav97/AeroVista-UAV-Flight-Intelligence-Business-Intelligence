import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ============================================================
# AEROVISTA
# MULTI-SOURCE UAV FLIGHT INTELLIGENCE & ANOMALY DETECTION
# DATA MINING ANALYSIS
# ============================================================

INPUT_FILE = "data/processed/aerovista_master_dataset.csv"

OUTPUT_DIR = "data/processed"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "aerovista_analysis_dataset.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("============================================================")
print("AEROVISTA DATA MINING ANALYSIS")
print("============================================================")

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

print("\nLoading master dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Records loaded: {len(df)}")
print(f"Columns loaded: {len(df.columns)}")

# ------------------------------------------------------------
# 2. SELECT FEATURES FOR DATA MINING
# ------------------------------------------------------------

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

print("\nFeatures selected for data mining:")

for feature in features:
    print("-", feature)

# ------------------------------------------------------------
# 3. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\nChecking missing values...")

missing = df[features].isnull().sum()

print(missing)

# ------------------------------------------------------------
# 4. CLEAN FEATURE DATA
# ------------------------------------------------------------

analysis_df = df.copy()

# Replace infinite values
analysis_df[features] = analysis_df[features].replace(
    [np.inf, -np.inf],
    np.nan
)

# Fill any remaining missing feature values
# using median values calculated from Source 2.
for feature in features:

    median_value = analysis_df[feature].median()

    analysis_df[feature] = analysis_df[feature].fillna(
        median_value
    )

print("\nFeature cleaning completed.")

print(
    "Remaining missing feature values:",
    analysis_df[features].isnull().sum().sum()
)

# ------------------------------------------------------------
# 5. STANDARDIZATION
# ------------------------------------------------------------

print("\nStandardizing features...")

scaler = StandardScaler()

X = scaler.fit_transform(
    analysis_df[features]
)

print("Standardization completed.")

# ------------------------------------------------------------
# 6. K-MEANS CLUSTERING
# ------------------------------------------------------------

print("\n============================================================")
print("K-MEANS CLUSTERING")
print("============================================================")

print("\nTesting cluster numbers from 2 to 6...")

silhouette_results = {}

for k in range(2, 7):

    kmeans_test = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels_test = kmeans_test.fit_predict(X)

    score = silhouette_score(
        X,
        labels_test
    )

    silhouette_results[k] = score

    print(
        f"K = {k} | Silhouette Score = {score:.4f}"
    )

# Select best K
best_k = max(
    silhouette_results,
    key=silhouette_results.get
)

print(
    f"\nSelected number of clusters: {best_k}"
)

# Final KMeans
kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

analysis_df["flight_cluster"] = kmeans.fit_predict(X)

print("\nCluster distribution:")

print(
    analysis_df["flight_cluster"].value_counts().sort_index()
)

# ------------------------------------------------------------
# 7. ISOLATION FOREST
# ------------------------------------------------------------

print("\n============================================================")
print("ISOLATION FOREST ANOMALY DETECTION")
print("============================================================")

print("\nDetecting potentially unusual flight behavior...")

isolation_forest = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=42
)

anomaly_prediction = isolation_forest.fit_predict(X)

anomaly_score = isolation_forest.decision_function(X)

# Isolation Forest:
#  1  = normal
# -1  = anomaly

analysis_df["anomaly_label"] = np.where(
    anomaly_prediction == -1,
    "Anomaly",
    "Normal"
)

analysis_df["anomaly_score"] = anomaly_score

print("\nAnomaly distribution:")

print(
    analysis_df["anomaly_label"].value_counts()
)

anomaly_count = (
    analysis_df["anomaly_label"] == "Anomaly"
).sum()

print(
    f"\nPotential anomalies detected: {anomaly_count}"
)

print(
    f"Potential anomaly percentage: "
    f"{(anomaly_count / len(analysis_df)) * 100:.2f}%"
)

# ------------------------------------------------------------
# 8. PCA
# ------------------------------------------------------------

print("\n============================================================")
print("PCA DIMENSIONALITY REDUCTION")
print("============================================================")

pca = PCA(
    n_components=2,
    random_state=42
)

pca_result = pca.fit_transform(X)

analysis_df["PCA1"] = pca_result[:, 0]

analysis_df["PCA2"] = pca_result[:, 1]

print(
    f"Variance explained by PCA1: "
    f"{pca.explained_variance_ratio_[0] * 100:.2f}%"
)

print(
    f"Variance explained by PCA2: "
    f"{pca.explained_variance_ratio_[1] * 100:.2f}%"
)

print(
    f"Total variance explained: "
    f"{pca.explained_variance_ratio_.sum() * 100:.2f}%"
)

# ------------------------------------------------------------
# 9. SAVE ANALYSIS DATASET
# ------------------------------------------------------------

analysis_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n============================================================")
print("AEROVISTA DATA MINING COMPLETE")
print("============================================================")

print(f"\nFinal records: {len(analysis_df)}")

print(
    f"Final columns: {len(analysis_df.columns)}"
)

print("\nNew analytical columns:")

print("flight_cluster")
print("anomaly_label")
print("anomaly_score")
print("PCA1")
print("PCA2")

print("\nSaved to:")

print(OUTPUT_FILE)

print("\n============================================================")
print("DONE")
print("============================================================")