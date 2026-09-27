from datasets import load_dataset
import pandas as pd

print("Connecting to UAV trajectory dataset...")

dataset = load_dataset(
    "riotu-lab/Synthetic-UAV-Flight-Trajectories",
    split="train",
    streaming=True
)

print("Connection successful!")
print("Collecting 150,000 UAV trajectory records...")

records = []

for i, row in enumerate(dataset):

    records.append({
        "timestamp": row["timestamp"],
        "x": row["tx"],
        "y": row["ty"],
        "z": row["tz"]
    })

    if (i + 1) % 10000 == 0:
        print(f"Collected {i + 1} records...")

    if i + 1 == 150000:
        break

print("\nCreating DataFrame...")

df = pd.DataFrame(records)

output_path = "data/raw/source2_uav_trajectories.csv"

df.to_csv(output_path, index=False)

print("\nDataset created successfully!")
print(f"Records: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"Saved to: {output_path}")

print("\nFirst 5 records:")
print(df.head())