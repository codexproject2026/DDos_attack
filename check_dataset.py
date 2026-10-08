import pandas as pd

# ==========================================
# LOAD DATASET
# ==========================================

dataset_path = "dataset/ML-EdgeIIoT-dataset.csv"

print("\nLoading dataset...")

df = pd.read_csv(dataset_path, low_memory=False)

print("\n====================================")
print("DATASET INFORMATION")
print("====================================")

print("Dataset Shape:", df.shape)

print("\nTotal Rows:", df.shape[0])
print("Total Columns:", df.shape[1])

# ==========================================
# COLUMN NAMES
# ==========================================

print("\n====================================")
print("COLUMN NAMES")
print("====================================")

for i, column in enumerate(df.columns, start=1):
    print(i, column)

# ==========================================
# ATTACK LABEL
# ==========================================

print("\n====================================")
print("ATTACK LABEL DISTRIBUTION")
print("====================================")

if "Attack_label" in df.columns:
    print(df["Attack_label"].value_counts(dropna=False))
else:
    print("Attack_label column NOT FOUND")

# ==========================================
# ATTACK TYPE
# ==========================================

print("\n====================================")
print("ATTACK TYPE DISTRIBUTION")
print("====================================")

if "Attack_type" in df.columns:
    print(df["Attack_type"].value_counts(dropna=False))
else:
    print("Attack_type column NOT FOUND")

print("\n====================================")
print("CHECK COMPLETED")
print("====================================")