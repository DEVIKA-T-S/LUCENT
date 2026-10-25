import pandas as pd
import os


# ==========================================
# LUCENT DATASET PREPARATION
# ==========================================

print("==========================================")
print("       LUCENT DATASET PREPARATION")
print("==========================================")

# Project paths
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "dataset.csv"
)

PROCESSED_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "dataset_processed.csv"
)


# ==========================================
# CHECK DATASET
# ==========================================

if not os.path.exists(DATASET_PATH):
    print("Dataset file not found!")
    print(DATASET_PATH)
    exit()

print("\nDataset found:")
print(DATASET_PATH)


# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully.")
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))


# ==========================================
# DISPLAY DATA
# ==========================================

print("\nDataset:")
print(df)


# ==========================================
# CHECK REQUIRED COLUMNS
# ==========================================

required_columns = [
    "blink_rate",
    "avg_blink_duration",
    "avg_blink_interval",
    "face_size",
    "session_duration",
    "state"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    print("\nMissing columns:")
    print(missing_columns)
    exit()

print("\nAll required columns are present.")


# ==========================================
# CHECK MISSING VALUES
# ==========================================

print("\nMissing values:")
print(df[required_columns].isnull().sum())


# ==========================================
# REMOVE INVALID ROWS
# ==========================================

df = df.dropna(subset=required_columns)

print("\nRows after removing invalid data:", len(df))


# ==========================================
# CHECK STATES
# ==========================================

print("\nState distribution:")
print(df["state"].value_counts())


# ==========================================
# CONVERT STATES TO NUMERIC LABELS
# ==========================================

state_mapping = {
    "Normal": 0,
    "Low Blinking": 1,
    "Mild Fatigue": 2,
    "High Fatigue": 3
}

df["label"] = df["state"].map(state_mapping)


# ==========================================
# CHECK UNKNOWN STATES
# ==========================================

if df["label"].isnull().any():
    print("\nUnknown state detected.")
    print(df[df["label"].isnull()]["state"])
    exit()

print("\nState labels created successfully.")


# ==========================================
# CREATE PROCESSED DIRECTORY
# ==========================================

processed_directory = os.path.dirname(PROCESSED_PATH)

os.makedirs(
    processed_directory,
    exist_ok=True
)


# ==========================================
# SAVE PROCESSED DATASET
# ==========================================

df.to_csv(
    PROCESSED_PATH,
    index=False
)

print("\n==========================================")
print("       DATASET PREPARATION COMPLETE")
print("==========================================")

print("\nProcessed dataset saved at:")
print(PROCESSED_PATH)

print("\nState mapping:")
print("Normal        = 0")
print("Low Blinking  = 1")
print("Mild Fatigue  = 2")
print("High Fatigue  = 3")

print("\nLUCENT dataset preparation completed successfully!")