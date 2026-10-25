import pandas as pd
import os
import joblib

from sklearn.ensemble import RandomForestClassifier


# ==========================================
# LUCENT MODEL TRAINING
# ==========================================

print("==========================================")
print("          LUCENT MODEL TRAINING")
print("==========================================")


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "dataset_processed.csv"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "fatigue_model.pkl"
)


# ==========================================
# CHECK DATASET
# ==========================================

if not os.path.exists(DATASET_PATH):
    print("Processed dataset not found!")
    print(DATASET_PATH)
    exit()

print("\nProcessed dataset found:")
print(DATASET_PATH)


# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully.")
print("Number of samples:", len(df))


# ==========================================
# FEATURES
# ==========================================

feature_columns = [
    "blink_rate",
    "avg_blink_duration",
    "avg_blink_interval",
    "face_size",
    "session_duration"
]

X = df[feature_columns]
y = df["label"]


print("\nFeatures used for training:")
for feature in feature_columns:
    print("-", feature)


# ==========================================
# DISPLAY LABELS
# ==========================================

print("\nLabels:")
print(y.tolist())


# ==========================================
# CREATE MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ==========================================
# TRAIN MODEL
# ==========================================

print("\nTraining model...")

model.fit(X, y)

print("Model training completed successfully.")


# ==========================================
# CREATE MODELS DIRECTORY
# ==========================================

os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True
)


# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    model,
    MODEL_PATH
)

print("\n==========================================")
print("       MODEL TRAINING COMPLETE")
print("==========================================")

print("\nModel saved at:")
print(MODEL_PATH)

print("\nLUCENT ML model is ready!")