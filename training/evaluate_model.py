import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


print("==========================================")
print("       LUCENT PROPER MODEL EVALUATION")
print("==========================================")


DATASET_PATH = "data/processed/dataset_processed.csv"


# Load dataset
print()
print("Loading processed dataset...")

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.")
print(f"Number of samples: {len(df)}")


# Features
feature_columns = [
    "blink_rate",
    "avg_blink_duration",
    "avg_blink_interval",
    "face_size",
    "session_duration"
]

X = df[feature_columns]


# Convert labels
state_mapping = {
    "Normal": 0,
    "Low Blinking": 1,
    "Mild Fatigue": 2,
    "High Fatigue": 3
}

y = df["state"].map(state_mapping)


# Split dataset
print()
print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# Train a fresh model only on training data
print()
print("Training evaluation model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print("Evaluation model trained successfully.")


# Predict unseen test data
print()
print("Predicting test data...")

y_pred = model.predict(X_test)


# Accuracy
accuracy = accuracy_score(
    y_test,
    y_pred
)


# Confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1, 2, 3]
)


# Classification report
report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1, 2, 3],
    target_names=[
        "Normal",
        "Low Blinking",
        "Mild Fatigue",
        "High Fatigue"
    ],
    zero_division=0
)


# Results
print()
print("==========================================")
print("        PROPER EVALUATION RESULTS")
print("==========================================")

print()
print(f"Accuracy: {accuracy * 100:.2f}%")

print()
print("Confusion Matrix:")
print(cm)

print()
print("Classification Report:")
print(report)

print("==========================================")
print("       EVALUATION COMPLETED")
print("==========================================")