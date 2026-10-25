import os
import joblib
import pandas as pd


# ==========================================
# LUCENT REAL-TIME PREDICTOR
# ==========================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "fatigue_model.pkl"
)


# ==========================================
# STATE LABELS
# ==========================================

STATE_LABELS = {
    0: "Normal",
    1: "Low Blinking",
    2: "Mild Fatigue",
    3: "High Fatigue"
}


# ==========================================
# LOAD MODEL
# ==========================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# ==========================================
# FEATURE ORDER
# ==========================================

FEATURE_COLUMNS = [
    "blink_rate",
    "avg_blink_duration",
    "avg_blink_interval",
    "face_size",
    "session_duration"
]


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_state(
    blink_rate,
    avg_blink_duration,
    avg_blink_interval,
    face_size,
    session_duration
):
    """
    Predict the current LUCENT state.
    """

    data = pd.DataFrame(
        [[
            blink_rate,
            avg_blink_duration,
            avg_blink_interval,
            face_size,
            session_duration
        ]],
        columns=FEATURE_COLUMNS
    )

    prediction = model.predict(data)[0]

    state = STATE_LABELS.get(
        int(prediction),
        "Unknown"
    )

    return state


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    print("==========================================")
    print("       LUCENT PREDICTOR TEST")
    print("==========================================")

    print("\nModel loaded successfully.")

    test_state = predict_state(
        blink_rate=9.6,
        avg_blink_duration=0.41,
        avg_blink_interval=5.19,
        face_size=0.315,
        session_duration=31.2
    )

    print("\nTest prediction:")
    print("Predicted state:", test_state)

    print("\nLUCENT predictor is working successfully!")