import cv2
import mediapipe as mp
import time
import os
import json
import joblib
import pandas as pd

from src.features import calculate_ear, calculate_face_size
from src.wellness_score import calculate_wellness_score
from src.database import create_tables, save_session, save_alert

# ==========================================
# LUCENT LIVE PREDICTION
# ==========================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "fatigue_model.pkl"
)

SETTINGS_PATH = os.path.join(
    PROJECT_ROOT,
    "config",
    "settings.json"
)

def load_settings():

    if not os.path.exists(SETTINGS_PATH):
        return {
            "blink_reminder_minutes": 1,
            "enable_alerts": True,
            "alert_cooldown_seconds": 60,
            "alert_display_seconds": 5
        }

    with open(SETTINGS_PATH, "r") as file:
        return json.load(file)


settings = load_settings()

ENABLE_ALERTS = settings.get(
    "enable_alerts",
    True
)

ALERT_COOLDOWN_SECONDS = settings.get(
    "alert_cooldown_seconds",
    60
)

ALERT_DISPLAY_DURATION = settings.get(
    "alert_display_seconds",
    5
)

BLINK_REMINDER_MINUTES = settings.get(
    "blink_reminder_minutes",
    1
)

# ----------------------------------
# BLINK REMINDER TIMER
# ----------------------------------

BLINK_REMINDER_SECONDS = (
    BLINK_REMINDER_MINUTES * 60
)

last_blink_time = time.time()
blink_reminder_triggered = False

FACE_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "face_landmarker.task"
)


# ==========================================
# CHECK FILES
# ==========================================

if not os.path.exists(MODEL_PATH):
    print("Fatigue model not found!")
    exit()

if not os.path.exists(FACE_MODEL_PATH):
    print("Face landmark model not found!")
    exit()


# ==========================================
# LOAD ML MODEL
# ==========================================

model = joblib.load(MODEL_PATH)

print("Fatigue model loaded successfully.")

# ==========================================
# DATABASE INITIALIZATION
# ==========================================

create_tables()
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
# MEDIAPIPE SETUP
# ==========================================

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=FACE_MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


landmarker = FaceLandmarker.create_from_options(
    options
)


# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Could not open webcam.")
    landmarker.close()
    exit()

print("LUCENT Live Prediction Started.")
print("Press Q to stop.")


# ==========================================
# EYE LANDMARKS
# ==========================================

LEFT_EYE = [
    362, 385, 387,
    263, 373, 380
]

RIGHT_EYE = [
    33, 160, 158,
    133, 153, 144
]


# ==========================================
# BLINK SETTINGS
# ==========================================

EAR_THRESHOLD = 0.21
MIN_CLOSED_FRAMES = 1

# ==========================================
# ALERT ENGINE
# ==========================================

from src.alert_engine import AlertEngine

alert_engine = AlertEngine(
    cooldown_seconds=ALERT_COOLDOWN_SECONDS,
    persistence_count=3
)

alert_display_until = 0
# ==========================================
# SESSION VARIABLES
# ==========================================

start_time = time.time()

blink_count = 0
closed_frames = 0

blink_durations = []
blink_intervals = []

blink_start_time = None
last_blink_time = None

video_start_time = time.monotonic()
previous_timestamp = -1

alert_history = []

# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read webcam frame.")
        break

    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    frame_timestamp = int((time.monotonic() - video_start_time) * 1000)
    frame_timestamp = max(frame_timestamp, previous_timestamp + 1)
    previous_timestamp = frame_timestamp

    result = landmarker.detect_for_video(
        mp_image,
        frame_timestamp
    )

    # --------------------------------------
    # DEFAULT VALUES
    # --------------------------------------

    blink_rate = 0.0
    avg_blink_duration = 0.0
    avg_blink_interval = 0.0
    face_size = 0.0

    # --------------------------------------
    # FACE DETECTED
    # --------------------------------------

    if result.face_landmarks:

        face_landmarks = result.face_landmarks[0]

        # Calculate face size
        face_size = calculate_face_size(
            face_landmarks
        )

        # Get eye points
        left_eye_points = [
            face_landmarks[i]
            for i in LEFT_EYE
        ]

        right_eye_points = [
            face_landmarks[i]
            for i in RIGHT_EYE
        ]

        # Calculate EAR
        left_ear = calculate_ear(
            left_eye_points
        )

        right_ear = calculate_ear(
            right_eye_points
        )

        average_ear = (
            left_ear + right_ear
        ) / 2

        # ----------------------------------
        # BLINK DETECTION
        # ----------------------------------

        if average_ear < EAR_THRESHOLD:

            closed_frames += 1

            if (
                closed_frames >= MIN_CLOSED_FRAMES
                and blink_start_time is None
            ):
                blink_start_time = time.time()

        else:

            if (
                blink_start_time is not None
                and 0 < (time.time() - blink_start_time) <= 1.0
                and (
                    last_blink_time is None
                    or time.time() - last_blink_time >= 0.20
                )
            ):

                blink_end_time = time.time()

                duration = (
                    blink_end_time
                    - blink_start_time
                )

                blink_durations.append(
                    duration
                )

                blink_count += 1

                if last_blink_time is not None:

                    interval = (
                        blink_end_time
                        - last_blink_time
                    )

                    blink_intervals.append(
                        interval
                    )

                last_blink_time = (
                    blink_end_time
                )

                blink_reminder_triggered = False

            blink_start_time = None

            closed_frames = 0

        # ----------------------------------
        # CALCULATE FEATURES
        # ----------------------------------

        elapsed_time = (
            time.time() - start_time
        )

        if elapsed_time > 0:

            blink_rate = (
                blink_count
                / elapsed_time
                * 60
            )

        if blink_durations:

            avg_blink_duration = sum(
                blink_durations
            ) / len(blink_durations)

        if blink_intervals:

            avg_blink_interval = sum(
                blink_intervals
            ) / len(blink_intervals)

        # ----------------------------------
        # ML PREDICTION
        # ----------------------------------

        if blink_count > 0:

            prediction_data = pd.DataFrame(
                [[
                    blink_rate,
                    avg_blink_duration,
                    avg_blink_interval,
                    face_size,
                    elapsed_time
                ]],
                columns=[
                    "blink_rate",
                    "avg_blink_duration",
                    "avg_blink_interval",
                    "face_size",
                    "session_duration"
                ]
            )

            prediction = model.predict(
                prediction_data
            )[0]

            state = STATE_LABELS.get(
                int(prediction),
                "Unknown"
            )

        else:

            state = "Collecting..."
        # ----------------------------------
        # WELLNESS SCORE
        # ----------------------------------

        if state != "Collecting...":

            wellness_score = calculate_wellness_score(
                blink_rate,
                avg_blink_duration,
                avg_blink_interval,
                state
            )

        else:

            wellness_score = 0

        # ----------------------------------
        # ALERT CHECK
        # ----------------------------------

        alert_triggered = False

        if state != "Collecting..." and ENABLE_ALERTS:

            alert_triggered = alert_engine.check_alert(
                state
            )

            if alert_triggered:

                alert_display_until = (
                    time.time() + ALERT_DISPLAY_DURATION
                )

                alert_history.append({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "alert_type": state,
                    "message": "Take a short break"
                })
        # ----------------------------------
        # BLINK REMINDER CHECK
        # ----------------------------------

        if state != "Collecting...":

            time_since_blink = (
                time.time() - last_blink_time
            )

            if (
                time_since_blink >= BLINK_REMINDER_SECONDS
                and not blink_reminder_triggered
            ):

                blink_reminder_triggered = True

                alert_display_until = (
                    time.time() + ALERT_DISPLAY_DURATION
                )

                alert_history.append({
                    "timestamp": time.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "alert_type": "Blink Reminder",
                    "message": "Please blink your eyes"
                })
        # ----------------------------------
        # DISPLAY
        # ----------------------------------

        cv2.putText(
            frame,
            f"State: {state}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Blinks: {blink_count}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Blink Rate: {blink_rate:.1f}/min",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Face Size: {face_size:.3f}",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Wellness Score: {wellness_score}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        if time.time() < alert_display_until:

            cv2.putText(
                frame,
                "ALERT: TAKE A SHORT BREAK",
                (20, 195),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
    else:

        cv2.putText(
            frame,
            "No face detected",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # --------------------------------------
    # SHOW CAMERA
    # --------------------------------------

    cv2.imshow(
        "LUCENT Live Prediction",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        # Calculate final session duration
        final_session_duration = time.time() - start_time

        # Save the completed session
        session_id = save_session(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            session_duration=final_session_duration,
            blink_rate=blink_rate,
            avg_blink_duration=avg_blink_duration,
            avg_blink_interval=avg_blink_interval,
            face_size=face_size,
            state=state,
            wellness_score=wellness_score
        )

        print("Session saved to LUCENT database.")
        print("Session ID:", session_id)

        # Save all alerts generated during this session
        for alert in alert_history:

            save_alert(
                session_id=session_id,
                timestamp=alert["timestamp"],
                alert_type=alert["alert_type"],
                message=alert["message"]
            )

        print("Alerts saved:", len(alert_history))

        break

# ==========================================
# CLEANUP
# ==========================================

cap.release()
cv2.destroyAllWindows()
landmarker.close()

print("\nLUCENT Live Prediction Stopped.")
print(f"Total blinks detected: {blink_count}")





