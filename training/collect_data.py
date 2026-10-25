import csv
import os
import sys
import cv2
import mediapipe as mp
import time


# ==================================================
# PROJECT PATH
# ==================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(__file__)
)

sys.path.append(PROJECT_ROOT)


# ==================================================
# FEATURE FUNCTIONS
# ==================================================

from src.features import (
    calculate_ear,
    calculate_face_size
)


# ==================================================
# DATASET PATH
# ==================================================

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "dataset_30sec.csv"
)


HEADERS = [
    "blink_rate",
    "avg_blink_duration",
    "avg_blink_interval",
    "face_size",
    "session_duration",
    "state"
]


# ==================================================
# MEDIAPIPE MODEL
# ==================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "face_landmarker.task"
)


if not os.path.exists(MODEL_PATH):

    print("ERROR: Face landmark model not found!")
    print(f"Expected location: {MODEL_PATH}")
    sys.exit()


print("Face landmark model found.")


# ==================================================
# PROGRAM INFORMATION
# ==================================================

print()
print("==========================================")
print("       LUCENT DATASET COLLECTOR")
print("==========================================")
print()
print(f"Dataset location: {DATASET_PATH}")
print()



# ==================================================
# MEDIAPIPE FACE LANDMARKER
# ==================================================

BaseOptions = mp.tasks.BaseOptions

FaceLandmarker = (
    mp.tasks.vision.FaceLandmarker
)

FaceLandmarkerOptions = (
    mp.tasks.vision.FaceLandmarkerOptions
)

RunningMode = (
    mp.tasks.vision.RunningMode
)


options = FaceLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=RunningMode.VIDEO,

    num_faces=1,

    min_face_detection_confidence=0.5,

    min_face_presence_confidence=0.5,

    min_tracking_confidence=0.5
)


landmarker = (
    FaceLandmarker.create_from_options(
        options
    )
)


print("Face landmarker initialized successfully.")
print()


# ==================================================
# WEBCAM
# ==================================================

camera = cv2.VideoCapture(0)


if not camera.isOpened():

    print("ERROR: Could not open webcam.")

    landmarker.close()

    sys.exit()


# ==================================================
# SESSION VARIABLES
# ==================================================

start_time = time.time()

last_sample_time = start_time
sample_count = 0

face_size = 0.0

elapsed_time = 0.0

blink_count_window = 0
blink_durations_window = []

blink_intervals_window = []

blink_rate = 0.0

average_blink_duration = 0.0

average_blink_interval = 0.0


# ==================================================
# BLINK SETTINGS
# ==================================================

EAR_THRESHOLD = 0.21

MIN_CLOSED_FRAMES = 2

# Maximum time allowed for a normal blink
MAX_BLINK_DURATION = 1.0

# ==================================================
# DATA COLLECTION SETTINGS
# ==================================================

SAMPLE_INTERVAL = 30
COLLECTION_DURATION = 120


closed_frames = 0

blink_count = 0
last_ear_print = time.time()


blink_start_time = None

blink_durations = []


last_blink_time = None

blink_intervals = []


# ==================================================
# EYE LANDMARK INDICES
# ==================================================

LEFT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380
]


RIGHT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144
]

# ==================================================
# ASK FOR STATE LABEL
# ==================================================

print()
print("Select the state for this session:")
print()
print("1 - Normal")
print("2 - Low Blinking")
print("3 - Mild Fatigue")
print("4 - High Fatigue")
print("5 - Skip this session")
print()

state_choice = input(
    "Enter choice (1-5): "
).strip()

state_map = {
    "1": "Normal",
    "2": "Low Blinking",
    "3": "Mild Fatigue",
    "4": "High Fatigue"
}

state = state_map.get(
    state_choice
)

if state is None:
    print("Invalid choice. Collection cancelled.")
    camera.release()
    landmarker.close()
    sys.exit()

print()
print(f"Selected state: {state}")
print()

# ==================================================
# START WEBCAM
# ==================================================

print("Webcam started.")
print("Blink normally during the session.")
print("Press Q to stop collecting data.")
print()


frame_timestamp = 0


# ==================================================
# MAIN LOOP
# ==================================================

while True:

    success, frame = camera.read()


    if not success:

        print("ERROR: Could not read frame.")

        break


    # ----------------------------------------------
    # Convert BGR to RGB
    # ----------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # ----------------------------------------------
    # Convert to MediaPipe image
    # ----------------------------------------------

    mp_image = mp.Image(

        image_format=mp.ImageFormat.SRGB,

        data=rgb_frame
    )


    # ----------------------------------------------
    # Timestamp
    # ----------------------------------------------

    frame_timestamp += 33


    # ----------------------------------------------
    # Detect face landmarks
    # ----------------------------------------------

    result = landmarker.detect_for_video(

        mp_image,

        frame_timestamp
    )


    # ----------------------------------------------
    # Calculate session time
    # ----------------------------------------------

    elapsed_time = (
        time.time()
        -
        start_time
    )


    # ----------------------------------------------
    # Face detected
    # ----------------------------------------------

    if result.face_landmarks:

        landmarks = result.face_landmarks[0]


        # ------------------------------------------
        # Face size
        # ------------------------------------------

        face_size = calculate_face_size(
            landmarks
        )


        # ------------------------------------------
        # Eye landmarks
        # ------------------------------------------

        left_eye = [

            landmarks[index]

            for index in LEFT_EYE

        ]


        right_eye = [

            landmarks[index]

            for index in RIGHT_EYE

        ]


        # ------------------------------------------
        # Calculate EAR
        # ------------------------------------------

        left_ear = calculate_ear(
            left_eye
        )


        right_ear = calculate_ear(
            right_eye
        )


        average_ear = (

            left_ear
            +
            right_ear

        ) / 2.0

        # Print EAR approximately once every second
        if time.time() - last_ear_print >= 1.0:
            print(f"Average EAR: {average_ear:.3f}")
            last_ear_print = time.time()



        # ------------------------------------------
        # Blink detection
        # ------------------------------------------

        if average_ear < EAR_THRESHOLD:

            closed_frames += 1


            if blink_start_time is None:

                blink_start_time = time.time()


        else:

            if closed_frames >= MIN_CLOSED_FRAMES:

                current_blink_time = time.time()

                # Calculate blink duration
                if blink_start_time is not None:

                    blink_duration = (
                        current_blink_time
                        - blink_start_time
                    )

                    # Count only normal-duration blinks
                    if blink_duration <= MAX_BLINK_DURATION:

                        blink_count += 1
                        blink_count_window += 1

                        blink_durations.append(
                            blink_duration
                        )

                        blink_durations_window.append(
                            blink_duration
                        )

                        # Calculate blink interval
                        if last_blink_time is not None:

                            blink_interval = (
                                current_blink_time
                                - last_blink_time
                            )

                            blink_intervals.append(
                                blink_interval
                            )

                            blink_intervals_window.append(
                                blink_interval
                            )

                        last_blink_time = (
                            current_blink_time
                        )

                blink_start_time = None



            closed_frames = 0


        # ------------------------------------------
        # Blink rate for current window
        # ------------------------------------------

        window_elapsed_time = (
            time.time() - last_sample_time
        )

        if window_elapsed_time > 0:

            blink_rate = (

                blink_count_window
                /
                window_elapsed_time

            ) * 60

        else:

            blink_rate = 0.0


        # ------------------------------------------
        # Average blink duration for current window
        # ------------------------------------------

        if blink_durations_window:

            average_blink_duration = (

                sum(blink_durations_window)
                /
                len(blink_durations_window)

            )

        else:

            average_blink_duration = 0.0


        # ------------------------------------------
        # Average blink interval for current window
        # ------------------------------------------

        if blink_intervals_window:

            average_blink_interval = (

                sum(blink_intervals_window)
                /
                len(blink_intervals_window)

            )

        else:

            average_blink_interval = 0.0


        # ------------------------------------------
        # Draw eye landmarks
        # ------------------------------------------

        for point in (
            left_eye + right_eye
        ):

            x = int(
                point.x * frame.shape[1]
            )

            y = int(
                point.y * frame.shape[0]
            )


            cv2.circle(

                frame,

                (x, y),

                3,

                (0, 255, 0),

                -1
            )


        # ------------------------------------------
        # Display EAR
        # ------------------------------------------

        cv2.putText(

            frame,

            f"Left EAR: {left_ear:.2f}",

            (20, 35),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 255, 0),

            2
        )


        cv2.putText(

            frame,

            f"Right EAR: {right_ear:.2f}",

            (20, 70),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 255, 0),

            2
        )


        cv2.putText(

            frame,

            f"Average EAR: {average_ear:.2f}",

            (20, 105),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 255, 0),

            2
        )


        # ------------------------------------------
        # Blink count
        # ------------------------------------------

        cv2.putText(

            frame,

            f"Blinks: {blink_count}",

            (20, 140),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 255, 255),

            2
        )


        # ------------------------------------------
        # Blink rate
        # ------------------------------------------

        cv2.putText(

            frame,

            f"Blink Rate: {blink_rate:.1f}/min",

            (20, 175),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 165, 0),

            2
        )


        # ------------------------------------------
        # Face size
        # ------------------------------------------

        cv2.putText(

            frame,

            f"Face Size: {face_size:.3f}",

            (20, 210),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 255, 0),

            2
        )


        # ------------------------------------------
        # Session duration
        # ------------------------------------------

        cv2.putText(

            frame,

            f"Session: {elapsed_time:.0f}s",

            (20, 245),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 255, 255),

            2
        )


    else:

        # ------------------------------------------
        # No face detected
        # ------------------------------------------

        cv2.putText(

            frame,

            "No face detected",

            (20, 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 0, 255),

            2
        )


    # ==================================================
    # SHOW WEBCAM
    # ==================================================

    cv2.imshow(
        "LUCENT Dataset Collector",
        frame
    )


    # ==================================================
    # PRESS Q TO STOP
    # ==================================================
    key = cv2.waitKey(1) & 0xFF


    # ==================================================
    # AUTOMATIC SAMPLE COLLECTION
    # ==================================================

    current_time = time.time()

    if (
        current_time - last_sample_time >= SAMPLE_INTERVAL
        or (
            elapsed_time >= COLLECTION_DURATION
            and current_time > last_sample_time
        )
    ):

        sample_count += 1

        print()
        print(
            f"Sample {sample_count} collected "
            f"at {elapsed_time:.1f} seconds"
        )

        # ----------------------------------------------
        # Save current sample to dataset
        # ----------------------------------------------

        os.makedirs(
            os.path.dirname(DATASET_PATH),
            exist_ok=True
        )

        file_exists = (
            os.path.exists(DATASET_PATH)
            and os.path.getsize(DATASET_PATH) > 0
        )

        with open(
            DATASET_PATH,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            if not file_exists:

                writer.writerow(
                    HEADERS
                )

            writer.writerow([

                round(blink_rate, 3),

                round(
                    average_blink_duration,
                    3
                ),

                round(
                    average_blink_interval,
                    3
                ),

                round(face_size, 3),

                round(elapsed_time, 3),

                state

            ])

        print(
            f"Sample {sample_count} saved to dataset."
        )

        last_sample_time = current_time
        blink_count_window = 0
        blink_durations_window.clear()
        blink_intervals_window.clear()
    
    # ==================================================
    # AUTOMATIC COLLECTION TIME LIMIT
    # ==================================================

    if elapsed_time >= COLLECTION_DURATION:

        print()
        print(
            "Collection duration reached."
        )

        break


    if key == ord("q"):

        break


# ==================================================
# CLEANUP
# ==================================================

camera.release()

cv2.destroyAllWindows()

landmarker.close()


# ==================================================
# FINAL FEATURE CALCULATIONS
# ==================================================

if elapsed_time > 0:

    blink_rate = (

        blink_count
        /
        elapsed_time

    ) * 60

else:

    blink_rate = 0.0


if blink_durations:

    average_blink_duration = (

        sum(blink_durations)
        /
        len(blink_durations)

    )

else:

    average_blink_duration = 0.0


if blink_intervals:

    average_blink_interval = (

        sum(blink_intervals)
        /
        len(blink_intervals)

    )

else:

    average_blink_interval = 0.0


# ==================================================
# SESSION SUMMARY
# ==================================================

print()
print("==========================================")
print("          SESSION COMPLETED")
print("==========================================")
print()

print(
    f"Total blinks detected: {blink_count}"
)

print(
    f"Session duration: {elapsed_time:.1f} seconds"
)

print(
    f"Average blink duration: "
    f"{average_blink_duration:.2f} seconds"
)

print(
    f"Average blink interval: "
    f"{average_blink_interval:.2f} seconds"
)

print(
    f"Final blink rate: "
    f"{blink_rate:.1f} blinks/min"
)

print(
    f"Final face size: "
    f"{face_size:.3f}"
)



