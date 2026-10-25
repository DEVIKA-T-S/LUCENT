import cv2
import mediapipe as mp
import os

import time

from src.features import (
    distance,
    calculate_ear,
    calculate_face_size
)
# --------------------------------------------------
# 1. Find the MediaPipe face landmark model
# --------------------------------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "face_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    print("Face landmark model not found!")
    exit()

print("Model found successfully!")

# --------------------------------------------------
# 2. MediaPipe setup
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# --------------------------------------------------
# 3. Eye landmark points
# --------------------------------------------------

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# --------------------------------------------------
# 4. Calculate distance between two landmarks
# --------------------------------------------------

# --------------------------------------------------
# 5. Calculate Eye Aspect Ratio (EAR)
# --------------------------------------------------

# --------------------------------------------------
# 6. Blink detection settings
# --------------------------------------------------

EAR_THRESHOLD = 0.21

closed_frames = 0
blink_count = 0
last_ear_print = time.time()

# Number of frames eyes must remain closed
# before we consider it a blink.
MIN_CLOSED_FRAMES = 1

# Maximum time allowed for a normal blink
MAX_BLINK_DURATION = 1.0

# Blink duration tracking
blink_start_time = None
blink_durations = []

# Blink interval tracking
last_blink_time = None
blink_intervals = []
# --------------------------------------------------
# 7. Start webcam
# --------------------------------------------------

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not camera.isOpened():
    print("Could not open the webcam.")
    exit()

print("LUCENT Blink Detection Started.")
print("Press Q to close the webcam.")

start_time = time.time()

# --------------------------------------------------
# 8. Start MediaPipe
# --------------------------------------------------

with FaceLandmarker.create_from_options(options) as landmarker:

    min_ear = None
    video_start_time = time.monotonic()
    previous_timestamp = -1
    fps_start_time = time.time()
    fps_frame_count = 0

    while True:

        success, frame = camera.read()

        if not success:
            print("Could not read frame.")
            break

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Create MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect face landmarks
        frame_timestamp = int((time.monotonic() - video_start_time) * 1000)
        frame_timestamp = max(frame_timestamp, previous_timestamp + 1)
        previous_timestamp = frame_timestamp
        result = landmarker.detect_for_video(
            mp_image,
            frame_timestamp
        )

        fps_frame_count += 1
        fps_elapsed = time.time() - fps_start_time
        if fps_elapsed >= 1.0:
            print(f"Processing FPS: {fps_frame_count / fps_elapsed:.1f}")
            fps_frame_count = 0
            fps_start_time = time.time()


        # --------------------------------------------------
        # 9. Process detected face
        # --------------------------------------------------

        if result.face_landmarks:

            landmarks = result.face_landmarks[0]
            face_size = calculate_face_size(landmarks)

            # Get eye landmarks
            left_eye_points = [
                landmarks[i]
                for i in LEFT_EYE
            ]

            right_eye_points = [
                landmarks[i]
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
            ) / 2.0

            # Track minimum EAR during each second
            if min_ear is None:
                min_ear = average_ear
            else:
                min_ear = min(min_ear, average_ear)

            if time.time() - last_ear_print >= 1.0:
                print(f"Minimum EAR: {min_ear:.3f} | Current EAR: {average_ear:.3f}")
                min_ear = None
                last_ear_print = time.time()
            # --------------------------------------------------
            # 10. Detect closed eyes
            # --------------------------------------------------
            if average_ear < EAR_THRESHOLD:

                # Eyes are closed
                closed_frames += 1

                # Record the time when the eyes first close
                if blink_start_time is None:
                    blink_start_time = time.time()

            else:

                # Eyes are open again
                if closed_frames >= MIN_CLOSED_FRAMES:

                    # Current time when the eyes open again
                    current_blink_time = time.time()

                    # Calculate how long the eyes were closed
                    if blink_start_time is not None:

                        blink_duration = (
                            current_blink_time - blink_start_time
                        )

                        # Count only normal-duration blinks
                        if (
                            blink_duration <= MAX_BLINK_DURATION
                            and (
                                last_blink_time is None
                                or current_blink_time - last_blink_time >= 0.20
                            )
                        ):

                            blink_count += 1

                            blink_durations.append(
                                blink_duration
                            )

                            # Calculate time since previous valid blink
                            if last_blink_time is not None:

                                blink_interval = (
                                    current_blink_time - last_blink_time
                                )

                                blink_intervals.append(
                                    blink_interval
                                )

                            # Save current valid blink time
                            last_blink_time = current_blink_time

                    # Reset blink start time
                    blink_start_time = None

                closed_frames = 0
                blink_start_time = None
           # --------------------------------------------------
           # 11. Calculate session time and blink rate
           # --------------------------------------------------

            elapsed_time = time.time() - start_time

            if  blink_count > 0 and elapsed_time > 0:
                blink_rate = (blink_count / elapsed_time) * 60
            else:
                blink_rate = 0.0

            # --------------------------------------------------
            # 12. Display information
            # --------------------------------------------------

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
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Average EAR: {average_ear:.2f}",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                f"Face Size: {face_size:.3f}",
                (20, 375),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                f"Blinks: {blink_count}",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Session: {elapsed_time:.0f}s",
                (20, 165),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            cv2.putText(
                frame,
                f"Blink Rate: {blink_rate:.1f}/min",
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
            # Calculate blink duration statistics
            if blink_durations:

                last_blink_duration = blink_durations[-1]

                average_blink_duration = (
                    sum(blink_durations) /
                    len(blink_durations)
                )

            else:

                last_blink_duration = 0.0
                average_blink_duration = 0.0

            cv2.putText(
                frame,
                f"Last Blink: {last_blink_duration:.2f}s",
                (20, 235),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Avg Blink: {average_blink_duration:.2f}s",
                (20, 270),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            # Calculate blink interval statistics
            if blink_intervals:

                last_blink_interval = blink_intervals[-1]

                average_blink_interval = (
                    sum(blink_intervals) /
                    len(blink_intervals)
                )

            else:

                last_blink_interval = 0.0
                average_blink_interval = 0.0


            cv2.putText(
                frame,
                f"Last Interval: {last_blink_interval:.2f}s",
                (20, 305),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 165, 0),
                2
            )

            cv2.putText(
                frame,
                f"Avg Interval: {average_blink_interval:.2f}s",
                (20, 340),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 165, 0),
                2
            )
            # --------------------------------------------------
            # 13. Show eye landmarks
            # --------------------------------------------------

            for point in left_eye_points:
                x = int(point.x * frame.shape[1])
                y = int(point.y * frame.shape[0])

                cv2.circle(
                    frame,
                    (x, y),
                    2,
                    (0, 255, 0),
                    -1
                )

            for point in right_eye_points:
                x = int(point.x * frame.shape[1])
                y = int(point.y * frame.shape[0])

                cv2.circle(
                    frame,
                    (x, y),
                    2,
                    (0, 255, 0),
                    -1
                )

        # --------------------------------------------------
        # 14. Display webcam
        # --------------------------------------------------

        cv2.imshow(
            "LUCENT - Blink Detection",
            frame
        )

        # Press Q to quit
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


# --------------------------------------------------
# 15. Release resources
# --------------------------------------------------

camera.release()
cv2.destroyAllWindows()

print("LUCENT Blink Detection Stopped.")
print(f"Total blinks detected: {blink_count}")





