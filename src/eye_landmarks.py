import cv2
import mediapipe as mp
import os


# --------------------------------------------------
# 1. Find the Face Landmarker model
# --------------------------------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "face_landmarker.task"
)


# --------------------------------------------------
# 2. Check whether the model exists
# --------------------------------------------------

if not os.path.exists(MODEL_PATH):
    print("ERROR: face_landmarker.task was not found.")
    print("Expected location:")
    print(MODEL_PATH)
    exit()

print("Model found successfully!")


# --------------------------------------------------
# 3. MediaPipe setup
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


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


# --------------------------------------------------
# 4. Eye landmark indexes
# --------------------------------------------------
#
# These are MediaPipe Face Mesh landmark positions
# around the left and right eye.
#

LEFT_EYE = [
    362, 385, 387, 263, 373, 380
]

RIGHT_EYE = [
    33, 160, 158, 133, 153, 144
]


# --------------------------------------------------
# 5. Open webcam
# --------------------------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("LUCENT Eye Landmark Detection Started.")
print("Press Q to close the webcam.")


# --------------------------------------------------
# 6. Start Face Landmarker
# --------------------------------------------------

with FaceLandmarker.create_from_options(options) as landmarker:

    frame_number = 0

    while True:

        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read webcam frame.")
            break

        # Convert BGR → RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Create MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Timestamp
        timestamp_ms = int(
            frame_number * 1000 / 30
        )

        # Detect landmarks
        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )


        # --------------------------------------------------
        # 7. Draw eye landmarks
        # --------------------------------------------------

        if result.face_landmarks:

            face_landmarks = result.face_landmarks[0]

            height, width, _ = frame.shape


            # Draw LEFT eye points
            for index in LEFT_EYE:

                landmark = face_landmarks[index]

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1
                )


            # Draw RIGHT eye points
            for index in RIGHT_EYE:

                landmark = face_landmarks[index]

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1
                )


            # Display message
            cv2.putText(
                frame,
                "Eyes Detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "No Face Detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )


        # --------------------------------------------------
        # 8. Show webcam
        # --------------------------------------------------

        cv2.imshow(
            "LUCENT - Eye Landmarks",
            frame
        )

        frame_number += 1


        # Press Q to stop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# --------------------------------------------------
# 9. Close webcam
# --------------------------------------------------

camera.release()
cv2.destroyAllWindows()

print("LUCENT Eye Landmark Detection Stopped.")