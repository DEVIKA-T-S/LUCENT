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
# 3. MediaPipe Face Landmarker setup
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
# 4. Open webcam
# --------------------------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("LUCENT Face Landmark Detection Started.")
print("Press Q to close the webcam.")


# --------------------------------------------------
# 5. Start Face Landmarker
# --------------------------------------------------

with FaceLandmarker.create_from_options(options) as landmarker:

    frame_number = 0

    while True:

        # Read webcam frame
        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read webcam frame.")
            break

        # OpenCV uses BGR.
        # MediaPipe expects RGB.
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Convert OpenCV frame into MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Create increasing timestamp
        timestamp_ms = int(
            frame_number * 1000 / 30
        )

        # Detect face landmarks
        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )


        # --------------------------------------------------
        # 6. Draw face landmarks
        # --------------------------------------------------

        if result.face_landmarks:

            for face_landmarks in result.face_landmarks:

                height, width, _ = frame.shape

                for landmark in face_landmarks:

                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    cv2.circle(
                        frame,
                        (x, y),
                        1,
                        (0, 255, 0),
                        -1
                    )

            cv2.putText(
                frame,
                "Face Detected",
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
        # 7. Display webcam
        # --------------------------------------------------

        cv2.imshow(
            "LUCENT - Face Landmarks",
            frame
        )

        frame_number += 1


        # Press Q to stop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# --------------------------------------------------
# 8. Close webcam
# --------------------------------------------------

camera.release()
cv2.destroyAllWindows()

print("LUCENT Face Landmark Detection Stopped.")