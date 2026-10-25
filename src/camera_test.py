import cv2

# Open the laptop webcam
camera = cv2.VideoCapture(0)

# Check whether the webcam opened
if not camera.isOpened():
    print("Could not open the webcam.")
    exit()

print("LUCENT webcam started.")
print("Press Q to close the webcam window.")

while True:

    # Read one frame from the webcam
    success, frame = camera.read()

    if not success:
        print("Could not read frame from webcam.")
        break

    # Display the webcam frame
    cv2.imshow("LUCENT - Webcam Test", frame)

    # Wait for a keyboard key
    key = cv2.waitKey(1) & 0xFF

    # Press Q to quit
    if key == ord("q"):
        break

# Release the webcam
camera.release()

# Close OpenCV windows
cv2.destroyAllWindows()

print("LUCENT webcam stopped.")