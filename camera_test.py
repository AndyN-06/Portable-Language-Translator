from picamera2 import Picamera2
import cv2

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration({"format": "RGB888"}))
picam2.start()

while True:
    frame = picam2.capture_array()
    # Now 'frame' is a numpy array you can pass to Mediapipe or show with OpenCV
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break
cv2.destroyAllWindows()
