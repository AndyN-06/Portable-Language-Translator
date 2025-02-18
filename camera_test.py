import cv2

cap = cv2.VideoCapture(
    "libcamerasrc ! videoconvert ! appsink",
    cv2.CAP_GSTREAMER
)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Camera not found!")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Pi Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
