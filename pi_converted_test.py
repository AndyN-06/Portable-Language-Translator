import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import threading
import queue
from picamera2 import Picamera2  # Import the Picamera2 module

actions = np.array(["hello", "thanks", "iloveyou"])

# -- Load the TFLite model --
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def tflite_predict(sequence):
    """Runs TFLite inference on a given input sequence."""
    sequence = np.expand_dims(sequence, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], sequence)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

# Mediapipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, holistic_model):
    """
    Processes an image using MediaPipe Holistic.
    Ensures the input is a 3-channel RGB image.
    Returns:
      - drawn_frame: image converted to BGR for OpenCV display.
      - results: MediaPipe detection results.
    """
    # If image has 4 channels (e.g. BGRA), convert from BGRA to RGB.
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    # If image has 3 channels, assume it's BGR and convert to RGB.
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("Input image does not have 3 or 4 channels.")

    results = holistic_model.process(image)
    # Convert the image back to BGR for display with OpenCV.
    drawn_frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return drawn_frame, results


def extract_keypoints(results):
    """Extracts pose, face, left-hand, and right-hand landmarks into a single array."""
    pose = np.array([[res.x, res.y, res.z, res.visibility] 
                     for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)

    face = np.array([[res.x, res.y, res.z]
                     for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)

    lh = np.array([[res.x, res.y, res.z]
                   for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)

    rh = np.array([[res.x, res.y, res.z]
                   for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)

    return np.concatenate([pose, face, lh, rh])

def draw_styled_landmarks(image, results):
    """Draw pose and hand landmarks (face landmarks are commented out)."""
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
    )
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
    )
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
    )

# === MULTITHREADING SETUP ===
sequence_queue = queue.Queue(maxsize=5)  # Queue to store sequences for inference
result_queue = queue.Queue(maxsize=5)    # Queue to store predictions
stop_thread = False

def inference_worker():
    """Runs in a separate thread to process sequences asynchronously."""
    while not stop_thread:
        try:
            sequence = sequence_queue.get(timeout=1)  # Get the next sequence (blocking)
            res = tflite_predict(sequence)
            predicted_action = np.argmax(res)
            result_queue.put((predicted_action, res[predicted_action]))  # Store result
        except queue.Empty:
            continue  # Timeout expired, loop again

# Start the worker thread
thread = threading.Thread(target=inference_worker, daemon=True)
thread.start()

# === PICAMERA2 SETUP ===
picam2 = Picamera2()
# Create a preview configuration (adjust resolution as needed)
preview_config = picam2.create_preview_configuration(main={"size": (1280, 720)})
picam2.configure(preview_config)
picam2.start()

# === MAIN LOOP ===
sequence = []
predictions = []
sentence = []
threshold = 0.5

frame_count = 0
start_time = time.time()

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:
    while True:
        # Capture frame from PiCamera2
        # The captured frame is in RGB format
        frame = picam2.capture_array()
        if frame is None:
            break

        frame_count += 1
        # Process the image (no need to convert from BGR to RGB)
        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]  # Keep only the last 30 frames

        if len(sequence) == 30 and not sequence_queue.full():
            sequence_queue.put_nowait(np.array(sequence))  # Send sequence to worker

        # Check if there's a new prediction result
        if not result_queue.empty():
            predicted_action, confidence = result_queue.get_nowait()
            predictions.append(predicted_action)

            # Ensure at least 10 consistent predictions before displaying
            if (len(predictions) > 10 and 
                np.unique(predictions[-10:])[0] == predicted_action and 
                confidence > threshold):
                
                action_name = actions[predicted_action]
                if not sentence or (action_name != sentence[-1]):
                    sentence.append(action_name)

            if len(sentence) > 3:
                sentence = sentence[-3:]

        # Display the current sentence on screen
        cv2.putText(
            image, ' '.join(sentence), (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
        )

        cv2.imshow("OpenCV Feed", image)

        # FPS calculation
        current_time = time.time()
        if current_time - start_time >= 1.0:
            fps = frame_count / (current_time - start_time)
            print(f"FPS: {fps:.2f}")
            frame_count = 0
            start_time = current_time

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

picam2.stop()
cv2.destroyAllWindows()
stop_thread = True
thread.join()  # Ensure the thread stops before exiting
