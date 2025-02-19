import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import threading
import queue

actions = np.array(["hello", "thanks", "iloveyou"])

# -- Load the TFLite model --
interpreter = tf.lite.Interpreter(model_path="hands.tflite")
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
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, hands_model):
    """Runs Mediapipe Hands on a frame and returns (drawn_frame, results)."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = hands_model.process(image_rgb)
    image_rgb.flags.writeable = True
    drawn_frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return drawn_frame, results

def extract_keypoints(results):
    # Each hand has 21 landmarks with 3 coordinates (x, y, z)
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # handedness.classification is a list; we take the first element.
            label = handedness.classification[0].label
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            if label == 'Left':
                lh = keypoints
            elif label == 'Right':
                rh = keypoints
    # Concatenate left and right hand keypoints into one array (total length = 126)
    return np.concatenate([lh, rh])

def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
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

# === MAIN LOOP ===
sequence = []
predictions = []
sentence = []
threshold = 0.5

# Initialize last detection time
last_detection_time = time.time()

cap = cv2.VideoCapture(0)
frame_count = 0
start_time = time.time()

with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.8
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        image, results = mediapipe_detection(frame, hands)
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
                    last_detection_time = time.time()  # Update the time on new detection

            # Limit the sentence length if desired
            if len(sentence) > 3:
                sentence = sentence[-3:]

        # Check if no gesture has been detected for 2 seconds.
        if time.time() - last_detection_time > 2 and sentence:
            # Append the sentence to a file and clear it
            with open("sentences.txt", "a") as f:
                f.write(' '.join(sentence) + "\n")
            sentence = []

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

cap.release()
cv2.destroyAllWindows()
stop_thread = True
thread.join()  # Ensure the thread stops before exiting
