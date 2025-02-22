import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import threading
import queue
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
from translator_device import TranslatorDevice  # Adjust the import path as needed

# ==================== ASL SETUP ====================
actions = np.array(["hello", "thanks", "iloveyou"])

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="hands.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def tflite_predict(sequence):
    """Run TFLite inference on a given input sequence."""
    sequence = np.expand_dims(sequence, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], sequence)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, hands_model):
    """Runs Mediapipe Hands on a frame and returns the drawn image and results."""
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
            label = handedness.classification[0].label
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            if label == 'Left':
                lh = keypoints
            elif label == 'Right':
                rh = keypoints
    return np.concatenate([lh, rh])

def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
            )

# Setup queues and threading for asynchronous ASL inference
sequence_queue = queue.Queue(maxsize=5)  # Queue for sequences to be inferred
result_queue = queue.Queue(maxsize=5)    # Queue for storing prediction results
stop_thread = False

def inference_worker():
    """Processes sequences asynchronously in a separate thread."""
    while not stop_thread:
        try:
            sequence = sequence_queue.get(timeout=1)
            res = tflite_predict(sequence)
            predicted_action = np.argmax(res)
            result_queue.put((predicted_action, res[predicted_action]))
        except queue.Empty:
            continue

asl_thread = threading.Thread(target=inference_worker, daemon=True)
asl_thread.start()

# ==================== SPEECH SETUP ====================
# Initialize the translator device and Flask app
translator_device = TranslatorDevice()
app = Flask(__name__)
CORS(app)

@app.route('/set_settings', methods=['POST'])
def set_settings():
    data = request.get_json()
    base_language = data.get('baseLanguage')
    gender = data.get('gender')
    if not base_language or not gender:
        return jsonify({'status': 'error', 'message': 'Invalid settings.'}), 400
    translator_device.set_settings(base_language, gender)
    return jsonify({'status': 'success', 'message': 'Settings updated.'}), 200

def speech_mode_logic():
    """Any additional logic to activate speech mode can be placed here."""
    print("Switched to Speech Mode. Translator device is active and listening.")
    translator_device.active = True  # Resume processing

def asl_mode_logic():
    """Logic to (re)initialize settings for ASL mode."""
    print("Switched to ASL Mode. Camera activated for gesture detection.")
    translator_device.active = False  # Pause processing

# Start the translator device and Flask server in separate threads
translator_thread = threading.Thread(target=translator_device.start, daemon=True)
translator_thread.start()

flask_thread = threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5000), daemon=True)
flask_thread.start()

# ==================== MAIN LOOP (MODE SWITCHING) ====================
mode = "SPEECH"  # Start in Speech mode
cap = None  # Camera handle; only used in ASL mode

# Variables for ASL detection
sequence = []
predictions = []
sentence = []
threshold = 0.8
last_detection_time = time.time()
frame_count = 0
start_time = time.time()

# Create the Mediapipe Hands instance for ASL mode (reuse it when needed)
hands_instance = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.8)

while True:
    key = cv2.waitKey(10) & 0xFF

    # Debug: print current mode
    # (Uncomment the next line if you want to see the mode on every loop iteration)
    # print(f"Current mode: {mode}")

    # Check for Enter key (on some systems, Enter might be 10 or 13)
    if key in [10, 13]:
        # Destroy any existing windows to clear the previous mode's display
        cv2.destroyAllWindows()
        if mode == "ASL":
            mode = "SPEECH"
            speech_mode_logic()
            if cap is not None:
                cap.release()
                cap = None
        else:
            mode = "ASL"
            asl_mode_logic()
            if cap is None:
                cap = cv2.VideoCapture(0)
        time.sleep(0.3)  # Prevent rapid toggling

    if mode == "ASL":
        # Ensure camera is active
        if cap is None:
            cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_count += 1
        image, results = mediapipe_detection(frame, hands_instance)
        draw_styled_landmarks(image, results)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]  # Keep the last 30 frames

        if len(sequence) == 30 and not sequence_queue.full():
            sequence_queue.put_nowait(np.array(sequence))

        if not result_queue.empty():
            predicted_action, confidence = result_queue.get_nowait()
            predictions.append(predicted_action)
            if (len(predictions) > 10 and 
                np.unique(predictions[-10:])[0] == predicted_action and 
                confidence > threshold):
                action_name = actions[predicted_action]
                if not sentence or (action_name != sentence[-1]):
                    sentence.append(action_name)
                    last_detection_time = time.time()
            if len(sentence) > 3:
                sentence = sentence[-3:]
        
        if time.time() - last_detection_time > 2 and sentence:
            with open("sentences.txt", "w") as f:
                f.write(' '.join(sentence) + "\n")
                
            # Plays detected ASL gesture 
            translator_device.synthesize_speech(' '.join(sentence), translator_device.base_language)
            
            # Listen for audio and store transcript in a text file
            transcript = translator_device.record_and_transcribe(duration=5)
            with open("audio_transcript.txt", "w") as f:
                f.write(transcript)
            
            sentence = []

        cv2.putText(
            image, ' '.join(sentence), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.imshow("OpenCV Feed", image)

        current_time = time.time()
        if current_time - start_time >= 1.0:
            fps = frame_count / (current_time - start_time)
            frame_count = 0
            start_time = current_time

    else:
        # In speech mode, ensure the camera is off and show a placeholder window
        if cap is not None:
            cap.release()
            cap = None
        blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank_image, "Speech Mode Active", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("OpenCV Feed", blank_image)

    # Exit the loop when 'q' is pressed
    if key == ord('q'):
        break

if cap is not None:
    cap.release()
cv2.destroyAllWindows()
stop_thread = True
asl_thread.join()
translator_thread.join()
flask_thread.join()
