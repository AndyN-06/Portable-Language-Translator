import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import threading
import queue
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from gpiozero import Button
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
    """Activate speech mode."""
    print("Switched to Speech Mode. Translator device is active and listening.")
    translator_device.active = True  # Resume processing

def asl_mode_logic():
    """Initialize settings for ASL mode."""
    print("Switched to ASL Mode. Camera activated for gesture detection.")
    translator_device.active = False  # Pause processing

# Start the translator device and Flask server in separate threads
translator_thread = threading.Thread(target=translator_device.start, daemon=True)
translator_thread.start()

flask_thread = threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5000), daemon=True)
flask_thread.start()

# ==================== PHYSICAL BUTTON SETUP ====================
# Volume control functions
def set_volume(level):
    """Set volume level (0-100%)"""
    os.system(f"amixer -D pulse sset Master {level}%")

def increase_volume(step=5):
    """Increase volume by a step"""
    os.system(f"amixer -D pulse sset Master {step}%+")

def decrease_volume(step=5):
    """Decrease volume by a step"""
    os.system(f"amixer -D pulse sset Master {step}%-")

def get_volume():
    """Get current volume level"""
    result = os.popen("amixer -D pulse get Master").read()
    volume = int(result.split('[')[1].split('%')[0])
    return volume

# GPIO pin setup (adjust pin numbers as needed)
PIN_MODE = 4
PIN_UP = 17
PIN_DOWN = 27

# Global mode variable. We use the same mode strings as in your original code.
mode = "SPEECH"  # initial mode

# Note: 'cap' is the camera handle used in ASL mode.
cap = None

def change_mode():
    global mode, cap
    cv2.destroyAllWindows()  # Clear any open windows when switching modes
    if mode == "ASL":
        mode = "SPEECH"
        speech_mode_logic()
        if cap is not None:
            cap.release()
            cap = None
        print("Mode changed to SPEECH")
    else:
        mode = "ASL"
        asl_mode_logic()
        if cap is None:
            cap = cv2.VideoCapture(0)
        print("Mode changed to ASL")

def volume_up():
    print("Increased Volume")
    increase_volume()

def volume_down():
    print("Decreased Volume")
    decrease_volume()

# Initialize buttons with pull-up resistors
button_mode = Button(PIN_MODE, pull_up=True, bounce_time=0.2)
button_up = Button(PIN_UP, pull_up=True, bounce_time=0.2)
button_down = Button(PIN_DOWN, pull_up=True, bounce_time=0.2)

# Attach event listeners to buttons
button_mode.when_pressed = change_mode
button_up.when_pressed = volume_up
button_down.when_pressed = volume_down

# ==================== MAIN LOOP (MODE SWITCHING) ====================
# Variables for ASL detection
sequence = []
predictions = []
sentence = []
threshold = 0.8
last_detection_time = time.time()
frame_count = 0
start_time = time.time()

# Create the Mediapipe Hands instance for ASL mode (reuse when needed)
hands_instance = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.8)

while True:
    # Check for a 'q' key press to quit the loop
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break

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
            # Plays detected ASL gesture 
            translator_device.synthesize_speech(' '.join(sentence), translator_device.base_language)
            # Listen for audio and store transcript in a text file
            translator_device.vad_active = True
            transcript = translator_device.listen_and_save_transcription(file_path="als_speech_audio_transcription.txt")
            translator_device.vad_active = False
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

# Cleanup on exit
if cap is not None:
    cap.release()
cv2.destroyAllWindows()
stop_thread = True
asl_thread.join()
translator_thread.join()
flask_thread.join()
