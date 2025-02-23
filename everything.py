import sys
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import threading
import queue
import os
import sounddevice as sd
from flask import Flask, request, jsonify
from flask_cors import CORS
from gpiozero import Button
from ui import CameraTextViewer
from PyQt5.QtWidgets import QApplication
from translator_device import TranslatorDevice  # Adjust the import path as needed
from shared import latest_frame
import shared

# ==================== ASL & SPEECH SETUP ====================

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

# Queues and threading for asynchronous inference
sequence_queue = queue.Queue(maxsize=5)
result_queue = queue.Queue(maxsize=5)
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

# ==================== FLASK & TRANSLATOR SETUP ====================

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
    translator_device.vad_active = True
    translator_device.vad_active = False
    time.sleep(1)
    translator_device.vad_active = True
    translator_device.active = True
    

def asl_mode_logic():
    """Initialize ASL mode."""
    print("Switched to ASL Mode. Camera activated for gesture detection.")
    translator_device.active = False
    translator_device.vad_active = False

translator_thread = threading.Thread(target=translator_device.start, daemon=True)
translator_thread.start()

flask_thread = threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5000), daemon=True)
flask_thread.start()

# ==================== PHYSICAL BUTTON & VOLUME SETUP ====================

def set_volume(level):
    os.system(f"amixer -D pulse sset Master {level}%")

def increase_volume(step=10):
    os.system(f"amixer -D pulse sset Master {step}%+")

def decrease_volume(step=10):
    os.system(f"amixer -D pulse sset Master {step}%-")

def get_volume():
    result = os.popen("amixer -D pulse get Master").read()
    volume = int(result.split('[')[1].split('%')[0])
    return volume

# Adjust these pin numbers as needed
PIN_MODE = 4
PIN_UP = 17
PIN_DOWN = 27

# Global mode variable and camera handle (for ASL mode)
mode = "SPEECH"  # Initial mode
cap = None

def flush_audio_stream():
    # Open a temporary stream to read and discard frames
    with sd.InputStream(samplerate=translator_device.SAMPLE_RATE, 
                        channels=translator_device.NUM_CHANNELS, dtype='int16') as flush_stream:
        # Read and discard frames for 2 seconds
        flush_end = time.time() + 2
        while time.time() < flush_end:
            try:
                _ = flush_stream.read(int(translator_device.SAMPLE_RATE * (translator_device.FRAME_DURATION / 1000.0)))
            except Exception as e:
                pass

def change_mode():
    global mode, cap, sequence, predictions, sentence
    # Flush ASL buffers/queues
    while not sequence_queue.empty():
        sequence_queue.get_nowait()
    while not result_queue.empty():
        result_queue.get_nowait()
    sequence.clear()
    predictions.clear()
    sentence.clear()
    
    cv2.destroyAllWindows()
    if mode == "ASL":
        mode = "SPEECH"
        speech_mode_logic()
        if cap is not None:
            cap.release()
            cap = None      
        print("Mode changed to SPEECH")
        translator_device.reset()
        translator_device.active = True
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

button_mode = Button(PIN_MODE, pull_up=True, bounce_time=0.2)
button_up = Button(PIN_UP, pull_up=True, bounce_time=0.2)
button_down = Button(PIN_DOWN, pull_up=True, bounce_time=0.2)

button_mode.when_pressed = change_mode
button_up.when_pressed = volume_up
button_down.when_pressed = volume_down

# ==================== ASL PROCESSING (Non-UI) ====================

# Variables used in ASL processing
sequence = []
predictions = []
sentence = []
threshold = 0.8
last_detection_time = time.time()
frame_count = 0
start_time = time.time()
latest_frame = None

hands_instance = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.8)

def asl_processing_loop():
    global cap, sequence, predictions, sentence, last_detection_time, frame_count, start_time, latest_frame
    while True:
        # Exit condition can be defined via a key press or external signal
        # Here, we simply break if a flag is set (you can adjust as needed)
        if mode == "ASL":
            if cap is None:
                cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            if not ret:
                continue
            
            shared.latest_frame = frame.copy()

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
                # When ASL gesture detected, synthesize speech and write to text file
                text_out = ' '.join(sentence)
                translator_device.synthesize_speech(text_out, translator_device.base_language)

                # Listen for audio and store transcript in a text file
                translator_device.vad_active = True
                transcript = translator_device.listen_and_save_transcription(file_path="als_speech_audio_transcription.txt")
                translator_device.vad_active = False
                sentence = []

            # Sleep briefly to yield control (adjust as needed)
            time.sleep(0.03)
        else:
            # When in speech mode, ensure the camera is released
            if cap is not None:
                cap.release()
                cap = None
            time.sleep(0.1)

asl_proc_thread = threading.Thread(target=asl_processing_loop, daemon=True)
asl_proc_thread.start()

# ==================== THREAD CLEANUP FUNCTION ====================
def cleanup():
    global stop_thread, cap
    print("Initiating cleanup...")
    stop_thread = True  # Signal all loops to exit
    # Release the camera if in use
    if cap is not None:
        cap.release()
    # Join threads
    asl_proc_thread.join()
    asl_thread.join()
    translator_thread.join()
    flask_thread.join()
    print("Cleanup complete.")

# ==================== APPLICATION ENTRY POINT ====================

if __name__ == "__main__":
    file_path = "als_speech_audio_transcription.txt"  # This file is updated by the ASL processing thread
    app_qt = QApplication(sys.argv)
    window = CameraTextViewer(file_path)
    window.show()
    try:
        exit_code = app_qt.exec_()
    except KeyboardInterrupt:
        exit_code = 0
    finally:
        cleanup()
    sys.exit(exit_code)
