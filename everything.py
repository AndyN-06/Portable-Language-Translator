# everything.py
import sys
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import shared
import time
import threading
import queue
import os
import sounddevice as sd
from flask import Flask, request, jsonify
from flask_cors import CORS
from gpiozero import Button
from TabularUI import MainWindow
from PyQt5.QtWidgets import QApplication, QMessageBox
from translator_device import TranslatorDevice  # Adjust the import path as needed
from shared import latest_frame

# ==================== ASL & SPEECH SETUP ====================
actions = np.array(["hello", "thanks", "nothing", "help", "yes", "bathroom"])

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="newest.tflite")
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
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def mediapipe_detection(image, model):
    """Runs MediaPipe Holistic on a frame and returns the drawn image and results."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    drawn_frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return drawn_frame, results

def extract_keypoints(results):
    """Extract keypoints from MediaPipe Holistic results."""
    # Extract pose landmarks (33 landmarks * 4 values (x,y,z,visibility))
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    
    # Extract left hand landmarks (21 landmarks * 3 values (x,y,z))
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    
    # Extract right hand landmarks (21 landmarks * 3 values (x,y,z))
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    
    return np.concatenate([pose, lh, rh])

def draw_styled_landmarks(image, results):
    """Draw landmarks and connections for pose and hands."""
    # Draw pose connections
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
    )
    # Draw left hand connections
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
    )
    # Draw right hand connections
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
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
    shared.ui_mode = "CAMERA"

translator_thread = threading.Thread(target=translator_device.start, daemon=True)
translator_thread.start()

flask_thread = threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5000), daemon=True)
flask_thread.start()

# ==================== PHYSICAL BUTTON & VOLUME SETUP ====================

def set_volume(level):
    # Ensure level doesn't exceed 90%
    capped_level = min(90, max(0, level))
    os.system(f"amixer -D pulse sset Master {capped_level}%")

def increase_volume(step=5):
    current = get_volume()
    # Calculate new volume but don't exceed 90%
    new_volume = min(90, current + step)
    set_volume(new_volume)

def decrease_volume(step=5):
    current = get_volume()
    # Ensure volume doesn't go below 0
    new_volume = max(0, current - step)
    set_volume(new_volume)

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
        shared.ui_mode = "TEXT"
    else:
        mode = "ASL"
        with open(file_path, 'w') as file:
            pass  # clear the file contents
        asl_mode_logic()
        if cap is None:
            cap = cv2.VideoCapture(0)
        print("Mode changed to ASL")
        shared.ui_mode = "CAMERA"

    shared.mode = mode

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
threshold = 0.9
last_detection_time = time.time()
frame_count = 0
start_time = time.time()
latest_frame = None
last_prediction_time = 0
min_prediction_interval = 0.5
HISTORY_LENGTH = 4  # Number of predictions to consider
MIN_CONSISTENT_PREDICTIONS = 3  # Minimum number of same predictions needed
prediction_history = []  # Store recent predictions

# hands_instance = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.8)

def asl_processing_loop():
    nothing_count = 0
    current_prediction = ""
    global cap, sequence, predictions, sentence, last_detection_time, frame_count
    global start_time, latest_frame, last_prediction_time, prediction_history

    while True:
        if mode == "ASL":
            if cap is None:
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set a fixed width
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400) # Set a fixed height
            ret, frame = cap.read()
            if not ret:
                continue
            
            image = cv2.resize(frame, (640, 400)) # Resize the frame
            
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            # Draw current sentence at the top
            sentence_text = ' '.join(sentence)
            cv2.putText(image, f"Sentence: {sentence_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Draw current prediction below the sentence
            cv2.putText(image, f"Predicting: {current_prediction}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            shared.latest_frame = image.copy()  # Update this line to use the annotated image
            frame_count += 1
            
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            # if len(sequence) >= 30 and frame_count % 5 == 0 and not sequence_queue.full():
            if len(sequence) >= 30 and not sequence_queue.full():
                sequence_queue.put_nowait(np.array(sequence[-30:]))

            if not result_queue.empty():
                predicted_action, confidence = result_queue.get_nowait()
                action_name = actions[predicted_action]
                current_time = time.time()
                time_since_last_prediction = current_time - last_prediction_time

                # Update current prediction display with more info
                current_prediction = f"{action_name} ({confidence:.2f})"

                prediction_history.append(action_name)
                prediction_history = prediction_history[-HISTORY_LENGTH:]

                if confidence > threshold:
                    prediction_counts = prediction_history.count(action_name)

                    if action_name == "nothing":
                        nothing_count += 1
                        last_prediction_time = current_time
                    elif (time_since_last_prediction >= min_prediction_interval and 
                        prediction_counts >= MIN_CONSISTENT_PREDICTIONS):  # Removed length check
                        nothing_count = 0
                        if not sentence or action_name != sentence[-1]:
                            sentence.append(action_name)
                            last_prediction_time = current_time
                            prediction_history.clear()  

                # Trigger synthesis on consecutive "nothing" gestures
                if nothing_count >= 2 and any(word != "nothing" for word in sentence):
                    text_out = ' '.join(sentence)
                    translator_device.synthesize_speech(text_out, translator_device.base_language)
                    shared.ui_mode = "TEXT"

                    # Reset all tracking variables
                    sentence.clear()
                    sequence.clear()
                    predictions.clear()
                    prediction_history.clear()
                    nothing_count = 0
                    
                    with open(file_path, 'w') as file:
                        pass  # Clear file contents

                    # Handle audio transcription
                    translator_device.vad_active = True
                    transcript = translator_device.listen_and_save_transcription(
                        file_path="als_speech_audio_transcription.txt")
                    translator_device.vad_active = False

                    time.sleep(3)
                    with open(file_path, 'w') as file:
                        pass
                    shared.ui_mode = "CAMERA"

            time.sleep(0.03)
        else:
            # Speech mode handling
            if cap is not None:
                cap.release()
                cap = None
            shared.ui_mode = "TEXT"
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
    with open(file_path, 'w') as file:
        pass  # clear the file contents

    app_qt = QApplication(sys.argv)
    window = MainWindow(file_path, translator_device)
    window.show()
    try:
        exit_code = app_qt.exec_()
    except KeyboardInterrupt:
        exit_code = 0
    finally:
        cleanup()
    sys.exit(exit_code)
