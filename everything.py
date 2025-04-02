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
# from flask import Flask, request, jsonify
# from flask_cors import CORS
from gpiozero import Button
from TabularUI import MainWindow
from PyQt5.QtWidgets import QApplication
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
    # Extract pose landmarks (33 landmarks * 4 values each)
    pose = np.array([[res.x, res.y, res.z, res.visibility]
                     for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    
    # Extract left hand landmarks (21 landmarks * 3 values each)
    lh = np.array([[res.x, res.y, res.z]
                   for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    
    # Extract right hand landmarks (21 landmarks * 3 values each)
    rh = np.array([[res.x, res.y, res.z]
                   for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    
    return np.concatenate([pose, lh, rh])

def draw_styled_landmarks(image, results):
    """Draw landmarks and connections for pose and hands."""
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
# If you're not using Flask, commented-out code is fine
# app = Flask(__name__)
# CORS(app)

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

# If you're not serving Flask, you can comment this out as well
# flask_thread = threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5000), daemon=True)
# flask_thread.start()

# ==================== PHYSICAL BUTTON & VOLUME SETUP ====================

def set_volume(level):
    capped_level = min(90, max(0, level))
    os.system(f"amixer -D pulse sset Master {capped_level}%")

def increase_volume(step=5):
    current = get_volume()
    new_volume = min(90, current + step)
    set_volume(new_volume)

def decrease_volume(step=5):
    current = get_volume()
    new_volume = max(0, current - step)
    set_volume(new_volume)

def get_volume():
    result = os.popen("amixer -D pulse get Master").read()
    volume = int(result.split('[')[1].split('%')[0])
    return volume

PIN_MODE = 4
PIN_UP = 17
PIN_DOWN = 27

mode = "SPEECH"  # Initial mode
cap = None

def flush_audio_stream():
    """Open & discard frames for 2 seconds to clear audio input buffers."""
    with sd.InputStream(samplerate=translator_device.SAMPLE_RATE,
                        channels=translator_device.NUM_CHANNELS, dtype='int16') as flush_stream:
        flush_end = time.time() + 2
        while time.time() < flush_end:
            try:
                _ = flush_stream.read(
                    int(translator_device.SAMPLE_RATE * (translator_device.FRAME_DURATION / 1000.0))
                )
            except Exception:
                pass

def change_mode():
    global mode, cap, sequence, sentence

    # Flush ASL buffers/queues
    while not sequence_queue.empty():
        sequence_queue.get_nowait()
    while not result_queue.empty():
        result_queue.get_nowait()
    sequence.clear()
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

sequence = []
sentence = []
threshold = 0.9

# --- If you no longer need these, you can remove them ---
min_prediction_interval = 0.5
HISTORY_LENGTH = 6
MIN_CONSISTENT_PREDICTIONS = 5
TRANSITION_FRAMES = 15
transition_counter = 0
# --------------------------------------------------------

COOLDOWN_PERIOD = 0.7  # Time (s) to wait before allowing another gesture
last_gesture_time = 0.0

# NEW variables for consecutive-frame approach:
CONSECUTIVE_THRESHOLD = 15  # e.g. 15 frames ~ 0.5s at ~30 FPS
current_action = None        # track which action we are currently verifying
current_streak = 0           # how many consecutive frames it's been seen

def asl_processing_loop():
    nothing_count = 0
    global cap, sequence, sentence
    global COOLDOWN_PERIOD, last_gesture_time
    global current_action, current_streak

    while True:
        if mode == "ASL":
            if cap is None:
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.resize(frame, (640, 480))
            
            # MediaPipe detection
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            # Display text overlays
            sentence_text = ' '.join(sentence)
            cv2.putText(image, f"Sentence: {sentence_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, f"Predicting: {current_action}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            shared.latest_frame = image.copy()
            
            # Collect keypoints
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            # Put into queue for inference
            if len(sequence) >= 30 and not sequence_queue.full():
                sequence_queue.put_nowait(np.array(sequence[-30:]))

            # Check if the inference worker has a result
            if not result_queue.empty():
                predicted_action, confidence = result_queue.get_nowait()
                action_name = actions[predicted_action]
                current_time = time.time()

                # If it's a real sign with enough confidence
                if action_name != "nothing" and confidence > threshold:
                    # 1) Check cooldown
                    if current_time - last_gesture_time > COOLDOWN_PERIOD:
                        # 2) Consecutive-frame logic
                        if current_action is None:
                            # Start tracking this action
                            current_action = action_name
                            current_streak = 1
                        else:
                            # If it's the same action as the one we're tracking, increment
                            if action_name == current_action:
                                current_streak += 1
                            else:
                                # Different action encountered, reset
                                current_action = action_name
                                current_streak = 1

                        # Once we hit the required consecutive frames
                        if current_streak >= CONSECUTIVE_THRESHOLD:
                            if not sentence or sentence[-1] != current_action:
                                print(f"Adding gesture: {current_action}")
                                sentence.append(current_action)
                                last_gesture_time = current_time
                            # Reset for next sign
                            current_action = None
                            current_streak = 0
                    else:
                        # Cooldown not finished => reset
                        current_action = None
                        current_streak = 0
                else:
                    # If it's "nothing" or below threshold => reset
                    current_action = None
                    current_streak = 0
                    nothing_count += 1

                    # If we have at least 2 "nothing" in a row and there's something in the sentence
                    if nothing_count >= 2 and any(word != "nothing" for word in sentence):
                        text_out = ' '.join(sentence)
                        translator_device.synthesize_speech(text_out, translator_device.base_language)
                        shared.ui_mode = "TEXT"

                        sentence.clear()
                        sequence.clear()
                        current_action = None
                        current_streak = 0
                        nothing_count = 0

                        with open(file_path, 'w') as file:
                            pass

                        translator_device.vad_active = True
                        transcript = translator_device.listen_and_save_transcription(
                            file_path="als_speech_audio_transcription.txt"
                        )
                        translator_device.vad_active = False

                        time.sleep(3)
                        with open(file_path, 'w') as file:
                            pass
                        shared.ui_mode = "CAMERA"
            
            time.sleep(0.03)
        else:
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
    stop_thread = True
    if cap is not None:
        cap.release()
    asl_proc_thread.join()
    asl_thread.join()
    translator_thread.join()
    # If Flask were running, you'd also join flask_thread
    print("Cleanup complete.")

# ==================== APPLICATION ENTRY POINT ====================
if __name__ == "__main__":
    file_path = "als_speech_audio_transcription.txt"
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
