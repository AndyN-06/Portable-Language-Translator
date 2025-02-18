import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time

from picamera2 import Picamera2

actions = np.array(["hello", "thanks", "iloveyou"])

# -- Load the TFLite model --
interpreter = tf.lite.Interpreter(model_path="model.tflite", num_threads=4)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def tflite_predict(sequence):
    if sequence.ndim == 2:
        sequence = np.expand_dims(sequence, axis=0)
    sequence = sequence.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], sequence)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]

# Mediapipe Holistic and Drawing
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, holistic_model):
    """Runs Mediapipe Holistic on a frame and returns (drawn_frame, results)."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = holistic_model.process(image_rgb)
    image_rgb.flags.writeable = True
    drawn_frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return drawn_frame, results

def extract_keypoints(results):
    """Extracts pose, face, left-hand, right-hand landmarks into one flattened array."""
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
    """Draw only pose and hands."""
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

sequence = []
predictions = []
sentence = []
threshold = 0.5

# ----------------------
# 1) Initialize Picamera2
# ----------------------
picam2 = Picamera2()

# Example: 640x480 with RGB888
camera_config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
picam2.configure(camera_config)
picam2.start()

frame_count = 0
start_time = time.time()

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:
    while True:
        # ----------------------
        # 2) Capture frame from Picamera2
        # ----------------------
        frame = picam2.capture_array()

        frame_count += 1
        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)

        # Extract keypoints and store them in sequence buffer
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]  # Keep the last 30 frames

        # Once we have 30 frames, run the TFLite model
        if len(sequence) == 30:
            res = tflite_predict(np.array(sequence))
            predicted_action = np.argmax(res)
            predictions.append(predicted_action)

            # Simple logic to update sentence if there's confidence
            if (len(predictions) > 10 and
                np.unique(predictions[-10:])[0] == predicted_action and
                res[predicted_action] > threshold):

                action_name = actions[predicted_action]
                if not sentence or (action_name != sentence[-1]):
                    sentence.append(action_name)

            if len(sentence) > 3:
                sentence = sentence[-3:]

            cv2.putText(
                image, ' '.join(sentence), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
            )

        # Display the resulting image
        cv2.imshow("Picamera2 Feed", image)

        # FPS calculation
        current_time = time.time()
        if current_time - start_time >= 1.0:
            fps = frame_count / (current_time - start_time)
            print(f"FPS: {fps:.2f}")
            frame_count = 0
            start_time = current_time

        # 'q' to quit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
picam2.stop()
