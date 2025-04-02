import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time

# Setup
actions = np.array(["hello", "thanks", "nothing", "help", "yes", "bathroom"])
threshold = 0.8

# Initialize model
interpreter = tf.lite.Interpreter(model_path="newest.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# MediaPipe setup for holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
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

def extract_keypoints(results):
    # Extract pose landmarks
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    # Extract left hand landmarks
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    # Extract right hand landmarks
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, lh, rh])

def predict(sequence):
    sequence = np.expand_dims(sequence, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], sequence)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

# Main loop
cap = cv2.VideoCapture(0)
sequence = []
predictions = []
last_prediction_time = 0
min_prediction_interval = 1
last_prediction = None
sentence = []
prediction_history = []  # Store recent predictions
HISTORY_LENGTH = 5  # Number of predictions to consider
MIN_CONSISTENT_PREDICTIONS = 3  # Minimum number of same predictions needed

while True:
    ret, frame = cap.read()
    if not ret:
        continue
        
    # Make detections
    image, results = mediapipe_detection(frame, holistic)
    
    # Draw landmarks
    draw_styled_landmarks(image, results)
    
    # Extract keypoints and make prediction
    keypoints = extract_keypoints(results)
    sequence.append(keypoints)
    sequence = sequence[-30:]  # Keep only last 30 frames
    

    if len(sequence) == 30:
        res = predict(np.array(sequence))
        predicted_idx = np.argmax(res)
        confidence = res[predicted_idx]
        predicted_action = actions[predicted_idx]
        
        current_time = time.time()
        time_since_last_prediction = current_time - last_prediction_time
        
        # Add current prediction to history
        prediction_history.append(predicted_action)
        prediction_history = prediction_history[-HISTORY_LENGTH:]  # Keep last N predictions
        
        if confidence > threshold:
            # Count occurrences of current prediction in history
            prediction_counts = prediction_history.count(predicted_action)
            
            # Allow "nothing" to be detected anytime
            if predicted_action == "nothing":
                prediction_text = f"Current: {predicted_action} ({confidence:.2f})"
                last_prediction = predicted_action
                last_prediction_time = current_time
            # For other gestures, check time interval and consistency
            elif (time_since_last_prediction >= min_prediction_interval and 
                prediction_counts >= MIN_CONSISTENT_PREDICTIONS):
                prediction_text = f"Current: {predicted_action} ({confidence:.2f})"
                last_prediction = predicted_action
                last_prediction_time = current_time
                if predicted_action != "nothing":
                    # Add to sentence if it's not already the last word
                    if not sentence or sentence[-1] != predicted_action:
                        sentence.append(predicted_action)
                        prediction_history.clear()  # Clear history after adding to sentence
            else:
                # Keep showing the previous prediction
                prediction_text = f"Current: {last_prediction} ({confidence:.2f})" if last_prediction else "Waiting..."
        else:
            prediction_text = "Waiting..."
            
        # Display current prediction
        cv2.putText(image, prediction_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display formed sentence
        sentence_text = f"Sentence: {' '.join(sentence)}"
        cv2.putText(image, sentence_text, (10, 70),  # Position below the prediction
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Optional: Add instructions for clearing sentence
        cv2.putText(image, "Press 'c' to clear sentence", (10, image.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Show frame
    cv2.imshow('ASL Detection', image)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        sentence.clear()  # Clear the sentence when 'c' is pressed

cap.release()
cv2.destroyAllWindows()