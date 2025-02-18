import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time

# 1. Define your actions in the same order used during training
actions = np.array(["hello", "thanks", "iloveyou"])

# 2. Load your trained model (either .keras or .h5, etc.)
model = tf.keras.models.load_model("model.keras")

# 3. Initialize Mediapipe modules
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# 4. Helper functions
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
    # Pose: 33 landmarks * 4 values (x, y, z, visibility) = 132
    pose = np.array([[res.x, res.y, res.z, res.visibility] 
                     for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)

    # Face: 468 landmarks * 3 values (x, y, z) = 1404
    face = np.array([[res.x, res.y, res.z]
                     for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)

    # Left hand: 21 landmarks * 3 values = 63
    lh = np.array([[res.x, res.y, res.z]
                   for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)

    # Right hand: 21 landmarks * 3 values = 63
    rh = np.array([[res.x, res.y, res.z]
                   for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)

    return np.concatenate([pose, face, lh, rh])

def draw_styled_landmarks(image, results):
    """Draw only pose and hands (face landmarks commented out)."""
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
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

# 5. Variables for running inference on sequences of frames
sequence = []
predictions = []
sentence = []
threshold = 0.5

# 6. Start video capture (adjust index if needed)
cap = cv2.VideoCapture(0)


# FPS counters
frame_count = 0
start_time = time.time()

# 7. Main loop
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1

        # Run Mediapipe detection
        image, results = mediapipe_detection(frame, holistic)

        # Draw landmarks (optional)
        draw_styled_landmarks(image, results)

        # Extract keypoints and build the 30-frame sequence
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]  # keep only last 30 frames

        # Once we have 30 frames, run inference
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predicted_action = np.argmax(res)
            predictions.append(predicted_action)

            # Simple voting logic: check if the last 10 predictions agree
            if len(predictions) > 10 and np.unique(predictions[-10:])[0] == predicted_action:
                if res[predicted_action] > threshold:
                    # Only add new word if it's different from last recognized word
                    action_name = actions[predicted_action]
                    if not sentence or (action_name != sentence[-1]):
                        sentence.append(action_name)

            # Keep sentence length short
            if len(sentence) > 3:
                sentence = sentence[-3:]

            # Put recognized words on top of the image
            cv2.putText(
                image, ' '.join(sentence), (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
            )

        # Display result
        cv2.imshow("OpenCV Feed", image)

        # FPS calculation
        current_time = time.time()
        if current_time - start_time >= 1.0:
            fps = frame_count / (current_time - start_time)
            print(f"FPS: {fps:.2f}")
            frame_count = 0
            start_time = current_time

        # Press 'q' to quit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
