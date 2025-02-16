import cv2
import mediapipe as mp
import numpy as np
import hailo

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Load Hailo Model
hef = hailo.load_model("my_model.hef")

# Define actions
actions = np.array(["hello", "thanks", "iloveyou"])

# Helper functions
def mediapipe_detection(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    drawn_frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return drawn_frame, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] 
                     for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face = np.array([[res.x, res.y, res.z] 
                     for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    lh = np.array([[res.x, res.y, res.z] 
                   for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] 
                   for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, face, lh, rh])

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

# Initialize variables
sequence, predictions, sentence = [], [], []
threshold = 0.5

# Initialize video capture
cap = cv2.VideoCapture(0)

# Run MediaPipe with Hailo inference
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        image, results = mediapipe_detection(frame, holistic)
        draw_landmarks(image, results)

        # Extract keypoints and maintain a sequence of 30 frames
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        # Run inference when 30 frames are collected
        if len(sequence) == 30:
            # Run inference on the Hailo model
            input_data = np.expand_dims(sequence, axis=0)
            prediction = hef.predict(input_data)[0]

            predicted_action = np.argmax(prediction)
            predictions.append(predicted_action)

            # Voting mechanism
            if len(predictions) > 10 and np.unique(predictions[-10:])[0] == predicted_action:
                if prediction[predicted_action] > threshold:
                    action_name = actions[predicted_action]
                    if not sentence or action_name != sentence[-1]:
                        sentence.append(action_name)

            # Display recognized sentence
            if len(sentence) > 3:
                sentence = sentence[-3:]

            cv2.putText(image, ' '.join(sentence), (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the video feed
        cv2.imshow("ASL Recognition", image)

        # Quit when 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
