import cv2 as cv
import mediapipe as mp
import csv
import copy
import numpy as np
import re
import time  # Import time for controlling letter addition frequency
from autocorrect import Speller  # Import autocorrect
from collections import Counter  # Import for counting occurrences of letters

from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier

# Initialize the autocorrect spell checker
spell = Speller()

def main():
    # Camera preparation ###############################################################
    cap_device = 0  # Default camera
    cap_width = 960
    cap_height = 540

    # Initialize camera
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Load MediaPipe hands model #######################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    # Initialize keypoint classifier ###################################################
    keypoint_classifier = KeyPointClassifier()

    # Load gesture labels (ASL A-Z) ####################################################
    with open(
        "model/keypoint_classifier/keypoint_classifier_label.csv", encoding="utf-8-sig"
    ) as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

    detected_string = ""  # To store the detected letters
    last_added_time = time.time()  # Track the last time a letter was added

    while True:
        # Capture frame from the camera
        ret, image = cap.read()
        if not ret:
            break

        image = cv.flip(image, 1)  # Mirror display for easier interaction
        debug_image = copy.deepcopy(image)

        # Convert the image color format from BGR to RGB for MediaPipe processing
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Variables to hold detected hand and letter information
        detected_letter = ""
        handedness_label = ""

        # Check if hand landmarks are detected
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Get hand landmarks and process for classification
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # Classify hand gesture (ASL letter)
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                detected_letter = keypoint_classifier_labels[hand_sign_id]

                # Determine if it is the right or left hand
                handedness_label = handedness.classification[0].label  # 'Right' or 'Left'

                # Draw landmarks and bounding box on the screen
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                debug_image = draw_bounding_rect(True, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)

                # Add the detected letter to the string if it's repeated no more than twice in a row
                # and only if 1 second has passed since the last letter was added
                current_time = time.time()
                if current_time - last_added_time > 1:  # Check if 1 second has passed
                    if len(detected_string) < 2 or detected_string[-2:] != detected_letter * 2:
                        detected_string += detected_letter
                        last_added_time = current_time  # Update the last added time

        # Display detected hand and letter on the screen
        if handedness_label != "" and detected_letter != "":
            debug_image = draw_info_text(debug_image, handedness_label, detected_letter)

        # Show the camera feed with real-time detection
        cv.imshow("Hand Gesture Recognition", debug_image)

        # Exit on pressing ESC
        key = cv.waitKey(10)
        if key == 27:  # ESC key
            break

        # Send the string to the spell checker and correct repetitions when 'd' is pressed
        if key == ord('d'):
            if detected_string:
                # Remove letters that only appear once
                filtered_string = remove_single_occurrences(detected_string)

                # Send the filtered string to the autocorrect function
                corrected_sentence = decipher_with_autocorrect(filtered_string)
                
                # Print both the unprocessed string and the processed string
                print(f"Unprocessed String: {detected_string}")
                print(f"Filtered String (before autocorrect): {filtered_string}")
                print(f"Processed String: {corrected_sentence}")
                
                # Clear the string after processing
                detected_string = ""

    cap.release()
    cv.destroyAllWindows()


# Function to remove letters that appear only once
def remove_single_occurrences(text):
    # Count occurrences of each letter
    letter_count = Counter(text)

    # Create a new string, including only letters that appear more than once
    filtered_text = ''.join([char for char in text if letter_count[char] > 1])

    return filtered_text


# Reduce repetitions of characters and correct the sentence with autocorrect
def reduce_repetitions(text, max_repeats=2):
    # Use regex to reduce repetitions of characters to a maximum of `max_repeats`
    return re.sub(r'(.)\1+', lambda m: m.group(1) * max_repeats, text)

def decipher_with_autocorrect(text):
    # Reduce repetitions to make it easier to decipher
    reduced_text = reduce_repetitions(text)
    
    # Correct the sentence using the autocorrect library
    corrected_sentence = spell(reduced_text)
    
    return corrected_sentence


# Calculate the hand landmarks' coordinates from MediaPipe results
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


# Pre-process the landmarks by converting them to relative and normalized coordinates
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = temp_landmark_list[0]
    for index, landmark_point in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] = landmark_point[0] - base_x
        temp_landmark_list[index][1] = landmark_point[1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(np.ravel(temp_landmark_list))

    # Normalize the values
    max_value = max(map(abs, temp_landmark_list))

    def normalize(n):
        return n / max_value

    temp_landmark_list = list(map(normalize, temp_landmark_list))

    return temp_landmark_list


# Draw bounding box around the hand
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]


# Draw landmarks on the hand
def draw_landmarks(image, landmark_point):
    for index, landmark in enumerate(landmark_point):
        cv.circle(image, (landmark[0], landmark[1]), 5, (0, 255, 0), -1)
    return image


# Draw bounding box around the hand
def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 255), 2)
    return image


# Draw the detected hand (right/left) and the detected letter on the screen
def draw_info_text(image, handedness_label, hand_sign_text):
    cv.putText(
        image,
        f"Hand: {handedness_label}, Letter: {hand_sign_text}",
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )
    return image


if __name__ == "__main__":
    main()


# import cv2 as cv
# import mediapipe as mp
# import csv
# import copy
# import numpy as np

# from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier


# def main():
#     # Camera preparation ###############################################################
#     cap_device = 0  # Default camera
#     cap_width = 960
#     cap_height = 540

#     # Initialize camera
#     cap = cv.VideoCapture(cap_device)
#     cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
#     cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

#     # Model load #############################################################
#     mp_hands = mp.solutions.hands
#     hands = mp_hands.Hands(
#         static_image_mode=False,
#         max_num_hands=1,
#         min_detection_confidence=0.7,
#         min_tracking_confidence=0.5,
#     )

#     # Initialize keypoint classifier
#     keypoint_classifier = KeyPointClassifier()

#     # Load gesture labels (ASL A-Z) ####################################################
#     with open(
#         "model/keypoint_classifier/keypoint_classifier_label.csv", encoding="utf-8-sig"
#     ) as f:
#         keypoint_classifier_labels = csv.reader(f)
#         keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

#     while True:
#         # Capture frame from the camera
#         ret, image = cap.read()
#         if not ret:
#             break

#         image = cv.flip(image, 1)  # Mirror display
#         debug_image = copy.deepcopy(image)

#         # Convert the image color format from BGR to RGB
#         image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
#         results = hands.process(image_rgb)

#         # Check if hand landmarks are detected
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 # Get hand landmarks and process for classification
#                 landmark_list = calc_landmark_list(debug_image, hand_landmarks)
#                 pre_processed_landmark_list = pre_process_landmark(landmark_list)

#                 # Classify hand gesture (ASL letter)
#                 hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
#                 detected_letter = keypoint_classifier_labels[hand_sign_id]

#                 # Print the detected letter
#                 print(f"Detected letter: {detected_letter}")

#         # Exit on pressing ESC
#         key = cv.waitKey(10)
#         if key == 27:  # ESC key
#             break

#     cap.release()
#     cv.destroyAllWindows()


# # Calculate the hand landmarks' coordinates from MediaPipe results
# def calc_landmark_list(image, landmarks):
#     image_width, image_height = image.shape[1], image.shape[0]
#     landmark_point = []

#     for _, landmark in enumerate(landmarks.landmark):
#         landmark_x = min(int(landmark.x * image_width), image_width - 1)
#         landmark_y = min(int(landmark.y * image_height), image_height - 1)
#         landmark_point.append([landmark_x, landmark_y])

#     return landmark_point


# # Pre-process the landmarks by converting them to relative and normalized coordinates
# def pre_process_landmark(landmark_list):
#     temp_landmark_list = copy.deepcopy(landmark_list)

#     # Convert to relative coordinates
#     base_x, base_y = temp_landmark_list[0]
#     for index, landmark_point in enumerate(temp_landmark_list):
#         temp_landmark_list[index][0] = landmark_point[0] - base_x
#         temp_landmark_list[index][1] = landmark_point[1] - base_y

#     # Convert to a one-dimensional list
#     temp_landmark_list = list(np.ravel(temp_landmark_list))

#     # Normalize the values
#     max_value = max(map(abs, temp_landmark_list))

#     def normalize(n):
#         return n / max_value

#     temp_landmark_list = list(map(normalize, temp_landmark_list))

#     return temp_landmark_list


# if __name__ == "__main__":
#     main()
