import cv2 as cv
import mediapipe as mp
import csv
import copy
import numpy as np
import re
import time  # Import time for controlling letter addition frequency and word separation
from collections import Counter  # Import for counting occurrences of letters

from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier

from spellchecker import SpellChecker

from autotest import load_model, fix_text

def correct_text(text):
    spell = SpellChecker()
    
    # Split into words
    words = text.split()
    
    # Find misspelled words
    misspelled = spell.unknown(words)
    
    # Replace misspelled words with corrections
    corrected_words = []
    for word in words:
        if word in misspelled:
            corrected_words.append(spell.correction(word))
        else:
            corrected_words.append(word)
    
    # Join back into text
    return ' '.join(corrected_words)

def main():
    tokenizer, model, device = load_model()
    
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
    word_list = ""        # To store the list of words

    # Variables for letter addition logic
    current_letter = None
    letter_start_time = None
    last_added_time = 0

    # Variables for word separation logic
    last_word_time = time.time()
    last_letter_detected_time = time.time()

    # New variable to track when detected_string becomes empty
    empty_string_start_time = None

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

        current_time = time.time()

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

                # Letter Stability Logic
                if detected_letter == current_letter:
                    # Same letter detected again
                    if letter_start_time is None:
                        # First time detecting this letter
                        letter_start_time = current_time
                    elif current_time - letter_start_time >= 0.75:
                        # Letter has been stable for at least 0.75 seconds
                        detected_string += detected_letter
                        print(f"Letter Added: {detected_letter}")
                        print(f"Detected String: {detected_string}")
                        last_added_time = current_time
                        # Reset for the next letter detection
                        current_letter = None
                        letter_start_time = None
                else:
                    # Different letter detected
                    current_letter = detected_letter
                    letter_start_time = current_time

        else:
            # No hand detected; optional: handle if needed
            pass

        # After processing the current frame and detecting letters
        current_time = time.time()

        if detected_letter:
            # Letter detected
            if current_letter == detected_letter:
                # Same letter detected, continue counting
                pass
            else:
                # Different letter detected
                current_letter = detected_letter
                letter_start_time = current_time
            # Reset the timer since a letter is being detected
            last_letter_detected_time = current_time

            # Since a letter is detected, reset the empty string timer
            empty_string_start_time = None
        else:
            # No letter detected
            if detected_string:
                # If there is a detected string, handle word separation
                if (current_time - last_letter_detected_time > 2):
                    # Append the detected string as a new word to the word list
                    word_list += detected_string + " "
                    print(f"New Word Added: {detected_string}")
                    print(f"Word List: {word_list}")
                    detected_string = ""  # Clear the detected string
                    last_letter_detected_time = current_time  # Reset the timer
            else:
                # detected_string is already empty
                if empty_string_start_time is None:
                    # Start the timer since detected_string is empty
                    empty_string_start_time = current_time
                elif (current_time - empty_string_start_time > 3):
                    # 3 seconds have passed with detected_string empty
                    if word_list:
                        word_list = word_list.lower()
                        
                        corrected_word_list = correct_text(word_list)
                        print(corrected_word_list)
                        print(f"Final Word List: {corrected_word_list}")
                    else:
                        print("No words detected.")

                    # Reset everything
                    detected_string = ""
                    word_list = ""
                    empty_string_start_time = None
                    current_letter = None
                    letter_start_time = None
                    last_added_time = current_time
                    last_word_time = current_time
                    last_letter_detected_time = current_time

        # Check if 5 seconds have passed since the last letter was added
        if detected_string and (current_time - last_added_time > 2):
            # Append the detected string as a new word to the word list
            word_list += detected_string + " "
            print(f"New Word Added: {detected_string}")
            print(f"Word List: {word_list}")
            detected_string = ""  # Clear the detected string
            last_word_time = current_time

        # Display detected hand and letter on the screen
        if handedness_label != "" and detected_letter != "":
            debug_image = draw_info_text(debug_image, handedness_label, detected_letter)

        # Optionally display the current detected string and word list
        cv.putText(
            debug_image,
            f"Detected String: {detected_string}",
            (10, 70),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )

        cv.putText(
            debug_image,
            f"Word List: {word_list}",
            (10, 110),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )

        # Show the camera feed with real-time detection
        cv.imshow("Hand Gesture Recognition", debug_image)

        # Exit on pressing ESC
        key = cv.waitKey(10)
        if key == 27:  # ESC key
            break

    cap.release()
    cv.destroyAllWindows()

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
        return n / max_value if max_value != 0 else 0

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
