from __future__ import print_function
import os
import logging

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Suppress additional TensorFlow and related library warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import imageio
import pygame
import mediapipe as mp
import csv
import copy
import numpy as np
import re
import time
from collections import Counter

from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier

from autocorrect import correct_sentence
import io

def main():
    # Initialize Pygame
    pygame.init()
    cap_device = 0  # Default camera
    cap_width = 960
    cap_height = 540
    screen = pygame.display.set_mode((cap_width, cap_height))
    pygame.display.set_caption("Hand Gesture Recognition")

    # Load MediaPipe hands model
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    # Initialize keypoint classifier
    keypoint_classifier = KeyPointClassifier()

    # Load gesture labels (ASL A-Z)
    with io.open(
        "model/keypoint_classifier/keypoint_classifier_label.csv", encoding="utf-8-sig"
    ) as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

    detected_string = ""
    word_list = ""

    current_letter = None
    letter_start_time = None
    last_added_time = 0

    last_word_time = time.time()
    last_letter_detected_time = time.time()

    empty_string_start_time = None

    # Initialize ImageIO reader
    reader = imageio.get_reader("<video0>")  # Adjust device string if necessary

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        try:
            frame = reader.get_next_data()
        except StopIteration:
            break

        image = np.flip(frame, axis=1)  # Mirror image
        debug_image = copy.deepcopy(image)

        # Convert RGB to BGR for MediaPipe
        image_rgb = image[:, :, ::-1]
        results = hands.process(image_rgb)

        detected_letter = ""
        handedness_label = ""

        current_time = time.time()

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                detected_letter = keypoint_classifier_labels[hand_sign_id]

                handedness_label = handedness.classification[0].label

                brect = calc_bounding_rect(debug_image, hand_landmarks)
                debug_image = draw_bounding_rect(True, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)

                if detected_letter == current_letter:
                    if letter_start_time is None:
                        letter_start_time = current_time
                    elif current_time - letter_start_time >= 0.75:
                        detected_string += detected_letter
                        print("Letter Added: {}".format(detected_letter))
                        print("Detected String: {}".format(detected_string))
                        last_added_time = current_time
                        current_letter = None
                        letter_start_time = None
                else:
                    current_letter = detected_letter
                    letter_start_time = current_time

        else:
            pass

        current_time = time.time()

        if detected_letter:
            if current_letter == detected_letter:
                pass
            else:
                current_letter = detected_letter
                letter_start_time = current_time
            last_letter_detected_time = current_time
            empty_string_start_time = None
        else:
            if detected_string:
                if (current_time - last_letter_detected_time > 2):
                    word_list += detected_string + " "
                    print("New Word Added: {}".format(detected_string))
                    print("Word List: {}".format(word_list))
                    detected_string = ""
                    last_letter_detected_time = current_time
            else:
                if empty_string_start_time is None:
                    empty_string_start_time = current_time
                elif (current_time - empty_string_start_time > 3):
                    if word_list:
                        word_list = word_list.lower()
                        corrected_word_list = correct_sentence(word_list)
                        print(corrected_word_list)
                        print("Final and corrected word list: {}".format(corrected_word_list))
                    else:
                        print("No words detected.")

                    detected_string = ""
                    word_list = ""
                    empty_string_start_time = None
                    current_letter = None
                    letter_start_time = None
                    last_added_time = current_time
                    last_word_time = current_time
                    last_letter_detected_time = current_time

        if detected_string and (current_time - last_added_time > 2):
            word_list += detected_string + " "
            print("New Word Added: {}".format(detected_string))
            print("Word List: {}".format(word_list))
            detected_string = ""
            last_word_time = current_time

        if handedness_label != "" and detected_letter != "":
            debug_image = draw_info_text(debug_image, handedness_label, detected_letter)

        debug_image = cv2_to_pygame(debug_image)
        screen.blit(debug_image, (0, 0))
        pygame.display.flip()

    reader.close()
    pygame.quit()

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = temp_landmark_list[0]
    for index, landmark_point in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] = landmark_point[0] - base_x
        temp_landmark_list[index][1] = landmark_point[1] - base_y

    temp_landmark_list = list(np.ravel(temp_landmark_list))
    max_value = max(map(abs, temp_landmark_list))

    def normalize(n):
        return n / max_value if max_value != 0 else 0

    temp_landmark_list = list(map(normalize, temp_landmark_list))
    return temp_landmark_list

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def draw_landmarks(image, landmark_point):
    for index, landmark in enumerate(landmark_point):
        pygame.draw.circle(image, (0, 255, 0), tuple(landmark), 5)
    return image

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        pygame.draw.rect(image, (255, 0, 0), pygame.Rect(brect[0], brect[1], brect[2]-brect[0], brect[3]-brect[1]), 2)
    return image

def draw_info_text(image, handedness_label, hand_sign_text):
    font = pygame.font.SysFont(None, 36)
    text_surface = font.render(f"Hand: {handedness_label}, Letter: {hand_sign_text}", True, (255, 255, 255))
    image.blit(text_surface, (10, 30))
    return image

def cv2_to_pygame(image):
    return pygame.surfarray.make_surface(image.swapaxes(0,1))

if __name__ == "__main__":
    main()