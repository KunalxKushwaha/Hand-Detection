# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import mediapipe as mp
# import cv2
# # import opencv as cv
# import numpy as np

# mp_drawing = mp.solutions.drawing_utils
# mp_hands = mp.solutions.hands

# capture = cv2.VideoCapture(1)

# with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence = 0.5) as hands:
#     while capture.isOpened():
#         ret, frame = capture.read()
#         if not ret:
#             print("Ignoring empty camera frame.")
#             continue

#         # Flip the image horizontally for a later selfie-view display
#         frame = cv2.flip(frame, 1)

#         # Convert the BGR image to RGB
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         detected_image = hands.process(image)
#         # Convert the RGB image back to BGR
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#         # Draw the hand annotations on the image.
#         if detected_image.multi_hand_landmarks:
#             for hand_landmarks in detected_image.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(
#                     image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
#                     landmark_drawing_spec=mp.solutions.drawing_utils.
#                     DrawingSpec(color=(255, 0, 255), thickness=4, circle_radius=2),
#                     connection_drawing_spec=mp.solutions.drawing_utils.
#                     DrawingSpec(color=(20, 180, 90), thickness=2, circle_radius=2)
#                 )
#         # Show the image
#         cv2.imshow('Hand Tracking', image)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
                
# capture.release()
# cv2.destroyAllWindows()

import os
import cv2
import mediapipe as mp

# Optional: Suppress TensorFlow/MediaPipe logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Setup MediaPipe hands and drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Open webcam (try index 0)
capture = cv2.VideoCapture(0)

# Check if webcam opens successfully
if not capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = capture.read()
        if not ret:
            print("Ignoring empty frame.")
            continue

        # Flip frame horizontally for selfie view
        frame = cv2.flip(frame, 1)

        # Convert BGR image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and find hands
        result = hands.process(image_rgb)

        # Convert back to BGR for OpenCV display
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Draw hand landmarks
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image_bgr,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=3, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )

        # Show output
        cv2.imshow("Hand Tracking", image_bgr)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Clean up
capture.release()
cv2.destroyAllWindows()
