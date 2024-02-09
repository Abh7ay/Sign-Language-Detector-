import os
import pickle
import mediapipe as mp
import cv2
import tensorflow as tf

# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

mp_hands = mp.solutions.hands

# Set up MediaPipe Hands module
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    # Full path to the current directory
    dir_path = os.path.join(DATA_DIR, dir_)

    # Skip files, only process directories
    if not os.path.isdir(dir_path):
        continue

    for img_path in os.listdir(dir_path):
        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(dir_path, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Use try-except block to handle cases where hands are not detected
        try:
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                        # Change color of the line connecting landmarks to white
                        if i < len(hand_landmarks.landmark) - 1:
                            x1, y1 = int(x * img.shape[1]), int(y * img.shape[0])
                            x2, y2 = int(hand_landmarks.landmark[i + 1].x * img.shape[1]), int(hand_landmarks.landmark[i + 1].y * img.shape[0])
                            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

                    # Normalize coordinates before appending to data_aux
                    for i in range(len(hand_landmarks.landmark)):
                        x_normalized = (hand_landmarks.landmark[i].x - min(x_))
                        y_normalized = (hand_landmarks.landmark[i].y - min(y_))
                        data_aux.extend([x_normalized, y_normalized])

                        # Change color of the landmark points to red
                        x_pixel, y_pixel = int(x * img.shape[1]), int(y * img.shape[0])
                        cv2.circle(img, (x_pixel, y_pixel), 5, (0, 0, 255), -1)

                data.append(data_aux)
                labels.append(dir_)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

# Save the data to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
