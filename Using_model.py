import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
from playsound import playsound
import threading
import time

# === Load Trained Model ===
model_path = r"eye_state_mobilenetv2_model.h5" #insert your path here
model = tf.keras.models.load_model(model_path)
IMG_SIZE = (96, 96)
import pygame

# Initialize Pygame mixer only once
pygame.mixer.init()

# Load sound
ALERT_SOUND_PATH = r"alert.mp3" #insert your path here

def play_alert():
    try:
        pygame.mixer.music.load(ALERT_SOUND_PATH)
        pygame.mixer.music.play()
    except Exception as e:
        print("Error playing alert sound:", e)

# === Sound Path ===
ALERT_SOUND_PATH = r"alert.mp3" #insert your path here


# === Initialize MTCNN ===
detector = MTCNN()


# === Eye State Vars ===
eye_closed_start_time = None
alert_triggered = False

# === Webcam Start ===
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(rgb_frame)

    eyes_closed = False

    for detection in detections:
        if detection['confidence'] < 0.9:
            continue

        keypoints = detection['keypoints']
        left_eye = keypoints['left_eye']
        right_eye = keypoints['right_eye']

        # Eye Region Size
        eye_box_size = 40  # Change if needed

        for eye in [left_eye, right_eye]:
            x, y = eye
            x1 = max(0, x - eye_box_size // 2)
            y1 = max(0, y - eye_box_size // 2)
            x2 = x1 + eye_box_size
            y2 = y1 + eye_box_size

            eye_crop = frame[y1:y2, x1:x2]
            if eye_crop.shape[0] != eye_box_size or eye_crop.shape[1] != eye_box_size:
                continue

            eye_resized = cv2.resize(eye_crop, IMG_SIZE)
            eye_input = np.expand_dims(eye_resized, axis=0).astype(np.float32)
            eye_input = tf.keras.applications.mobilenet_v2.preprocess_input(eye_input)

            prediction = model.predict(eye_input, verbose=0)[0][0]
            label = "Closed" if prediction < 0.3 else "Open"
            color = (0, 0, 255) if label == "Closed" else (0, 255, 0)

            if label == "Closed":
                eyes_closed = True

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # === Drowsiness Logic ===
    if eyes_closed:
        if eye_closed_start_time is None:
            eye_closed_start_time = time.time()
        elif time.time() - eye_closed_start_time > 0.3 and not alert_triggered:
            alert_triggered = True
            threading.Thread(target=play_alert, daemon=True).start()
    else:
        eye_closed_start_time = None
        alert_triggered = False

    # === Red Overlay Alert ===
    if alert_triggered:
        red_overlay = np.full_like(frame, (0, 0, 255), dtype=np.uint8)
        cv2.addWeighted(red_overlay, 0.3, frame, 0.7, 0, frame)

    cv2.imshow("Drowsiness Detection - Yashu Mowa 😎🔥", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
