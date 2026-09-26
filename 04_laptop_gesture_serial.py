import cv2
import mediapipe as mp
import numpy as np
import serial
import time

SERIAL_PORT = "COM14"     # CHANGE THIS
BAUD = 115200

GESTURES = ["FIST", "ONE", "TWO", "PALM"]

ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0.05)
time.sleep(2)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def normalize_landmarks(lm):
    p = np.array([[x, y, z] for x, y, z in lm], dtype=np.float32)
    p -= p[0]
    scale = np.max(np.linalg.norm(p[:, :2], axis=1))
    if scale < 1e-6:
        scale = 1.0
    p /= scale
    return p.flatten()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera.")

last_send = None
last_time = 0

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        status = "NO HAND"

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            feats = normalize_landmarks(
                [(p.x, p.y, p.z) for p in hand.landmark]
            )

            # Send only when a hand is present.
            # Comma-separated 63 floats + newline.
            msg = ",".join(f"{v:.5f}" for v in feats) + "\n"

            # Limit traffic to about 15 messages/sec.
            now = time.time()
            if now - last_time > 0.065:
                ser.write(msg.encode("ascii"))
                last_time = now

            mp_draw.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS
            )
            status = "FEATURES -> ESP32 TinyML"

        cv2.putText(
            frame, status, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2
        )
        cv2.putText(
            frame, "Q = quit", (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2
        )

        cv2.imshow("Laptop OpenCV -> ESP32 TinyML", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
ser.close()
cv2.destroyAllWindows()
