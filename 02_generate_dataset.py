import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time

OUT = "gesture_data.csv"
CLASSES = ["FIST", "ONE", "TWO", "PALM"]
SAMPLES_PER_CLASS = 120

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def normalize_landmarks(lm):
    p = np.array([[x, y, z] for x, y, z in lm], dtype=np.float32)
    p -= p[0]                         # wrist as origin
    scale = np.max(np.linalg.norm(p[:, :2], axis=1))
    if scale < 1e-6:
        scale = 1.0
    p /= scale
    return p.flatten()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open laptop camera.")

rows = []
print("Show each gesture when prompted. Press Q to quit.")

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    for label_id, label in enumerate(CLASSES):
        print(f"\nGet ready: {label}")
        for t in range(3, 0, -1):
            print(t)
            time.sleep(1)

        count = 0
        while count < SAMPLES_PER_CLASS:
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                feats = normalize_landmarks(
                    [(p.x, p.y, p.z) for p in hand.landmark]
                )
                rows.append([label_id] + feats.tolist())
                count += 1

                mp_draw.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS
                )

            cv2.putText(
                frame, f"{label}: {count}/{SAMPLES_PER_CLASS}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2
            )
            cv2.putText(
                frame, "Keep hand visible | Q = quit",
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2
            )
            cv2.imshow("Dataset Collection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                raise SystemExit

cap.release()
cv2.destroyAllWindows()

header = ["label"] + [f"f{i}" for i in range(63)]
with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(rows)

print(f"Saved {len(rows)} samples to {OUT}")
