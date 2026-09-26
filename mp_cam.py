import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Open webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Flip image
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand
    result = hands.process(rgb)

    finger_count = 0

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Get landmark positions
            landmarks = hand_landmarks.landmark

            # Thumb
            if landmarks[4].x < landmarks[3].x:
                finger_count += 1

            # Index finger
            if landmarks[8].y < landmarks[6].y:
                finger_count += 1

            # Middle finger
            if landmarks[12].y < landmarks[10].y:
                finger_count += 1

            # Ring finger
            if landmarks[16].y < landmarks[14].y:
                finger_count += 1

            # Little finger
            if landmarks[20].y < landmarks[18].y:
                finger_count += 1

    # Display finger count
    cv2.putText(
        frame,
        "Fingers: " + str(finger_count),
        (30, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 255, 0),
        3
    )

    # Show output
    cv2.imshow("Finger Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()