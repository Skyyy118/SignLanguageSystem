import cv2
import joblib
from collections import deque, Counter
from hand_tracking.hand_detector import HandDetector
from hand_tracking.landmark_processor import extract_landmarks


# Load model
model = joblib.load("models/sign_model.pkl")

detector = HandDetector()
cap = cv2.VideoCapture(0)

# Store last 10 predictions
prediction_buffer = deque(maxlen=10)

print("Live Prediction Started")
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame, landmarks = detector.detect(frame)

    if landmarks is not None:
        for hand_landmarks in landmarks:
            data = extract_landmarks(hand_landmarks)

            prediction = model.predict([data])[0]

            # Add to buffer
            prediction_buffer.append(prediction)

            # Majority vote
            if len(prediction_buffer) > 0:
                most_common = Counter(prediction_buffer).most_common(1)[0][0]

                cv2.putText(
                    frame,
                    f"Prediction: {most_common}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 0),
                    3
                )

    cv2.imshow("Live Sign Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
