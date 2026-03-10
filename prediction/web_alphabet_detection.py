import joblib
import numpy as np
from hand_tracking.hand_detector import HandDetector
from hand_tracking.landmark_processor import extract_landmarks

model = joblib.load("models/sign_model.pkl")

detector = HandDetector()

latest_letter = ""

def process_alphabet_frame(frame):

    global latest_letter

    frame, landmarks = detector.detect(frame)

    if landmarks is not None:

        for hand_landmarks in landmarks:

            data = extract_landmarks(hand_landmarks)

            if data is None:
                continue

            prediction = model.predict([data])[0]

            latest_letter = prediction

    return frame, latest_letter