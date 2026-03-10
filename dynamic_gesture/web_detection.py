import numpy as np
import time
from tensorflow.keras.models import load_model

from hand_tracking.hand_detector import HandDetector
from hand_tracking.landmark_processor import extract_landmarks


model = load_model("models/dynamic_gesture_model.h5")

GESTURES = np.array([
    'HELLO','THANKYOU','YES','NO','PLEASE',
    'STOP','HELP','SORRY','LOVE','OK','WAIT'
])

sequence = []
sequence_length = 30

detector = HandDetector()

prediction_buffer = []
buffer_size = 15

latest_word = ""

last_prediction_time = 0
prediction_interval = 0.75


def process_frame(frame):

    global sequence, prediction_buffer, latest_word, last_prediction_time

    frame, landmarks = detector.detect(frame)

    if landmarks is not None:

        for hand_landmarks in landmarks:

            data = extract_landmarks(hand_landmarks)

            sequence.append(data)

            if len(sequence) > sequence_length:
                sequence.pop(0)

            if len(sequence) == sequence_length:

                current_time = time.time()

                if current_time - last_prediction_time > prediction_interval:

                    input_data = np.expand_dims(sequence, axis=0)

                    prediction = model.predict(input_data, verbose=0)[0]

                    confidence = np.max(prediction)
                    gesture = GESTURES[np.argmax(prediction)]

                    if confidence > 0.7:

                        prediction_buffer.append(gesture)

                        if len(prediction_buffer) > buffer_size:
                            prediction_buffer.pop(0)

                        latest_word = max(
                            set(prediction_buffer),
                            key=prediction_buffer.count
                        )

                    last_prediction_time = current_time

    return frame, latest_word