import cv2
import numpy as np
import time
import threading
import asyncio
import edge_tts
import pygame
import os
import uuid

from tensorflow.keras.models import load_model
from hand_tracking.hand_detector import HandDetector
from hand_tracking.landmark_processor import extract_landmarks


pygame.mixer.init()

model = load_model("models/dynamic_gesture_model.h5")

GESTURES = np.array([
    'HELLO',
    'THANKYOU',
    'YES',
    'NO',
    'PLEASE',
    'STOP',
    'HELP',
    'SORRY',
    'LOVE',
    'OK',
    'WAIT'
])

sequence = []
sequence_length = 30

detector = HandDetector()

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

sentence = []
latest_word = ""

# timing
prediction_interval = 0.75
gesture_cooldown = 1.2

last_prediction_time = 0
last_gesture_time = 0

# smoothing buffer
prediction_buffer = []
buffer_size = 15


def speak(text):

    filename = f"speech_{uuid.uuid4().hex}.mp3"

    async def generate():
        communicate = edge_tts.Communicate(
            text,
            voice="en-IN-NeerjaNeural"
        )
        await communicate.save(filename)

    asyncio.run(generate())

    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

    try:
        os.remove(filename)
    except:
        pass


while True:

    ret, frame = cap.read()
    if not ret:
        break

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

                    if confidence > 0.70:

                        prediction_buffer.append(gesture)

                        if len(prediction_buffer) > buffer_size:
                            prediction_buffer.pop(0)

                        stable_gesture = max(
                            set(prediction_buffer),
                            key=prediction_buffer.count
                        )

                        latest_word = stable_gesture

                        if (
                            (len(sentence) == 0 or sentence[-1] != stable_gesture)
                            and (current_time - last_gesture_time > gesture_cooldown)
                        ):

                            sentence.append(stable_gesture)

                            last_gesture_time = current_time

                            prediction_buffer.clear()

                    last_prediction_time = current_time


    height, width, _ = frame.shape

    dashboard = np.zeros((height, width + 300, 3), dtype=np.uint8)

    dashboard[:, :width] = frame
    dashboard[:, width:] = (30,30,30)

    cv2.putText(
        dashboard,
        "SIGN LANGUAGE AI",
        (width + 20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,255),
        2
    )

    cv2.putText(
        dashboard,
        "Detected Word:",
        (width + 20,120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.putText(
        dashboard,
        latest_word,
        (width + 20,160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0,255,0),
        2
    )

    cv2.putText(
        dashboard,
        "Sentence:",
        (width + 20,230),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.putText(
        dashboard,
        " ".join(sentence),
        (width + 20,270),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )

    cv2.putText(
        dashboard,
        "Controls",
        (width + 20,350),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.putText(
        dashboard,
        "S : Speak",
        (width + 20,390),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,255),
        2
    )

    cv2.putText(
        dashboard,
        "C : Clear",
        (width + 20,430),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,255),
        2
    )

    cv2.putText(
        dashboard,
        "Q : Quit",
        (width + 20,470),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,255),
        2
    )


    cv2.imshow("Sign Language to text Translator", dashboard)

    key = cv2.waitKey(1) & 0xFF


    if key == ord('q'):
        break

    elif key == ord('s'):

        if len(sentence) > 0:

            speech = " ".join(sentence)

            print("Speaking:", speech)

            threading.Thread(
                target=speak,
                args=(speech,)
            ).start()

    elif key == ord('c'):

        sentence = []
        prediction_buffer.clear()

        print("Sentence cleared")


cap.release()
cv2.destroyAllWindows()