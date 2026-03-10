import cv2
import numpy as np
import os

from hand_tracking.hand_detector import HandDetector
from hand_tracking.landmark_processor import extract_landmarks


DATASET_PATH = "dataset_gesture"
SEQUENCE_LENGTH = 30

detector = HandDetector()
cap = cv2.VideoCapture(0)

current_label = None


gestures = {
    '1': "HELLO",
    '2': "THANKYOU",
    '3': "YES",
    '4': "NO",
    '5': "PLEASE",
    '6': "STOP",
    '7': "HELP",
    'a': "SORRY",
    'b': "LOVE",
    'c': "OK",
    'd': "WAIT"
}


print("\nSelect gesture label:")
for k,v in gestures.items():
    print(k, v)

print("\nControls:")
print("R → Record 30 frame gesture")
print("Q → Quit\n")


while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame, landmarks = detector.detect(frame)

    if current_label:
        cv2.putText(
            frame,
            f"Label: {current_label}",
            (10,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

    cv2.imshow("Gesture Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF


    # Select gesture label
    if chr(key) in gestures:
        current_label = gestures[chr(key)]
        print("Selected Label:", current_label)


    # Record sequence
    elif key == ord('r'):

        if current_label is None:
            print("Select a label first!")
            continue

        sequence = []

        print("Recording 30 frames...")

        while len(sequence) < SEQUENCE_LENGTH:

            ret, frame = cap.read()
            frame, landmarks = detector.detect(frame)

            if landmarks:
                data = extract_landmarks(landmarks[0])
                sequence.append(data)

            cv2.putText(
                frame,
                f"Recording {len(sequence)}/{SEQUENCE_LENGTH}",
                (10,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                2
            )

            cv2.imshow("Gesture Data Collection", frame)
            cv2.waitKey(1)

        sequence = np.array(sequence)

        save_dir = os.path.join(DATASET_PATH, current_label)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        sample_count = len(os.listdir(save_dir))

        filename = os.path.join(save_dir, f"sequence_{sample_count}.npy")

        np.save(filename, sequence)

        print("Saved:", filename)


    elif key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()