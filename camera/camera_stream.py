print("RUNNING NEW CAMERA STREAM FILE")

import cv2
import csv
import os
from hand_tracking.hand_detector import HandDetector
from hand_tracking.landmark_processor import extract_landmarks


DATASET_PATH = "dataset/dataset.csv"


def save_to_csv(label, data):
    file_exists = os.path.isfile(DATASET_PATH)

    with open(DATASET_PATH, mode='a', newline='') as f:
        writer = csv.writer(f)

        if not file_exists:
            header = ["label"] + [f"f{i}" for i in range(63)]
            writer.writerow(header)

        writer.writerow([label] + data)


def start_camera():

    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    current_label = None
    data = None

    print("Press a–i, k–y to select gesture label")
    print("Press 1 to save sample")
    print("Press 2 to quit")

    cv2.namedWindow("Dataset Collection", cv2.WINDOW_NORMAL)

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame, landmarks = detector.detect(frame)

        if landmarks is not None:
            for hand_landmarks in landmarks:

                data = extract_landmarks(hand_landmarks)

                if current_label:
                    cv2.putText(
                        frame,
                        f"Label: {current_label}",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

        cv2.imshow("Dataset Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        # A–I
        if key == ord('a'):
            current_label = "A"
        elif key == ord('b'):
            current_label = "B"
        elif key == ord('c'):
            current_label = "C"
        elif key == ord('d'):
            current_label = "D"
        elif key == ord('e'):
            current_label = "E"
        elif key == ord('f'):
            current_label = "F"
        elif key == ord('g'):
            current_label = "G"
        elif key == ord('h'):
            current_label = "H"
        elif key == ord('i'):
            current_label = "I"

        # Skip J

        # K–M
        elif key == ord('k'):
            current_label = "K"
        elif key == ord('l'):
            current_label = "L"
        elif key == ord('m'):
            current_label = "M"

        # N–P
        elif key == ord('n'):
            current_label = "N"
        elif key == ord('o'):
            current_label = "O"
        elif key == ord('p'):
            current_label = "P"

        # Q–S
        elif key == ord('q'):
            current_label = "Q"
        elif key == ord('r'):
            current_label = "R"
        elif key == ord('s'):
            current_label = "S"

        # T–V
        elif key == ord('t'):
            current_label = "T"
        elif key == ord('u'):
            current_label = "U"
        elif key == ord('v'):
            current_label = "V"

        # W–Y (NEW)
        elif key == ord('w'):
            current_label = "W"
        elif key == ord('x'):
            current_label = "X"
        elif key == ord('y'):
            current_label = "Y"

        # Print selected label
        if key >= ord('a') and key <= ord('z'):
            if current_label:
                print(f"Selected Label: {current_label}")

        # Save sample
        elif key == ord('1'):

            if data is not None and current_label is not None:
                save_to_csv(current_label, data)
                print(f"Saved sample for {current_label}")
            else:
                print("No hand detected or label not selected")

        # Quit
        elif key == ord('2'):
            break

    cap.release()
    cv2.destroyAllWindows()
