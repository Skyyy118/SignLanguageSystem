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

        # Write header only if file doesn't exist
        if not file_exists:
            header = ["label"] + [f"f{i}" for i in range(63)]
            writer.writerow(header)

        writer.writerow([label] + data)


def start_camera():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    current_label = None
    data = None  # 👈 store latest features safely

    print("Press a / b / c to select gesture label")
    print("Press s to save sample")
    print("Press q to quit")

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

        # Select label (lowercase only)
        if key == ord('a'):
            current_label = "A"
            print("Selected Label: A")

        elif key == ord('b'):
            current_label = "B"
            print("Selected Label: B")

        elif key == ord('c'):
            current_label = "C"
            print("Selected Label: C")

        # Save sample safely
        elif key == ord('s'):
            if data is not None and current_label is not None:
                save_to_csv(current_label, data)
                print(f"Saved sample for {current_label}")
            else:
                print("No hand detected or label not selected")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()