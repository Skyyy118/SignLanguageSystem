import os
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint


# -----------------------------
# Dataset settings
# -----------------------------

DATA_PATH = "dataset_gesture"

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

sequence_length = 30


# -----------------------------
# Load dataset
# -----------------------------

print("Loading dataset...\n")

sequences = []
labels = []

for gesture in GESTURES:

    gesture_path = os.path.join(DATA_PATH, gesture)

    if not os.path.exists(gesture_path):
        print("Missing folder:", gesture)
        continue

    for file in os.listdir(gesture_path):

        if not file.endswith(".npy"):
            continue

        file_path = os.path.join(gesture_path, file)

        sequence = np.load(file_path)

        # verify correct shape
        if sequence.shape != (30,63):
            print("Skipping invalid file:", file_path)
            continue

        sequences.append(sequence)

        labels.append(np.where(GESTURES == gesture)[0][0])


X = np.array(sequences)
y = to_categorical(labels).astype(int)

print("Dataset loaded")
print("Total samples:", len(X))
print("Input shape:", X.shape)


# -----------------------------
# Train test split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=True,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# -----------------------------
# TensorBoard log
# -----------------------------

log_dir = os.path.join("logs")
tb_callback = TensorBoard(log_dir=log_dir)


# -----------------------------
# Early stopping
# -----------------------------

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)


# -----------------------------
# Save best model automatically
# -----------------------------

checkpoint = ModelCheckpoint(
    "models/best_dynamic_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)


# -----------------------------
# Build LSTM Model
# -----------------------------

print("\nBuilding model...\n")

model = Sequential()

model.add(LSTM(64, return_sequences=True, input_shape=(30,63)))
model.add(Dropout(0.2))

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(64))

model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

model.add(Dense(GESTURES.shape[0], activation='softmax'))


model.compile(
    optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# -----------------------------
# Train model
# -----------------------------

print("\nTraining started...\n")

model.fit(
    X_train,
    y_train,
    epochs=80,
    validation_data=(X_test, y_test),
    callbacks=[tb_callback, early_stop, checkpoint]
)


# -----------------------------
# Save final model
# -----------------------------

model.save("models/dynamic_gesture_model.h5")

print("\nTraining completed")
print("Model saved successfully")