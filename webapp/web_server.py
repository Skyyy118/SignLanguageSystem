import sys
import os

# allow project root imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, render_template, jsonify, Response
import cv2
import asyncio
import edge_tts
import uuid
import pygame
import time

# AI modules
from dynamic_gesture.web_detection import process_frame
from prediction.web_alphabet_detection import process_alphabet_frame


app = Flask(__name__)

# ==========================
# GLOBAL STATE
# ==========================

current_mode = "none"
latest_translation = ""

sentence = []
last_added = ""

last_add_time = 0
gesture_cooldown = 1.2

# camera
camera = cv2.VideoCapture(0)
camera.set(3,640)
camera.set(4,480)

pygame.mixer.init()


# ==========================
# TEXT TO SPEECH
# ==========================
def speak_text(text):

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


# ==========================
# HOME PAGE
# ==========================
@app.route("/")
def home():
    return render_template("index.html")


# ==========================
# CAMERA STREAM
# ==========================
def generate_frames():

    global current_mode
    global latest_translation
    global sentence
    global last_added
    global last_add_time

    while True:

        success, frame = camera.read()

        if not success:
            break

        detected = False

        # =====================
        # WORD MODE
        # =====================
        if current_mode == "words":

            frame, word = process_frame(frame)

            if word:

                # keep original for UI
                latest_translation = word
                detected = True

                # remove spaces between letters
                clean_word = word.replace(" ", "").strip()

                current_time = time.time()

                if clean_word != last_added and (current_time - last_add_time > gesture_cooldown):

                    sentence.append(clean_word)

                    last_added = clean_word
                    last_add_time = current_time

                cv2.putText(
                    frame,
                    f"Word: {word}",
                    (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2
                )


        # =====================
        # ALPHABET MODE
        # =====================
        elif current_mode == "alphabet":

            frame, letter = process_alphabet_frame(frame)

            if letter:

                latest_translation = letter
                detected = True

                current_time = time.time()

                if letter != last_added and (current_time - last_add_time > gesture_cooldown):

                    sentence.append(letter)

                    last_added = letter
                    last_add_time = current_time

                cv2.putText(
                    frame,
                    f"Letter: {letter}",
                    (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255,200,0),
                    2
                )


        if not detected and current_mode != "none":
            latest_translation = ""

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        )


# ==========================
# VIDEO STREAM
# ==========================
@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# ==========================
# TRANSLATION API
# ==========================
@app.route("/get_translation")
def get_translation():

    return jsonify({
        "text": latest_translation
    })


# ==========================
# SENTENCE API
# ==========================
@app.route("/get_sentence")
def get_sentence():

    return jsonify({
        "sentence": " ".join(sentence)
    })


@app.route("/clear_sentence")
def clear_sentence():

    global sentence
    global last_added

    sentence = []
    last_added = ""

    return jsonify({
        "message":"cleared"
    })


# ==========================
# SPEAK SENTENCE
# ==========================
@app.route("/speak")
def speak():

    if sentence:
        speak_text(" ".join(sentence))

    return jsonify({
        "message":"speaking"
    })


# ==========================
# MODE CONTROLS
# ==========================
@app.route("/alphabet")
def start_alphabet():

    global current_mode
    global sentence

    sentence = []
    current_mode = "alphabet"

    return jsonify({"message":"alphabet mode"})


@app.route("/words")
def start_words():

    global current_mode
    global sentence

    sentence = []
    current_mode = "words"

    return jsonify({"message":"word mode"})


@app.route("/stop")
def stop_all():

    global current_mode

    current_mode = "none"

    return jsonify({"message":"stopped"})


# ==========================
# RUN SERVER
# ==========================
if __name__ == "__main__":
    app.run(debug=True)