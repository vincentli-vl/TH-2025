from flask import Flask, render_template, request, jsonify, Response
import speech_recognition as sr
import json
import os
from datetime import datetime
import pyttsx3
import threading
import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
import numpy as np
import time
from shutil import move

CAM0 = 0
CAM1 = 1
app = Flask(__name__)

# =================== Setup ===================

recognizer = sr.Recognizer()
conversation_history = []
conversations_folder = "conversations"

if not os.path.exists(conversations_folder):
    os.makedirs(conversations_folder)

# Text-to-Speech
try:
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 150)
    tts_engine.setProperty('volume', 0.9)
    tts_available = True
except Exception as e:
    print(f"Text-to-speech not available: {e}")
    tts_available = False

# =================== ROUTES ===================

cam0 = cv2.VideoCapture(CAM0 + cv2.CAP_DSHOW)
cam1 = cv2.VideoCapture(CAM1 + cv2.CAP_DSHOW)

def scan(cam):
    while True:
        r, f = cam.read()
        if not r:
            continue

        try:
            analysis = DeepFace.analyze(f, actions=['emotion'], enforce_detection=False)
            if analysis and isinstance(analysis, list):
                face = analysis[0]
                region = face.get('region', {})
                x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
                frame_height, frame_width, _ = f.shape
                is_valid_face = w > 0 and h > 0 and w < frame_width - 1 and h < frame_height - 1
                if is_valid_face:
                    cv2.rectangle(f, (x, y), (x + w, y + h), (255, 255, 255), 1)
        except Exception as e:
            print(f"Face scan error: {e}")

        ret, buffer = cv2.imencode('.jpg', f)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n') 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/c0')
def c0():
    return Response(scan(cam0),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/c1')
def c1():
    return Response(scan(cam0),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/voice')
def voice():
    return render_template('voice.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/profile')
def profile():
    return render_template('profile.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/journal')
def journal():
    return render_template('journal.html')

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text data received'}), 400

        text = data['text']
        conversation_history.append({'timestamp': datetime.now().isoformat(), 'user_input': text, 'type': 'user'})
        response = generate_response(text)
        conversation_history.append({'timestamp': datetime.now().isoformat(), 'response': response, 'type': 'system'})
        save_conversation_to_file()

        if tts_available:
            threading.Thread(target=play_audio_response, args=(response,), daemon=True).start()

        return jsonify({'transcription': text, 'response': response, 'success': True})

    except Exception as e:
        return jsonify({'error': f'Error processing transcription: {e}'}), 500

@app.route('/get_history', methods=['GET'])
def get_history():
    return jsonify(conversation_history)

# =================== CAMERA LOOP ===================



# def loop():
#     while True:
#         scan("./static/output0", cam0)
#         scan("./static/output1", cam1)

# =================== EMOTION TRACKING ADDITION ===================

latest_emotions = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}
emoji_dir = "emojis"
emoji_map = {}
emoji_size = (80, 80)
emotion_tracker_active = False

for emo in latest_emotions.keys():
    path = os.path.join(emoji_dir, f"{emo}.png")
    if os.path.exists(path):
        emoji_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if emoji_img is not None:
            emoji_img = cv2.resize(emoji_img, emoji_size)
            emoji_map[emo] = emoji_img

CYAN = (255, 255, 0)

def overlay_png_alpha(img, png_img, x, y):
    h, w = png_img.shape[:2]
    if y + h > img.shape[0] or x + w > img.shape[1] or x < 0 or y < 0:
        return img
    png_rgb = png_img[:, :, :3]
    alpha = png_img[:, :, 3] / 255.0
    roi = img[y:y+h, x:x+w]
    for c in range(3):
        roi[:, :, c] = roi[:, :, c] * (1 - alpha) + png_rgb[:, :, c] * alpha
    img[y:y+h, x:x+w] = roi
    return img

def emotion_plotter():
    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(latest_emotions.keys(), latest_emotions.values(), color='skyblue')
    ax.set_ylim(0, 100)
    ax.set_ylabel("Confidence (%)")
    ax.set_title("Real-Time Emotion Confidence")
    plt.show(block=False)
    while emotion_tracker_active:
        for bar, key in zip(bars, latest_emotions.keys()):
            bar.set_height(latest_emotions[key])
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.1)
    plt.ioff()
    plt.close()

def emotion_tracker():
    global emotion_tracker_active
    cap = cam0
    if not cap.isOpened():
        print("Webcam not found.")
        return
    plot_thread = threading.Thread(target=emotion_plotter)
    plot_thread.start()
    cv2.namedWindow("Facial Tracker", cv2.WINDOW_NORMAL)
    while emotion_tracker_active:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (960, 720))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        analysis = DeepFace.analyze(frame_rgb, actions=['emotion'], enforce_detection=False)
        if analysis and isinstance(analysis, list):
            face = analysis[0]
            region = face.get('region', {})
            x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
            if w > 0 and h > 0:
                latest_emotions.update(face.get('emotion', latest_emotions))
                dominant_emotion = face.get('dominant_emotion', 'neutral').lower()
                conf = latest_emotions.get(dominant_emotion, 0)
                percent_text = f"Acc: {conf:.1f}%"
                emoji_x, emoji_y = x, max(y - emoji_size[1] - 20, 0)
                text_y = emoji_y + emoji_size[1] + 5
                if dominant_emotion in emoji_map:
                    frame = overlay_png_alpha(frame, emoji_map[dominant_emotion], emoji_x, emoji_y)
                cv2.putText(frame, dominant_emotion.capitalize(), (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, CYAN, 2)
                cv2.putText(frame, percent_text, (x + w - 100, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, CYAN, 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), CYAN, 2)
        cv2.imshow("Facial Tracker", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            emotion_tracker_active = False
            break
    cap.release()
    cv2.destroyAllWindows()
    plot_thread.join()

@app.route('/start_emotion_tracker')
def start_emotion_tracker():
    global emotion_tracker_active
    if not emotion_tracker_active:
        emotion_tracker_active = True
        threading.Thread(target=emotion_tracker, daemon=True).start()
    return "Emotion Tracker Started"

@app.route('/stop_emotion_tracker')
def stop_emotion_tracker():
    global emotion_tracker_active
    emotion_tracker_active = False
    return "Emotion Tracker Stopped"

# =================== UTILITIES ===================

def generate_response(user_input):
    lower = user_input.lower()
    if 'hello' in lower or 'hi' in lower:
        return "Hello! How can I help you today?"
    elif 'how are you' in lower:
        return "I'm doing well, thank you for asking!"
    elif 'bye' in lower:
        return "Goodbye! Have a great day!"
    elif 'weather' in lower:
        return "I'm sorry, I don't have access to weather information right now."
    elif 'time' in lower:
        return f"The current time is {datetime.now().strftime('%H:%M')}."
    else:
        return f"I heard you say: '{user_input}'. That's interesting!"

def play_audio_response(text):
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception as e:
        print(f"Error playing audio: {e}")

def save_conversation_to_file():
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(conversations_folder, f"conversation_{timestamp}.txt"), 'w', encoding='utf-8') as f:
            f.write(f"Conversation Session - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" + "="*50 + "\n\n")
            for entry in conversation_history:
                if entry['type'] == 'user':
                    f.write(f"User ({entry['timestamp']}): {entry['user_input']}\n")
                else:
                    f.write(f"Assistant ({entry['timestamp']}): {entry['response']}\n")
                f.write("\n")
    except Exception as e:
        print(f"Error saving conversation: {e}")

# =================== MAIN ===================

if __name__ == '__main__':
    print("Starting Flask app...")
    # a = threading.Thread(target=loop, daemon=True)
    # a.start()
    app.run(debug=False)