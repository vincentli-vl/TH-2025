from flask import Flask, render_template, request, jsonify, Response
import speech_recognition as sr
import json
import os
from datetime import datetime
import pyttsx3
import threading
import cv2
from deepface import DeepFace
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CAM0 = 0
CAM1 = 1
app = Flask(__name__)

# Setup
recognizer = sr.Recognizer()
conversation_history = []
conversations_folder = "conversations"
if not os.path.exists(conversations_folder):
    os.makedirs(conversations_folder)

try:
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 150)
    tts_engine.setProperty('volume', 0.9)
    tts_available = True
except Exception as e:
    print(f"Text-to-speech not available: {e}")
    tts_available = False

cam0 = cv2.VideoCapture(CAM0 + cv2.CAP_DSHOW)
cam1 = cv2.VideoCapture(CAM1 + cv2.CAP_DSHOW)

# Reference embeddings
reference_embeddings = {
    "Sayeed": [],
    "Vivian": []
}
for img_name in ["sayeed.jpg", "sayeed2.jpg"]:
    embedding = DeepFace.represent(img_path=img_name, model_name="Facenet", enforce_detection=False)[0]["embedding"]
    reference_embeddings["Sayeed"].append(embedding)

for img_name in ["vivian.jpg", "vivian2.jpg"]:
    embedding = DeepFace.represent(img_path=img_name, model_name="Facenet", enforce_detection=False)[0]["embedding"]
    reference_embeddings["Vivian"].append(embedding)

# Emotion data
latest_emotions = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}
emoji_dir = "emojis"
emoji_map = {}
emoji_size = (80, 80)
CYAN = (255, 255, 0)
stop_plot_thread = False

for emo in latest_emotions.keys():
    path = os.path.join(emoji_dir, f"{emo}.png")
    if os.path.exists(path):
        emoji_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if emoji_img is not None:
            emoji_img = cv2.resize(emoji_img, emoji_size)
            emoji_map[emo] = emoji_img

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

def recognize_identity(face_embedding):
    for name, embeddings in reference_embeddings.items():
        for ref_emb in embeddings:
            dist = np.linalg.norm(np.array(ref_emb) - np.array(face_embedding))
            if dist < 10:  # threshold
                return name
    return None

def scan(cam):
    frame_count = 0
    analysis_results = []
    scale_factor = 0.5
    while True:
        r, f = cam.read()
        if not r:
            continue

        frame_count += 1
        if frame_count % 10 == 0:
            try:
                small_f = cv2.resize(f, (0, 0), fx=scale_factor, fy=scale_factor)
                analysis = DeepFace.analyze(
                    small_f,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend="opencv"
                )
                if isinstance(analysis, dict):
                    analysis_results = [analysis]
                elif isinstance(analysis, list):
                    analysis_results = analysis
            except Exception as e:
                print(f"Face scan error: {e}")

        for face in analysis_results:
            region = face.get('region', {})
            x = int(region.get('x', 0) / scale_factor)
            y = int(region.get('y', 0) / scale_factor)
            w = int(region.get('w', 0) / scale_factor)
            h = int(region.get('h', 0) / scale_factor)
            if w > 0 and h > 0:
                latest_emotions.update(face.get('emotion', latest_emotions))
                dominant_emotion = face.get('dominant_emotion', 'neutral').lower()
                conf = latest_emotions.get(dominant_emotion, 0)
                percent_text = f"Acc: {conf:.1f}%"
                emoji_x, emoji_y = x, max(y - emoji_size[1] - 20, 0)
                text_y = emoji_y + emoji_size[1] + 5
                if dominant_emotion in emoji_map:
                    f = overlay_png_alpha(f, emoji_map[dominant_emotion], emoji_x, emoji_y)
                cv2.putText(f, dominant_emotion.capitalize(), (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, CYAN, 2)
                cv2.putText(f, percent_text, (x + w - 100, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, CYAN, 2)
                cv2.rectangle(f, (x, y), (x + w, y + h), CYAN, 2)

        ret, buffer = cv2.imencode('.jpg', f)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/c0')
def c0():
    return Response(scan(cam0),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/c1')
def c1():
    return Response(scan(cam1),
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

@app.route('/chart')
def chart():
    import io
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    fig, ax = plt.subplots(figsize=(3, 3))
    values = list(latest_emotions.values())
    labels = list(latest_emotions.keys())
    total = sum(values)

    if total == 0:
        values = [1]
        labels = ['None']

    colors = ['#D7263D', '#8B5E3C', '#5D50A0', '#FFD700', '#2E86AB', '#FF6B35', "#219B7C"]
    ax.pie(values, labels=labels, colors=colors[:len(values)], autopct='%1.1f%%', startangle=90)
    ax.axis('equal')

    canvas = FigureCanvas(fig)
    output = io.BytesIO()
    canvas.print_png(output)
    plt.close(fig)

    return Response(output.getvalue(), mimetype='image/png')

# Utilities
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

def emotion_plotter():
    plt.ion()
    previous_emotions = None

    fig = plt.figure(figsize=(10, 6))
    pie_ax = fig.add_axes([0.05, 0.1, 0.5, 0.8])
    legend_ax = fig.add_axes([0.62, 0.05, 0.35, 0.9])
    legend_ax.axis('off')

    while not stop_plot_thread:
        current = latest_emotions.copy()
        if current == previous_emotions:
            time.sleep(0.1)
            continue

        pie_ax.clear()
        values = list(current.values())
        labels = list(current.keys())
        total = sum(values)
        if total == 0:
            time.sleep(0.1)
            continue

        percentages = [v / total * 100 for v in values]
        custom_colors = {
            'angry': '#D7263D',
            'disgust': '#8B5E3C',
            'fear': '#5D50A0',
            'happy': '#FFD700',
            'sad': '#2E86AB',
            'surprise': '#FF6B35',
            'neutral': "#219B7C"
        }
        colors = [custom_colors.get(label, '#CCCCCC') for label in labels]

        wedges, _ = pie_ax.pie(percentages, colors=colors, startangle=140)
        pie_ax.set_title("Real-Time Emotion Pie Chart", fontsize=14)

        legend_ax.clear()
        legend_ax.axis('off')
        spacing = 0.09
        total_items = len(labels)
        start_y = 0.5 + (spacing * (total_items - 1)) / 2

        for i, (label, pct, color) in enumerate(zip(labels, percentages, colors)):
            y = start_y - i * spacing
            legend_ax.add_patch(mpatches.Rectangle((0.0, y - 0.03), 0.08, 0.06, color=color, transform=legend_ax.transAxes))
            legend_ax.text(0.12, y, f"{label.capitalize():<10} {pct:5.1f}%", fontsize=12, va='center', transform=legend_ax.transAxes)

        plt.draw()
        plt.pause(0.1)
        previous_emotions = current.copy()

    plt.close()

if __name__ == '__main__':
    print("Starting Flask app...")
    plot_thread = threading.Thread(target=emotion_plotter, daemon=True)
    plot_thread.start()
    app.run(debug=False)
