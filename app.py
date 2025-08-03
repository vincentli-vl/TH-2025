from flask import Flask, render_template, request, jsonify, Response
import speech_recognition as sr
import json
import os
from datetime import datetime
import threading
import cv2
from deepface import DeepFace
import numpy as np
import time
import matplotlib.pyplot as plt


CAM0 = 0
CAM1 = 1
app = Flask(__name__)

# Setup
recognizer = sr.Recognizer()
conversation_history = []
conversations_folder = "conversations"
if not os.path.exists(conversations_folder):
    os.makedirs(conversations_folder)

# Audio files mapping
audio_responses = {
    "where am i": "1.wav",
    "who are you": "2.wav",
    "why am i here": "3.wav",
    "i want to go home": "4.wav",
    "what is going on": "5.wav",
    "hungry": "6.wav",
    "thirsty": "6.wav",
    "bed": "7.wav",
    "i miss": "8.wav",
    "doing today": "9.wav",
    "remember": "10.wav",
    "medicine": "11.wav",
    "eat": "12.wav",
    "niece": "13.wav",
    "default": "13.wav"
}

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
emotion_snapshots = []
emoji_dir = "emojis"
emoji_map = {}
emoji_size = (80, 80)
CYAN = (255, 255, 0)

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
    scale_factor = 0.5  # Must match fx, fy in resize
    while True:
        r, f = cam.read()
        if not r:
            continue

        frame_height, frame_width, _ = f.shape
        frame_count += 1
        # Only analyze every 10th frame for performance
        if frame_count % 10 == 0:
            try:
                small_f = cv2.resize(f, (0, 0), fx=scale_factor, fy=scale_factor)
                analysis = DeepFace.analyze(
                    small_f,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend="opencv"
                )
                # DeepFace returns a dict for one face, or a list for multiple
                if isinstance(analysis, dict):
                    analysis_results = [analysis]
                elif isinstance(analysis, list):
                    analysis_results = analysis
            except Exception as e:
                print(f"Face scan error: {e}")

        # Draw results for all detected faces
        for face in analysis_results:
            region = face.get('region', {})
            # Scale coordinates back to original frame size
            x = int(region.get('x', 0) / scale_factor)
            y = int(region.get('y', 0) / scale_factor)
            w = int(region.get('w', 0) / scale_factor)
            h = int(region.get('h', 0) / scale_factor)
            if w > 0 and h > 0 and w < frame_width - 3 and h < frame_height - 3:
                latest_emotions.update(face.get('emotion', latest_emotions))
                dominant_emotion = face.get('dominant_emotion', 'neutral').lower()
                conf = latest_emotions.get(dominant_emotion, 0)
                emotion_snapshots.append(latest_emotions.copy())
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

# Routes
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
        
        # Generate response (now returns audio filename)
        audio_filename = generate_response(text)
        
        # Add the response to conversation history
        conversation_history.append({'timestamp': datetime.now().isoformat(), 'response': audio_filename, 'type': 'system'})
        save_conversation_to_file()

        # Return the audio filename to the frontend
        return jsonify({
            'transcription': text, 
            'response': audio_filename, 
            'audio_file': audio_filename,
            'success': True
        })
    except Exception as e:
        return jsonify({'error': f'Error processing transcription: {e}'}), 500

@app.route('/get_history', methods=['GET'])
def get_history():
    return jsonify(conversation_history)


@app.route('/static/audio/<filename>')
def serve_audio(filename):
    from flask import send_from_directory
    return send_from_directory('static/audio', filename)

# @app.route('/chart')
# def chart():
#     import io
#     from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

#     fig, ax = plt.subplots(figsize=(3, 3))
#     values = list(latest_emotions.values())
#     labels = list(latest_emotions.keys())
#     total = sum(values)

#     if total == 0:
#         values = [1]
#         labels = ['None']

#     colors = ['#D7263D', '#8B5E3C', '#5D50A0', '#FFD700', '#2E86AB', '#FF6B35', "#219B7C"]
#     ax.pie(values, labels=labels, colors=colors[:len(values)], autopct='%1.1f%%', startangle=90)
#     ax.axis('equal')

#     canvas = FigureCanvas(fig)
#     output = io.BytesIO()
#     canvas.print_png(output)
#     plt.close(fig)

#     return Response(output.getvalue(), mimetype='image/png')
# @app.route('/chart-anim')
# def chart_anim():
#     from matplotlib import pyplot as plt
#     from matplotlib.animation import FuncAnimation, PillowWriter
#     import io

#     if not emotion_snapshots:
#         return Response("No emotion data to animate.", status=204)

#     fig, ax = plt.subplots(figsize=(4, 4))

#     def update(frame):
#         ax.clear()
#         emotions = emotion_snapshots[frame]
#         labels = list(emotions.keys())
#         values = list(emotions.values())
#         total = sum(values)

#         if total == 0:
#             labels = ['None']
#             values = [1]

#         colors = ['#D7263D', '#8B5E3C', '#5D50A0', '#FFD700', '#2E86AB', '#FF6B35', '#219B7C']
#         ax.pie(values, labels=labels, colors=colors[:len(values)], autopct='%1.1f%%', startangle=90)
#         ax.set_title(f"Frame {frame + 1}")
#         ax.axis('equal')

#     ani = FuncAnimation(fig, update, frames=min(len(emotion_snapshots), 30), interval=1000)

#     gif_io = io.BytesIO()
#     ani.save(gif_io, format='gif', writer=PillowWriter(fps=1))
#     gif_io.seek(0)

#     return Response(gif_io.getvalue(), mimetype='image/gif')

# Utilities
def generate_response(user_input):
    lower = user_input.lower().strip()
    
    # Check for exact matches first
    if lower in audio_responses:
        return audio_responses[lower]
    
    # Check for partial matches
    if any(word in lower for word in ['where am', 'where are', 'where is', 'where i']):
        return audio_responses['where am i']
    elif any(phrase in lower for phrase in ['who', 'are you']):
        return audio_responses['who are you']
    elif any(word in lower for word in ['why am', 'i here']):
        return audio_responses['why am i here']
    elif any(word in lower for word in ['go home', 'to go']):
        return audio_responses['i want to go home']
    elif any(word in lower for word in ['going on']):
        return audio_responses['what is going on']
    elif any(word in lower for word in ['hungry', 'thirsty']):
        return audio_responses['hungry']
    elif any(word in lower for word in ['bed', 'sleep', 'time to go']):
        return audio_responses['bed']
    elif any(word in lower for word in ['i miss', 'family', 'friend']):
        return audio_responses['i miss']
    elif any(word in lower for word in ['doing today', 'we doing']):
        return audio_responses['doing today']
    elif any(word in lower for word in ['can\'t remember', 'cannot remember', 'remember']):
        return audio_responses['remember']
    elif any(word in lower for word in ['medicine', 'medication']):
        return audio_responses['medicine']
    elif any(word in lower for word in ['eat', 'food']):
        return audio_responses['eat']
    elif any(word in lower for word in ['niece', 'nephew']):
        return audio_responses['niece']
    else:
        return audio_responses['default']

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

if __name__ == '__main__':
    print("Starting Flask app...")
    #plot_thread = threading.Thread(target=emotion_plotter, daemon=True)
    #plot_thread.start()
    app.run(debug=False)
