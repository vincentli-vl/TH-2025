from flask import Flask, render_template, request, jsonify
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

CAM0 = 0
CAM1 = 1

app = Flask(__name__)

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Initialize text-to-speech engine
try:
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 150)  # Speed of speech
    tts_engine.setProperty('volume', 0.9)  # Volume level
    tts_available = True
except Exception as e:
    print(f"Text-to-speech not available: {e}")
    tts_available = False

# Store conversation history
conversation_history = []

# Create conversations folder if it doesn't exist
conversations_folder = "conversations"
if not os.path.exists(conversations_folder):
    os.makedirs(conversations_folder)
    print(f"Created conversations folder: {conversations_folder}")

@app.route('/')
def home():
	return render_template('index.html')

cam0 = cv2.VideoCapture(CAM0 + cv2.CAP_DSHOW)
cam1 = cv2.VideoCapture(CAM1 + cv2.CAP_DSHOW)

def scan(out, cam):
	r, f = cam.read()
	if not r:
		return
		
	try:
		# 2. Analyze the in-memory frame directly
		analysis = DeepFace.analyze(f, actions=['emotion'], enforce_detection=False)
			
		if analysis and isinstance(analysis, list):
			face = analysis[0]
			region = face.get('region', {})
			x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
			frame_height, frame_width, _ = f.shape
			is_valid_face = w > 0 and h > 0 and w < frame_width - 1 and h < frame_height - 1 # Use a simple pixel threshold

			if is_valid_face:
				# print("VALID FACE")
					
				# 3. Draw the rectangle on the SAME frame
				cv2.rectangle(f, (x, y), (x + w, y + h), (255, 255, 255), 1)
					
			# else:
			# 	print("no face")
			
	except Exception as e:
		# This will catch errors from DeepFace if no face is found
		print(f"No face detected or error occurred: {e}")
		pass	

	cv2.imwrite(f"./{out}", f)

def loop():
	while True:
		scan("output0.png", cam0)
		scan("output1.png", cam1)
		

@app.route('/voice')
def voice():
    return render_template('voice.html')

@app.route('/journal')
def journal():
    return render_template('journal.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/profile')
def profile():
    return render_template('profile.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        # Get text data from request (since we're using browser speech recognition)
        data = request.get_json()
        
        if not data or 'text' not in data:
            print("ERROR: No text data received")
            return jsonify({'error': 'No text data received'}), 400
        
        text = data['text']
        print(f"Received: '{text}'")
        
        # Add to conversation history
        conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'user_input': text,
            'type': 'user'
        })
        
        # Generate response
        response = generate_response(text)
        print(f"Response: '{response}'")
        
        # Add response to conversation history
        conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'response': response,
            'type': 'system'
        })
        
        # Save conversation to file
        save_conversation_to_file()
        
        # Play audio response in a separate thread (only if TTS is available)
        if tts_available:
            threading.Thread(target=play_audio_response, args=(response,), daemon=True).start()
        
        return jsonify({
            'transcription': text,
            'response': response,
            'success': True
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': f'Error processing transcription: {e}'}), 500

@app.route('/get_history', methods=['GET'])
def get_history():
    return jsonify(conversation_history)

def generate_response(user_input):
    """Generate a response based on user input"""
    user_input_lower = user_input.lower()
    
    # Simple response logic - you can expand this
    if 'hello' in user_input_lower or 'hi' in user_input_lower:
        return "Hello! How can I help you today?"
    elif 'how are you' in user_input_lower:
        return "I'm doing well, thank you for asking!"
    elif 'bye' in user_input_lower or 'goodbye' in user_input_lower:
        return "Goodbye! Have a great day!"
    elif 'weather' in user_input_lower:
        return "I'm sorry, I don't have access to weather information right now."
    elif 'time' in user_input_lower:
        current_time = datetime.now().strftime("%H:%M")
        return f"The current time is {current_time}."
    else:
        return f"I heard you say: '{user_input}'. That's interesting!"

def play_audio_response(text):
    """Play audio response using text-to-speech"""
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()

    except Exception as e:
        print(f"Error playing audio response: {e}")

def save_conversation_to_file():
    """Save conversation history to a text file in the conversations folder"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(conversations_folder, f"conversation_{timestamp}.txt")
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Conversation Session - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            
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
    print("Available routes:")
    print("  - / (home page)")
    print("  - /voice (voice recognition)")
    print("  - /test (test endpoint)")
    print("  - /transcribe (POST - voice processing)")
    print("  - /get_history (GET - conversation history)")
    a = threading.Thread(target=loop, daemon=True)
    a.start()
    app.run(debug=False)