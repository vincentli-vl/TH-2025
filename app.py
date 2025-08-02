from flask import Flask, render_template
import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
import numpy as np
import threading

CAM0 = 0
CAM1 = 1

app = Flask(__name__)

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
		

 

if __name__ == '__main__':
	a = threading.Thread(target=loop, daemon=True)
	a.start()
	app.run(debug=False)
	