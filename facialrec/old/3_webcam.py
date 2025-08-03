import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
import numpy as np
import threading
import time
import os

# Globals
latest_emotions = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}
stop_thread = False

emoji_dir = "emojis"    #load emoji PNGs into a dict (resize them for overlay)
emoji_map = {}
emoji_size = (80, 80)   #gonna increase the size for better visibility

for emo in latest_emotions.keys():
    path = os.path.join(emoji_dir, f"{emo}.png")
    if os.path.exists(path):
        emoji_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  #loading w/ alpha channel
        if emoji_img is not None:
            emoji_img = cv2.resize(emoji_img, emoji_size)
            emoji_map[emo] = emoji_img


#matplotlib emotion bar chart in separate thread / window, can comment this out if dont want
def emotion_plotter():
    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(latest_emotions.keys(), latest_emotions.values(), color='skyblue')
    ax.set_ylim(0, 100)
    ax.set_ylabel("Confidence (%)")
    ax.set_title("Real-Time Emotion Confidence")
    plt.show(block=False)

    while not stop_thread:
        for bar, key in zip(bars, latest_emotions.keys()):
            bar.set_height(latest_emotions[key])
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.1)

    plt.ioff()
    plt.close()

plot_thread = threading.Thread(target=emotion_plotter)
plot_thread.start()

cap = cv2.VideoCapture(0)
cv2.namedWindow("Facial Tracker", cv2.WINDOW_NORMAL)

if not cap.isOpened():
    print("Webcam not found.")
    stop_thread = True
    exit()

CYAN = (255, 255, 0)

def overlay_png_alpha(img, png_img, x, y):
    """Overlay png_img with alpha channel on img at position x,y."""
    h, w = png_img.shape[:2]

    if y + h > img.shape[0] or x + w > img.shape[1] or x < 0 or y < 0:
        #out of bounds, skip overlay
        return img

    #splitting png into RGB and alpha
    png_rgb = png_img[:, :, :3]
    alpha = png_img[:, :, 3] / 255.0

    roi = img[y:y+h, x:x+w]

    #blend
    for c in range(3):
        roi[:, :, c] = roi[:, :, c] * (1 - alpha) + png_rgb[:, :, c] * alpha

    img[y:y+h, x:x+w] = roi
    return img

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (960, 720))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    analysis = DeepFace.analyze(frame_rgb, actions=['emotion'], enforce_detection=False)

    #checking if a face was detected and if its size is reasonable
    if analysis and isinstance(analysis, list):
        face = analysis[0]
        region = face.get('region', {})
        x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)

        #validation threshold for the face box size, due to no face being present results the traces, text and emoji still being present
        frame_height, frame_width, _ = frame.shape
        #validation threshold set to 95%, will adjust if need be
        is_valid_face = w > 0 and h > 0 and w < frame_width * 0.95 and h < frame_height * 0.95

        if is_valid_face:
            latest_emotions = face.get('emotion', latest_emotions)
            dominant_emotion = face.get('dominant_emotion', 'neutral').lower()

            #trace face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), CYAN, 2)

            #calculate positions for the emoji and text to prevent overlap and make them larger
            text_font_scale = 1.2
            text_thickness = 2
            text_size, _ = cv2.getTextSize(dominant_emotion.capitalize(), cv2.FONT_HERSHEY_SIMPLEX, text_font_scale, text_thickness)
            
            #repositioning emoji slightly above the text
            emoji_y = max(y - emoji_size[1] - text_size[1] - 15, 0)
            emoji_x = x
            
            #repositioning text slightly below the emoji
            text_y = emoji_y + emoji_size[1] + 5
            text_x = x

            #overlay emoji png (basically only if its available)
            if dominant_emotion in emoji_map:
                frame = overlay_png_alpha(frame, emoji_map[dominant_emotion], emoji_x, emoji_y)
                
            #emotion text label
            cv2.putText(frame, dominant_emotion.capitalize(), (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, text_font_scale, CYAN, text_thickness)

            #trace eyes
            left_eye = (x + int(w * 0.3), y + int(h * 0.4))
            right_eye = (x + int(w * 0.7), y + int(h * 0.4))
            cv2.circle(frame, left_eye, 10, CYAN, -1)
            cv2.circle(frame, right_eye, 10, CYAN, -1)

            #trace eyebrows
            left_brow = np.array([
                [x + int(w * 0.22), y + int(h * 0.24)],
                [x + int(w * 0.3), y + int(h * 0.22)],
                [x + int(w * 0.38), y + int(h * 0.24)]
            ], np.int32)

            right_brow = np.array([
                [x + int(w * 0.62), y + int(h * 0.24)],
                [x + int(w * 0.7), y + int(h * 0.22)],
                [x + int(w * 0.78), y + int(h * 0.24)]
            ], np.int32)

            cv2.polylines(frame, [left_brow], False, CYAN, 2)
            cv2.polylines(frame, [right_brow], False, CYAN, 2)

            #trace mouth curve
            mouth_top = y + int(h * 0.75)
            offset = 10 if dominant_emotion == "happy" else -10 if dominant_emotion == "sad" else 0
            mouth_curve = np.array([
                [x + int(w * 0.30), mouth_top],
                [x + int(w * 0.40), mouth_top + offset],
                [x + int(w * 0.50), mouth_top + offset],
                [x + int(w * 0.60), mouth_top + offset],
                [x + int(w * 0.70), mouth_top]
            ], np.int32)
            cv2.polylines(frame, [mouth_curve], False, CYAN, 2)

    cv2.imshow("Facial Tracker", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        stop_thread = True
        break
    elif key == ord('s'):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        cv2.imwrite(filename, frame)
        print(f"[Saved] {filename}")

cap.release()
cv2.destroyAllWindows()
stop_thread = True
plot_thread.join()