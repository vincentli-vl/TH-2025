import cv2
from deepface import DeepFace
import pandas as pd
import threading
import time
import os

# ---------------- Configuration ----------------
CSV_FILE = "emotion_data.csv"
emoji_dir = "emojis"
emoji_size = (80, 80)
CYAN = (255, 255, 0)
# -----------------------------------------------

latest_emotions = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}
emoji_map = {}
stop_thread = False

# Load emojis (PNG with alpha channel)
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

def write_emotion_to_csv_loop():
    with open(CSV_FILE, "w") as f:
        header = "Time," + ",".join(latest_emotions.keys()) + "\n"
        f.write(header)
    while not stop_thread:
        timestamp = time.strftime("%H:%M:%S")
        row = f"{timestamp}," + ",".join(f"{latest_emotions[k]:.2f}" for k in latest_emotions) + "\n"
        with open(CSV_FILE, "a") as f:
            f.write(row)
        time.sleep(1)

def start_tracker():
    global stop_thread
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not found.")
        return

    # Start CSV writer
    csv_thread = threading.Thread(target=write_emotion_to_csv_loop)
    csv_thread.start()

    cv2.namedWindow("Facial Tracker", cv2.WINDOW_NORMAL)

    while not stop_thread:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (960, 720))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            analysis = DeepFace.analyze(frame_rgb, actions=['emotion'], enforce_detection=False)
            if analysis and isinstance(analysis, list):
                face = analysis[0]
                region = face.get('region', {})
                x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
                frame_height, frame_width, _ = frame.shape
                is_valid_face = w > 0 and h > 0 and w < frame_width * 0.95 and h < frame_height * 0.95

                if is_valid_face:
                    latest_emotions.update(face.get('emotion', latest_emotions))
                    dominant = face.get('dominant_emotion', 'neutral').lower()
                    conf = latest_emotions.get(dominant, 0)
                    conf_text = f"Acc: {conf:.1f}%"

                    cv2.rectangle(frame, (x, y), (x + w, y + h), CYAN, 2)

                    emoji_y = max(y - emoji_size[1] - 20, 0)
                    text_y = emoji_y + emoji_size[1] + 5
                    if dominant in emoji_map:
                        frame = overlay_png_alpha(frame, emoji_map[dominant], x, emoji_y)

                    cv2.putText(frame, dominant.capitalize(), (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, CYAN, 2)
                    cv2.putText(frame, conf_text, (x + w - 100, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, CYAN, 2)

        except Exception as e:
            print(f"[Error] {e}")

        cv2.imshow("Facial Tracker", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            stop_thread = True
            break

    cap.release()
    cv2.destroyAllWindows()
    csv_thread.join()

if __name__ == '__main__':
    print("Launching emotion tracker with sjvisualizer CSV output...")
    start_tracker()
