import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import threading
import time
import os

# Globals
latest_emotions = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}
previous_emotions = None  # for smoothing pie chart
emoji_dir = "emojis"
emoji_map = {}
emoji_size = (80, 80)
stop_thread = False
CYAN = (255, 255, 0)

# Load emojis
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

def smooth_emotions(current, new, alpha=0.5):
    return {k: alpha * new.get(k, 0) + (1 - alpha) * current.get(k, 0) for k in current}

def emotion_plotter():
    global previous_emotions
    plt.ion()

    fig = plt.figure(figsize=(10, 6))
    pie_ax = fig.add_axes([0.05, 0.1, 0.5, 0.8])     # [left, bottom, width, height]
    legend_ax = fig.add_axes([0.62, 0.05, 0.35, 0.9])  # Lowered vertically to center legend
    legend_ax.axis('off')

    while not stop_thread:
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
            'angry':    '#D7263D',  # crimson red
            'disgust':  '#8B5E3C',  # brown
            'fear':     '#5D50A0',  # deep purple
            'happy':    '#FFD700',  # bright yellow
            'sad':      '#2E86AB',  # sky blue
            'surprise': '#FF6B35',  # bright orange
            'neutral':  "#219B7C"   # mint green
        }
        colors = [custom_colors.get(label, '#CCCCCC') for label in labels]

        wedges, _ = pie_ax.pie(
            percentages,
            colors=colors,
            startangle=140
        )
        pie_ax.set_title("Real-Time Emotion Pie Chart", fontsize=14)

        # T-chart Legend: Lowered & evenly spaced
        legend_ax.clear()
        legend_ax.axis('off')

        spacing = 0.09
        total_items = len(labels)
        start_y = 0.5 + (spacing * (total_items - 1)) / 2  # center vertically

        for i, (label, pct, color) in enumerate(zip(labels, percentages, colors)):
            y = start_y - i * spacing
            legend_ax.add_patch(mpatches.Rectangle((0.0, y - 0.03), 0.08, 0.06, color=color, transform=legend_ax.transAxes))
            legend_ax.text(0.12, y, f"{label.capitalize():<10} {pct:5.1f}%", fontsize=12, va='center', transform=legend_ax.transAxes)

        plt.draw()
        plt.pause(0.1)
        previous_emotions = current.copy()

    plt.close()

def get_face_embedding(img):
    # Extract embedding vector for a given face image
    embedding = DeepFace.represent(img, enforce_detection=False)
    if isinstance(embedding, list) and len(embedding) > 0:
        return embedding[0]["embedding"]
    else:
        return None

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    if vec1.shape != vec2.shape:
        return -1
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def start_tracker():
    global stop_thread, latest_emotions

    # Load and encode known images for each person (multiple per person)
    known_faces = {
        "Sayeed": [],
        "Vivian": []
    }
    # For each person, load both images and get embeddings
    for name, files in {
        "Sayeed": ["sayeed.jpg", "sayeed2.jpg"],
        "Vivian": ["vivian.jpg", "vivian2.jpg"]
    }.items():
        for img_file in files:
            if not os.path.exists(img_file):
                print(f"[Warning] Known face image '{img_file}' not found!")
                continue
            img = cv2.imread(img_file)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            embedding = get_face_embedding(img_rgb)
            if embedding is not None:
                known_faces[name].append(embedding)
            else:
                print(f"[Warning] Could not compute embedding for {img_file}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not found.")
        return

    plot_thread = threading.Thread(target=emotion_plotter)
    plot_thread.start()

    cv2.namedWindow("Facial Tracker", cv2.WINDOW_NORMAL)

    while not stop_thread:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (960, 720))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            analyses = DeepFace.analyze(frame_rgb, actions=['emotion'], enforce_detection=False)
        except Exception as e:
            analyses = []

        if not isinstance(analyses, list):
            analyses = [analyses]

        face_count = 0

        for face in analyses:
            region = face.get('region', {})
            x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
            frame_height, frame_width, _ = frame.shape
            is_valid_face = w > 0 and h > 0 and w < frame_width * 0.95 and h < frame_height * 0.95

            if not is_valid_face:
                continue

            face_count += 1

            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(frame_width, x + w), min(frame_height, y + h)
            face_img = frame_rgb[y1:y2, x1:x2]

            face_embedding = get_face_embedding(face_img)

            matched_name = "Unknown"
            if face_embedding is not None:
                best_score = -1
                best_person = None
                for person, embeddings in known_faces.items():
                    for emb in embeddings:
                        sim = cosine_similarity(face_embedding, emb)
                        if sim > best_score:
                            best_score = sim
                            best_person = person
                if best_score >= 0.5:  # Adjust threshold as needed
                    matched_name = best_person
                else:
                    matched_name = "Unknown"

            # Emotion for first face only
            if face_count == 1:
                detected_emotions = face.get('emotion', latest_emotions)
                latest_emotions = smooth_emotions(latest_emotions, detected_emotions)
                dominant = face.get('dominant_emotion', 'neutral').lower()
                conf = latest_emotions.get(dominant, 0)
                conf_text = f"Acc: {conf:.1f}%"
            else:
                dominant = None
                conf_text = ""

            cv2.rectangle(frame, (x, y), (x + w, y + h), CYAN, 2)
            label_y = y + h + 20 if (y + h + 20 < frame_height) else y + h - 10
            cv2.putText(frame, matched_name, (x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, CYAN, 2)

            if face_count == 1 and dominant in emoji_map:
                emoji_y = max(y - emoji_size[1] - 20, 0)
                frame = overlay_png_alpha(frame, emoji_map[dominant], x, emoji_y)
                cv2.putText(frame, dominant.capitalize(), (x, emoji_y + emoji_size[1] + 5),
                            cv2.FONT_HERSHEY_DUPLEX, 1.2, CYAN, 2)
                cv2.putText(frame, conf_text, (x + w - 100, y + h + 45),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, CYAN, 2)

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

        time.sleep(0.05)

    cap.release()
    cv2.destroyAllWindows()
    plot_thread.join()

if __name__ == '__main__':
    print("Launching emotion tracker with multi-face recognition...")
    start_tracker()
