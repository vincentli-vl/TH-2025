import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
import os

# === Setup ===
image_path = 'disgust_face.jpg'
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

try:
    # Analyze the image
    analysis = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=True)

    if analysis:
        # If only one face is detected, DeepFace returns a dict directly (not a list)
        face_analysis = analysis[0] if isinstance(analysis, list) else analysis

        dominant_emotion = face_analysis['dominant_emotion']
        emotion_scores = face_analysis['emotion']

        print(f"Analysis for the image '{image_path}':")
        print(f"Dominant Emotion: {dominant_emotion.capitalize()}")
        print("\nEmotion Scores:")
        for emotion, score in emotion_scores.items():
            print(f"- {emotion.capitalize()}: {score:.2f}%")

        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get face region from 'region' instead of 'face_area'
        region = face_analysis.get('region')
        if region:
            x, y, w, h = region['x'], region['y'], region['w'], region['h']

            # Draw bounding box and label
            cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image_rgb, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 0, 0), 2)

            # Optional: Approximate eyes and mouth positions
            eye_y = y + int(h * 0.3)
            eye_w = int(w * 0.2)
            eye_h = int(h * 0.1)
            mouth_y = y + int(h * 0.75)
            mouth_h = int(h * 0.1)

            # Draw eye boxes
            cv2.rectangle(image_rgb, (x + int(w * 0.2), eye_y),
                          (x + int(w * 0.2) + eye_w, eye_y + eye_h), (255, 255, 0), 1)
            cv2.rectangle(image_rgb, (x + int(w * 0.6), eye_y),
                          (x + int(w * 0.6) + eye_w, eye_y + eye_h), (255, 255, 0), 1)

            # Draw mouth box
            cv2.rectangle(image_rgb, (x + int(w * 0.3), mouth_y),
                          (x + int(w * 0.7), mouth_y + mouth_h), (0, 255, 255), 1)
        else:
            print("Warning: Face region not found, skipping face box drawing.")

        # Save annotated image
        annotated_image_path = os.path.join(results_dir, 'emotion_annotated.png')
        cv2.imwrite(annotated_image_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        print(f"Annotated image saved to: {annotated_image_path}")

        # Save emotion score chart
        emotions = list(emotion_scores.keys())
        scores = list(emotion_scores.values())

        plt.figure(figsize=(8, 4))
        plt.bar(emotions, scores, color='skyblue')
        plt.title('Emotion Confidence Scores')
        plt.ylabel('Confidence (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()

        chart_path = os.path.join(results_dir, 'emotion_scores_chart.png')
        plt.savefig(chart_path)
        plt.close()
        print(f"Emotion chart saved to: {chart_path}")

    else:
        print("No faces were detected in the image.")

except Exception as e:
    print(f"An error occurred: {e}")
