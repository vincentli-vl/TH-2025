import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt

image_path = 'disgust_face.jpg' # Make sure you have an image with this name in the same directory.

try:
    # Analyze the image for emotions. DeepFace handles face detection automatically.
    # The 'actions' parameter specifies what we want to analyze.
    analysis = DeepFace.analyze(image_path, actions=['emotion'])

    # The 'analysis' object is a list of dictionaries, one for each face detected.
    # Let's assume there's at least one face and print its analysis.
    if analysis:
        # Get the first face's analysis.
        face_analysis = analysis[0]

        # Extract the dominant emotion.
        dominant_emotion = face_analysis['dominant_emotion']

        # Extract the emotion scores (the confidence percentages for each emotion).
        emotion_scores = face_analysis['emotion']

        print(f"Analysis for the image '{image_path}':")
        print(f"Dominant Emotion: {dominant_emotion.capitalize()}")
        print("\nEmotion Scores:")
        for emotion, score in emotion_scores.items():
            print(f"- {emotion.capitalize()}: {score:.2f}%")

        # --- Optional: Display the image with the results ---
        # Load the image using OpenCV.
        image = cv2.imread(image_path)

        # Convert the image from BGR (OpenCV default) to RGB for matplotlib.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get the facial area (bounding box) from the analysis.
        x, y, w, h = face_analysis['face_area']['x'], face_analysis['face_area']['y'], \
                     face_analysis['face_area']['w'], face_analysis['face_area']['h']

        # Draw a rectangle around the face.
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Put the dominant emotion text on the image.
        cv2.putText(image, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the image using matplotlib.
        plt.imshow(image)
        plt.title('Facial Emotion Recognition')
        plt.axis('off')
        plt.show()

    else:
        print("No faces were detected in the image.")

except Exception as e:
    print(f"An error occurred: {e}")