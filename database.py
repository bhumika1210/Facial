import cv2
import numpy as np
import sqlite3
import tensorflow as tf
from tensorflow import keras
import time

# Connect to the database
conn = sqlite3.connect('emotions.db')
cursor = conn.cursor()

# Create a table to store the emotions
cursor.execute('''CREATE TABLE IF NOT EXISTS emotions
                  (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  emotion TEXT)''')

# Load the model architecture from JSON file
with open('model/emotion_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
model = keras.models.model_from_json(loaded_model_json)

# Load the trained weights from H5 file
model.load_weights('model/emotion_model.h5')

# Define the emotions labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open a video capture
video_capture = cv2.VideoCapture(-1)

# Define the delay between each frame (in seconds)
delay = 2

while True:
    # Read the video frame
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face region
        face = gray[y:y + h, x:x + w]

        # Preprocess the face image
        face = cv2.resize(face, (48, 48))
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)
        face = face / 255.0  # Normalize pixel values

        # Perform emotion detection
        emotion_probs = model.predict(face)[0]
        emotion_label = emotions[np.argmax(emotion_probs)]

        # Store the emotion in the database
        cursor.execute("INSERT INTO emotions (emotion) VALUES (?)", (emotion_label,))
        conn.commit()

        # Draw a rectangle around the face and display the detected emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Check for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

     # Delay between frames
    time.sleep(delay)

# Connect to the database
conn = sqlite3.connect('emotions.db')
cursor = conn.cursor()

# Retrieve all emotions from the database
cursor.execute("SELECT emotion FROM emotions")
emotions = cursor.fetchall()

# Print the retrieved emotions
for emotion in emotions:
    print(emotion[0])

# Close the database connection
conn.close()

# Release the video capture and close the database connection
video_capture.release()
cv2.destroyAllWindows()
conn.close()
