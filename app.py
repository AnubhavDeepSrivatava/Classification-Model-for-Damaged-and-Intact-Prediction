import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
import os
import gdown
from tensorflow.keras.models import load_model


# Define the path to the model
MODEL_PATH = "resnet34_model.h5"

# Check if the model file exists, if not, download it
if not os.path.exists(MODEL_PATH):
    # URL of the model from Google Drive
    url = "https://drive.google.com/uc?export=download&id=1emFBZipAFq4z5TnJQuyyaHUvG3eTvoKx"
    gdown.download(url, MODEL_PATH, quiet=False)

# Load the trained model
model = load_model(MODEL_PATH)

# Define image preprocessing function
def preprocess_image(frame, img_size=(256, 256)):
    frame_resized = cv2.resize(frame, img_size)
    frame_normalized = frame_resized / 255.0  # Normalize if your model expects it
    frame_expanded = np.expand_dims(frame_normalized, axis=0)
    return frame_expanded

# Streamlit UI
st.title("Real-Time Package Damage Detection")
run = st.checkbox("Start Webcam")

# Open webcam and start the frame capture
cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()

if run:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame")
            break
        
        # Preprocess and predict
        processed_frame = preprocess_image(frame)
        prediction = model.predict(processed_frame)
        
        # Extract prediction
        predicted_value = prediction[0][0]  # Extract scalar value
        label = "Damaged" if predicted_value > 0.5 else "Intact"
        confidence = predicted_value if label == "Damaged" else 1 - predicted_value
        
        # Display results
        text = f"{label} ({confidence:.2f})"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame_placeholder.image(frame, channels="BGR")

        # Exit condition
        if not run:
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
