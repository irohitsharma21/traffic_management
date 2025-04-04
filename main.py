# app.py
import streamlit as st
import torch
import numpy as np
import cv2
import tempfile
import librosa
from keras.models import load_model
from ultralytics import YOLO
import moviepy.editor as mp
import os

# Load models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model = YOLO("best.pt")  # Ensure this file is in your root directory
audio_model = load_model("ambulance_siren_model.h5")  # Ensure this file is in your root directory

SAMPLE_RATE = 22050

def extract_audio(video_path):
    try:
        video = mp.VideoFileClip(video_path)
        temp_audio_path = os.path.join(os.path.dirname(video_path), "temp_audio.wav")
        video.audio.write_audiofile(temp_audio_path, fps=SAMPLE_RATE)
        return temp_audio_path
    except Exception as e:
        st.error(f"Audio extraction failed: {e}")
        return None

def classify_audio(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, 128 - log_mel_spec.shape[1])), mode='constant')
        mel_spec_reshaped = np.expand_dims(log_mel_spec, axis=-1)
        prediction = audio_model.predict(np.expand_dims(mel_spec_reshaped, axis=0))
        return prediction[0][0] > 0.5
    except Exception as e:
        st.error(f"Audio processing error: {e}")
        return False

def detect_ambulance(video_path):
    cap = cv2.VideoCapture(video_path)
    detected = False
    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 5 != 0:
            continue  # Skip frames for faster processing

        resized_frame = cv2.resize(frame, (640, 360))
        results = yolo_model.predict(resized_frame, conf=0.65, device=device)

        for result in results:
            for i, box in enumerate(result.boxes.xyxy):
                x1, y1, x2, y2 = map(int, box[:4])
                confidence = result.boxes.conf[i].item() * 100
                if confidence > 70:
                    cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(resized_frame, f"Ambulance ({confidence:.1f}%)", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    detected = True
        frames.append(resized_frame)

    cap.release()
    return detected, frames

# Streamlit UI
st.title("🚑 Ambulance Detection from Video")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_video:
        temp_video.write(uploaded_file.read())
        video_path = temp_video.name

    st.info("Processing video...")
    ambulance_detected, frames = detect_ambulance(video_path)

    if ambulance_detected:
        audio_path = extract_audio(video_path)
        if audio_path:
            if classify_audio(audio_path):
                st.success("🚨 Emergency Ambulance Detected (Siren present)!")
            else:
                st.warning("🚑 Ambulance Detected (No Siren)")
        else:
            st.error("Failed to extract audio.")
    else:
        st.info("No Ambulance Detected.")

    st.video(video_path)
