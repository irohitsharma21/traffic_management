import cv2
import torch
import numpy as np
import threading
import time
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import librosa
from keras.models import load_model
import os

# Load Models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model = YOLO("best.pt")  # Ensure best.pt is in the directory
audio_model = load_model("ambulance_siren_model.h5")  # Siren model

# Audio Processing Parameters
SAMPLE_RATE = 22050
DURATION = 3  # 3 seconds audio clip


class AmbulanceDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🚑 Ambulance Detection System")

        # Frame for UI
        self.frame = tk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Label
        self.label = tk.Label(self.frame, text="Select a Video for Detection", font=("Arial", 14))
        self.label.pack(pady=10)

        # Browse Button
        self.button = tk.Button(self.frame, text="📂 Browse Video", command=self.select_video, font=("Arial", 12), bg="lightblue")
        self.button.pack(pady=10)

        # Canvas for Video Display (Dynamic Resizing)
        self.canvas = tk.Canvas(self.frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Variables
        self.video_path = None
        self.cap = None
        self.frame_width = 640  # Default width
        self.frame_height = 480  # Default height
        self.running = False

    def select_video(self):
        """Select a video file and start processing."""
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
        if self.video_path:
            self.running = True
            self.process_video()

    def process_video(self):
        """Process and display video in real-time with optimized performance."""
        self.cap = cv2.VideoCapture(self.video_path)

        # Get Video Dimensions
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Resize the Canvas to Fit Video Dimensions
        self.canvas.config(width=self.frame_width, height=self.frame_height)

        # Run Detection in a Separate Thread
        threading.Thread(target=self.run_detection, daemon=True).start()

    def run_detection(self):
        """Handles real-time video processing without lag."""
        skip_frames = 2  # Process every 2nd frame to improve speed
        frame_count = 0

        while self.cap.isOpened() and self.running:
            ret, frame = self.cap.read()
            if not ret:
                break  # Stop if video ends

            frame_count += 1
            if frame_count % skip_frames == 0:
                # Resize frame for faster YOLO processing (optional)
                small_frame = cv2.resize(frame, (640, 360))

                # YOLO Object Detection
                results = yolo_model.predict(small_frame, conf=0.65, device=device)

                detected = False
                for result in results:
                    for i, box in enumerate(result.boxes.xyxy):
                        x1, y1, x2, y2 = map(int, box[:4])
                        confidence = result.boxes.conf[i].item() * 100

                        if confidence > 70:
                            detected = True
                            x1, y1, x2, y2 = map(lambda v: int(v * self.frame_width / 640), [x1, y1, x2, y2])  # Scale back

                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            cv2.putText(frame, f"Ambulance ({confidence:.2f}%)", 
                                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # If detected, verify using audio (asynchronous)
                if detected:
                    threading.Thread(target=self.verify_siren, daemon=True).start()

            # Convert Frame for Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tk = ImageTk.PhotoImage(frame_pil)

            # Update Canvas
            self.canvas.create_image(0, 0, anchor=tk.NW, image=frame_tk)
            self.canvas.image = frame_tk

            time.sleep(0.01)  # Small delay to prevent UI freezing

        self.cap.release()

    def verify_siren(self):
        """Analyzes a short audio clip to check for an ambulance siren asynchronously."""
        temp_audio_path = self.extract_audio(self.video_path)
        if temp_audio_path:
            is_siren = self.classify_audio(temp_audio_path)
            if is_siren:
                print("🚨 Emergency Ambulance Detected!")
            else:
                print("🚑 Off-duty Ambulance (No Siren)")

    def extract_audio(self, video_path):
        """Extracts a short audio segment from the video."""
        try:
            import moviepy.editor as mp
            video = mp.VideoFileClip(video_path)
            temp_audio_path = os.path.join(os.path.dirname(video_path), "temp_audio.wav")
            video.audio.write_audiofile(temp_audio_path, fps=SAMPLE_RATE)
            return temp_audio_path
        except Exception as e:
            print(f"Audio extraction failed: {e}")
            return None

    def classify_audio(self, audio_path):
        """Classifies whether the extracted audio contains an ambulance siren."""
        try:
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, 128 - log_mel_spec.shape[1])), mode='constant')
            mel_spec_reshaped = np.expand_dims(log_mel_spec, axis=-1)
            prediction = audio_model.predict(np.expand_dims(mel_spec_reshaped, axis=0))
            return prediction[0][0] > 0.5
        except Exception as e:
            print(f"Audio processing error: {e}")
            return False


# Run the App
root = tk.Tk()
app = AmbulanceDetectorApp(root)
root.mainloop()