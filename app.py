import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import os

# Load the trained YOLOv8 model
model = YOLO("runs/detect/train/weights/best.pt")

# CSV file to store results
LOG_FILE = "detection_logs.csv"

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Neerakshak - Microplastic Detection", page_icon="🔬", layout="wide")

# ---------------------------
# Header with Logo + Title
# ---------------------------
col1, col2 = st.columns([1, 4])
with col1:
    st.image("logo.jpg", width=120)  # Place logo.jpg in same folder as app.py
with col2:
    st.markdown(
        """
        <h1 style="color:#2c3e50; margin-bottom:0;">Project-Neerakshak</h1>
        <h5 style="color:#2980b9; margin-top:5px;">Microplastic Detection with Machine Learning</h5>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# ---------------------------
# Function to Save Results
# ---------------------------
def save_results(results):
    boxes = results[0].boxes
    detected_items = []
    for box in boxes:
        cls = int(box.cls[0])   # class id
        conf = float(box.conf[0])  # confidence
        detected_items.append({"class": cls, "confidence": round(conf, 2)})

    particle_count = len(boxes)  # number of detected particles
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df_new = pd.DataFrame([{
        "time": timestamp,
        "detections": str(detected_items),
        "count": particle_count
    }])

    if os.path.exists(LOG_FILE):
        df_new.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df_new.to_csv(LOG_FILE, mode="w", header=True, index=False)

    st.success(f"✅ Results saved successfully! ({particle_count} particles found)")

# ---------------------------
# Upload Image Option
# ---------------------------
st.subheader("Upload a Microscope Image")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img_array = np.array(img)

    results = model.predict(img_array, conf=0.5)
    annotated = results[0].plot()

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    st.image(annotated_rgb, caption="Detection Result", width=300)

    # Display number of particles
    particle_count = len(results[0].boxes)
    st.write(f"**Number of particles detected:** {particle_count}")

    if st.button("Save Results (Image Detection)"):
        save_results(results)

# ---------------------------
# Live Detection Option
# ---------------------------
st.subheader("Live Microscope Feed")

start_live = st.button("Start Live Detection")
if start_live:
    cap = cv2.VideoCapture(0)  # Adjust index for your microscope camera
    stframe = st.empty()       # Placeholder for video
    stcount = st.empty()       # Placeholder for particle count
    save_button = st.empty()   # Placeholder for save button
    stop_button = st.button("Stop Live Detection")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("No video feed detected. Check camera connection.")
            break

        # Run YOLO detection
        results = model.predict(frame, conf=0.5)
        annotated_frame = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Show live video
        stframe.image(annotated_rgb, channels="RGB")

        # Show particle count dynamically
        particle_count = len(results[0].boxes)
        stcount.markdown(f"**Live Particle Count:** {particle_count}")

        # Save current frame’s results
        if save_button.button("Save Results (Live Detection)"):
            save_results(results)

        # Stop live detection
        if stop_button:
            break

    cap.release()

# ---------------------------
# View Saved Data
# ---------------------------
st.subheader("View Saved Data")
if st.button("Show Detection Logs"):
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        st.dataframe(df)    
    else:
        st.warning("No saved data found yet.")
        
# ---------------------------
# Footer with Copyright
# ---------------------------
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f0f2f6;
        color: #333333;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        border-top: 1px solid #e6e6e6;
    }
    </style>
    <div class="footer">
        © 2025 Team VARUNA. All Rights Reserved.
    </div>
    """,
    unsafe_allow_html=True
)
  