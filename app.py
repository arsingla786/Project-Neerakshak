import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import os

# ---------------------------
# Load the trained YOLOv8 model
# ---------------------------
model = YOLO("runs/detect/train/weights/best.pt")

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
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        detected_items.append({"class": cls, "confidence": round(conf, 2)})

    particle_count = len(boxes)
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
# Upload Image Detection
# ---------------------------
st.subheader("Upload a Microscope Image")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    # ✅ Always convert to RGB (fixes 4 channel issue)
    img = img.convert("RGB")

    img_array = np.array(img)

    # Run YOLO detection
    results = model.predict(img_array, conf=0.5)

    # Annotate result
    annotated = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    st.image(annotated_rgb, caption="Detection Result", width=300)

    # Particle count
    particle_count = len(results[0].boxes)
    st.write(f"*Number of particles detected:* {particle_count}")

    # Save button
    if st.button("Save Results (Image Detection)"):
        save_results(results)

# ---------------------------
# Live Detection
# ---------------------------
st.subheader("Live Microscope Feed")

run_live = st.checkbox("Start Live Detection")

if run_live:
    cap = cv2.VideoCapture(2)  # try 0, 1, or 2 depending on your camera index

    if not cap.isOpened():
        st.error("Camera could not be opened. Try changing the index (0/1/2) or check connection.")
    else:
        stframe = st.empty()
        stcount = st.empty()
        save_btn = st.button("Save Results (Live Detection)")

        while run_live:
            ret, frame = cap.read()
            if not ret:
                st.warning("No video feed detected. Check camera connection.")
                break

            # ✅ Ensure frame is in RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run YOLO detection
            results = model.predict(frame_rgb, conf=0.5)
            annotated_frame = results[0].plot()
            annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # Display live video
            stframe.image(annotated_rgb, channels="RGB", use_container_width=True)

            # Show particle count dynamically
            particle_count = len(results[0].boxes)
            stcount.markdown(f"*Live Particle Count:* {particle_count}")

            # Save if button clicked
            if save_btn:
                save_results(results)

            # Check if user unchecked the box to stop
            run_live = st.session_state.get("Start Live Detection", False)

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
        st.warning(f"No saved data found at {LOG_FILE}")

# ---------------------------
# Footer
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