import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import os
import time

# ---------------------------
# Load the trained YOLOv8 model
# ---------------------------
model = YOLO("runs/detect/train/weights/best.pt")
LOG_FILE = "detection_logs.csv"

# (Your header / upload / save_results code remains the same)
# ... (keep the earlier parts of your app here)

# ---------------------------
# Helper: try to open camera using multiple backends
# ---------------------------
def try_open_camera(index):
    # try a few common backends (Windows: DSHOW, MSMF / Linux: V4L2)
    backend_names = []
    backends = []
    for name in ("CAP_DSHOW", "CAP_MSMF", "CAP_V4L2", "CAP_FFMPEG", "CAP_ANY"):
        if hasattr(cv2, name):
            backend_names.append(name)
            backends.append(getattr(cv2, name))
    # Always try without specifying backend too (cv2 will use default)
    backends.append(None)

    last_exc = None
    for b in backends:
        try:
            if b is None:
                cap = cv2.VideoCapture(index)
            else:
                cap = cv2.VideoCapture(index, b)
            if cap is None:
                continue
            # small wait for some backends
            time.sleep(0.2)
            if cap.isOpened():
                return cap, b  # success
            else:
                cap.release()
        except Exception as e:
            last_exc = e
    return None, last_exc

# ---------------------------
# Live Detection (Start/Stop buttons + improved handling)
# ---------------------------
st.subheader("Live Microscope Feed (Start / Stop)")

# camera device selection & performance options
camera_index = st.selectbox("Camera device index", options=[0, 1, 2, 3, 4], index=0)
frame_width = st.selectbox("Frame width (px)", options=[320, 480, 640, 800], index=2)
detect_every_n_frames = st.number_input("Run detection every N frames (1 = every frame)", min_value=1, max_value=60, value=3)
use_resize_for_inference = st.checkbox("Resize frame for faster inference (recommended)", value=True)

col_start, col_stop = st.columns([1, 1])
with col_start:
    if st.button("Start Live"):
        st.session_state["live_running"] = True
with col_stop:
    if st.button("Stop Live"):
        st.session_state["live_running"] = False

if "live_running" not in st.session_state:
    st.session_state["live_running"] = False

if st.session_state["live_running"]:
    st.info("Attempting to open camera. If it fails, try different device index or check system camera permissions.")
    cap, backend_or_exc = try_open_camera(camera_index)

    if not cap:
        # backend_or_exc can be an exception or backend value from last trial
        st.error("Could not open camera. Possible causes:\n"
                 "- Wrong device index\n- Camera in use by another app\n- Running inside WSL / remote server or Docker without webcam passthrough\n- Missing OS permissions\n\n"
                 "Backend trial result: " + str(backend_or_exc))
        # show quick probe for indices 0..4
        probe = []
        for i in range(5):
            c = cv2.VideoCapture(i, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)
            probe.append((i, bool(c.isOpened())))
            c.release()
        st.write("Quick probe (index, opened):", probe)
        st.session_state["live_running"] = False
    else:
        st.success(f"Camera opened (index={camera_index}, backend={backend_or_exc}). Resize to {frame_width}px for display.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(frame_width * 3/4))
        stframe = st.empty()
        stcount = st.empty()
        frame_idx = 0

        try:
            while st.session_state.get("live_running", False):
                ret, frame = cap.read()
                if not ret:
                    st.warning("Frame grab failed (ret=False). Camera may have disconnected.")
                    break

                frame_idx += 1

                # Optionally run detection only every N frames to reduce CPU/GPU usage
                if frame_idx % int(detect_every_n_frames) == 0:
                    # prepare a resized copy for faster inference
                    inference_frame = frame
                    if use_resize_for_inference:
                        inference_frame = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])))

                    try:
                        results = model.predict(inference_frame, conf=0.5)
                    except Exception as e:
                        st.error(f"Model inference error: {e}")
                        break

                    # results[0].plot() returns annotated image for the inference_frame,
                    # so if we resized for inference, we show the resized annotated image.
                    annotated = results[0].plot()
                    try:
                        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    except Exception:
                        # sometimes plot already returns RGB
                        annotated_rgb = annotated

                    # particle count
                    particle_count = len(results[0].boxes)
                    stcount.markdown(f"**Live Particle Count:** {particle_count}")

                    # show image
                    stframe.image(annotated_rgb, channels="RGB", use_container_width=True)
                else:
                    # Show raw frame (no detection) to keep UI responsive between inferences
                    try:
                        preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    except Exception:
                        preview = frame
                    stframe.image(preview, channels="RGB", use_container_width=True)

                # tiny sleep so Streamlit can process button presses and UI
                time.sleep(0.01)

        finally:
            cap.release()
            st.session_state["live_running"] = False
            st.success("Camera released. Live detection stopped.")

# ---------------------------
# View Saved Data (unchanged)
# ---------------------------
st.subheader("View Saved Data")
if st.button("Show Detection Logs"):
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        st.dataframe(df)
    else:
        st.warning(f"No saved data found at {LOG_FILE}")
 