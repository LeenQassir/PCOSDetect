import streamlit as st
import os
import tempfile
from PIL import Image
from ultralytics import YOLO
import numpy as np

# Set YOLO cache directory to a unique writable temp folder
temp_cache_dir = tempfile.mkdtemp(prefix="yolo_cache_")
os.environ['YOLO_CACHE_DIR'] = temp_cache_dir

# Path to your uploaded YOLO weights file
weights_path = "/mnt/data/yolov8n.pt"  # Make sure this file is uploaded to this path

# Load YOLO model (no caching to isolate permission issues)
def load_yolo(weights_file):
    return YOLO(weights_file)

yolo_model = load_yolo(weights_path)

st.title("YOLO Follicle Detection")

uploaded_file = st.file_uploader("Upload Ultrasound Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Ultrasound Image", use_container_width=True)
    
    temp_path = os.path.join(tempfile.gettempdir(), "uploaded_image.png")
    img.save(temp_path)

    if st.button("Detect Follicles"):
        results = yolo_model.predict(temp_path, conf=0.1, iou=0.45)
        follicle_count = len(results[0].boxes) if results else 0
        st.write(f"Follicle count detected: **{follicle_count}**")

        img_with_boxes = results[0].plot()
        st.image(img_with_boxes, caption="Detections with bounding boxes", use_container_width=True)




