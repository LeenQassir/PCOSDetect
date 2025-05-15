import streamlit as st
import os
import tempfile
from PIL import Image
import numpy as np
import sqlite3
from datetime import datetime
from tensorflow.keras.models import load_model

# Path to your uploaded YOLO weights file
WEIGHTS_PATH = "/mnt/data/yolov8n.pt"  # Adjust if needed

@st.cache_resource(show_spinner=False)
def get_classification_model():
    return load_model("best_mobilenet_model.h5")

@st.cache_resource(show_spinner=False)
def get_yolo_model(weights_path):
    # Set cache directory env var before importing YOLO
    os.environ['YOLO_CACHE_DIR'] = '/mnt/data/yolo_cache'
    # Ultralytics import delayed until here to respect env var
    from ultralytics import YOLO
    return YOLO(weights_path)

model = get_classification_model()
yolo_model = get_yolo_model(WEIGHTS_PATH)

# Database setup
def init_db():
    conn = sqlite3.connect('patients.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            patient_id TEXT PRIMARY KEY,
            name TEXT,
            age INTEGER,
            last_prediction TEXT,
            confidence REAL,
            follicle_count INTEGER,
            last_update TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def get_patient_record(patient_id):
    conn = sqlite3.connect('patients.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
    record = cursor.fetchone()
    conn.close()
    return record

def update_patient_record(patient_id, name, age, prediction, confidence, follicle_count):
    conn = sqlite3.connect('patients.db')
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT OR REPLACE INTO patients
        (patient_id, name, age, last_prediction, confidence, follicle_count, last_update)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (patient_id, name, age, prediction, confidence, follicle_count, now))
    conn.commit()
    conn.close()

def preprocess_image(image_file):
    img = image_file.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def detect_and_visualize_follicles(image_path):
    results = yolo_model.predict(image_path, conf=0.1, iou=0.45)
    count = len(results[0].boxes) if results else 0
    img_with_boxes = results[0].plot()
    return count, img_with_boxes

# Streamlit UI
st.markdown("""
<style>
.landing-container {
    text-align: left;
    padding: 40px;
    background-color: #fff;
    border-radius: 16px;
    margin-bottom: 30px;
    box-shadow: 0 0 15px rgba(0,0,0,0.1);
}
h1 { color: #2c3e50; font-size: 48px; margin-bottom: 10px; }
h3 { color: #333; font-size: 20px; margin-bottom: 10px; }
.note {
    font-size: 18px; color: #2c3e50;
    background-color: #e0e0e0; padding: 10px;
    border-radius: 8px; display: inline-block;
    margin-top: 15px;
}
</style>
<div class='landing-container'>
    <h1>AI MEETS PCOS</h1>
    <h3>AI-Powered PCOS Detection Platform</h3>
    <div>Upload an ultrasound image to perform AI-based screening for <strong>Polycystic Ovary Syndrome (PCOS)</strong>.</div>
    <div class='note'><em>Note: For preliminary analysis only. Not a substitute for professional medical advice.</em></div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    patient_id = st.text_input("Enter Patient ID (Unique)", placeholder="e.g., P12345").strip()
    patient_name = st.text_input("Enter Patient Name", placeholder="e.g., Jane Doe").strip()
with col2:
    patient_age = st.number_input("Enter Patient Age", min_value=18, max_value=45, step=1)

prev_record = get_patient_record(patient_id) if patient_id and patient_name else None

if prev_record and prev_record[1].strip().lower() != patient_name.lower():
    st.warning(f"⚠️ Patient ID **{patient_id}** is assigned to **{prev_record[1]}**. Name mismatch — analysis blocked.")
else:
    uploaded_file = st.file_uploader("Upload an Ultrasound Image", type=["jpg","jpeg","png"])

    if uploaded_file and patient_id and patient_name:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Ultrasound Image", use_container_width=True)

        temp_path = os.path.join(tempfile.gettempdir(), f"{patient_id}_uploaded.png")
        img.save(temp_path)

        st.markdown("---")

        if 18 <= patient_age <= 45:
            if st.button("Analyze Image"):
                st.subheader("AI Diagnostic Result")

                follicle_count, img_with_boxes = detect_and_visualize_follicles(temp_path)
                st.write(f"Follicle count detected: **{follicle_count}**")

                st.image(img_with_boxes, caption="Detected follicles", use_container_width=True)

                processed_img = preprocess_image(img)
                prediction = model.predict(processed_img)
                result = "PCOS Detected" if prediction[0][0] > 0.5 else "No PCOS Detected"
                confidence = prediction[0][0] * 100

                update_patient_record(patient_id, patient_name, patient_age, result, confidence, follicle_count)

                st.success(f"**{result}** for **{patient_name}**, Age: **{int(patient_age)}**.")
                st.info(f"*Model Confidence: {confidence:.2f}%*")

                if prev_record:
                    st.markdown("---")
                    st.subheader("📊 Previous Diagnostic Result Found:")
                    st.write(f"**Last Diagnosis:** {prev_record[3]}")
                    try:
                        confidence_value = float(prev_record[4]) if prev_record[4] is not None else 0.0
                    except:
                        confidence_value = 0.0
                    st.write(f"**Confidence:** {confidence_value:.2f}%")
                    st.write(f"**Follicle Count:** {prev_record[5]}")
                    st.write(f"**Last Update:** {prev_record[6]}")

                    if prev_record[3] != result:
                        st.warning("⚠️ Diagnosis changed from last test. Consider consulting a medical professional.")
                    else:
                        st.success("✅ Diagnosis consistent with previous test.")
        else:
            st.warning("⚠️ Age must be between 18 and 45 to proceed.")
    else:
        if uploaded_file and (patient_id == "" or patient_name == ""):
            st.warning("⚠️ Please provide both Patient ID and Name.")

st.markdown("---")
st.markdown("<div style='text-align:center;'>© 2025 PCOS Detection AI | For Medical Research Use Only.</div>", unsafe_allow_html=True)



