import streamlit as st
from PIL import Image
import numpy as np
import sqlite3
from datetime import datetime
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import os

# --- Page Configuration ---
st.set_page_config(page_title="AI MEETS PCOS | AI Diagnostic", layout="centered", page_icon="🩺")

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    html, body, .stApp, .main {
        background-color: #f0f0f0 !important;
        color: #2c3e50;
    }
    h1, h2, h3 {
        color: #2c3e50 !important;
        text-align: left !important;
    }
    .landing-container {
        text-align: left !important;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input {
        border-radius: 8px;
    }
    footer, header {visibility: hidden;}
    .block-container {padding-top: 0rem;}
    </style>
""", unsafe_allow_html=True)

# --- Load PCOS Classification Model ---
@st.cache_resource
def load_classification_model():
    return load_model("best_mobilenet_model.h5")

model = load_classification_model()

# --- Load YOLO Follicle Detection Model ---
@st.cache_resource
def load_yolo_model():
    weights_path = "/mnt/data/yolov8n.pt"  # Path where your uploaded weights are stored
    return YOLO(weights_path)

yolo_model = load_yolo_model()

# --- Initialize SQLite Database ---
def init_db():
    conn = sqlite3.connect('patients.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS patients (
                        patient_id TEXT PRIMARY KEY,
                        name TEXT,
                        age INTEGER,
                        last_prediction TEXT,
                        confidence REAL,
                        follicle_count INTEGER,
                        last_update TEXT
                    )''')
    conn.commit()
    conn.close()

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
    cursor.execute('''INSERT OR REPLACE INTO patients 
                      (patient_id, name, age, last_prediction, confidence, follicle_count, last_update) 
                      VALUES (?, ?, ?, ?, ?, ?, ?)''', 
                   (patient_id, name, age, prediction, confidence, follicle_count, now))
    conn.commit()
    conn.close()

init_db()

# --- Preprocess Uploaded Image for Classification ---
def preprocess_image(image_file):
    img = image_file.resize((224, 224))  
    img_array = np.array(img) / 255.0    
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array

# --- Count follicles in image using YOLO model ---
def count_follicles(image_path):
    results = yolo_model.predict(image_path, conf=0.25, iou=0.45)
    count = len(results[0].boxes) if results else 0
    return count

# --- Custom Title and Description ---
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap" rel="stylesheet">
    <style>
    html, body, .stApp {
        background-color: #f0f0f0;
        font-family: 'Cairo', sans-serif;
    }
    .landing-container {
        text-align: left !important;
        padding: 40px;
        background-color: #ffffff;
        border-radius: 16px;
        margin-bottom: 30px;
        box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #2c3e50;
        font-size: 48px;
        margin-bottom: 10px;
    }
    h3 {
        color: #333;
        font-size: 20px;
        margin-bottom: 10px;
    }
    .note {
        font-size: 18px;
        color: #2c3e50;
        background-color: #e0e0e0;
        padding: 10px;
        border-radius: 8px;
        display: inline-block;
        margin-top: 15px;
    }
    </style>
    <div class='landing-container'>
        <h1>AI MEETS PCOS</h1>
        <h3>Welcome to the <strong>AI-Powered PCOS Detection Platform</strong></h3>
        <div>
            Upload an ultrasound image to perform an initial AI-based screening for 
            <strong>Polycystic Ovary Syndrome (PCOS)</strong>.
        </div>
        <div class='note'>
            <em>Note: This tool is for preliminary analysis and does not replace professional medical advice.</em>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- Patient Information Input ---
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    patient_id = st.text_input("Enter Patient ID (Unique)", placeholder="e.g., P12345").strip()
    patient_name = st.text_input("Enter Patient Name", placeholder="e.g., Jane Doe").strip()
with col2:
    patient_age = st.number_input("Enter Patient Age", min_value=18, max_value=45, step=1)

prev_record = None
if patient_id and patient_name:
    prev_record = get_patient_record(patient_id)

if prev_record and prev_record[1].strip().lower() != patient_name.lower():
    st.warning(f"⚠️ Patient ID **{patient_id}** is already assigned to **{prev_record[1]}**. Name mismatch detected! Analysis is blocked.")
else:
    uploaded_file = st.file_uploader("Upload an Ultrasound Image", type=["jpg", "jpeg", "png"])

    if uploaded_file and patient_id and patient_name:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Ultrasound Image", use_container_width=True)
        
        # Save temporarily for YOLO prediction (YOLO expects a file path)
        temp_path = f"/tmp/{patient_id}_uploaded.png"
        img.save(temp_path)

        st.markdown("---")

        if 18 <= patient_age <= 45:
            if st.button("Analyze Image"):
                st.subheader("AI Diagnostic Result")

                # Count follicles via YOLO
                follicle_count = count_follicles(temp_path)
                st.write(f"Follicle count detected: **{follicle_count}**")

                # PCOS classification
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
                    except (ValueError, TypeError):
                        confidence_value = 0.0
                    st.write(f"**Confidence:** {confidence_value:.2f}%")
                    st.write(f"**Follicle Count:** {prev_record[5]}")
                    st.write(f"**Last Update:** {prev_record[6]}")

                    if prev_record[3] != result:
                        st.warning("⚠️ Diagnosis has changed from the last test. Consider medical consultation.")
                    else:
                        st.success("✅ Diagnosis remains consistent with previous test.")
        else:
            st.warning("⚠️ Age must be between 18 and 45 to proceed with the analysis.")
    else:
        if uploaded_file and (patient_id == "" or patient_name == ""):
            st.warning("⚠️ Please provide both Patient ID and Name to proceed with the analysis.")

# --- Footer ---
st.markdown("---")
st.markdown("<div style='text-align: center;'>© 2025 PCOS Detection AI | For Medical Research Use Only.</div>", unsafe_allow_html=True)

