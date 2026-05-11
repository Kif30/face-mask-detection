import streamlit as st
import numpy as np
from PIL import Image
import os
import time

import tf_keras
from tf_keras.models import load_model
from tf_keras.preprocessing import image as keras_image

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="MaskScan AI",
    page_icon="😷",
    layout="centered"
)

# =========================================================
# CUSTOM CSS — DARK CYBER AESTHETIC
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;400;600;700&display=swap');

/* ---- GLOBAL ---- */
html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: #050A0F;
    color: #C8D8E8;
}

.stApp {
    background: #050A0F;
    background-image:
        radial-gradient(ellipse at 20% 10%, rgba(0,255,200,0.04) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 90%, rgba(0,120,255,0.05) 0%, transparent 50%);
}

/* ---- SCANLINE OVERLAY ---- */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,255,180,0.015) 2px,
        rgba(0,255,180,0.015) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* ---- HEADER ---- */
.cyber-header {
    text-align: center;
    padding: 2.5rem 0 1rem 0;
    position: relative;
}

.cyber-header h1 {
    font-family: 'Share Tech Mono', monospace;
    font-size: 3rem;
    letter-spacing: 0.2em;
    color: #00FFB4;
    text-shadow:
        0 0 10px rgba(0,255,180,0.8),
        0 0 30px rgba(0,255,180,0.4),
        0 0 60px rgba(0,255,180,0.2);
    margin: 0;
    animation: flicker 5s infinite;
}

@keyframes flicker {
    0%, 95%, 100% { opacity: 1; }
    96% { opacity: 0.8; }
    97% { opacity: 1; }
    98% { opacity: 0.6; }
    99% { opacity: 1; }
}

.cyber-header .subtitle {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #00A878;
    letter-spacing: 0.35em;
    margin-top: 0.4rem;
    text-transform: uppercase;
}

.cyber-header .divider {
    margin: 1.2rem auto 0 auto;
    width: 60%;
    height: 1px;
    background: linear-gradient(to right, transparent, #00FFB4, transparent);
    box-shadow: 0 0 8px rgba(0,255,180,0.5);
}

/* ---- STAT BOXES ---- */
.stat-row {
    display: flex;
    gap: 1rem;
    margin: 1.5rem 0;
    justify-content: center;
}

.stat-box {
    flex: 1;
    border: 1px solid rgba(0,255,180,0.2);
    background: rgba(0,255,180,0.03);
    padding: 0.9rem 1rem;
    position: relative;
    clip-path: polygon(0 0, calc(100% - 12px) 0, 100% 12px, 100% 100%, 12px 100%, 0 calc(100% - 12px));
}

.stat-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 12px; height: 12px;
    border-top: 1px solid #00FFB4;
    border-left: 1px solid #00FFB4;
}

.stat-box::after {
    content: '';
    position: absolute;
    bottom: 0; right: 0;
    width: 12px; height: 12px;
    border-bottom: 1px solid #00FFB4;
    border-right: 1px solid #00FFB4;
}

.stat-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    color: #00A878;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
}

.stat-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.3rem;
    color: #00FFB4;
    text-shadow: 0 0 8px rgba(0,255,180,0.5);
}

/* ---- UPLOAD ZONE ---- */
.upload-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    color: #00A878;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

[data-testid="stFileUploader"] {
    border: 1px solid rgba(0,255,180,0.25) !important;
    background: rgba(0,20,15,0.6) !important;
    border-radius: 0 !important;
    padding: 1rem !important;
    position: relative;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,255,180,0.6) !important;
    box-shadow: 0 0 20px rgba(0,255,180,0.1) !important;
}

/* ---- IMAGE DISPLAY ---- */
[data-testid="stImage"] {
    border: 1px solid rgba(0,255,180,0.3);
    box-shadow: 0 0 30px rgba(0,255,180,0.08), inset 0 0 30px rgba(0,0,0,0.5);
}

/* ---- BUTTON ---- */
.stButton > button {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: #050A0F !important;
    background: #00FFB4 !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0.75rem 2.5rem !important;
    width: 100% !important;
    clip-path: polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 10px 100%, 0 calc(100% - 10px)) !important;
    transition: all 0.2s ease !important;
    font-weight: 700 !important;
}

.stButton > button:hover {
    background: #00FFD5 !important;
    box-shadow: 0 0 25px rgba(0,255,180,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ---- RESULT PANELS ---- */
.result-panel {
    margin: 1.5rem 0;
    padding: 1.5rem 2rem;
    position: relative;
    animation: slideIn 0.4s ease;
}

@keyframes slideIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

.result-safe {
    border: 1px solid rgba(0,255,180,0.5);
    background: rgba(0,255,180,0.06);
    box-shadow: 0 0 40px rgba(0,255,180,0.1), inset 0 0 40px rgba(0,255,180,0.03);
}

.result-danger {
    border: 1px solid rgba(255,60,60,0.5);
    background: rgba(255,40,40,0.06);
    box-shadow: 0 0 40px rgba(255,60,60,0.1), inset 0 0 40px rgba(255,40,40,0.03);
}

.result-tag {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.result-safe .result-tag  { color: #00A878; }
.result-danger .result-tag { color: #FF5050; }

.result-verdict {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2.2rem;
    letter-spacing: 0.1em;
    line-height: 1;
    margin-bottom: 1rem;
}

.result-safe .result-verdict {
    color: #00FFB4;
    text-shadow: 0 0 20px rgba(0,255,180,0.6);
}

.result-danger .result-verdict {
    color: #FF4040;
    text-shadow: 0 0 20px rgba(255,60,60,0.6);
}

.confidence-bar-bg {
    height: 6px;
    background: rgba(255,255,255,0.05);
    border-radius: 0;
    overflow: hidden;
    margin-bottom: 0.5rem;
}

.confidence-bar-fill-safe {
    height: 100%;
    background: linear-gradient(to right, #00A878, #00FFB4);
    box-shadow: 0 0 10px rgba(0,255,180,0.5);
    transition: width 0.8s ease;
}

.confidence-bar-fill-danger {
    height: 100%;
    background: linear-gradient(to right, #A83000, #FF4040);
    box-shadow: 0 0 10px rgba(255,60,60,0.5);
    transition: width 0.8s ease;
}

.confidence-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    color: #556677;
    letter-spacing: 0.15em;
}

/* ---- SECTION HEADERS ---- */
.section-tag {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.3em;
    color: #00A878;
    text-transform: uppercase;
    border-left: 2px solid #00FFB4;
    padding-left: 0.6rem;
    margin-bottom: 0.8rem;
}

/* ---- FOOTER ---- */
.cyber-footer {
    text-align: center;
    padding: 2rem 0 1rem 0;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.3em;
    color: #1A3040;
    text-transform: uppercase;
}

/* ---- HIDE STREAMLIT DEFAULTS ---- */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 680px; }
[data-testid="stSpinner"] > div { border-top-color: #00FFB4 !important; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = "face_mask_detector.keras"

@st.cache_resource
def load_detection_model():
    return load_model(MODEL_PATH, compile=False)

model = load_detection_model()
IMG_SIZE = 128

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="cyber-header">
    <h1>MASKSCAN</h1>
    <div class="subtitle">// Neural Detection System v2.1 //</div>
    <div class="divider"></div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# STATS ROW
# =========================================================

st.markdown("""
<div class="stat-row">
    <div class="stat-box">
        <div class="stat-label">Model</div>
        <div class="stat-value">CNN</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">Input Size</div>
        <div class="stat-value">128×128</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">Classes</div>
        <div class="stat-value">2</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">Status</div>
        <div class="stat-value" style="color:#00FFB4">ONLINE</div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# FILE UPLOADER
# =========================================================

st.markdown('<div class="section-tag">// Input Feed</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop image here or click to upload",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_mask(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = keras_image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array, verbose=0)
    return prediction[0][0]

# =========================================================
# MAIN APP
# =========================================================

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    st.markdown('<div class="section-tag">// Visual Feed</div>', unsafe_allow_html=True)
    st.image(img, caption="", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("⬡  RUN ANALYSIS"):
        with st.spinner("Processing neural scan..."):
            time.sleep(0.4)  # slight dramatic pause
            prediction = predict_mask(img)

        st.markdown('<div class="section-tag">// Detection Result</div>', unsafe_allow_html=True)

        if prediction > 0.5:
            confidence = prediction * 100
            st.markdown(f"""
            <div class="result-panel result-danger">
                <div class="result-tag">▸ Alert — Violation Detected</div>
                <div class="result-verdict">NO MASK</div>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill-danger" style="width:{confidence:.1f}%"></div>
                </div>
                <div class="confidence-label">Confidence: {confidence:.2f}% &nbsp;|&nbsp; Raw score: {prediction:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            confidence = (1 - prediction) * 100
            st.markdown(f"""
            <div class="result-panel result-safe">
                <div class="result-tag">▸ Clearance — Compliant</div>
                <div class="result-verdict">MASK ON</div>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill-safe" style="width:{confidence:.1f}%"></div>
                </div>
                <div class="confidence-label">Confidence: {confidence:.2f}% &nbsp;|&nbsp; Raw score: {prediction:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="cyber-footer">
    ◈ &nbsp; Built with TensorFlow · CNN · Streamlit &nbsp; ◈
</div>
""", unsafe_allow_html=True)
