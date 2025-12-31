import streamlit as st
import tempfile
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import whisper

from To_view_Analysis import analyze_audio_file, generate_rating

st.set_page_config(page_title="Voice Analyzer", layout="centered")
st.markdown('<h1 style="text-align:center">Cognitive Stress Analyser</h1>', unsafe_allow_html=True)

# -----------------------------
# LOAD WHISPER ONCE
# -----------------------------
@st.cache_resource
def load_whisper():
    return whisper.load_model("tiny")  # use "base" if you want better accuracy

model = load_whisper()

# -----------------------------
# CURVED METER
# -----------------------------
def curved_meter(rating):
    fig, ax = plt.subplots(figsize=(6,3))
    ax.set_aspect('equal')
    ax.axis('off')

    zones = [(0,3,'green'), (3,7,'yellow'), (7,10,'red')]
    for start, end, color in zones:
        theta = np.linspace(math.pi*(1-start/10), math.pi*(1-end/10), 100)
        ax.plot(np.cos(theta), np.sin(theta),
                linewidth=15, color=color, solid_capstyle='round')

    angle = math.pi*(1-rating/10)
    ax.plot([0, np.cos(angle)*0.9], [0, np.sin(angle)*0.9], color='red', linewidth=4)
    ax.scatter(0,0, color='black', s=80)

    for i in range(11):
        a = math.pi*(1-i/10)
        ax.text(1.1*np.cos(a), 1.1*np.sin(a), str(i), ha='center')

    ax.set_xlim(-1.2,1.2)
    ax.set_ylim(-0.2,1.2)
    st.pyplot(fig)

# -----------------------------
# UI
# -----------------------------
mode = st.radio("Choose input method", ["Upload Audio", "Record Voice"])
audio_path = None

# ---------- Upload ----------
if mode == "Upload Audio":
    uploaded = st.file_uploader("Upload audio", type=["wav","mp3","webm"])
    if uploaded:
        st.audio(uploaded)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded.read())
            audio_path = tmp.name

# ---------- Voice Input ----------
if mode == "Record Voice":
    audio = st.audio_input("Click and speak")
    if audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio.getbuffer())
            audio_path = tmp.name

# -----------------------------
# ANALYSIS
# -----------------------------
if audio_path:
    with st.spinner("Analyzing audio..."):
        df = analyze_audio_file(audio_path, model)
        rating = generate_rating(df)

    st.success("Analysis complete")
    curved_meter(rating)

    if rating <= 3:
        st.markdown('<h3 style="text-align:center">Normal Level</h3>', unsafe_allow_html=True)
    elif rating <= 6:
        st.markdown('<h3 style="text-align:center">Little bit stressed</h3>', unsafe_allow_html=True)
    else:
        st.markdown('<h3 style="text-align:center">Highly Stressed</h3>', unsafe_allow_html=True)

    st.dataframe(df)

    st.download_button(
        "⬇ Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        "analysis_report.csv",
        "text/csv"
    )

    os.remove(audio_path)


