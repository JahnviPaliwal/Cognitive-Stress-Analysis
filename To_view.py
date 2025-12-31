import streamlit as st
import tempfile
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import av
import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
from To_view_Analysis import analyze_audio_file, generate_rating

st.set_page_config(page_title="Voice Analyzer", layout="centered")

st.markdown('<h1 style="text-align:center">Cognitive Stress Analyser</h1>', unsafe_allow_html=True)

# -----------------------------
# LIVE AUDIO RECORDER
# -----------------------------

class AudioRecorder(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame):
        audio = frame.to_ndarray()
        self.frames.append(audio)
        return frame

# -----------------------------
# CURVED METER
# -----------------------------

def curved_meter(value):
    """
    2D semicircle meter with color zones and needle.
    rating: 0-10
    """
    fig, ax = plt.subplots(figsize=(6,3), facecolor="none")  # transparent figure
    ax.set_aspect('equal')
    ax.axis('off')

    # Create color zones: 0-3 green, 4-7 yellow, 8-10 red
    zones = [(0,3,'green'), (3,7,'yellow'), (7,10,'red')]

    for start, end, color in zones:
        theta = np.linspace(math.pi*(1-start/10), math.pi*(1-end/10), 100)
        x = np.cos(theta)
        y = np.sin(theta)
        ax.plot(x, y, linewidth=15, color=color, solid_capstyle='round')

    # Needle
    needle_angle = math.pi*(1-rating/10)
    ax.plot([0, np.cos(needle_angle)*0.9],
            [0, np.sin(needle_angle)*0.9],
            color='red', linewidth=4)

    # Center point
    ax.scatter(0,0, color='black', s=100)

    # Tick labels
    for i in range(0, 11):
        angle = math.pi*(1-i/10)
        ax.text(1.1*np.cos(angle), 1.1*np.sin(angle), str(i),
                ha='center', va='center', fontsize=10)

    ax.set_xlim(-1.2,1.2)
    ax.set_ylim(-0.2,1.2)

    st.pyplot(fig, transparent=True)

# -----------------------------
# UI
# -----------------------------

mode = st.radio("Choose input method", ["Upload Audio", "Record Live"])
audio_path = None

# ---------- Upload ----------
if mode == "Upload Audio":
    uploaded_file = st.file_uploader(
        "Upload audio file", type=["wav", "mp3", "webm"]
    )
    if uploaded_file:
        st.audio(uploaded_file)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            audio_path = tmp.name

# ---------- Live Recording ----------
if mode == "Record Live":
    st.info(
        "- Click START → Speak\n"
        "- Click Analyse to get results\n"
        "- Download CSV report\n"
        "- Click STOP for restart"
    )


    ctx = webrtc_streamer(
        key="voice-recorder",
        audio_processor_factory=AudioRecorder,
        media_stream_constraints={"audio": True, "video": False},
    )

    if st.button("Analyse") and ctx.audio_processor:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            audio_data = np.concatenate(ctx.audio_processor.frames, axis=1).T
            sf.write(tmp.name, audio_data, 44100)
            audio_path = tmp.name

# -----------------------------
# ANALYSIS
# -----------------------------

if audio_path:
    with st.spinner("Analyzing audio..."):
        df = analyze_audio_file(audio_path)
        rating = generate_rating(df)

    st.success("Analysis complete")
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📄 Analysis Report")

    curved_meter(rating)
    print(rating)
    if rating in [0, 1, 2, 3]:
        st.markdown('<h3 style="text-align:center">Normal Level</h3>', unsafe_allow_html=True)
    if rating in [4, 5, 6]:
        st.markdown('<h3 style="text-align:center">Stressed</h3>', unsafe_allow_html=True)
    if rating in [7, 8, 9, 10]:
        st.markdown('<h3 style="text-align:center">Highly Stressed</h3>', unsafe_allow_html=True)

    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download CSV",
        csv,
        "analysis_report.csv",
        "text/csv",
    )

    # st.subheader("📊 Performance Meter")
    # curved_meter(rating)

    os.remove(audio_path)
