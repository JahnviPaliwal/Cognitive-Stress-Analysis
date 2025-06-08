import librosa
import numpy as np
import pandas as pd
import speech_recognition as sr
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt', quiet=True)

def audio_to_text(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio)
        except Exception:
            return ""

def extract_acoustic_features(audio_path):
    y, sr = librosa.load(audio_path)
    duration = librosa.get_duration(y=y)
    words = len(word_tokenize(audio_to_text(audio_path)))
    speech_rate = words / (duration / 60)
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitch_mean = np.mean(pitches[pitches > 0])
    return {
        "Speech Rate (wpm)": round(speech_rate, 2),
        "Pitch Mean": round(pitch_mean, 2)
    }

def process_audio(path):
    data = extract_acoustic_features(path)
    df = pd.DataFrame([data])
    return df.to_csv(index=False)