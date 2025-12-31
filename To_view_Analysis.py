import os
import numpy as np
import pandas as pd

from pydub import AudioSegment
import librosa
import speech_recognition as sr
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity

# Download tokenizer
nltk.download("punkt", quiet=True)

# -----------------------------
# AUDIO PROCESSING
# -----------------------------

def convert_to_wav(input_path, output_path):
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format="wav")

def audio_to_text(file_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.8)
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            print("\n Text recived is :" ,text)
            print("sm shi")
            if not text.strip():  # empty string check
                return ""
            return text
    except sr.UnknownValueError as e1:
        # Could not understand audio
        print(f"{e1}❌ Speech was detected, but could not be understood.")
        return ""
    except sr.RequestError as e:
        # API request failed
        print(f"❌ Google API error: {e}")
        return ""

def extract_acoustic_features(audio_path):
    y, sr_ = librosa.load(audio_path)
    duration = librosa.get_duration(y=y)

    text = audio_to_text(audio_path)
    words = len(word_tokenize(text))
    speech_rate = words / (duration / 60 + 1e-6)

    pitches, _ = librosa.piptrack(y=y, sr=sr_)
    pitches = pitches[pitches > 0]

    intervals = librosa.effects.split(y, top_db=20)

    pause_durations = [
        librosa.samples_to_time(i[1] - i[0], sr=sr_)
        for i in intervals
    ]

    return {
        "speech_rate": speech_rate,
        "pitch_mean": np.mean(pitches) if len(pitches) else 0,
        "pitch_std": np.std(pitches) if len(pitches) else 0,
        "avg_pause_duration": np.mean(pause_durations) if pause_durations else 0,
        "pause_frequency": len(pause_durations),
    }

def linguistic_features(text):
    if not text or text.strip() == "":
        # Return default values for empty text
        return {
            'hesitation_count': 0,
            'word_count': 0,
            'avg_sentence_length': 0,
            'hesitation_ratio': 0
        }

    hesitation_count = text.lower().count('uh') + text.lower().count('um')
    words = word_tokenize(text)
    word_count = len(words)
    sentences = sent_tokenize(text)
    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0

    return {
        'hesitation_count': hesitation_count,
        'word_count': word_count,
        'avg_sentence_length': avg_sentence_length,
        'hesitation_ratio': hesitation_count / word_count if word_count else 0
    }


def process_audio(file_path):

    text = audio_to_text(file_path)
    # print("Raw text from audio:", repr(text))
    # print("Length:", len(text))
    acoustic = extract_acoustic_features(file_path)
    linguistic = linguistic_features(text)
    return {**acoustic, **linguistic}

def apply_ml(df):
    df_norm = (df - df.mean()) / df.std()
    df_norm = df_norm.fillna(0)

    if len(df) > 1:
        df["cluster"] = KMeans(n_clusters=2, random_state=42).fit_predict(df_norm)
    else:
        df["cluster"] = 0

    iso = IsolationForest(contamination=0.1, random_state=42)
    df["anomaly_score"] = iso.fit_predict(df_norm)

    ref = df_norm.mean().values.reshape(1, -1)
    df["similarity_score"] = cosine_similarity(df_norm, ref)

    return df

def analyze_audio_file(input_audio_path):
    wav_path = input_audio_path + ".wav"
    convert_to_wav(input_audio_path, wav_path)

    data = [process_audio(wav_path)]
    df = pd.DataFrame(data)
    df = apply_ml(df)

    os.remove(wav_path)
    return df

def generate_rating(df):
    """
    Produces a 0-10 stress/tension rating based on multiple speech features.
    Higher values indicate more stress-like patterns.
    """

    x = df.iloc[0]  # single row of features

    # Normalize each feature relative to a plausible range
    # You can tune these constants based on your dataset
    def norm(val, lo, hi):
        return np.clip((val - lo) / (hi - lo + 1e-9), 0, 1)

    pitch_mean_score = norm(x["pitch_mean"], 100, 300)
    pitch_std_score  = norm(x["pitch_std"], 5, 50)
    hes_ratio_score  = norm(x["hesitation_ratio"], 0, 0.2)

    # speech rate: too slow or too fast may be stress-like
    sr = x["speech_rate"]
    # assume normal comfortable speech ~ 110-160 wpm
    speech_rate_score = abs(norm(sr, 0, 300) - 0.5) * 2  # penalize extreme rates

    pause_duration_score = norm(x["avg_pause_duration"], 0, 2)
    pause_freq_score     = norm(x["pause_frequency"], 0, 10)

    anomaly_score = norm(x["anomaly_score"], -1, 1)

    # Weighted sum of features
    combined = (
        1.5 * pitch_mean_score +
        1.5 * pitch_std_score +
        2.0 * hes_ratio_score +
        1.0 * speech_rate_score +
        1.0 * pause_duration_score +
        0.5 * pause_freq_score +
        1.0 * anomaly_score
    )

    # Convert to normalized 0-10 rating
    rating = int(np.clip(combined / 8.0 * 10, 0, 10))
    return rating

