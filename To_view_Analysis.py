import os
import numpy as np
import pandas as pd
import librosa
from pydub import AudioSegment
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity

# nltk.download("punkt", quiet=True)
import nltk

# Download punkt tokenizer at runtime if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# -----------------------------
# AUDIO → TEXT (WHISPER)
# -----------------------------
def audio_to_text(file_path, model):
    result = model.transcribe(file_path, language="en")
    return result["text"].strip()

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def extract_acoustic_features(audio_path, text):
    y, sr = librosa.load(audio_path)
    duration = librosa.get_duration(y=y, sr=sr)

    words = len(word_tokenize(text)) if text else 0
    speech_rate = words / (duration / 60 + 1e-6)

    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitches = pitches[pitches > 0]

    intervals = librosa.effects.split(y, top_db=20)
    pauses = [librosa.samples_to_time(i[1]-i[0], sr=sr) for i in intervals]

    return {
        "speech_rate": speech_rate,
        "pitch_mean": np.mean(pitches) if len(pitches) else 0,
        "pitch_std": np.std(pitches) if len(pitches) else 0,
        "avg_pause_duration": np.mean(pauses) if pauses else 0,
        "pause_frequency": len(pauses),
    }

def linguistic_features(text):
    if not text:
        return dict(hesitation_count=0, word_count=0,
                    avg_sentence_length=0, hesitation_ratio=0)

    hesitation = text.lower().count("uh") + text.lower().count("um")
    words = word_tokenize(text)
    sentences = sent_tokenize(text)

    return {
        "hesitation_count": hesitation,
        "word_count": len(words),
        "avg_sentence_length": sum(len(s.split()) for s in sentences) / len(sentences),
        "hesitation_ratio": hesitation / len(words)
    }

# -----------------------------
# PIPELINE
# -----------------------------
def analyze_audio_file(audio_path, model):
    text = audio_to_text(audio_path, model)
    acoustic = extract_acoustic_features(audio_path, text)
    linguistic = linguistic_features(text)

    df = pd.DataFrame([{**acoustic, **linguistic}])

    df_norm = (df - df.mean()) / df.std()
    df_norm = df_norm.fillna(0)

    if len(df) >= 2:
        df["cluster"] = KMeans(n_clusters=2, random_state=42).fit_predict(df_norm)
    else:
        df["cluster"] = 0
    df["anomaly_score"] = IsolationForest(random_state=42).fit_predict(df_norm)
    df["similarity_score"] = cosine_similarity(df_norm, df_norm.mean().values.reshape(1,-1))

    return df

# -----------------------------
# RATING
# -----------------------------
def generate_rating(df):
    x = df.iloc[0]

    def norm(v,l,h): return np.clip((v-l)/(h-l+1e-9),0,1)

    score = (
        1.5*norm(x["pitch_mean"],100,300) +
        1.5*norm(x["pitch_std"],5,50) +
        2.0*norm(x["hesitation_ratio"],0,0.2) +
        1.0*abs(norm(x["speech_rate"],0,300)-0.5)*2 +
        1.0*norm(x["avg_pause_duration"],0,2) +
        0.5*norm(x["pause_frequency"],0,10) +
        1.0*norm(x["anomaly_score"],-1,1)
    )

    return int(np.clip(score/8*10,0,10))


