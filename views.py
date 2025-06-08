import os
import tempfile
import csv
import io
from pydub import AudioSegment

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

import librosa
import numpy as np
import pandas as pd
import speech_recognition as sr
import nltk
from nltk.tokenize import word_tokenize

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt', quiet=True)


from django.shortcuts import render

def index(request):
    return render(request, 'analyzer/index.html')

def convert_to_wav(input_path, output_path):
    # pydub will use ffmpeg to convert webm/opus to wav
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format="wav")

def audio_to_text(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except Exception as e:
            print(f"Error: {e}")
            return ""

def extract_acoustic_features(audio_path):
    y, sr_ = librosa.load(audio_path)
    duration = librosa.get_duration(y=y)
    text = audio_to_text(audio_path)
    words = len(word_tokenize(text))
    speech_rate = words / (duration / 60 + 1e-6)  # avoid division by zero

    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr_)
    pitches_nonzero = pitches[pitches > 0]
    pitch_mean = np.mean(pitches_nonzero) if len(pitches_nonzero) > 0 else 0
    pitch_std = np.std(pitches_nonzero) if len(pitches_nonzero) > 0 else 0

    intervals = librosa.effects.split(y, top_db=20)
    pause_durations = [librosa.samples_to_time(i[1]-i[0], sr=sr_) for i in intervals]
    avg_pause_duration = np.mean(pause_durations) if pause_durations else 0

    return {
        'speech_rate': speech_rate,
        'pitch_mean': pitch_mean,
        'pitch_std': pitch_std,
        'avg_pause_duration': avg_pause_duration,
        'pause_frequency': len(pause_durations)
    }

def linguistic_features(text):
    hesitation_count = text.lower().count('uh') + text.lower().count('um')
    words = word_tokenize(text)
    word_count = len(words)
    sentences = nltk.sent_tokenize(text)
    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0

    return {
        'hesitation_count': hesitation_count,
        'word_count': word_count,
        'avg_sentence_length': avg_sentence_length,
        'hesitation_ratio': hesitation_count / word_count if word_count else 0
    }

def process_audio_sample(file_path):
    text = audio_to_text(file_path)
    acoustic = extract_acoustic_features(file_path)
    linguistic = linguistic_features(text)
    return {**acoustic, **linguistic}

def apply_unsupervised_learning(df):
    df_normalized = (df - df.mean()) / df.std()
    df_normalized = df_normalized.fillna(0)
    if df_normalized.shape[0] >= 2:
        kmeans = KMeans(n_clusters=2, random_state=42)
        df['cluster'] = kmeans.fit_predict(df_normalized)
    else:
        df['cluster'] = 0
    iso_forest = IsolationForest(contamination=0.1)
    df['anomaly_score'] = iso_forest.fit_predict(df_normalized)
    reference = df_normalized.mean().values.reshape(1, -1)
    df['similarity_score'] = cosine_similarity(df_normalized, reference)
    return df

@csrf_exempt  # Remove this if you use CSRF tokens in your AJAX/form
def analyze_audio(request):
    if request.method == 'POST' and request.FILES.get('audio'):
        # Save uploaded file to a temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp:
            for chunk in request.FILES['audio'].chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        wav_path = tmp_path.replace('.webm', '.wav')
        convert_to_wav(tmp_path, wav_path)

        text = audio_to_text(wav_path)

        # Process audio and extract features
        features = [process_audio_sample(wav_path)]
        df = pd.DataFrame(features)
        df = apply_unsupervised_learning(df)

        # Prepare CSV response
        output = io.StringIO()
        df.to_csv(output, index=False)
        csv_data = output.getvalue()
        os.unlink(tmp_path)
        os.unlink(wav_path)  # Clean up

        return HttpResponse(csv_data, content_type='text/csv')
    return HttpResponse('Invalid request', status=400)
