# Audio feature extraction utilities
import librosa
import numpy as np

def reduce_to_spotify_features(features):
    selected_keys = [
        'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms'
    ]
    return {k: features[k] for k in selected_keys if k in features}

def extract_audio_features(audio_path, sample_rate=22050, duration=None):
    y, sr = librosa.load(audio_path, sr=sample_rate, duration=duration)
    features = {}
    # Tempo and rhythm
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = float(tempo)
    # Energy and loudness
    rms = librosa.feature.rms(y=y)[0]
    features['energy'] = float(np.mean(rms) * 10)
    features['loudness'] = float(min(0, -60 + np.mean(rms) * 100))
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    # Voice and instrument detection
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features['mfcc'] = mfccs.mean(axis=1).tolist()
    features['speechiness'] = float(np.mean(np.abs(mfccs[1:5])) * 0.3)
    # Zero crossing rate for noisiness
    zcr = librosa.feature.zero_crossing_rate(y)
    features['instrumentalness'] = float(1.0 - min(1.0, np.mean(zcr) * 10))
    # Live performance detection
    flatness = librosa.feature.spectral_flatness(y=y)
    features['liveness'] = float(np.mean(flatness) * 5)
    # Derived features
    features['acousticness'] = float(1.0 - (np.mean(spectral_centroids) / (sr/2)))
    features['danceability'] = float(np.mean(spectral_contrast) * 0.5 + 0.5)
    # Emotional content
    valence = 0.5
    if np.mean(rms) > 0.1:
        valence = float(min(1.0, np.mean(rms) * 5))
    features['valence'] = valence
    # Key and mode (not robust, set to 0 for now)
    features['key'] = 0
    features['mode'] = 0
    # Duration
    features['duration_ms'] = float(len(y) / sr * 1000)
    # Reduce to standard features for compatibility
    return reduce_to_spotify_features(features) 