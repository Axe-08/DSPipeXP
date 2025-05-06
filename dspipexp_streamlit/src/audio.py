# Audio feature extraction utilities
import librosa
import numpy as np
import logging

# Configure logging
logger = logging.getLogger(__name__)

def reduce_to_spotify_features(features):
    selected_keys = [
        'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms'
    ]
    reduced = {k: features[k] for k in selected_keys if k in features}
    logger.info(f"Spotify-compatible features: {reduced}")
    return reduced

def extract_audio_features(audio_path, sample_rate=22050, duration=None):
    logger.info(f"Extracting audio features from {audio_path}")
    y, sr = librosa.load(audio_path, sr=sample_rate, duration=duration)
    logger.info(f"Loaded audio: {len(y)/sr:.2f} seconds, sample rate: {sr}")
    
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
    logger.info(f"Raw spectral centroids: mean={np.mean(spectral_centroids):.4f}, std={np.std(spectral_centroids):.4f}")
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    logger.info(f"Raw spectral bandwidth: mean={np.mean(spectral_bandwidth):.4f}, std={np.std(spectral_bandwidth):.4f}")
    
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    logger.info(f"Raw spectral contrast: mean={np.mean(spectral_contrast):.4f}, std={np.std(spectral_contrast):.4f}")
    
    # Voice and instrument detection
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features['mfcc'] = mfccs.mean(axis=1).tolist()
    features['speechiness'] = float(np.mean(np.abs(mfccs[1:5])) * 0.3)
    logger.info(f"MFCC stats: mean={np.mean(mfccs):.4f}, std={np.std(mfccs):.4f}")
    
    # Zero crossing rate for noisiness
    zcr = librosa.feature.zero_crossing_rate(y)
    features['instrumentalness'] = float(1.0 - min(1.0, np.mean(zcr) * 10))
    logger.info(f"Raw ZCR: mean={np.mean(zcr):.4f}, using for instrumentalness: {features['instrumentalness']:.4f}")
    
    # Live performance detection
    flatness = librosa.feature.spectral_flatness(y=y)
    features['liveness'] = float(np.mean(flatness) * 5)
    logger.info(f"Raw flatness: mean={np.mean(flatness):.4f}, using for liveness: {features['liveness']:.4f}")
    
    # Derived features
    features['acousticness'] = float(1.0 - (np.mean(spectral_centroids) / (sr/2)))
    logger.info(f"Derived acousticness: {features['acousticness']:.4f}")
    
    features['danceability'] = float(np.mean(spectral_contrast) * 0.5 + 0.5)
    logger.info(f"Derived danceability: {features['danceability']:.4f}")
    
    # Emotional content
    valence = 0.5
    if np.mean(rms) > 0.1:
        valence = float(min(1.0, np.mean(rms) * 5))
    features['valence'] = valence
    logger.info(f"Derived valence: {features['valence']:.4f}")
    
    # Key and mode (not robust, set to 0 for now)
    features['key'] = 0
    features['mode'] = 0
    
    # Duration
    features['duration_ms'] = float(len(y) / sr * 1000)
    logger.info(f"Duration: {features['duration_ms']/1000:.2f} seconds")
    
    # Normalize features to match database ranges before returning
    normalized_features = normalize_features(features)
    logger.info(f"Raw features: {features}")
    logger.info(f"Normalized features: {normalized_features}")
    
    # Reduce to standard features for compatibility
    return reduce_to_spotify_features(normalized_features)

def normalize_features(features):
    """Normalize feature values to match database value ranges based on Spotify-like scales"""
    normalized = features.copy()
    
    # Fix danceability which has way too high values (12+) to be in 0-1 range
    if 'danceability' in normalized and normalized['danceability'] > 1:
        # If coming from spectral contrast calculation
        normalized['danceability'] = min(1.0, max(0.0, normalized['danceability'] / 26.0))
    
    # Fix speechiness which has way too high values (12+) to be in 0-1 range
    if 'speechiness' in normalized and normalized['speechiness'] > 1:
        normalized['speechiness'] = min(0.95, max(0.0, normalized['speechiness'] / 30.0))
    
    # Fix energy which can sometimes be too high
    if 'energy' in normalized and normalized['energy'] > 1:
        normalized['energy'] = min(1.0, max(0.0, normalized['energy'] / 10.0))
    
    # Fix valence to ensure it's in 0-1 range
    if 'valence' in normalized:
        normalized['valence'] = min(1.0, max(0.0, normalized['valence']))
    
    # Fix loudness to be in a reasonable range (-60 to 0 dB)
    if 'loudness' in normalized and normalized['loudness'] < -60:
        normalized['loudness'] = max(-60, normalized['loudness'])
    
    # Fix tempo to be in a reasonable range
    if 'tempo' in normalized and normalized['tempo'] > 250:
        normalized['tempo'] = min(250, normalized['tempo'])
    
    return normalized 