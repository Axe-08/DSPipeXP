import pandas as pd
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    features = {}
    
    features['duration_ms'] = len(y) / sr * 1000
    
    tempo, _ = librosa.beat.beat_track(y = y, sr = sr)
    features['tempo'] = tempo
    features['tempo'] = sum([f for f in features['tempo']]) / len([f for f in features['tempo']])
    rms = librosa.feature.rms(y=y)[0]
    features['energy'] = np.mean(rms) * 10
    
    # Spectral centroid for brightness/energy
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['acousticness'] = 1.0 - (np.mean(spectral_centroids) / (sr/2))  # Inverse of brightness

    # Chroma features for key detection
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features['key'] = np.argmax(np.mean(chroma, axis=1))
        
    # Spectral contrast for energy distribution
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features['danceability'] = np.mean(contrast) * 0.5 + 0.5  # Scale to 0-1
        
    # MFCC for speech content
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    features['speechiness'] = np.mean(np.abs(mfccs[1:5])) * 0.3  # Scale to match dataset
        
    # Zero crossing rate for noisiness
    zcr = librosa.feature.zero_crossing_rate(y)
    features['instrumentalness'] = 1.0 - min(1.0, np.mean(zcr) * 10)  # Inverse of ZCR, scaled
        
    # Spectral flatness for liveness approximation
    flatness = librosa.feature.spectral_flatness(y=y)
    features['liveness'] = np.mean(flatness) * 5  # Scale to match dataset

    # Approximating valence based on spectral statistics
    features['valence'] = 0.5  # Default middle value
    if np.mean(rms) > 0.1:  # If energy is high
        features['valence'] = min(1.0, np.mean(rms) * 5)  # Higher energy often correlates with positive valence
        
    # Loudness approximation (dB scale)
    features['loudness'] = min(0, -60 + np.mean(rms) * 100)  # Scale to negative dB values
        
    # Mode (major/minor)
    features['mode'] = 1 if np.mean(chroma[0]) > np.mean(chroma[3]) else 0  # Very rough approximation
        
    # Time signature (default to 4)
    features['time_signature'] = 4

    return features

def knn(scaled_features, n_neighbours = 5):
    knn_model = NearestNeighbors(n_neighbors = n_neighbours, algorithm = 'auto', metric = 'euclidean')
    knn_model.fit(scaled_features)
    
    return knn_model
    
def kmeans(scaled_features, n_clusters = 10):
    kmeans_model = KMeans(n_clusters = n_clusters, random_state = 42)
    df['cluster'] = kmeans_model.fit_predict(scaled_features)
    
    return kmeans_model

def knn_recommend(df, song_features, features, scaled_features, n_recommendations = 5):
    knn_model = knn(scaled_features, n_recommendations)
    
    feature_values = []
    for f in features:
        value = song_features.get(f, 0)
        if hasattr(value, '__len__') and not isinstance(value, (str, bytes)):
            value = float(np.mean(value)) 
        feature_values.append(value)
    
    feature_array = np.array(feature_values).reshape(1, -1)
    scaled_song_features = scaler.transform(feature_array)
    # feature_array = np.array([song_features.get(f, 0) for f in features]).reshape(1, -1)
    # scaled_song_features = scaler.transform(feature_array)
    
    dist, ind = knn_model.kneighbors(scaled_song_features)
    
    recommendations = df.iloc[ind[0]][['track_name', 'artists', 'track_genre', 'popularity']]
    
    return recommendations, dist[0]

def kmeans_recommend(df, song_features, features, scaled_features):
    kmeans_model = kmeans(scaled_features)
    
    feature_values = []
    for f in features:
        value = song_features.get(f, 0)
        if hasattr(value, '__len__') and not isinstance(value, (str, bytes)):
            value = float(np.mean(value))
        feature_values.append(value)
    
    feature_array = np.array(feature_values).reshape(1, -1)
    scaled_song_features = scaler.transform(feature_array)
    
    # feature_array = np.array([song_features.get(f, 0) for f in features]).reshape(1, -1)
    # scaled_song_features = scaler.transform(feature_array)
    
    cluster = kmeans_model.predict(scaled_song_features)[0]
    
    recommendations = df[df['cluster'] == cluster][['track_name', 'artists', 'track_genre', 'popularity']]
    
    return recommendations.sample(min(5, len(recommendations)))    
    
def cosine_recommend(df, song_features, features, scaled_features, n_recommendations = 5):
    # feature_array = np.array([song_features.get(f, 0) for f in features]).reshape(1, -1)
    # scaled_song_features = scaler.transform(feature_array)
   
    feature_values = []
    for f in features:
        value = song_features.get(f, 0)
        if hasattr(value, '__len__') and not isinstance(value, (str, bytes)):
            value = float(np.mean(value))
        feature_values.append(value)
    
    feature_array = np.array(feature_values).reshape(1, -1)
    scaled_song_features = scaler.transform(feature_array)
   
    similarity_scores = cosine_similarity(scaled_song_features, scaled_features)[0]
    top_ind = similarity_scores.argsort()[-n_recommendations:][::-1]
    
    recommendations = df.iloc[top_ind][['track_name', 'artists', 'track_genre', 'popularity']]
    
    return recommendations, similarity_scores[top_ind]
    
def add_to_dataset(df, audio_path, popularity = 50, explicit = False):
    features = extract_features(audio_path)
    
    new_song = {
        'Unnamed: 0': float(df.shape[0]),
        'track_id': '2hETkH7cOfqmz3LqZDHZf6',
        'artists': 'pcrc',
        'album_name': 'beta',
        'track_name': audio_path.strip('.mp3'),
        'popularity': popularity,
        'explicit': explicit,
        'danceability': features['danceability'],
        'energy': features['energy'],
        'key': features['key'],
        'loudness': features['loudness'],
        'mode': features['mode'],
        'speechiness': features['speechiness'],
        'acousticness': features['acousticness'],
        'instrumentalness': features['instrumentalness'],
        'liveness': features['liveness'],
        'valence': features['valence'],
        'tempo': features['tempo'],
        'duration_ms': features['duration_ms'],
        'track_genre': 'indian-indie'   ,
        'time_signature': features['time_signature']     
    }    
    
    df = pd.concat([df, pd.DataFrame([new_song])], ignore_index = True)
    df.to_csv('spotify_dataset/dataset.csv', index = False)
    return df

dataset_path = 'spotify_dataset/dataset.csv'
song_path = 'Foolmuse.mp3'

df = pd.read_csv(dataset_path)
df = add_to_dataset(df, song_path)

print(df.shape)

features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
print((df.columns))

df = df.drop_duplicates(subset=['track_name', 'artists'])
df.to_csv('spotify_dataset/dataset.csv', index = False)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])
song_features = extract_features(song_path)

knn_rec, dist = knn_recommend(df, song_features, features, scaled_features, 5)
print("KNN Recommendations:")
print(knn_rec)

kmeans_rec = kmeans_recommend(df, song_features, features, scaled_features)
print("\nK-means cluster recommendations:")
print(kmeans_rec)

cosine_rec, scores = cosine_recommend(df, song_features, features, scaled_features, 5)
print("\nCosine similarity recommendations:")
print(cosine_rec)

