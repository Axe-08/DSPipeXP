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

    # Approximating valence based 
    features['valence'] = 0.5  # 
    if np.mean(rms) > 0.1:  # If 
        features['valence'] = min(1.0, np.mean(rms) * 5)  # Higher energy often correlates with positive valence
        
    features['loudness'] = min(0, -60 + np.mean(rms) * 100)  # Scale to negative dB values
        
    features['mode'] = 1 if np.mean(chroma[0]) > np.mean(chroma[3]) else 0  # Very rough approximation
        
    # Time signature (default to 4)
    features['time_signature'] = 4

    return features

def knn(scaled_features, n_neighbours = 5):
    knn_model = NearestNeighbors(n_neighbors = n_neighbours, algorithm = 'auto', metric = 'euclidean')
    knn_model.fit(scaled_features)
    
    return knn_model
    
def kmeans(scaled_features, n_clusters=10):
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_model.fit(scaled_features)
    
    return kmeans_model

def knn_recommend(df, song_features, features, scaled_features, n_recommendations=5):
    feature_values = []
    for f in features:
        value = song_features.get(f, 0)
        if hasattr(value, '__len__') and not isinstance(value, (str, bytes)):
            value = float(np.mean(value)) 
        feature_values.append(value)
    
    feature_df = pd.DataFrame([feature_values], columns=features)
    scaled_song_features = scaler.transform(feature_df)
    
    similarity_scores = cosine_similarity(scaled_song_features, scaled_features)[0]
    
    similarity_df = pd.DataFrame({
        'index': range(len(similarity_scores)),
        'similarity_score': similarity_scores
    })
    
    similarity_df = similarity_df.sort_values('similarity_score', ascending=False)
    
    start_idx = 1 if similarity_df.iloc[0]['similarity_score'] > 0.999 else 0
    top_indices = similarity_df.iloc[start_idx:start_idx+n_recommendations]['index'].values
    
    recommendations = df.iloc[top_indices].copy()
    recommendations['similarity_score'] = similarity_scores[top_indices]
    
    final_recommendations = recommendations[['track_name', 'artists', 'track_genre', 'popularity', 'similarity_score']]
    
    return final_recommendations, similarity_scores[top_indices]

def kmeans_recommend(df, song_features, features, scaled_features, n_recommendations=5, n_clusters=20):
    feature_values = []
    for f in features:
        value = song_features.get(f, 0)
        if hasattr(value, '__len__') and not isinstance(value, (str, bytes)):
            value = float(np.mean(value))
        feature_values.append(value)
    
    feature_df = pd.DataFrame([feature_values], columns=features)
    scaled_song_features = scaler.transform(feature_df)
    
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans_model.fit_predict(scaled_features)
    
    temp_df = df.copy()
    temp_df['cluster'] = clusters
    
    cluster = kmeans_model.predict(scaled_song_features)[0]
    
    cluster_songs = temp_df[temp_df['cluster'] == cluster].copy()
    
    if len(cluster_songs) <= 1:  
        print(f"Cluster {cluster} has too few songs. Finding nearby clusters...")
        
        cluster_distances = []
        for i, center in enumerate(kmeans_model.cluster_centers_):
            if i != cluster:  
                dist = np.linalg.norm(scaled_song_features - center.reshape(1, -1))
                cluster_distances.append((i, dist))
        
        cluster_distances.sort(key=lambda x: x[1])
        nearest_cluster = cluster_distances[0][0]
        
        print(f"Using nearest cluster {nearest_cluster} instead")
        cluster_songs = temp_df[temp_df['cluster'] == nearest_cluster].copy()
    
    similarities = cosine_similarity(scaled_song_features, 
                                    scaled_features[cluster_songs.index])[0]
    
    cluster_songs['similarity_score'] = similarities
    
    cluster_songs = cluster_songs[cluster_songs['similarity_score'] < 0.999]
    
    recommendations = cluster_songs.sort_values('similarity_score', ascending=False)
    recommendations = recommendations[['track_name', 'artists', 'track_genre', 'popularity', 'similarity_score']]
    
    return recommendations.head(n_recommendations)
    
def cosine_recommend(df, song_features, features, scaled_features, n_recommendations=5):
    feature_values = []
    for f in features:
        value = song_features.get(f, 0)
        if hasattr(value, '__len__') and not isinstance(value, (str, bytes)):
            value = float(np.mean(value))
        feature_values.append(value)
    
    feature_df = pd.DataFrame([feature_values], columns=features)
    scaled_song_features = scaler.transform(feature_df)
   
    similarity_scores = cosine_similarity(scaled_song_features, scaled_features)[0]
    
    sorted_indices = np.argsort(-similarity_scores)
    
    if similarity_scores[sorted_indices[0]] > 0.999:
        top_indices = sorted_indices[1:n_recommendations+1]
    else:
        top_indices = sorted_indices[:n_recommendations]
    
    recommendations = df.iloc[top_indices][['track_name', 'artists', 'track_genre', 'popularity']]
    scores_to_return = similarity_scores[top_indices]
    
    recommendations = recommendations.copy()
    recommendations['similarity_score'] = scores_to_return
    recommendations = recommendations.sort_values('similarity_score', ascending=False)
    
    return recommendations, scores_to_return    
    
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
# df = add_to_dataset(df, song_path)  

print(df.shape)

features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
print((df.columns))

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

