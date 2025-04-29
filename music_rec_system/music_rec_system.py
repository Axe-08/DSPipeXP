import pandas as pd
import numpy as np
import librosa
import os
import ast
import soundfile as sf
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import subprocess
import tempfile

class MusicRecommender:
    def __init__(self, dataset_path):
        """Initialize the music recommendation system."""
        self.dataset_path = dataset_path
        self.df = pd.read_csv(dataset_path, low_memory=False)
        
        self.df = self.df.drop_duplicates(subset=['track_name', 'artists'])
        
        self.features = [
            'danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 
            'valence', 'tempo'
        ]
        
        for feature in self.features:
            self.df[feature] = self.df[feature].apply(self._convert_to_float)
        
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.df[self.features])
        
        self.knn_model = self._build_knn_model()
    
    def _convert_to_float(self, value):
        """Convert string-formatted lists or other value types to float."""
        if isinstance(value, str):
            try:
                if value.startswith('[') and value.endswith(']'):
                    parsed = ast.literal_eval(value)
                    if isinstance(parsed, list):
                        return float(parsed[0])
                
                return float(value)
            except (ValueError, SyntaxError):
                return 0.0
        return float(value)
    
    def _build_knn_model(self, n_neighbors=10):
        """Build the K-nearest neighbors model."""
        knn_model = NearestNeighbors(
            n_neighbors=n_neighbors, 
            algorithm='auto', 
            metric='euclidean'
        )
        knn_model.fit(self.scaled_features)
        return knn_model
    
    def _load_audio(self, file_path):
        """
        Load audio file with proper handling for different formats.
        Convert unsupported formats to wav using ffmpeg when necessary.
        """
        file_path = Path(file_path)
        
        # Extensions that soundfile can handle directly
        sf_compatible = ['.wav', '.flac', '.aiff', '.ogg']
        
        if file_path.suffix.lower() in sf_compatible:
            try:
                # Try using soundfile directly
                y, sr = sf.read(file_path)
                # Convert to float32 and mono if needed
                if y.ndim > 1:
                    y = y.mean(axis=1)
                return y.astype(np.float32), sr
            except Exception as e:
                print(f"SoundFile failed: {e}, falling back to ffmpeg conversion")
        
        # For non-compatible formats, convert to temp WAV using ffmpeg
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Use ffmpeg to convert to WAV
            print(f"Converting {file_path} to temporary WAV using ffmpeg")
            subprocess.run(
                ['ffmpeg', '-i', str(file_path), '-ar', '44100', '-ac', '1', '-y', temp_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            
            # Read the converted file
            y, sr = sf.read(temp_path)
            return y.astype(np.float32), sr
        
        except Exception as e:
            print(f"FFmpeg conversion failed: {e}, falling back to librosa's default loader")
            # If all else fails, use librosa's default loader (which will use audioread)
            y, sr = librosa.load(file_path, sr=None)
            return y, sr
        
        finally:
            # Clean up temp file if it exists
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def extract_features(self, file_path):
        """Extract audio features from a music file using librosa."""
        try:
            # Use our custom audio loading function
            y, sr = self._load_audio(file_path)
            features = {}
            
            # Basic features
            features['duration_ms'] = len(y) / sr * 1000
            
            # Tempo and rhythm features
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = tempo
            
            # Energy and loudness
            rms = librosa.feature.rms(y=y)[0]
            features['energy'] = np.mean(rms) * 10
            features['loudness'] = min(0, -60 + np.mean(rms) * 100)  # Scale to negative dB values
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            
            # Derived features
            features['acousticness'] = 1.0 - (np.mean(spectral_centroids) / (sr/2))
            features['danceability'] = np.mean(spectral_contrast) * 0.5 + 0.5
            
            # Harmonics
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['key'] = np.argmax(np.mean(chroma, axis=1))
            features['mode'] = 1 if np.mean(chroma[0]) > np.mean(chroma[3]) else 0
            
            # Voice and instrument detection
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['speechiness'] = np.mean(np.abs(mfccs[1:5])) * 0.3
            
            # Zero crossing rate for noisiness
            zcr = librosa.feature.zero_crossing_rate(y)
            features['instrumentalness'] = 1.0 - min(1.0, np.mean(zcr) * 10)
            
            # Live performance detection
            flatness = librosa.feature.spectral_flatness(y=y)
            features['liveness'] = np.mean(flatness) * 5
            
            # Emotional content
            features['valence'] = 0.5  # Default value
            if np.mean(rms) > 0.1:
                features['valence'] = min(1.0, np.mean(rms) * 5)
            
            # Time signature (default to 4)
            features['time_signature'] = 4
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {file_path}: {e}")
            # Return default features in case of error
            return {feature: 0 for feature in self.features}
    
    def _prepare_song_features(self, song_features):
        """Prepare and scale song features for recommendation."""
        feature_values = []
        for feature in self.features:
            value = song_features.get(feature, 0)
            # Handle array-like features by taking their mean
            if hasattr(value, '__len__') and not isinstance(value, (str, bytes)):
                value = float(np.mean(value))
            feature_values.append(value)
        
        feature_array = np.array(feature_values).reshape(1, -1)
        scaled_song_features = self.scaler.transform(feature_array)
        return scaled_song_features
    
    def recommend_songs(self, song_path, n_recommendations=5, method='hybrid'):
        """
        Recommend songs based on the input song.
        
        Parameters:
        - song_path: Path to the song file
        - n_recommendations: Number of songs to recommend
        - method: 'knn', 'cosine', or 'hybrid' (default)
        
        Returns:
        - DataFrame with recommended songs
        """
        # Extract features from the input song
        song_features = self.extract_features(song_path)
        scaled_song_features = self._prepare_song_features(song_features)
        
        if method == 'knn':
            return self._knn_recommend(scaled_song_features, n_recommendations)
        elif method == 'cosine':
            return self._cosine_recommend(scaled_song_features, n_recommendations)
        else:  # hybrid
            return self._hybrid_recommend(scaled_song_features, n_recommendations)
    
    def _knn_recommend(self, scaled_song_features, n_recommendations):
        """KNN-based song recommendation."""
        distances, indices = self.knn_model.kneighbors(scaled_song_features)
        
        recommendations = self.df.iloc[indices[0]][
            ['track_name', 'artists', 'track_genre', 'popularity']
        ].copy()
        
        recommendations['distance'] = distances[0]
        return recommendations
    
    def _cosine_recommend(self, scaled_song_features, n_recommendations):
        """Cosine similarity-based song recommendation."""
        similarity_scores = cosine_similarity(scaled_song_features, self.scaled_features)[0]
        top_indices = similarity_scores.argsort()[-n_recommendations:][::-1]
        
        recommendations = self.df.iloc[top_indices][
            ['track_name', 'artists', 'track_genre', 'popularity']
        ].copy()
        
        recommendations['similarity'] = similarity_scores[top_indices]
        return recommendations
    
    def _hybrid_recommend(self, scaled_song_features, n_recommendations):
        """Hybrid recommendation combining KNN and cosine similarity."""
        # Get twice as many recommendations from each method
        knn_recs = self._knn_recommend(scaled_song_features, n_recommendations * 2)
        cosine_recs = self._cosine_recommend(scaled_song_features, n_recommendations * 2)
        
        # Normalize distances and similarities for fair comparison
        knn_recs['normalized_score'] = 1 - (knn_recs['distance'] / knn_recs['distance'].max())
        cosine_recs['normalized_score'] = cosine_recs['similarity']
        
        # Combine recommendations
        combined_recs = pd.concat([knn_recs, cosine_recs])
        combined_recs = combined_recs.drop_duplicates(subset=['track_name', 'artists'])
        
        # Sort by normalized score and take top recommendations
        return combined_recs.sort_values('normalized_score', ascending=False).head(n_recommendations)
    
    def add_song_to_dataset(self, audio_path, metadata=None):
        """
        Add a new song to the dataset with extracted features.
        
        Parameters:
        - audio_path: Path to the audio file
        - metadata: Dictionary with song metadata (optional)
        """
        features = self.extract_features(audio_path)
        
        # Default metadata
        if metadata is None:
            metadata = {
                'artists': Path(audio_path).stem,
                'album_name': 'Unknown',
                'track_name': Path(audio_path).stem,
                'popularity': 50,
                'explicit': False,
                'track_genre': 'Unknown'
            }
        
        # Create new song entry
        new_song = {
            'track_id': f"local_{len(self.df)}",
            'artists': metadata.get('artists', 'Unknown'),
            'album_name': metadata.get('album_name', 'Unknown'),
            'track_name': metadata.get('track_name', Path(audio_path).stem),
            'popularity': metadata.get('popularity', 50),
            'explicit': metadata.get('explicit', False),
            'danceability': features.get('danceability', 0),
            'energy': features.get('energy', 0),
            'key': features.get('key', 0),
            'loudness': features.get('loudness', 0),
            'mode': features.get('mode', 0),
            'speechiness': features.get('speechiness', 0),
            'acousticness': features.get('acousticness', 0),
            'instrumentalness': features.get('instrumentalness', 0),
            'liveness': features.get('liveness', 0),
            'valence': features.get('valence', 0),
            'tempo': features.get('tempo', 0),
            'duration_ms': features.get('duration_ms', 0),
            'track_genre': metadata.get('track_genre', 'Unknown'),
            'time_signature': features.get('time_signature', 4)
        }
        
        # Add to dataframe
        self.df = pd.concat([self.df, pd.DataFrame([new_song])], ignore_index=True)
        
        # Update scaled features
        self.scaled_features = self.scaler.fit_transform(self.df[self.features])
        
        # Update KNN model
        self.knn_model = self._build_knn_model()
        
        # Save updated dataset
        self.df.to_csv(self.dataset_path, index=False)
        
        return self.df.shape[0] - 1  # Return index of the newly added song

    def search_song(self, query):
        """
        Search for a song in the dataset by name or artist.
        
        Parameters:
        - query: Search query string
        
        Returns:
        - DataFrame with matching songs
        """
        # Case-insensitive search in track_name and artists
        mask = (
            self.df['track_name'].str.lower().str.contains(query.lower()) | 
            self.df['artists'].str.lower().str.contains(query.lower())
        )
        return self.df[mask][['track_name', 'artists', 'track_genre', 'popularity']]
    
    def recommend_from_index(self, index, n_recommendations=5, method='hybrid'):
        """
        Recommend songs based on a song index from the dataset.
        
        Parameters:
        - index: Index of the song in the dataset
        - n_recommendations: Number of songs to recommend
        - method: 'knn', 'cosine', or 'hybrid' (default)
        
        Returns:
        - DataFrame with recommended songs
        """
        # Get the features of the song at the given index
        song_features = self.scaled_features[index].reshape(1, -1)
        
        if method == 'knn':
            return self._knn_recommend(song_features, n_recommendations)
        elif method == 'cosine':
            return self._cosine_recommend(song_features, n_recommendations)
        else:  # hybrid
            return self._hybrid_recommend(song_features, n_recommendations)


# Example usage
if __name__ == "__main__":
    dataset_path = 'spotify_dataset/dataset.csv'
    song_path = 'Foolmuse.mp3'
    
    # Initialize recommender
    recommender = MusicRecommender(dataset_path)
    
    # Add song to dataset if not already present
    search_results = recommender.search_song(Path(song_path).stem)
    if search_results.empty:
        recommender.add_song_to_dataset(song_path)
    
    # Get recommendations using hybrid method
    recommendations = recommender.recommend_songs(song_path, n_recommendations=5, method='hybrid')
    print("\nHybrid Recommendations:")
    print(recommendations[['track_name', 'artists', 'track_genre', 'popularity', 'normalized_score']])
    
    # Alternative recommendation methods
    knn_recs = recommender.recommend_songs(song_path, method='knn')
    print("\nKNN Recommendations:")
    print(knn_recs[['track_name', 'artists', 'track_genre', 'popularity', 'distance']])
    
    cosine_recs = recommender.recommend_songs(song_path, method='cosine')
    print("\nCosine Similarity Recommendations:")
    print(cosine_recs[['track_name', 'artists', 'track_genre', 'popularity', 'similarity']])

