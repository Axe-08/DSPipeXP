import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import librosa
import soundfile as sf
import tempfile
import subprocess
import os
from pathlib import Path
from .database import db_manager
from app.models.models import Song as SongModel, RecommendationResponse
import json
import faiss
import logging
from app.models.recommendation import Recommendation
from sqlalchemy.sql import text
from ..services.lyrics_service import lyrics_service

logger = logging.getLogger(__name__)

class RecommendationEngine:
    def __init__(self):
        self.scaler = StandardScaler()
        self.knn_model = None
        self._init_knn_model()
        
    def _init_knn_model(self, n_neighbors: int = 10):
        """Initialize KNN model for hybrid recommendations"""
        if db_manager.vector_store.features_matrix is not None:
            self.knn_model = NearestNeighbors(
                n_neighbors=n_neighbors,
                algorithm='auto',
                metric='euclidean'
            )
            self.knn_model.fit(db_manager.vector_store.features_matrix)
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using StandardScaler"""
        if features.shape[0] == 1:
            self.scaler.fit(features)
            return self.scaler.transform(features)
        return self.scaler.fit_transform(features)
    
    def _compute_similarity(self, query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query vector and matrix"""
        return cosine_similarity(query_vector.reshape(1, -1), matrix)[0]
    
    def _get_top_k_indices(self, similarities: np.ndarray, k: int, exclude_indices: List[int] = None) -> List[int]:
        """Get indices of top k similar items, excluding specified indices"""
        if exclude_indices:
            similarities[exclude_indices] = -1
        return similarities.argsort()[-k:][::-1]
    
    def _combine_features(self, audio_features: Dict, lyrics_features: Optional[Dict] = None) -> np.ndarray:
        """Combine audio and lyrics features into a single vector"""
        # Convert audio features to vector in consistent order
        feature_keys = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                       'speechiness', 'acousticness', 'instrumentalness', 
                       'liveness', 'valence', 'tempo', 'duration_ms']
        return np.array([float(audio_features.get(key, 0.0)) for key in feature_keys])
    
    def _load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file with proper handling for different formats"""
        file_path = Path(file_path)
        sf_compatible = ['.wav', '.flac', '.aiff', '.ogg']
        
        if file_path.suffix.lower() in sf_compatible:
            try:
                y, sr = sf.read(file_path)
                if y.ndim > 1:
                    y = y.mean(axis=1)
                return y.astype(np.float32), sr
            except Exception as e:
                print(f"SoundFile failed: {e}, falling back to ffmpeg conversion")
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            subprocess.run(
                ['ffmpeg', '-i', str(file_path), '-ar', '44100', '-ac', '1', '-y', temp_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            y, sr = sf.read(temp_path)
            return y.astype(np.float32), sr
        except Exception as e:
            print(f"FFmpeg conversion failed: {e}, falling back to librosa")
            y, sr = librosa.load(file_path, sr=None)
            return y, sr
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def extract_audio_features(self, file_path: str) -> Dict[str, float]:
        """Extract audio features from a music file"""
        try:
            y, sr = self._load_audio(file_path)
            features = {}
            
            # Basic features
            features['duration_ms'] = len(y) / sr * 1000
            
            # Tempo and rhythm
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = tempo
            
            # Energy and loudness
            rms = librosa.feature.rms(y=y)[0]
            features['energy'] = float(np.mean(rms) * 10)
            features['loudness'] = float(min(0, -60 + np.mean(rms) * 100))
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            
            # Derived features
            features['acousticness'] = float(1.0 - (np.mean(spectral_centroids) / (sr/2)))
            features['danceability'] = float(np.mean(spectral_contrast) * 0.5 + 0.5)
            
            # Voice and instrument detection
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['speechiness'] = float(np.mean(np.abs(mfccs[1:5])) * 0.3)
            
            # Zero crossing rate for noisiness
            zcr = librosa.feature.zero_crossing_rate(y)
            features['instrumentalness'] = float(1.0 - min(1.0, np.mean(zcr) * 10))
            
            # Live performance detection
            flatness = librosa.feature.spectral_flatness(y=y)
            features['liveness'] = float(np.mean(flatness) * 5)
            
            # Emotional content
            features['valence'] = 0.5
            if np.mean(rms) > 0.1:
                features['valence'] = float(min(1.0, np.mean(rms) * 5))
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {file_path}: {e}")
            return {
                'duration_ms': 0, 'tempo': 0, 'energy': 0, 'loudness': 0,
                'acousticness': 0, 'danceability': 0, 'speechiness': 0,
                'instrumentalness': 0, 'liveness': 0, 'valence': 0
            }
    
    async def recommend_by_features(
        self,
        features: Dict[str, float],
        feature_type: str,
        k: int = 5,
        exclude_songs: List[int] = None
    ) -> List[Tuple[SongModel, float]]:
        """Recommend songs based on specific feature type"""
        # Get feature matrix
        matrix = db_manager.vector_store.features_matrix
        if matrix is None:
            return []
            
        # Convert features dict to numpy array in consistent order
        if isinstance(features, dict):
            feature_keys = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                          'speechiness', 'acousticness', 'instrumentalness', 
                          'liveness', 'valence', 'tempo', 'duration_ms']
            query_vector = np.array([float(features.get(key, 0.0)) for key in feature_keys])
        else:
            query_vector = np.array(features)
            
        # Normalize features
        matrix_norm = self._normalize_features(matrix)
        query_vector_norm = self.scaler.transform(query_vector.reshape(1, -1))
        
        # Get recommendations using both KNN and cosine similarity
        results = []
        
        # KNN recommendations
        if self.knn_model is not None:
            distances, indices = self.knn_model.kneighbors(query_vector_norm)
            knn_scores = 1 - (distances[0] / distances[0].max())
            
            for idx, score in zip(indices[0], knn_scores):
                if exclude_songs and db_manager.vector_store.song_ids[idx] in exclude_songs:
                    continue
                results.append((idx, float(score), 'knn'))
        
        # Cosine similarity recommendations
        similarities = self._compute_similarity(query_vector_norm, matrix_norm)
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        for idx in top_k_indices:
            if exclude_songs and db_manager.vector_store.song_ids[idx] in exclude_songs:
                continue
            results.append((idx, float(similarities[idx]), 'cosine'))
        
        # Combine and sort results
        combined_results = []
        seen_indices = set()
        
        for idx, score, method in sorted(results, key=lambda x: x[1], reverse=True):
            if idx not in seen_indices:
                seen_indices.add(idx)
                song_id = db_manager.vector_store.song_ids[idx]
                song = await db_manager.get_song(song_id)
                if song:
                    # Get YouTube URL if not present
                    if not song.youtube_url and song.track_name and song.track_artist:
                        from ..api.endpoints.youtube import youtube_service
                        query = f"{song.track_name} {song.track_artist} official audio"
                        song.youtube_url = youtube_service.search(query)
                    combined_results.append((song, score))
                    if len(combined_results) >= k:
                        break
        
        return combined_results[:k]
    
    def recommend_by_mood(
        self,
        mood: str,
        k: int = 5,
        exclude_songs: List[int] = None
    ) -> List[Tuple[SongModel, float]]:
        """Recommend songs based on mood"""
        # Define mood feature vectors (these could be learned from data)
        mood_vectors = {
            "happy": {"valence": 0.8, "energy": 0.7, "danceability": 0.7},
            "sad": {"valence": 0.2, "energy": 0.3, "danceability": 0.3},
            "energetic": {"valence": 0.6, "energy": 0.9, "danceability": 0.8},
            "relaxed": {"valence": 0.5, "energy": 0.2, "danceability": 0.3},
            "angry": {"valence": 0.3, "energy": 0.8, "danceability": 0.4},
            "romantic": {"valence": 0.6, "energy": 0.4, "danceability": 0.5},
            "melancholic": {"valence": 0.3, "energy": 0.4, "danceability": 0.3},
            "party": {"valence": 0.8, "energy": 0.8, "danceability": 0.9}
        }
        
        if mood.lower() not in mood_vectors:
            raise ValueError(f"Unknown mood: {mood}")
            
        return self.recommend_by_features(
            features=mood_vectors[mood.lower()],
            feature_type="audio",
            k=k,
            exclude_songs=exclude_songs
        )
    
    async def recommend_similar_songs(
        self,
        song_id: int,
        k: int = 5,
        weights: Dict[str, float] = None
    ) -> List[RecommendationResponse]:
        """Get similar songs based on audio features and lyrics"""
        try:
            # Get the input song
            song = await db_manager.get_song(song_id)
            if not song:
                raise ValueError(f"Song with ID {song_id} not found")

            # Get audio features
            if song.audio_features:
                audio_features = json.loads(song.audio_features) if isinstance(song.audio_features, str) else song.audio_features
            else:
                audio_features = {}

            # Get or fetch lyrics features
            lyrics_features = None
            if song.lyrics and song.sentiment_features and song.topic_features:
                lyrics_features = {
                    'sentiment': json.loads(song.sentiment_features) if isinstance(song.sentiment_features, str) else song.sentiment_features,
                    'topics': json.loads(song.topic_features) if isinstance(song.topic_features, str) else song.topic_features
                }
            elif song.track_name and song.track_artist:
                # Fetch lyrics and analyze
                lyrics_data = await lyrics_service.get_lyrics(song.track_name, song.track_artist)
                if lyrics_data:
                    lyrics_features = {
                        'sentiment': lyrics_data['sentiment'],
                        'topics': lyrics_data['topics']
                    }
                    # Update song with lyrics data
                    await db_manager.update_song(song_id, {
                        'lyrics': lyrics_data['lyrics'],
                        'clean_lyrics': lyrics_data['clean_lyrics'],
                        'sentiment_features': lyrics_data['sentiment'],
                        'topic_features': lyrics_data['topics']
                    })

            # Combine features
            combined_features = self._combine_features(audio_features, lyrics_features)

            # Get recommendations
            similar_songs = await self.recommend_by_features(
                features=combined_features,
                feature_type='combined',
                k=k,
                exclude_songs=[song_id]
            )

            # Convert to RecommendationResponse objects
            recommendations = []
            for similar_song, score in similar_songs:
                # Get YouTube URL if not present
                if not similar_song.youtube_url and similar_song.track_name and similar_song.track_artist:
                    from ..api.endpoints.youtube import youtube_service
                    query = f"{similar_song.track_name} {similar_song.track_artist} official audio"
                    similar_song.youtube_url = youtube_service.search(query)

                recommendation = {
                    'id': similar_song.id,
                    'track_name': similar_song.track_name,
                    'track_artist': similar_song.track_artist,
                    'track_album_name': similar_song.track_album_name,
                    'playlist_genre': similar_song.playlist_genre,
                    'similarity_score': score,
                    'component_scores': {
                        'audio': score * 0.6,
                        'lyrics': score * 0.4 if lyrics_features else 0
                    },
                    'external_links': {'youtube': similar_song.youtube_url} if similar_song.youtube_url else None,
                    'added_date': str(similar_song.added_date)
                }
                recommendations.append(recommendation)

            return recommendations

        except Exception as e:
            logger.error(f"Error getting similar songs: {e}")
            raise

class MusicRecommender:
    def __init__(self):
        self.engine = RecommendationEngine()
        self.default_weights = {
            'audio': 0.6,
            'lyrics': 0.4
        }

    async def get_recommendations(self, song_id: int, limit: int = 10) -> Recommendation:
        """Get recommendations for a song"""
        try:
            # Get song from database
            song = await db_manager.get_song(song_id)
            if not song:
                raise ValueError(f"Song with ID {song_id} not found")

            # Get recommendations
            recommendations = await self.engine.recommend_similar_songs(
                song_id=song_id,
                k=limit,
                weights=self.default_weights
            )

            # Convert song to dict
            song_dict = {
                'id': song.id,
                'track_name': song.track_name,
                'track_artist': song.track_artist,
                'track_album_name': song.track_album_name,
                'playlist_genre': song.playlist_genre,
                'youtube_url': song.youtube_url,
                'audio_path': song.audio_path,
                'is_original': song.is_original,
                'added_date': str(song.added_date)
            }

            return Recommendation(
                input_song=song_dict,
                recommendations=recommendations,
                similarity_scores=[r['similarity_score'] for r in recommendations]
            )

        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            raise

    async def get_recommendations_by_features(self, features: Dict, limit: int = 10) -> Recommendation:
        """Get recommendations based on audio features"""
        try:
            recommendations = await self.engine.recommend_by_features(
                features=features,
                feature_type='audio',
                k=limit
            )

            # Convert to RecommendationResponse objects
            recommendation_responses = []
            for song, score in recommendations:
                # Get YouTube URL if not present
                if not song.youtube_url and song.track_name and song.track_artist:
                    from ..api.endpoints.youtube import youtube_service
                    query = f"{song.track_name} {song.track_artist} official audio"
                    song.youtube_url = youtube_service.search(query)

                recommendation_responses.append(
                    RecommendationResponse(
                        track_name=song.track_name,
                        track_artist=song.track_artist,
                        track_album_name=song.track_album_name,
                        playlist_genre=song.playlist_genre,
                        similarity_score=score,
                        component_scores={'audio': score},
                        external_links={'youtube': song.youtube_url} if song.youtube_url else None
                    )
                )

            return Recommendation(
                input_features=features,
                recommendations=recommendation_responses
            )

        except Exception as e:
            logger.error(f"Error getting recommendations by features: {e}")
            raise

    def add_songs(self, songs: List[SongModel]):
        """Add songs to the recommendation engine"""
        # Extract features and add to vector store
        for song in songs:
            if song.audio_features:
                features = json.loads(song.audio_features) if isinstance(song.audio_features, str) else song.audio_features
                db_manager.vector_store.add_song(song.id, features)

# Create global instance
recommender = MusicRecommender()
