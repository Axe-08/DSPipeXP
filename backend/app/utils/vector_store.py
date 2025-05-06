import numpy as np
import os
import json
import logging
from typing import List, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        """Initialize an empty vector store"""
        self.features_matrix = None
        self.song_ids = []
        
    def add_song(self, song_id: int, features: Dict[str, float]) -> None:
        """Add a song's features to the vector store"""
        try:
            # Convert dict to array in consistent order
            feature_array = np.array([
                float(features.get('danceability', 0.0)),
                float(features.get('energy', 0.0)),
                float(features.get('key', 0)),
                float(features.get('loudness', 0.0)),
                float(features.get('mode', 0)),
                float(features.get('speechiness', 0.0)),
                float(features.get('acousticness', 0.0)),
                float(features.get('instrumentalness', 0.0)),
                float(features.get('liveness', 0.0)),
                float(features.get('valence', 0.0)),
                float(features.get('tempo', 0.0)),
                float(features.get('duration_ms', 0.0))
            ], dtype=np.float32)
            
            if self.features_matrix is None:
                self.features_matrix = feature_array.reshape(1, -1)
                self.song_ids = [song_id]
            else:
                # Check if song already exists
                if song_id in self.song_ids:
                    idx = self.song_ids.index(song_id)
                    self.features_matrix[idx] = feature_array
                else:
                    self.features_matrix = np.vstack([self.features_matrix, feature_array])
                    self.song_ids.append(song_id)
                
            logger.debug(f"Added song {song_id} to vector store")
            
        except Exception as e:
            logger.error(f"Error adding song {song_id} to vector store: {e}")
            raise

    def find_similar_songs(self, features: Dict[str, float], k: int = 5) -> List[int]:
        """Find k most similar songs based on feature similarity"""
        try:
            if self.features_matrix is None or len(self.song_ids) == 0:
                return []

            # Convert query features to array in same order as add_song
            query_array = np.array([
                float(features.get('danceability', 0.0)),
                float(features.get('energy', 0.0)),
                float(features.get('key', 0)),
                float(features.get('loudness', 0.0)),
                float(features.get('mode', 0)),
                float(features.get('speechiness', 0.0)),
                float(features.get('acousticness', 0.0)),
                float(features.get('instrumentalness', 0.0)),
                float(features.get('liveness', 0.0)),
                float(features.get('valence', 0.0)),
                float(features.get('tempo', 0.0)),
                float(features.get('duration_ms', 0.0))
            ], dtype=np.float32).reshape(1, -1)

            # Calculate similarities
            similarities = cosine_similarity(query_array, self.features_matrix)[0]
            
            # Get indices of top k similar songs
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            
            # Return song IDs
            return [self.song_ids[i] for i in top_k_indices]

        except Exception as e:
            logger.error(f"Error finding similar songs: {e}")
            return []

    def clear(self) -> None:
        """Clear the vector store"""
        self.features_matrix = None
        self.song_ids = []
        logger.info("Vector store cleared") 