from typing import Dict, List, Optional, Tuple
import numpy as np
from annoy import AnnoyIndex
from app.models.song import Song
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        # Dimensions for different feature types
        self.word2vec_dim = settings.WORD2VEC_DIM
        self.audio_dim = settings.AUDIO_DIM
        self.sentiment_dim = settings.SENTIMENT_DIM
        self.topic_dim = settings.TOPIC_DIM

        # Initialize Annoy indices for different feature types
        self.word2vec_index: Optional[AnnoyIndex] = None
        self.audio_index: Optional[AnnoyIndex] = None
        self.sentiment_index: Optional[AnnoyIndex] = None
        self.topic_index: Optional[AnnoyIndex] = None

        # Store song IDs for mapping back results
        self.song_ids: List[int] = []
        self.id_to_index: Dict[int, int] = {}
        self.index_to_id: Dict[int, int] = {}

    def _init_annoy_index(self, dimension: int) -> AnnoyIndex:
        """Initialize an Annoy index with the specified dimension"""
        index = AnnoyIndex(dimension, 'angular')  # Using angular (cosine) distance
        index.set_seed(42)  # For reproducibility
        return index

    def _ensure_index(self, feature_type: str):
        """Ensure Annoy index is initialized for the given feature type"""
        if feature_type == 'word2vec' and self.word2vec_index is None:
            self.word2vec_index = self._init_annoy_index(self.word2vec_dim)
        elif feature_type == 'audio' and self.audio_index is None:
            self.audio_index = self._init_annoy_index(self.audio_dim)
        elif feature_type == 'sentiment' and self.sentiment_index is None:
            self.sentiment_index = self._init_annoy_index(self.sentiment_dim)
        elif feature_type == 'topic' and self.topic_index is None:
            self.topic_index = self._init_annoy_index(self.topic_dim)

    def build_indices(self, songs: List[Song]):
        """Load feature matrices and build Annoy indices from songs"""
        if not songs:
            logger.warning("No songs provided to build indices")
            return

        # Reset indices and mappings
        self.song_ids = []
        self.id_to_index = {}
        self.index_to_id = {}

        # Collect feature matrices
        word2vec_features = []
        audio_features = []
        sentiment_features = []
        topic_features = []

        for idx, song in enumerate(songs):
            self.song_ids.append(song.id)
            self.id_to_index[song.id] = idx
            self.index_to_id[idx] = song.id

            if song.word2vec_features:
                word2vec_features.append(song.word2vec_features)
            if song.audio_features:
                audio_features.append(song.audio_features)
            if song.sentiment_features:
                sentiment_features.append(song.sentiment_features)
            if song.topic_features:
                topic_features.append(song.topic_features)

        # Initialize and build Annoy indices
        if word2vec_features:
            self.word2vec_index = self._init_annoy_index(len(word2vec_features[0]))
            for i, features in enumerate(word2vec_features):
                self.word2vec_index.add_item(i, features)
            self.word2vec_index.build(10)  # 10 trees for better accuracy

        if audio_features:
            self.audio_index = self._init_annoy_index(len(audio_features[0]))
            for i, features in enumerate(audio_features):
                self.audio_index.add_item(i, features)
            self.audio_index.build(10)

        if sentiment_features:
            self.sentiment_index = self._init_annoy_index(len(sentiment_features[0]))
            for i, features in enumerate(sentiment_features):
                self.sentiment_index.add_item(i, features)
            self.sentiment_index.build(10)

        if topic_features:
            self.topic_index = self._init_annoy_index(len(topic_features[0]))
            for i, features in enumerate(topic_features):
                self.topic_index.add_item(i, features)
            self.topic_index.build(10)

        logger.info(f"Built indices for {len(songs)} songs")

    def add_song(self, song: Song):
        """Add a single song to the indices"""
        idx = len(self.song_ids)
        self.song_ids.append(song.id)
        self.id_to_index[song.id] = idx
        self.index_to_id[idx] = song.id

        # Add to Annoy indices
        if song.word2vec_features and self.word2vec_index:
            self.word2vec_index.add_item(idx, song.word2vec_features)
            self.word2vec_index.build(10)

        if song.audio_features and self.audio_index:
            self.audio_index.add_item(idx, song.audio_features)
            self.audio_index.build(10)

        if song.sentiment_features and self.sentiment_index:
            self.sentiment_index.add_item(idx, song.sentiment_features)
            self.sentiment_index.build(10)

        if song.topic_features and self.topic_index:
            self.topic_index.add_item(idx, song.topic_features)
            self.topic_index.build(10)

    def find_similar(self, query_features: np.ndarray, feature_type: str, k: int = 10) -> List[Tuple[int, float]]:
        """Find similar songs using Annoy"""
        if not query_features.any():
            return []

        # Get appropriate index
        index = None
        if feature_type == 'word2vec':
            index = self.word2vec_index
        elif feature_type == 'audio':
            index = self.audio_index
        elif feature_type == 'sentiment':
            index = self.sentiment_index
        elif feature_type == 'topic':
            index = self.topic_index

        if index is None:
            logger.warning(f"No index available for feature type: {feature_type}")
            return []

        # Search using Annoy
        indices, distances = index.get_nns_by_vector(query_features, k, include_distances=True)
        
        # Convert indices to song IDs and normalize distances
        results = [(self.index_to_id[idx], dist) for idx, dist in zip(indices, distances)]
        return results

# Create global instance
vector_store = VectorStore() 