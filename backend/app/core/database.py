from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, Boolean, DateTime, text, MetaData
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool
import numpy as np
import json
import os
from typing import Dict, List, Optional, AsyncGenerator, Generator
from contextlib import contextmanager
import pandas as pd
import logging
from .config import settings
from datetime import datetime
from ..models.models import Song as SongModel
from ..utils.vector_store import VectorStore
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy import desc, select
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

logger = logging.getLogger(__name__)

# Create Base class for SQLAlchemy models
Base = declarative_base()

class Song(Base):
    __tablename__ = "songs"
    
    id = Column(Integer, primary_key=True, index=True)
    track_name = Column(String, nullable=False, index=True)
    track_artist = Column(String, nullable=False, index=True)
    track_album_name = Column(String, nullable=True)
    playlist_genre = Column(String, nullable=True, index=True)
    lyrics = Column(String, nullable=True)
    clean_lyrics = Column(String, nullable=True)
    word2vec_features = Column(JSON, nullable=True)
    audio_features = Column(JSON, nullable=True)
    sentiment_features = Column(JSON, nullable=True)
    topic_features = Column(JSON, nullable=True)
    youtube_url = Column(String, nullable=True)
    audio_path = Column(String, nullable=True)
    is_original = Column(Boolean, default=False)
    added_date = Column(DateTime, server_default=text('now()'))

class DatabaseManager:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = create_async_engine(database_url)
        self.SessionLocal = async_sessionmaker(self.engine, expire_on_commit=False)
        self.vector_store = VectorStore()

    async def get_db(self) -> AsyncGenerator[AsyncSession, None]:
        async with self.SessionLocal() as session:
            try:
                yield session
            finally:
                await session.close()

    async def initialize(self):
        """Initialize the database and vector store"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await self.populate_vector_store()

    async def populate_vector_store(self):
        """Populate the vector store with features from existing songs"""
        try:
            async with self.SessionLocal() as session:
                # Get all songs with audio features
                query = select(Song).where(Song.audio_features.isnot(None))
                result = await session.execute(query)
                songs = result.scalars().all()
                
                # Add each song's features to the vector store
                for song in songs:
                    features = json.loads(song.audio_features) if isinstance(song.audio_features, str) else song.audio_features
                    self.vector_store.add_song(song.id, features)
                
                logger.info(f"Populated vector store with {len(songs)} songs")
                
        except Exception as e:
            logger.error(f"Error populating vector store: {e}")
            raise

    async def get_songs(self, skip: int = 0, limit: int = 10) -> List[Song]:
        """Get a list of songs with pagination."""
        async with self.SessionLocal() as session:
            try:
                query = select(Song).order_by(desc(Song.added_date)).offset(skip).limit(limit)
                result = await session.execute(query)
                songs = result.scalars().all()
                
                # Parse JSON fields
                for song in songs:
                    if song.audio_features and isinstance(song.audio_features, str):
                        song.audio_features = json.loads(song.audio_features)
                    if song.word2vec_features and isinstance(song.word2vec_features, str):
                        song.word2vec_features = json.loads(song.word2vec_features)
                    if song.sentiment_features and isinstance(song.sentiment_features, str):
                        song.sentiment_features = json.loads(song.sentiment_features)
                    if song.topic_features and isinstance(song.topic_features, str):
                        song.topic_features = json.loads(song.topic_features)
                
                return songs
            except Exception as e:
                logger.error(f"Error getting songs: {e}")
                return []

    async def get_song(self, song_id: int) -> Optional[Song]:
        """Get a song by ID."""
        async with self.SessionLocal() as session:
            try:
                query = select(Song).where(Song.id == song_id)
                result = await session.execute(query)
                song = result.scalar_one_or_none()
                
                if song and song.audio_features:
                    song.audio_features = json.loads(song.audio_features) if isinstance(song.audio_features, str) else song.audio_features
                
                return song
            except Exception as e:
                logger.error(f"Error getting song {song_id}: {e}")
                return None

    async def get_similar_songs(self, song_id: int, limit: int = 5) -> List[Song]:
        """Get similar songs based on audio features."""
        try:
            song = await self.get_song(song_id)
            if not song or not song.audio_features:
                return []

            features = song.audio_features
            similar_ids = self.vector_store.find_similar_songs(features, k=limit)
            
            if not similar_ids:
                return []
            
            async with self.SessionLocal() as session:
                query = select(Song).where(Song.id.in_(similar_ids))
                result = await session.execute(query)
                return result.scalars().all()

        except Exception as e:
            logger.error(f"Error finding similar songs: {e}")
            return []

    async def search_songs(self, query: Optional[str] = None, artist: Optional[str] = None,
                    genre: Optional[str] = None, mood: Optional[str] = None,
                    skip: int = 0, limit: int = 10) -> List[Song]:
        """Search for songs with various filters."""
        async with self.SessionLocal() as session:
            try:
                stmt = select(Song)
                
                if query:
                    stmt = stmt.where(Song.track_name.ilike(f"%{query}%"))
                if artist:
                    stmt = stmt.where(Song.track_artist.ilike(f"%{artist}%"))
                if genre:
                    stmt = stmt.where(Song.playlist_genre.ilike(f"%{genre}%"))
                    
                stmt = stmt.offset(skip).limit(limit)
                
                result = await session.execute(stmt)
                return result.scalars().all()
            except Exception as e:
                logger.error(f"Error searching songs: {e}")
                return []

    async def get_active_file_paths(self) -> List[str]:
        """Get list of active file paths from database"""
        async with self.SessionLocal() as session:
            try:
                result = await session.execute(
                    text("SELECT audio_path FROM songs WHERE audio_path IS NOT NULL")
                )
                paths = result.fetchall()
                return [path[0] for path in paths if path[0]]
            except Exception as e:
                logger.error(f"Error getting active file paths: {e}")
                return []

    async def update_song(self, song_id: int, update_data: Dict) -> Optional[Song]:
        """Update a song by ID with the provided data."""
        def convert_numpy_types(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(i) for i in obj]
            return obj

        async with self.SessionLocal() as session:
            try:
                # Get the song
                query = select(Song).where(Song.id == song_id)
                result = await session.execute(query)
                song = result.scalar_one_or_none()
                
                if not song:
                    return None
                
                # Update fields
                for key, value in update_data.items():
                    if hasattr(song, key):
                        # Convert dict to JSON string for JSON fields
                        if key in ['audio_features', 'word2vec_features', 'sentiment_features', 'topic_features']:
                            if isinstance(value, dict):
                                value = json.dumps(convert_numpy_types(value))
                        setattr(song, key, value)
                
                await session.commit()
                await session.refresh(song)
                
                # Parse JSON fields in response
                if song.audio_features and isinstance(song.audio_features, str):
                    song.audio_features = json.loads(song.audio_features)
                if song.word2vec_features and isinstance(song.word2vec_features, str):
                    song.word2vec_features = json.loads(song.word2vec_features)
                if song.sentiment_features and isinstance(song.sentiment_features, str):
                    song.sentiment_features = json.loads(song.sentiment_features)
                if song.topic_features and isinstance(song.topic_features, str):
                    song.topic_features = json.loads(song.topic_features)
                
                return song
                
            except Exception as e:
                logger.error(f"Error updating song {song_id}: {e}")
                await session.rollback()
                return None

# Create global instance with production settings
db_manager = DatabaseManager("postgresql+asyncpg://postgres:postgres@db:5432/dspipexp")

# Dependency for FastAPI
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session"""
    async with db_manager.SessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
