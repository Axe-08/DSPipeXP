from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, Boolean, DateTime, text, MetaData, func, or_, case
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool
import numpy as np
import json
import os
from typing import Dict, List, Optional, AsyncGenerator, Generator, Any
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

class DatabaseError(Exception):
    pass

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
    youtube_url_updated_at = Column(DateTime, nullable=True)
    youtube_url_status = Column(String, nullable=True)  # 'valid', 'invalid', 'pending'
    youtube_url_error = Column(String, nullable=True)  # Store any error messages
    audio_path = Column(String, nullable=True)
    is_original = Column(Boolean, default=False)
    added_date = Column(DateTime, server_default=text('now()'))

def get_async_database_url() -> str:
    """Convert DATABASE_URL to async format if needed"""
    url = settings.DATABASE_URL
    if not url:
        raise ValueError("DATABASE_URL is not set in environment variables")
    
    # Ensure we have a valid postgresql URL
    if not url.startswith(('postgresql://', 'postgresql+asyncpg://')):
        raise ValueError(f"Invalid database URL format: {url}")
        
    if url.startswith('postgresql://'):
        url = url.replace('postgresql://', 'postgresql+asyncpg://', 1)
    return url

class DatabaseManager:
    _instance = None
    _initialized = False

    def __new__(cls, database_url: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, database_url: Optional[str] = None):
        if not self._initialized:
            self.database_url = database_url if database_url else get_async_database_url()
            self.engine = create_async_engine(self.database_url)
            self.SessionLocal = async_sessionmaker(self.engine, expire_on_commit=False)
            self.vector_store = VectorStore()
            self._initialized = True

    async def initialize(self):
        """Initialize the database and vector store"""
        if not hasattr(self, '_setup_done'):
            try:
                # Test connection first
                async with self.engine.begin() as conn:
                    result = await conn.execute(text("SELECT 1"))
                    value = result.scalar()  # Don't await this - it's already a scalar
                    if value != 1:
                        raise DatabaseError("Database connection test failed")
                    
                    # If connection works, create tables
                    await conn.run_sync(Base.metadata.create_all)
                
                # Then populate vector store
                await self.populate_vector_store()
                self._setup_done = True
                logger.info("Database initialization completed successfully")
            except Exception as e:
                logger.error(f"Database initialization failed: {e}")
                raise DatabaseError(f"Failed to initialize database: {str(e)}")

    async def get_db(self) -> AsyncGenerator[AsyncSession, None]:
        if not hasattr(self, '_setup_done'):
            await self.initialize()
        async with self.SessionLocal() as session:
            try:
                yield session
            finally:
                await session.close()

    async def get_session(self) -> AsyncSession:
        """Get a single session for one-off operations"""
        if not hasattr(self, '_setup_done'):
            await self.initialize()
        return self.SessionLocal()

    async def populate_vector_store(self):
        """Populate the vector store with features from existing songs"""
        try:
            async with self.SessionLocal() as session:
                # Get all songs with audio features, but only select the columns we need
                query = select(
                    Song.id,
                    Song.audio_features
                ).where(Song.audio_features.isnot(None))
                
                result = await session.execute(query)
                songs = result.all()  # Returns list of tuples (id, audio_features)
                
                # Add each song's features to the vector store
                for song_id, audio_features in songs:
                    features = json.loads(audio_features) if isinstance(audio_features, str) else audio_features
                    self.vector_store.add_song(song_id, features)
                
                logger.info(f"Populated vector store with {len(songs)} songs")
                
        except Exception as e:
            logger.error(f"Error populating vector store: {e}")
            # Don't raise the error - vector store population is not critical
            # We can still function without it
            pass

    async def search_songs(
        self,
        query: Optional[str] = None,
        artist: Optional[str] = None,
        genre: Optional[str] = None,
        mood: Optional[str] = None,
        offset: int = 0,
        limit: int = 50
    ) -> List[Song]:
        """Search for songs using fuzzy matching and multiple criteria."""
        async with self.SessionLocal() as session:
            try:
                # Start with base query
                base_query = select(Song)
                conditions = []
                
                # Build search conditions
                if query:
                    # Fuzzy match on track name with similarity score
                    name_similarity = func.similarity(Song.track_name, query)
                    conditions.append(name_similarity >= 0.3)  # Minimum similarity threshold
                    
                    # Add weighted score for ordering
                    base_query = base_query.add_columns(
                        (case(
                            [(name_similarity >= 0.8, 3.0),  # High similarity
                             (name_similarity >= 0.5, 2.0)],  # Medium similarity
                            else_=name_similarity  # Lower similarity
                        )).label('relevance_score')
                    )
                
                if artist:
                    # Fuzzy match on artist name
                    artist_similarity = func.similarity(Song.track_artist, artist)
                    conditions.append(artist_similarity >= 0.3)
                    
                    if not query:  # Only add score if not already added
                        base_query = base_query.add_columns(
                            (case(
                                [(artist_similarity >= 0.8, 2.0),
                                 (artist_similarity >= 0.5, 1.5)],
                                else_=artist_similarity
                            )).label('relevance_score')
                        )
                
                # Exact matches for genre and mood
                if genre:
                    conditions.append(Song.playlist_genre == genre)
                if mood:
                    conditions.append(Song.mood == mood)
                
                # Apply conditions if any exist
                if conditions:
                    base_query = base_query.filter(or_(*conditions))
                
                # Order by relevance score if text search was performed
                if query or artist:
                    base_query = base_query.order_by(text('relevance_score DESC NULLS LAST'))
                else:
                    base_query = base_query.order_by(Song.track_name)
                
                # Apply pagination
                result = await session.execute(base_query.offset(offset).limit(limit))
                results = result.all()
                
                # Extract Song objects from results (ignoring relevance scores)
                songs = [result[0] if isinstance(result, tuple) else result for result in results]
                
                return songs
                
            except SQLAlchemyError as e:
                logger.error(f"Database error during song search: {str(e)}")
                raise DatabaseError(f"Error searching songs: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error during song search: {str(e)}")
                raise

# Create lazy database manager instance
db_manager = DatabaseManager() 