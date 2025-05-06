from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import json
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from app.models.song import Song
from app.core.config import settings

logger = logging.getLogger(__name__)

class YouTubeCache:
    def __init__(self, db: AsyncSession):
        """
        Initialize YouTube cache with database session.
        
        Args:
            db: SQLAlchemy async session
        """
        self.db = db
        self.cache_duration = timedelta(days=settings.YOUTUBE_CACHE_DAYS)
        
    async def get_url(self, song_id: int) -> Optional[str]:
        """
        Get cached YouTube URL for a song if it exists and is not expired.
        
        Args:
            song_id: ID of the song
            
        Returns:
            Optional[str]: Cached URL if valid, None otherwise
        """
        query = select(Song).where(Song.id == song_id)
        result = await self.db.execute(query)
        song = result.scalar_one_or_none()
        
        if not song or not song.youtube_url:
            return None
            
        # Check if URL is expired
        if not song.youtube_url_updated_at or \
           datetime.utcnow() - song.youtube_url_updated_at > self.cache_duration:
            return None
            
        return song.youtube_url
        
    async def set_url(self, song_id: int, url: str) -> bool:
        """
        Cache YouTube URL for a song.
        
        Args:
            song_id: ID of the song
            url: YouTube URL to cache
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            stmt = update(Song).where(Song.id == song_id).values(
                youtube_url=url,
                youtube_url_updated_at=datetime.utcnow()
            )
            await self.db.execute(stmt)
            await self.db.commit()
            return True
        except Exception as e:
            logger.error(f"Error caching YouTube URL for song {song_id}: {e}")
            await self.db.rollback()
            return False
            
    async def get_metadata(self, song_id: int) -> Optional[Dict[str, Any]]:
        """
        Get cached YouTube metadata for a song if it exists and is not expired.
        
        Args:
            song_id: ID of the song
            
        Returns:
            Optional[Dict[str, Any]]: Cached metadata if valid, None otherwise
        """
        query = select(Song).where(Song.id == song_id)
        result = await self.db.execute(query)
        song = result.scalar_one_or_none()
        
        if not song or not song.youtube_metadata:
            return None
            
        # Check if metadata is expired
        if not song.youtube_metadata_updated_at or \
           datetime.utcnow() - song.youtube_metadata_updated_at > self.cache_duration:
            return None
            
        try:
            return json.loads(song.youtube_metadata)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in YouTube metadata for song {song_id}")
            return None
            
    async def set_metadata(self, song_id: int, metadata: Dict[str, Any]) -> bool:
        """
        Cache YouTube metadata for a song.
        
        Args:
            song_id: ID of the song
            metadata: YouTube metadata to cache
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            stmt = update(Song).where(Song.id == song_id).values(
                youtube_metadata=json.dumps(metadata),
                youtube_metadata_updated_at=datetime.utcnow()
            )
            await self.db.execute(stmt)
            await self.db.commit()
            return True
        except Exception as e:
            logger.error(f"Error caching YouTube metadata for song {song_id}: {e}")
            await self.db.rollback()
            return False
            
    async def invalidate(self, song_id: int) -> bool:
        """
        Invalidate cached YouTube data for a song.
        
        Args:
            song_id: ID of the song
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            stmt = update(Song).where(Song.id == song_id).values(
                youtube_url=None,
                youtube_url_updated_at=None,
                youtube_metadata=None,
                youtube_metadata_updated_at=None
            )
            await self.db.execute(stmt)
            await self.db.commit()
            return True
        except Exception as e:
            logger.error(f"Error invalidating YouTube cache for song {song_id}: {e}")
            await self.db.rollback()
            return False 