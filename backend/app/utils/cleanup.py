import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import List
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings

logger = logging.getLogger(__name__)

async def cleanup_old_files(directory: str, max_age_hours: int = 24) -> None:
    """Delete files older than specified hours"""
    try:
        now = datetime.now()
        count = 0
        size_freed = 0

        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                if now - mtime > timedelta(hours=max_age_hours):
                    size_freed += os.path.getsize(filepath)
                    os.remove(filepath)
                    count += 1

        logger.info(
            f"Cleaned up {count} files from {directory}, freed {size_freed / 1024 / 1024:.2f} MB"
        )
    except Exception as e:
        logger.error(f"Error cleaning up {directory}: {str(e)}")

async def monitor_storage_usage() -> dict:
    """Monitor storage usage in temporary directories"""
    audio_size = sum(
        os.path.getsize(os.path.join(settings.AUDIO_STORAGE_PATH, f))
        for f in os.listdir(settings.AUDIO_STORAGE_PATH)
        if os.path.isfile(os.path.join(settings.AUDIO_STORAGE_PATH, f))
    )
    
    cache_size = sum(
        os.path.getsize(os.path.join(settings.CACHE_STORAGE_PATH, f))
        for f in os.listdir(settings.CACHE_STORAGE_PATH)
        if os.path.isfile(os.path.join(settings.CACHE_STORAGE_PATH, f))
    )
    
    return {
        "audio_storage_mb": audio_size / 1024 / 1024,
        "cache_storage_mb": cache_size / 1024 / 1024,
        "total_storage_mb": (audio_size + cache_size) / 1024 / 1024
    }

async def cleanup_database(db: AsyncSession) -> None:
    """Clean up old database records"""
    try:
        # Delete old songs that haven't been accessed in 30 days
        await db.execute(
            text("""
                DELETE FROM songs 
                WHERE last_accessed < NOW() - INTERVAL '30 days'
                AND is_original = false
            """)
        )
        
        # Delete orphaned audio features
        await db.execute(
            text("""
                DELETE FROM songs 
                WHERE id NOT IN (
                    SELECT DISTINCT song_id 
                    FROM recommendations
                    WHERE created_at > NOW() - INTERVAL '30 days'
                )
                AND is_original = false
            """)
        )
        
        # Clean up old recommendations
        await db.execute(
            text("""
                DELETE FROM recommendations
                WHERE created_at < NOW() - INTERVAL '7 days'
            """)
        )
        
        await db.commit()
        logger.info("Database cleanup completed successfully")
        
    except Exception as e:
        await db.rollback()
        logger.error(f"Error cleaning up database: {str(e)}")

class StorageManager:
    def __init__(self, max_storage_mb: float = 450):  # Leave some buffer from 512MB limit
        self.max_storage_mb = max_storage_mb
        self.cleanup_running = False
    
    async def check_and_cleanup(self, db: AsyncSession) -> None:
        """Check storage usage and trigger cleanup if needed"""
        if self.cleanup_running:
            return
            
        try:
            self.cleanup_running = True
            usage = await monitor_storage_usage()
            
            if usage["total_storage_mb"] > self.max_storage_mb:
                logger.warning(f"Storage usage high: {usage['total_storage_mb']:.2f}MB")
                
                # Aggressive cleanup for audio files (keep only 6 hours)
                await cleanup_old_files(settings.AUDIO_STORAGE_PATH, max_age_hours=6)
                
                # Normal cleanup for cache (24 hours)
                await cleanup_old_files(settings.CACHE_STORAGE_PATH, max_age_hours=24)
                
                # Database cleanup
                await cleanup_database(db)
                
                # Check usage after cleanup
                new_usage = await monitor_storage_usage()
                logger.info(f"Storage after cleanup: {new_usage['total_storage_mb']:.2f}MB")
                
        finally:
            self.cleanup_running = False

# Create global storage manager instance
storage_manager = StorageManager() 