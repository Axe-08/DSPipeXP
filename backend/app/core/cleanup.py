import os
import time
import logging
import asyncio
from datetime import datetime, timedelta
from typing import List
from .config import settings
from .database import db_manager

logger = logging.getLogger(__name__)

class FileCleanupManager:
    def __init__(self):
        self.audio_dir = settings.AUDIO_STORAGE_PATH
        self.cache_dir = os.path.dirname(settings.VECTOR_STORE_PATH)
        self.max_age = int(os.getenv('MAX_FILE_AGE', 86400))  # 24 hours default
        self.cleanup_interval = int(os.getenv('CLEANUP_INTERVAL', 3600))  # 1 hour default
        
    async def start_cleanup_task(self):
        """Start the periodic cleanup task"""
        while True:
            try:
                await self.cleanup_files()
                await asyncio.sleep(self.cleanup_interval)
            except Exception as e:
                logger.error(f"Error in cleanup task: {str(e)}")
                await asyncio.sleep(60)  # Wait a minute before retrying
                
    async def cleanup_files(self) -> dict:
        """Clean up old files from storage"""
        try:
            # Get list of active files from database
            active_files = await db_manager.get_active_file_paths()
            
            # Clean up audio files
            cleaned_audio = await self._cleanup_directory(
                self.audio_dir,
                active_files,
                ['.mp3', '.wav', '.m4a']
            )
            
            # Clean up cache files
            cleaned_cache = await self._cleanup_directory(
                self.cache_dir,
                active_files,
                ['.pkl', '.npy', '.json']
            )
            
            return {
                "status": "success",
                "cleaned_files": cleaned_audio + cleaned_cache,
                "count": len(cleaned_audio) + len(cleaned_cache)
            }
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
            
    async def _cleanup_directory(
        self,
        directory: str,
        active_files: List[str],
        extensions: List[str]
    ) -> List[str]:
        """Clean up old files in a directory"""
        cleaned_files = []
        current_time = time.time()
        
        try:
            for filename in os.listdir(directory):
                if not any(filename.endswith(ext) for ext in extensions):
                    continue
                    
                filepath = os.path.join(directory, filename)
                
                # Skip if file is still in use
                if filepath in active_files:
                    continue
                    
                # Check file age
                file_time = os.path.getmtime(filepath)
                if current_time - file_time > self.max_age:
                    try:
                        os.remove(filepath)
                        cleaned_files.append(filepath)
                        logger.info(f"Removed old file: {filepath}")
                    except Exception as e:
                        logger.error(f"Error removing file {filepath}: {str(e)}")
                        
            return cleaned_files
            
        except Exception as e:
            logger.error(f"Error cleaning directory {directory}: {str(e)}")
            return []
            
    async def get_storage_stats(self) -> dict:
        """Get storage statistics"""
        try:
            audio_size = sum(
                os.path.getsize(os.path.join(self.audio_dir, f))
                for f in os.listdir(self.audio_dir)
                if os.path.isfile(os.path.join(self.audio_dir, f))
            )
            
            cache_size = sum(
                os.path.getsize(os.path.join(self.cache_dir, f))
                for f in os.listdir(self.cache_dir)
                if os.path.isfile(os.path.join(self.cache_dir, f))
            )
            
            return {
                "audio_storage_mb": round(audio_size / (1024 * 1024), 2),
                "cache_storage_mb": round(cache_size / (1024 * 1024), 2),
                "total_storage_mb": round((audio_size + cache_size) / (1024 * 1024), 2),
                "last_cleanup": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting storage stats: {str(e)}")
            return {}

# Create global cleanup manager instance
cleanup_manager = FileCleanupManager() 