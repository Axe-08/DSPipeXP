from fastapi import APIRouter, HTTPException
from typing import Dict
import psutil
import logging
from ...core.cache import cache
from ...core.cleanup import cleanup_manager
from ...core.database import db_manager

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/stats")
async def get_system_stats() -> Dict:
    """Get system statistics including cache, storage, and performance metrics"""
    try:
        # Get cache stats
        cache_stats = await cache.get_stats()
        
        # Get storage stats
        storage_stats = await cleanup_manager.get_storage_stats()
        
        # Get database stats
        db_stats = await db_manager.get_stats()
        
        # Get system stats
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "cache": cache_stats,
            "storage": storage_stats,
            "database": db_stats,
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": round(memory.used / (1024**3), 2),
                "disk_percent": disk.percent,
                "disk_used_gb": round(disk.used / (1024**3), 2)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting system stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cleanup")
async def trigger_cleanup() -> Dict:
    """Manually trigger file cleanup"""
    try:
        cleaned_files = await cleanup_manager.cleanup_files()
        return {
            "status": "success",
            "cleaned_files": cleaned_files,
            "count": len(cleaned_files)
        }
    except Exception as e:
        logger.error(f"Error triggering cleanup: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 