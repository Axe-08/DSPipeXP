from fastapi import APIRouter, Depends
from typing import Dict
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
import redis.asyncio as redis
from app.core.config import settings
import logging
import traceback

logger = logging.getLogger(__name__)

router = APIRouter()

async def check_db_connection(db: AsyncSession) -> tuple[bool, str]:
    """Check if database connection is alive"""
    try:
        # Try to get database version to ensure connectivity
        result = await db.execute("SELECT version()")
        version = await result.scalar()
        logger.info(f"Database version: {version}")
        
        # Check if our tables exist
        result = await db.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'songs')")
        tables_exist = await result.scalar()
        if not tables_exist:
            return False, "Database schema not initialized - tables missing"
            
        return True, "Database connection healthy"
    except Exception as e:
        error_details = f"Database error: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_details)
        return False, error_details

async def check_redis_connection() -> tuple[bool, str]:
    """Check if Redis connection is alive"""
    try:
        redis_client = redis.from_url(settings.REDIS_URL)
        await redis_client.ping()
        info = await redis_client.info()
        logger.info(f"Redis info: {info.get('redis_version')}")
        return True, "Redis connection healthy"
    except Exception as e:
        error_details = f"Redis error: {str(e)}"
        logger.error(error_details)
        return False, error_details

@router.get("/health", response_model=Dict[str, str])
async def health_check(db: AsyncSession = Depends(get_db)):
    """
    Health check endpoint that verifies:
    - API is responding
    - Database connection is alive
    - Redis connection is alive
    """
    db_healthy, db_details = await check_db_connection(db)
    redis_healthy, redis_details = await check_redis_connection()
    
    logger.info(f"Health check - DB: {db_healthy}, Redis: {redis_healthy}")
    
    overall_status = "healthy" if all([db_healthy, redis_healthy]) else "unhealthy"
    
    return {
        "status": overall_status,
        "database": "healthy" if db_healthy else "unhealthy",
        "database_details": db_details,
        "redis": "healthy" if redis_healthy else "unhealthy",
        "redis_details": redis_details
    } 