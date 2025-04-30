from fastapi import APIRouter, Depends
from typing import Dict
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
import redis.asyncio as redis
from app.core.config import settings

router = APIRouter()

async def check_db_connection(db: AsyncSession) -> bool:
    """Check if database connection is alive"""
    try:
        await db.execute("SELECT 1")
        return True
    except Exception:
        return False

async def check_redis_connection() -> bool:
    """Check if Redis connection is alive"""
    try:
        redis_client = redis.from_url(settings.REDIS_URL)
        await redis_client.ping()
        return True
    except Exception:
        return False

@router.get("/health", response_model=Dict[str, str])
async def health_check(db: AsyncSession = Depends(get_db)):
    """
    Health check endpoint that verifies:
    - API is responding
    - Database connection is alive
    - Redis connection is alive
    """
    db_status = "healthy" if await check_db_connection(db) else "unhealthy"
    redis_status = "healthy" if await check_redis_connection() else "unhealthy"
    
    overall_status = "healthy" if all([
        db_status == "healthy",
        redis_status == "healthy"
    ]) else "unhealthy"
    
    return {
        "status": overall_status,
        "database": db_status,
        "redis": redis_status
    } 