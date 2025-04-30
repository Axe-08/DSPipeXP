from fastapi import APIRouter, Depends, HTTPException
from typing import Dict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.core.database import get_db, db_manager
from app.core.redis import get_redis
from app.core.config import settings
import logging
import traceback
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

router = APIRouter()

async def check_db_connection(db: AsyncSession) -> tuple[bool, str]:
    """Check if database connection is alive"""
    try:
        # Try to get database version to ensure connectivity
        result = await db.execute(text("SELECT version()"))
        version = await result.scalar()
        logger.info(f"Database version: {version}")
        
        # Check if our tables exist
        result = await db.execute(
            text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'songs')")
        )
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
        redis_client = await get_redis()
        await redis_client.ping()
        info = await redis_client.info()
        logger.info(f"Redis info: {info.get('redis_version')}")
        return True, "Redis connection healthy"
    except Exception as e:
        error_details = f"Redis error: {str(e)}"
        logger.error(error_details)
        return False, error_details

def get_redacted_url(url: str | None) -> str:
    """Return a redacted version of the URL with password hidden."""
    if not url:
        return "not set"
    try:
        parsed = urlparse(url)
        if parsed.password:
            # Replace password with ***
            netloc = parsed.netloc.replace(parsed.password, "***")
            return url.replace(parsed.netloc, netloc)
        return url
    except Exception as e:
        logger.error(f"Error redacting URL: {e}")
        return "invalid url format"

@router.get("")  # Changed from "/health" to "" since we're mounting at /api/v1/health
async def health_check():
    """Health check endpoint that checks database and Redis connectivity"""
    response = {
        "status": "healthy",
        "database": {
            "status": "unknown",
            "version": None,
            "schema_initialized": False,
            "migration_version": None,
            "error": None
        },
        "redis": {
            "status": "unknown",
            "version": None,
            "error": None,
            "url_format": None if not settings.REDIS_URL else settings.REDIS_URL.replace(
                urlparse(settings.REDIS_URL).password or "", "***"
            ) if settings.REDIS_URL else None
        }
    }
    
    # Check database
    try:
        async with db_manager.SessionLocal() as session:
            # Get database version
            result = await session.execute(text("SELECT version()"))
            version = await result.scalar()
            
            # Check if schema is initialized by checking alembic_version table
            try:
                result = await session.execute(text("SELECT version_num FROM alembic_version"))
                migration_version = await result.scalar()
                response["database"].update({
                    "status": "healthy",
                    "version": version,
                    "schema_initialized": True,
                    "migration_version": migration_version
                })
            except Exception as e:
                response["database"].update({
                    "status": "unhealthy",
                    "version": version,
                    "schema_initialized": False,
                    "error": f"Schema not initialized: {str(e)}"
                })
                response["status"] = "unhealthy"
                
    except Exception as e:
        response["database"].update({
            "status": "unhealthy",
            "error": f"Database connection failed: {str(e)}"
        })
        response["status"] = "unhealthy"

    # Check Redis
    if settings.REDIS_URL:
        try:
            redis = await get_redis()
            info = await redis.info()
            response["redis"].update({
                "status": "healthy",
                "version": info.get("redis_version", "unknown")
            })
        except Exception as e:
            error_msg = f"Redis connection failed: {str(e)}"
            logger.error(error_msg)
            response["redis"].update({
                "status": "unhealthy",
                "error": error_msg
            })
            response["status"] = "unhealthy"
    else:
        response["redis"].update({
            "status": "not configured",
            "error": "REDIS_URL not set"
        })

    return response 