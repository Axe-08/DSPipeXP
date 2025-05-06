from redis.asyncio import Redis
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

_redis_client: Redis | None = None

async def get_redis() -> Redis:
    """
    Get Redis client instance.
    Creates a new connection if one doesn't exist.
    """
    global _redis_client
    
    if _redis_client is None:
        try:
            _redis_client = Redis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True
            )
            # Test connection
            await _redis_client.ping()
            logger.info("Successfully connected to Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise
    
    return _redis_client

async def close_redis():
    """Close Redis connection if it exists."""
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        _redis_client = None 