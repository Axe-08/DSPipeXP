from fastapi import Request, HTTPException
import time
from typing import Dict, Tuple
import logging
from .cache import cache

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.window_size = 60  # 1 minute
        
    async def is_rate_limited(self, key: str) -> Tuple[bool, Dict]:
        """Check if request should be rate limited"""
        try:
            current = int(time.time())
            window_key = f"{key}:{current // self.window_size}"
            
            # Get current count
            count = await cache.get(window_key) or 0
            
            if count >= self.requests_per_minute:
                return True, {
                    "limit": self.requests_per_minute,
                    "remaining": 0,
                    "reset": (current // self.window_size + 1) * self.window_size
                }
            
            # Increment counter
            await cache.set(window_key, count + 1, expire=self.window_size)
            
            return False, {
                "limit": self.requests_per_minute,
                "remaining": self.requests_per_minute - count - 1,
                "reset": (current // self.window_size + 1) * self.window_size
            }
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {str(e)}")
            return False, {}  # Allow request on error

async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware"""
    limiter = RateLimiter()
    
    # Get client IP
    client_ip = request.client.host if request.client else "unknown"
    
    # Check rate limit
    is_limited, headers = await limiter.is_rate_limited(f"rate_limit:{client_ip}")
    
    if is_limited:
        raise HTTPException(
            status_code=429,
            detail="Too many requests"
        )
    
    # Add rate limit headers
    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(headers.get("limit", ""))
    response.headers["X-RateLimit-Remaining"] = str(headers.get("remaining", ""))
    response.headers["X-RateLimit-Reset"] = str(headers.get("reset", ""))
    
    return response 