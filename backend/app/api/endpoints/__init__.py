"""
API endpoints package initialization
"""
from fastapi import APIRouter
from . import songs, recommendations, youtube, health, search, monitoring

# Create main router
router = APIRouter()

# Include all sub-routers
router.include_router(health.router, prefix="/health", tags=["health"])
router.include_router(monitoring.router, prefix="/monitoring", tags=["monitoring"])
router.include_router(songs.router, prefix="/songs", tags=["songs"])
router.include_router(youtube.router, prefix="/youtube", tags=["youtube"])
router.include_router(search.router, prefix="/search", tags=["search"])
router.include_router(recommendations.router, prefix="/recommendations", tags=["recommendations"])

# Export all routers
__all__ = ['songs', 'recommendations', 'youtube', 'health', 'search', 'monitoring'] 