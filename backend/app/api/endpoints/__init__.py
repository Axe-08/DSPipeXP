"""
API endpoints package initialization
"""
from . import songs, recommendations, youtube, health, search, monitoring

# Export all routers
__all__ = ['songs', 'recommendations', 'youtube', 'health', 'search', 'monitoring']

# Import routers for direct access
from .songs import router as songs_router
from .recommendations import router as recommendations_router
from .youtube import router as youtube_router
from .health import router as health_router
from .search import router as search_router
from .monitoring import router as monitoring_router

# Re-export routers
router = songs_router
router.include_router(recommendations_router, prefix="/recommendations", tags=["recommendations"])
router.include_router(youtube_router, prefix="/youtube", tags=["youtube"])
router.include_router(health_router, prefix="/health", tags=["health"])
router.include_router(search_router, prefix="/search", tags=["search"])
router.include_router(monitoring_router, prefix="/monitoring", tags=["monitoring"]) 