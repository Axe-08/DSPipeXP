from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.endpoints import songs, recommendations, youtube, health, search, monitoring
from .core.config import PROJECT_NAME, API_V1_STR, settings
from .core.database import db_manager
from .core.cleanup import cleanup_manager
from .core.data_loader import DataLoader
import os
import logging
import asyncio

logger = logging.getLogger(__name__)

app = FastAPI(
    title=PROJECT_NAME,
    description="API for music recommendations using audio features and machine learning",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=settings.CORS_METHODS,
    allow_headers=settings.CORS_HEADERS,
)

# Include routers
app.include_router(health.router, prefix=API_V1_STR, tags=["health"])  # Health check at API_V1_STR level
app.include_router(songs.router, prefix=f"{API_V1_STR}/songs", tags=["songs"])
app.include_router(search.router, prefix=f"{API_V1_STR}/search", tags=["search"])
app.include_router(recommendations.router, prefix=f"{API_V1_STR}/recommendations", tags=["recommendations"])
app.include_router(youtube.router, prefix=f"{API_V1_STR}/youtube", tags=["youtube"])
app.include_router(monitoring.router, prefix=f"{API_V1_STR}/monitoring", tags=["monitoring"])

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        # Initialize database and vector store
        await db_manager.initialize()
        logger.info("Database and vector store initialized")
        
        # Start cleanup task
        asyncio.create_task(cleanup_manager.start_cleanup_task())
        logger.info("Cleanup task started")
        
        # Load initial data if needed
        data_loader = DataLoader()
        await data_loader.load_initial_data()
        logger.info("Initial data loaded")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

@app.on_event("shutdown")
def shutdown_event():
    """Cleanup on shutdown"""
    try:
        # No need to stop cleanup task as it will be stopped when the event loop is closed
        logger.info("Cleanup manager stopped successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Music Recommendation System API",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }
