from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.logging import setup_logging
from app.core.tasks import BackgroundTasks
from app.api.endpoints import youtube, search, health
import logging

logger = logging.getLogger(__name__)

app = FastAPI(
    title="DSPipeXP API",
    description="Music recommendation system API",
    version="1.0.0"
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
setup_logging()

# Include routers
app.include_router(youtube.router, prefix="/api/v1/youtube", tags=["youtube"])
app.include_router(search.router, prefix="/api/v1/search", tags=["search"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])

# Initialize background tasks
background_tasks = BackgroundTasks(app)

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    try:
        # Start background tasks
        await background_tasks.start()
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    try:
        # Stop background tasks
        await background_tasks.stop()
        logger.info("Application shutdown completed")
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")
        raise

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Music Recommendation System API",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }
