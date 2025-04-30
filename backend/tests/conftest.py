import pytest
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.core.database import Base
from app.core.config import settings
import os

# Test database URL
TEST_DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/dspipexp_test"

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def test_engine():
    """Create a test database engine."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    
    # Create test database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Clean up
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()

@pytest.fixture(scope="session")
async def test_session_factory(test_engine):
    """Create a test session factory."""
    return sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )

@pytest.fixture
async def test_session(test_session_factory):
    """Create a test database session."""
    async with test_session_factory() as session:
        yield session
        await session.rollback()

@pytest.fixture(scope="session", autouse=True)
def test_env():
    """Set up test environment variables."""
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL
    os.environ["REDIS_URL"] = "redis://localhost:6379"
    os.environ["GENIUS_API_KEY"] = "test_key"
    os.environ["YOUTUBE_API_KEY"] = "test_key"
    os.environ["AUDIO_STORAGE_PATH"] = "./data/test_audio"
    os.environ["CACHE_STORAGE_PATH"] = "./data/test_cache"
    
    # Create test directories
    os.makedirs("./data/test_audio", exist_ok=True)
    os.makedirs("./data/test_cache", exist_ok=True)
    
    return None

@pytest.fixture
def test_audio_file(tmp_path):
    """Create a test audio file."""
    import numpy as np
    import soundfile as sf
    
    # Create a simple sine wave
    sample_rate = 22050
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    # Save as WAV file
    file_path = tmp_path / "test_song.wav"
    sf.write(file_path, audio_data, sample_rate)
    
    return str(file_path)

@pytest.fixture
def test_song_data():
    """Return test song metadata."""
    return {
        "name": "Test Song",
        "artist": "Test Artist",
        "features": [0.5] * settings.FEATURE_DIMENSIONS,  # Fixed features for testing
        "audio_path": "test_song.mp3",
        "youtube_url": "https://youtube.com/test",
        "metadata": {
            "genre": "Test Genre",
            "year": 2024,
            "mood": "happy"
        }
    } 