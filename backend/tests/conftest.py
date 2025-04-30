import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
import sys
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(backend_dir))

from app.core.config import settings
from app.core.database import Base

# Use in-memory SQLite for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

@pytest.fixture(scope="session")
def test_db_engine():
    """Create a test database engine."""
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        connect_args={"check_same_thread": False}
    )
    
    Base.metadata.create_all(bind=engine)
    
    yield engine
    
    Base.metadata.drop_all(bind=engine)
    if os.path.exists("test.db"):
        os.remove("test.db")

@pytest.fixture(scope="function")
def test_db_session(test_db_engine):
    """Create a fresh database session for a test."""
    connection = test_db_engine.connect()
    transaction = connection.begin()
    session = sessionmaker(bind=connection)()
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()

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