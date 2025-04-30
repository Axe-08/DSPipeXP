import pytest
import numpy as np
import json
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.database import DatabaseManager, VectorStore, Song, Base
from app.core.config import settings

@pytest.fixture(scope="function")
def test_db():
    """Create a test database."""
    # Use SQLite for testing
    test_db_url = "sqlite:///./test.db"
    
    # Create test instance with test settings
    db = DatabaseManager(db_url=test_db_url)
    
    # Create tables
    Base.metadata.create_all(bind=db.engine)
    
    yield db
    
    # Cleanup
    Base.metadata.drop_all(bind=db.engine)
    if os.path.exists("test_vector_store.npz"):
        os.remove("test_vector_store.npz")
    if os.path.exists("test.db"):
        os.remove("test.db")

@pytest.fixture
def sample_song_data():
    """Create sample song data."""
    return {
        "name": "Test Song",
        "artist": "Test Artist",
        "features": np.random.rand(settings.FEATURE_DIMENSIONS),
        "audio_path": "test_song.mp3",
        "youtube_url": "https://youtube.com/test",
        "metadata": {  # This will be stored in song_metadata
            "genre": "Test Genre",
            "year": 2024,
            "mood": "happy"
        }
    }

@pytest.mark.asyncio
async def test_add_song(test_db, sample_song_data):
    """Test adding a song to the database."""
    song = await test_db.add_song(
        name=sample_song_data["name"],
        artist=sample_song_data["artist"],
        features=sample_song_data["features"],
        audio_path=sample_song_data["audio_path"],
        youtube_url=sample_song_data["youtube_url"],
        metadata=sample_song_data["metadata"]
    )
    
    assert song.id is not None
    assert song.name == sample_song_data["name"]
    assert song.artist == sample_song_data["artist"]
    assert json.loads(song.features) == sample_song_data["features"].tolist()
    assert song.youtube_url == sample_song_data["youtube_url"]
    assert json.loads(song.song_metadata) == sample_song_data["metadata"]

def test_get_song_by_name(test_db, sample_song_data):
    """Test retrieving a song by name."""
    # First add a song
    song = test_db.get_session().__enter__().query(Song).first()
    if not song:
        song = Song(
            name=sample_song_data["name"],
            artist=sample_song_data["artist"],
            features=json.dumps(sample_song_data["features"].tolist()),
            audio_path=str(settings.AUDIO_DIR / sample_song_data["audio_path"]),
            youtube_url=sample_song_data["youtube_url"],
            song_metadata=json.dumps(sample_song_data["metadata"])
        )
        with test_db.get_session() as session:
            session.add(song)
            session.commit()
    
    # Then retrieve it
    retrieved_song = test_db.get_song_by_name(sample_song_data["name"])
    assert retrieved_song is not None
    assert retrieved_song.name == sample_song_data["name"]

def test_search_songs(test_db, sample_song_data):
    """Test searching songs."""
    # Add multiple songs
    songs_data = [
        {
            "name": "Happy Song",
            "artist": "Happy Artist",
            "features": np.random.rand(settings.FEATURE_DIMENSIONS),
            "audio_path": "happy.mp3",
            "metadata": {"genre": "Pop", "mood": "happy"}
        },
        {
            "name": "Sad Song",
            "artist": "Sad Artist",
            "features": np.random.rand(settings.FEATURE_DIMENSIONS),
            "audio_path": "sad.mp3",
            "metadata": {"genre": "Blues", "mood": "sad"}
        }
    ]
    
    with test_db.get_session() as session:
        for data in songs_data:
            song = Song(
                name=data["name"],
                artist=data["artist"],
                features=json.dumps(data["features"].tolist()),
                audio_path=str(settings.AUDIO_DIR / data["audio_path"]),
                song_metadata=json.dumps(data["metadata"])
            )
            session.add(song)
        session.commit()
    
    # Test different search scenarios
    happy_songs = test_db.search_songs(query="Happy")
    assert len(happy_songs) == 1
    assert happy_songs[0].name == "Happy Song"
    
    sad_songs = test_db.search_songs(artist="Sad Artist")
    assert len(sad_songs) == 1
    assert sad_songs[0].artist == "Sad Artist"
    
    pop_songs = test_db.search_songs(genre="Pop")
    assert len(pop_songs) == 1
    assert json.loads(pop_songs[0].song_metadata)["genre"] == "Pop"

def test_vector_store(test_db, sample_song_data, tmp_path):
    """Test vector store functionality."""
    # Set up temporary vector store path
    vector_store_path = tmp_path / "test_vector_store.npz"
    test_db.vector_store = VectorStore(str(vector_store_path))
    
    # Add a song and check vector store
    features = sample_song_data["features"]
    test_db.vector_store.add_song(1, features)
    
    assert test_db.vector_store.features_matrix is not None
    assert test_db.vector_store.features_matrix.shape == (1, len(features))
    assert test_db.vector_store.song_ids == [1]
    
    # Test persistence
    test_db.vector_store._save_store()
    new_vector_store = VectorStore(str(vector_store_path))
    assert new_vector_store.features_matrix is not None
    assert np.array_equal(new_vector_store.features_matrix, test_db.vector_store.features_matrix)
    assert new_vector_store.song_ids == test_db.vector_store.song_ids
    
    # Cleanup
    test_db.vector_store.clear()

def test_load_initial_data(test_db, tmp_path):
    """Test loading initial data from CSV."""
    # Create a test CSV file
    csv_path = tmp_path / "test_songs.csv"
    import pandas as pd
    
    df = pd.DataFrame([
        {
            "name": "CSV Song 1",
            "artist": "CSV Artist 1",
            "features": json.dumps(np.random.rand(settings.FEATURE_DIMENSIONS).tolist()),
            "audio_path": "song1.mp3",
            "metadata": json.dumps({"genre": "Rock"})
        },
        {
            "name": "CSV Song 2",
            "artist": "CSV Artist 2",
            "features": json.dumps(np.random.rand(settings.FEATURE_DIMENSIONS).tolist()),
            "audio_path": "song2.mp3",
            "metadata": json.dumps({"genre": "Jazz"})
        }
    ])
    
    df.to_csv(csv_path, index=False)
    
    # Test loading
    test_db.load_initial_data(str(csv_path))
    
    with test_db.get_session() as session:
        songs = session.query(Song).all()
        assert len(songs) == 2
        assert songs[0].name == "CSV Song 1"
        assert songs[1].name == "CSV Song 2"
        
    # Check vector store was updated
    assert test_db.vector_store.features_matrix is not None
    assert test_db.vector_store.features_matrix.shape == (2, settings.FEATURE_DIMENSIONS)
    assert len(test_db.vector_store.song_ids) == 2 