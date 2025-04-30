import pytest
from httpx import AsyncClient
from pathlib import Path
import json
import os
from app.main import app
from app.core.database import db_manager

@pytest.fixture(scope="module")
def test_audio_file():
    # Create a small test audio file
    current_dir = Path(__file__).parent
    test_file_path = current_dir / "test_audio.mp3"
    return str(test_file_path)

@pytest.fixture(scope="module")
async def test_client():
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.fixture(scope="module")
async def uploaded_song(test_client, test_audio_file):
    # Upload a test song and return its ID
    with open(test_audio_file, "rb") as f:
        files = {"audio_file": ("test_song.mp3", f, "audio/mpeg")}
        data = {
            "track_name": "Test Song",
            "track_artist": "Test Artist",
            "track_album_name": "Test Album",
            "playlist_genre": "Test Genre",
            "lyrics": "Test lyrics"
        }
        response = await test_client.post("/songs/", files=files, data=data)
    assert response.status_code == 200
    return response.json()["id"]

class TestSongEndpoints:
    @pytest.mark.asyncio
    async def test_create_song(self, test_client, test_audio_file):
        """Test creating a new song"""
        with open(test_audio_file, "rb") as f:
            files = {"audio_file": ("test_song.mp3", f, "audio/mpeg")}
            data = {
                "track_name": "Test Song",
                "track_artist": "Test Artist",
                "track_album_name": "Test Album",
                "playlist_genre": "Test Genre",
                "lyrics": "Test lyrics"
            }
            response = await test_client.post("/songs/", files=files, data=data)
            
        assert response.status_code == 200
        data = response.json()
        assert data["track_name"] == "Test Song"
        assert data["track_artist"] == "Test Artist"
        assert "audio_features" in data
        
    @pytest.mark.asyncio
    async def test_get_songs(self, test_client):
        """Test getting list of songs"""
        response = await test_client.get("/songs/")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        if len(data) > 0:
            assert "id" in data[0]
            assert "track_name" in data[0]
            
    @pytest.mark.asyncio
    async def test_get_song_by_id(self, test_client, uploaded_song):
        """Test getting a specific song by ID"""
        response = await test_client.get(f"/songs/{uploaded_song}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == uploaded_song
        assert "audio_features" in data
        
    @pytest.mark.asyncio
    async def test_update_song(self, test_client, uploaded_song):
        """Test updating a song"""
        update_data = {
            "track_name": "Updated Song",
            "track_artist": "Updated Artist",
            "track_album_name": "Updated Album",
            "playlist_genre": "Updated Genre"
        }
        response = await test_client.put(f"/songs/{uploaded_song}", json=update_data)
        assert response.status_code == 200
        data = response.json()
        assert data["track_name"] == "Updated Song"
        assert data["track_artist"] == "Updated Artist"
        
    @pytest.mark.asyncio
    async def test_search_songs(self, test_client, uploaded_song):
        """Test searching songs"""
        # Search by title
        response = await test_client.get("/songs/search", params={"query": "Updated"})
        assert response.status_code == 200
        data = response.json()
        assert len(data) > 0
        assert any(song["track_name"] == "Updated Song" for song in data)
        
        # Search by artist
        response = await test_client.get("/songs/search", params={"artist": "Updated"})
        assert response.status_code == 200
        data = response.json()
        assert len(data) > 0
        assert any(song["track_artist"] == "Updated Artist" for song in data)
        
        # Search by genre
        response = await test_client.get("/songs/search", params={"genre": "Updated"})
        assert response.status_code == 200
        data = response.json()
        assert len(data) > 0
        assert any(song["playlist_genre"] == "Updated Genre" for song in data)
        
    @pytest.mark.asyncio
    async def test_add_youtube_song(self, test_client):
        """Test adding a song from YouTube"""
        data = {
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "track_name": "Never Gonna Give You Up",
            "track_artist": "Rick Astley"
        }
        response = await test_client.post("/songs/youtube", json=data)
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["track_name"] == "Never Gonna Give You Up"
        
    @pytest.mark.asyncio
    async def test_batch_upload_songs(self, test_client, test_audio_file):
        """Test batch uploading songs"""
        with open(test_audio_file, "rb") as f1, open(test_audio_file, "rb") as f2:
            files = [
                ("files", ("song1.mp3", f1, "audio/mpeg")),
                ("files", ("song2.mp3", f2, "audio/mpeg"))
            ]
            metadata = {
                "song1.mp3": {
                    "track_name": "Batch Song 1",
                    "track_artist": "Batch Artist 1"
                },
                "song2.mp3": {
                    "track_name": "Batch Song 2",
                    "track_artist": "Batch Artist 2"
                }
            }
            data = {"metadata": json.dumps(metadata)}
            response = await test_client.post("/songs/batch", files=files, data=data)
            
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert all("id" in song for song in data)
        
    @pytest.mark.asyncio
    async def test_delete_song(self, test_client, uploaded_song):
        """Test deleting a song"""
        response = await test_client.delete(f"/songs/{uploaded_song}")
        assert response.status_code == 200
        
        # Verify deletion
        response = await test_client.get(f"/songs/{uploaded_song}")
        assert response.status_code == 404

class TestRecommendationEndpoints:
    @pytest.mark.asyncio
    async def test_get_similar_songs(self, test_client, uploaded_song):
        """Test getting similar songs"""
        response = await test_client.get(
            f"/recommendations/similar/{uploaded_song}",
            params={"k": 3}
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        if len(data) > 0:
            assert "track_name" in data[0]
            assert "similarity_score" in data[0]
            assert "component_scores" in data[0]
            
    @pytest.mark.asyncio
    async def test_get_mood_recommendations(self, test_client):
        """Test getting mood-based recommendations"""
        response = await test_client.get(
            "/recommendations/mood/happy",
            params={"k": 3}
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        if len(data) > 0:
            assert "track_name" in data[0]
            assert "similarity_score" in data[0]
            
    @pytest.mark.asyncio
    async def test_get_feature_recommendations(self, test_client):
        """Test getting feature-based recommendations"""
        features = {
            "features": {
                "danceability": 0.8,
                "energy": 0.6,
                "valence": 0.9
            },
            "feature_type": "audio",
            "k": 3
        }
        response = await test_client.post("/recommendations/features", json=features)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        if len(data) > 0:
            assert "track_name" in data[0]
            assert "similarity_score" in data[0]
            
    @pytest.mark.asyncio
    async def test_analyze_song(self, test_client, uploaded_song):
        """Test song analysis"""
        response = await test_client.get(f"/recommendations/analyze/{uploaded_song}")
        assert response.status_code == 200
        data = response.json()
        assert "audio_features" in data
        assert "sentiment_analysis" in data
        assert "topic_analysis" in data
        assert "mood_prediction" in data 