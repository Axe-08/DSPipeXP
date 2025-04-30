import pytest
from fastapi.testclient import TestClient
from app.main import app
import os
from app.core.config import settings

client = TestClient(app)

def test_youtube_search():
    """Test the YouTube search endpoint"""
    query = "Rick Astley Never Gonna Give You Up"
    response = client.get(f"/api/v1/youtube/search?query={query}")
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) > 0
    
    # Verify result structure
    first_result = data["results"][0]
    assert all(key in first_result for key in ["url", "title", "duration", "channel", "thumbnail", "video_id"])
    assert "youtube.com/watch?v=" in first_result["url"]
    assert "never gonna give you up" in first_result["title"].lower()

def test_youtube_metadata():
    """Test the YouTube metadata endpoint"""
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Never Gonna Give You Up
    response = client.get(f"/api/v1/youtube/metadata?url={url}")
    assert response.status_code == 200
    data = response.json()
    assert all(key in data for key in ["title", "duration", "view_count", "channel", "video_id"])
    assert "never gonna give you up" in data["title"].lower()

def test_youtube_process():
    """Test the YouTube process endpoint"""
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Never Gonna Give You Up
    response = client.post("/api/v1/youtube/process", json={"url": url, "quality": "64"})  # Lower quality for faster test
    assert response.status_code == 200
    data = response.json()
    assert "file_path" in data
    assert "metadata" in data
    assert os.path.exists(data["file_path"])
    assert "never gonna give you up" in data["metadata"]["title"].lower()

def test_youtube_download():
    """Test the YouTube download endpoint"""
    query = "Rick Astley Never Gonna Give You Up"
    response = client.post(f"/api/v1/youtube/download?query={query}&quality=64")  # Lower quality for faster test
    assert response.status_code == 200
    data = response.json()
    assert "file_path" in data
    assert "metadata" in data
    assert os.path.exists(data["file_path"])
    assert "never gonna give you up" in data["metadata"]["title"].lower()

def test_invalid_url():
    """Test handling of invalid YouTube URLs"""
    url = "https://www.youtube.com/watch?v=invalid_id"
    response = client.get(f"/api/v1/youtube/metadata?url={url}")
    assert response.status_code == 400

def test_empty_search():
    """Test search with empty query"""
    response = client.get("/api/v1/youtube/search?query=")
    assert response.status_code == 422  # FastAPI validation error

def test_cleanup():
    """Clean up downloaded files after tests"""
    test_files = [f for f in os.listdir(settings.AUDIO_STORAGE_PATH) if f.endswith('.mp3')]
    for file in test_files:
        try:
            os.remove(os.path.join(settings.AUDIO_STORAGE_PATH, file))
        except Exception:
            pass 