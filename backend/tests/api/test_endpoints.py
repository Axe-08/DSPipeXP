import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

@pytest.mark.asyncio
async def test_songs_endpoints():
    # Test song list
    response = client.get("/api/v1/songs")
    assert response.status_code == 200
    
    # Test song search
    response = client.get("/api/v1/songs/search?query=test")
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_recommendations():
    response = client.get("/api/v1/recommendations/similar/1")
    assert response.status_code in [200, 404]  # 404 if song doesn't exist

@pytest.mark.asyncio
async def test_youtube():
    # Test YouTube search
    response = client.get("/api/v1/youtube/search?query=test+song")
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_monitoring():
    response = client.get("/api/v1/monitoring/stats")
    assert response.status_code == 200 