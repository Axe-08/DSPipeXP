import pytest
from fastapi import status
import json
import os

def test_create_song(test_app, test_song_file, test_song_data):
    """Test song creation endpoint."""
    with open(test_song_file, 'rb') as f:
        files = {'file': ('test_song.wav', f, 'audio/wav')}
        response = test_app.post(
            "/api/v1/songs/",
            files=files,
            data={'metadata': json.dumps(test_song_data)}
        )
    
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data['title'] == test_song_data['title']
    assert data['artist'] == test_song_data['artist']
    assert 'id' in data

def test_get_songs(test_app):
    """Test getting all songs."""
    response = test_app.get("/api/v1/songs/")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert isinstance(data, list)

def test_get_song(test_app):
    """Test getting a specific song."""
    # First create a song
    with open(test_song_file, 'rb') as f:
        files = {'file': ('test_song.wav', f, 'audio/wav')}
        create_response = test_app.post(
            "/api/v1/songs/",
            files=files,
            data={'metadata': json.dumps(test_song_data)}
        )
    song_id = create_response.json()['id']
    
    # Then get it
    response = test_app.get(f"/api/v1/songs/{song_id}")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data['id'] == song_id

def test_search_songs(test_app, test_song_data):
    """Test song search endpoint."""
    response = test_app.get(
        "/api/v1/songs/search/",
        params={"query": test_song_data['title']}
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert isinstance(data, list)

def test_get_similar_songs(test_app):
    """Test similar songs endpoint."""
    # First create a song
    with open(test_song_file, 'rb') as f:
        files = {'file': ('test_song.wav', f, 'audio/wav')}
        create_response = test_app.post(
            "/api/v1/songs/",
            files=files,
            data={'metadata': json.dumps(test_song_data)}
        )
    song_id = create_response.json()['id']
    
    # Then get similar songs
    response = test_app.get(f"/api/v1/recommendations/similar/{song_id}")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert isinstance(data, list)

def test_get_songs_by_mood(test_app):
    """Test mood-based recommendations."""
    response = test_app.get("/api/v1/recommendations/mood/happy")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert isinstance(data, list)

def test_get_songs_by_features(test_app):
    """Test feature-based recommendations."""
    features = {
        "tempo_min": 120,
        "tempo_max": 140,
        "energy_min": 0.5,
        "energy_max": 1.0
    }
    response = test_app.post(
        "/api/v1/recommendations/features/",
        json=features
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert isinstance(data, list)

def test_invalid_song_id(test_app):
    """Test error handling for invalid song ID."""
    response = test_app.get("/api/v1/songs/999999")
    assert response.status_code == status.HTTP_404_NOT_FOUND

def test_invalid_file_format(test_app, test_song_data):
    """Test error handling for invalid file format."""
    # Create a text file instead of audio
    with open("test_file.txt", "w") as f:
        f.write("This is not an audio file")
    
    with open("test_file.txt", "rb") as f:
        files = {'file': ('test_file.txt', f, 'text/plain')}
        response = test_app.post(
            "/api/v1/songs/",
            files=files,
            data={'metadata': json.dumps(test_song_data)}
        )
    
    os.remove("test_file.txt")
    assert response.status_code == status.HTTP_400_BAD_REQUEST 