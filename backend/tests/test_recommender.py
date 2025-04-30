import pytest
import numpy as np
from app.core.recommender import recommender

def test_compute_similarity():
    """Test similarity computation."""
    # Create sample feature vectors
    v1 = np.array([1.0, 0.5, 0.3])
    v2 = np.array([0.9, 0.4, 0.2])
    
    similarity = recommender._compute_similarity(v1, v2)
    assert 0 <= similarity <= 1
    
    # Test identical vectors
    assert recommender._compute_similarity(v1, v1) == pytest.approx(1.0)
    
    # Test orthogonal vectors
    v3 = np.array([0.0, 1.0, 0.0])
    v4 = np.array([1.0, 0.0, 0.0])
    assert recommender._compute_similarity(v3, v4) == pytest.approx(0.0)

def test_get_similar_songs(mocker):
    """Test similar songs recommendation."""
    # Mock database query
    mock_songs = [
        {'id': 1, 'features': np.array([1.0, 0.5, 0.3])},
        {'id': 2, 'features': np.array([0.9, 0.4, 0.2])},
        {'id': 3, 'features': np.array([0.1, 0.2, 0.8])}
    ]
    mocker.patch('app.core.database.get_all_songs', return_value=mock_songs)
    
    # Test with reference song
    similar_songs = recommender.get_similar_songs(1, n=2)
    assert len(similar_songs) == 2
    assert similar_songs[0]['similarity'] >= similar_songs[1]['similarity']

def test_get_songs_by_mood(mocker):
    """Test mood-based recommendations."""
    # Mock database query
    mock_songs = [
        {'id': 1, 'sentiment': {'compound': 0.8}},
        {'id': 2, 'sentiment': {'compound': -0.5}},
        {'id': 3, 'sentiment': {'compound': 0.3}}
    ]
    mocker.patch('app.core.database.get_all_songs', return_value=mock_songs)
    
    # Test happy mood
    happy_songs = recommender.get_songs_by_mood('happy', n=2)
    assert len(happy_songs) == 2
    assert happy_songs[0]['sentiment']['compound'] >= happy_songs[1]['sentiment']['compound']
    
    # Test sad mood
    sad_songs = recommender.get_songs_by_mood('sad', n=2)
    assert len(sad_songs) == 2
    assert sad_songs[0]['sentiment']['compound'] <= sad_songs[1]['sentiment']['compound']

def test_get_songs_by_features(mocker):
    """Test feature-based recommendations."""
    # Mock database query
    mock_songs = [
        {'id': 1, 'features': {'tempo': 120, 'energy': 0.8}},
        {'id': 2, 'features': {'tempo': 90, 'energy': 0.4}},
        {'id': 3, 'features': {'tempo': 140, 'energy': 0.9}}
    ]
    mocker.patch('app.core.database.get_all_songs', return_value=mock_songs)
    
    # Test feature filtering
    features = {
        'tempo_min': 100,
        'tempo_max': 130,
        'energy_min': 0.7,
        'energy_max': 1.0
    }
    
    filtered_songs = recommender.get_songs_by_features(features)
    assert len(filtered_songs) == 1
    assert filtered_songs[0]['id'] == 1

def test_generate_playlist(mocker):
    """Test playlist generation."""
    # Mock database query
    mock_songs = [
        {'id': 1, 'features': np.array([1.0, 0.5, 0.3])},
        {'id': 2, 'features': np.array([0.9, 0.4, 0.2])},
        {'id': 3, 'features': np.array([0.1, 0.2, 0.8])}
    ]
    mocker.patch('app.core.database.get_all_songs', return_value=mock_songs)
    
    playlist = recommender.generate_playlist(seed_song_id=1, length=2)
    assert len(playlist) == 2
    assert playlist[0]['id'] != playlist[1]['id']

def test_invalid_mood():
    """Test error handling for invalid mood."""
    with pytest.raises(ValueError):
        recommender.get_songs_by_mood('invalid_mood')

def test_invalid_features():
    """Test error handling for invalid feature ranges."""
    features = {
        'tempo_min': 200,  # Too high
        'tempo_max': 100   # Lower than min
    }
    with pytest.raises(ValueError):
        recommender.get_songs_by_features(features) 