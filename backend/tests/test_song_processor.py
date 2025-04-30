import pytest
import numpy as np
from app.core.song_processor import song_processor

def test_process_file(test_song_file):
    """Test audio file processing."""
    features = song_processor.process_file(test_song_file)
    
    # Check all expected features are present
    assert 'mfcc' in features
    assert 'spectral_contrast' in features
    assert 'chroma' in features
    assert 'tempo' in features
    assert 'beats' in features
    assert 'harmonic' in features
    assert 'percussive' in features
    
    # Check aggregated features
    assert 'energy' in features
    assert 'harmonicity' in features
    assert 'rhythm_strength' in features
    assert 'complexity' in features
    
    # Check feature shapes
    assert isinstance(features['mfcc'], np.ndarray)
    assert len(features['mfcc']) == 20  # Number of MFCC coefficients
    assert isinstance(features['tempo'], float)
    assert features['tempo'] > 0

def test_process_with_lyrics(mocker):
    """Test lyrics processing with mocked Genius API."""
    mock_lyrics = {
        'lyrics': 'Test lyrics',
        'title': 'Test Song',
        'artist': 'Test Artist',
        'sentiment': {
            'compound': 0.5,
            'positive': 0.6,
            'negative': 0.1,
            'neutral': 0.3
        }
    }
    
    # Mock the lyrics service
    mocker.patch('app.core.lyrics_service.get_lyrics', return_value=mock_lyrics)
    
    result = song_processor.process_with_lyrics('Test Song', 'Test Artist')
    
    assert result == mock_lyrics
    assert 'lyrics' in result
    assert 'sentiment' in result
    
def test_process_file_invalid_path():
    """Test error handling for invalid file path."""
    with pytest.raises(Exception):
        song_processor.process_file('nonexistent_file.mp3')
        
def test_process_with_lyrics_no_results(mocker):
    """Test handling of no lyrics found."""
    # Mock the lyrics service to return None
    mocker.patch('app.core.lyrics_service.get_lyrics', return_value=None)
    
    result = song_processor.process_with_lyrics('Nonexistent Song', 'Unknown Artist')
    
    assert result == {} 