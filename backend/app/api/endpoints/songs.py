import os
import numpy as np
import librosa
from typing import Dict, Optional, Tuple, List
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query, Depends
from pydantic import BaseModel, HttpUrl
from . import youtube
from ...services.lyrics_service import lyrics_service
from ...core.database import db_manager, DatabaseManager, get_db
from ...models.models import Song, SongCreate, SongBase, SongUpdate
import tempfile
import json

router = APIRouter()
logger = logging.getLogger(__name__)

class YouTubeRequest(BaseModel):
    url: HttpUrl

class AudioFeatures(BaseModel):
    mfcc: List[float]
    spectral_contrast: List[float]
    chroma: List[float]
    tempo: float
    beats: List[int]
    harmonic: float
    percussive: float
    energy: float
    harmonicity: float
    rhythm_strength: float
    complexity: float

class SongResponse(SongBase):
    id: int
    added_date: str
    audio_features: Optional[AudioFeatures] = None
    
    class Config:
        from_attributes = True

class SongProcessor:
    def __init__(self, sample_rate: int = 22050, duration: Optional[int] = None):
        """Initialize the song processor.
        
        Args:
            sample_rate (int): Audio sample rate to use
            duration (Optional[int]): Duration to load in seconds, None for full song
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.hop_length = 512
        self.n_mels = 128
        self.n_fft = 2048
        
    def process_file(self, file_path: str) -> Dict:
        """Process an audio file and extract features.
        
        Args:
            file_path (str): Path to audio file
            
        Returns:
            Dict: Extracted features
        """
        try:
            # Load audio
            y, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
            
            # Extract features
            # Tempo and rhythm
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            
            # Energy and loudness
            rms = librosa.feature.rms(y=y)[0]
            energy = float(np.mean(rms) * 10)
            loudness = float(min(0, -60 + np.mean(rms) * 100))
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            
            # Voice and instrument detection
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            speechiness = float(np.mean(np.abs(mfccs[1:5])) * 0.3)
            
            # Zero crossing rate for noisiness
            zcr = librosa.feature.zero_crossing_rate(y)
            instrumentalness = float(1.0 - min(1.0, np.mean(zcr) * 10))
            
            # Live performance detection
            flatness = librosa.feature.spectral_flatness(y=y)
            liveness = float(np.mean(flatness) * 5)
            
            # Derived features
            acousticness = float(1.0 - (np.mean(spectral_centroids) / (sr/2)))
            danceability = float(np.mean(spectral_contrast) * 0.5 + 0.5)
            
            # Emotional content
            valence = 0.5
            if np.mean(rms) > 0.1:
                valence = float(min(1.0, np.mean(rms) * 5))
            
            # Return features in the expected format
            features = {
                'danceability': danceability,
                'energy': energy,
                'key': 0,  # We don't extract key for now
                'loudness': loudness,
                'mode': 0,  # We don't extract mode for now
                'speechiness': speechiness,
                'acousticness': acousticness,
                'instrumentalness': instrumentalness,
                'liveness': liveness,
                'valence': valence,
                'tempo': float(tempo),
                'duration_ms': float(len(y) / sr * 1000)
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error processing audio file {file_path}: {str(e)}")
            raise
            
    def process_youtube_url(self, url: str) -> Tuple[Dict, Dict]:
        """Process a YouTube URL - download audio and extract features.
        
        Args:
            url (str): YouTube URL
            
        Returns:
            Tuple[Dict, Dict]: (Audio features, Video metadata)
        """
        try:
            # Download audio
            audio_path, metadata = youtube.youtube_service.search_and_download(url)
            if not audio_path:
                raise ValueError(f"Failed to download audio from {url}")
                
            # Extract features
            features = self.process_file(audio_path)
            
            # Clean up downloaded file
            os.remove(audio_path)
            
            return features, metadata
            
        except Exception as e:
            logger.error(f"Error processing YouTube URL {url}: {str(e)}")
            raise
            
    def process_with_lyrics(self, title: str, artist: str) -> Dict:
        """Get lyrics and analyze them.
        
        Args:
            title (str): Song title
            artist (str): Artist name
            
        Returns:
            Dict: Lyrics and analysis results
        """
        try:
            lyrics_data = lyrics_service.get_lyrics(title, artist)
            if not lyrics_data:
                logger.warning(f"No lyrics found for {title} by {artist}")
                return {}
            return lyrics_data
            
        except Exception as e:
            logger.error(f"Error processing lyrics for {title} by {artist}: {str(e)}")
            return {}
            
    def _extract_mfcc(self, y: np.ndarray) -> List[float]:
        """Extract MFCC features."""
        return librosa.feature.mfcc(
            y=y,
            sr=self.sample_rate,
            n_mfcc=20,
            hop_length=self.hop_length
        ).mean(axis=1).tolist()
        
    def _extract_spectral_contrast(self, y: np.ndarray) -> List[float]:
        """Extract spectral contrast features."""
        return librosa.feature.spectral_contrast(
            y=y,
            sr=self.sample_rate,
            hop_length=self.hop_length
        ).mean(axis=1).tolist()
        
    def _extract_chroma(self, y: np.ndarray) -> List[float]:
        """Extract chromagram features."""
        return librosa.feature.chroma_stft(
            y=y,
            sr=self.sample_rate,
            hop_length=self.hop_length
        ).mean(axis=1).tolist()
        
    def _extract_tempo(self, y: np.ndarray) -> float:
        """Extract tempo."""
        return float(librosa.beat.tempo(
            y=y,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )[0])
        
    def _extract_beats(self, y: np.ndarray) -> List[int]:
        """Extract beat frames."""
        _, beats = librosa.beat.beat_track(
            y=y,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        return beats.tolist()
        
    def _extract_harmonic(self, y: np.ndarray) -> float:
        """Extract harmonic component."""
        return float(librosa.effects.harmonic(y).mean())
        
    def _extract_percussive(self, y: np.ndarray) -> float:
        """Extract percussive component."""
        return float(librosa.effects.percussive(y).mean())
        
    def _aggregate_features(self, features: Dict) -> Dict:
        """Create aggregated features from individual ones."""
        return {
            'energy': float(np.abs(features['percussive'])),
            'harmonicity': float(np.abs(features['harmonic'])),
            'rhythm_strength': float(len(features['beats']) / (self.duration or 30)),
            'complexity': float(np.std(features['mfcc'])),
        }

# Create global instance
song_processor = SongProcessor()

@router.post("/process/file", response_model=SongResponse)
async def process_audio_file(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    artist: Optional[str] = Form(None)
):
    """Process uploaded audio file and extract features."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name
            
        processor = SongProcessor()
        features = processor.process_file(temp_path)
        
        # Get lyrics if title and artist provided
        lyrics_data = {}
        if title and artist:
            lyrics_data = processor.process_with_lyrics(title, artist)
            
        # Create song entry
        song_data = {
            "track_name": title or file.filename,
            "track_artist": artist or "Unknown",
            "audio_features": features,  # Pass as dict, not JSON string
            "lyrics": lyrics_data.get("lyrics"),
            "clean_lyrics": lyrics_data.get("clean_lyrics"),
            "sentiment_features": lyrics_data.get("sentiment"),
            "topic_features": lyrics_data.get("topics"),
            "audio_path": temp_path,
            "is_original": True
        }
        
        # Save to database
        song = await db_manager.create_song(song_data)
        
        # Convert datetime to string for response
        response_data = {
            **song_data,
            "id": song.id,
            "added_date": song.added_date.isoformat(),
            "audio_features": AudioFeatures(**features) if features else None
        }
        return response_data
        
    except Exception as e:
        logger.error(f"Error processing audio file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'temp_path' in locals():
            os.remove(temp_path)

@router.post("/process/youtube", response_model=SongResponse)
async def process_youtube_url(request: YouTubeRequest):
    """Process YouTube URL and extract features."""
    try:
        processor = SongProcessor()
        features, metadata = processor.process_youtube_url(str(request.url))
        
        # Get lyrics if title and artist available in metadata
        lyrics_data = {}
        if metadata.get("title") and metadata.get("artist"):
            lyrics_data = processor.process_with_lyrics(
                metadata["title"], 
                metadata["artist"]
            )
            
        # Create song entry
        song_data = {
            "track_name": metadata.get("title", "Unknown"),
            "track_artist": metadata.get("artist", "Unknown"),
            "audio_features": features,  # Pass as dict, not JSON string
            "lyrics": lyrics_data.get("lyrics"),
            "clean_lyrics": lyrics_data.get("clean_lyrics"),
            "sentiment_features": lyrics_data.get("sentiment"),
            "topic_features": lyrics_data.get("topics"),
            "youtube_url": str(request.url),
            "is_original": False
        }
        
        # Save to database
        song = await db_manager.create_song(song_data)
        
        # Convert datetime to string for response
        response_data = {
            **song_data,
            "id": song.id,
            "added_date": song.added_date.isoformat(),
            "audio_features": AudioFeatures(**features) if features else None
        }
        return response_data
        
    except Exception as e:
        logger.error(f"Error processing YouTube URL: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=List[SongBase])
async def get_songs(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100)
):
    """
    Get a list of songs with pagination
    """
    try:
        songs = await db_manager.get_songs(skip=skip, limit=limit)
        return songs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{song_id}", response_model=SongResponse)
async def get_song(song_id: int):
    """
    Get a specific song by ID
    """
    try:
        song = await db_manager.get_song(song_id)
        if not song:
            raise HTTPException(status_code=404, detail="Song not found")
        return song
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/", response_model=Song)
async def create_song(
    song: SongCreate,
    db: DatabaseManager = Depends(get_db)
):
    """
    Create a new song
    """
    try:
        return await db.add_song(song)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{song_id}", response_model=Song)
async def update_song(
    song_id: str,
    song_update: SongUpdate,
    db: DatabaseManager = Depends(get_db)
):
    """
    Update a song
    """
    try:
        updated_song = await db.update_song(song_id, song_update)
        if not updated_song:
            raise HTTPException(status_code=404, detail="Song not found")
        return updated_song
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{song_id}")
async def delete_song(
    song_id: str,
    db: DatabaseManager = Depends(get_db)
):
    """
    Delete a song
    """
    try:
        success = await db.delete_song(song_id)
        if not success:
            raise HTTPException(status_code=404, detail="Song not found")
        return {"message": "Song deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search")
async def search_songs(
    query: Optional[str] = None,
    artist: Optional[str] = None,
    genre: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100)
):
    """Search for songs with optional filters"""
    try:
        songs = await db_manager.search_songs(
            query=query,
            artist=artist,
            genre=genre,
            skip=skip,
            limit=limit
        )
        return {
            "total": len(songs),
            "items": songs,
            "skip": skip,
            "limit": limit
        }
    except Exception as e:
        logger.error(f"Error searching songs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
