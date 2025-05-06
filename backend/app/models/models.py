from sqlalchemy import Column, Integer, String, JSON, Boolean, DateTime, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, HttpUrl

Base = declarative_base()

class Song(Base):
    __tablename__ = "songs"

    id = Column(Integer, primary_key=True, index=True)
    track_name = Column(String, nullable=False, index=True)
    track_artist = Column(String, nullable=False, index=True)
    track_album_name = Column(String, nullable=True)
    playlist_genre = Column(String, nullable=True, index=True)
    lyrics = Column(String, nullable=True)
    clean_lyrics = Column(String, nullable=True)
    word2vec_features = Column(JSON, nullable=True)
    audio_features = Column(JSON, nullable=True)
    sentiment_features = Column(JSON, nullable=True)
    topic_features = Column(JSON, nullable=True)
    youtube_url = Column(String, nullable=True)
    audio_path = Column(String, nullable=True)
    is_original = Column(Boolean, default=False)
    added_date = Column(DateTime, server_default=text('now()'))

    def __repr__(self):
        return f"<Song(id={self.id}, track_name='{self.track_name}', track_artist='{self.track_artist}')>"

# Pydantic models for API
class SongBase(BaseModel):
    track_name: str
    track_artist: str
    track_album_name: Optional[str] = None
    playlist_genre: Optional[str] = None
    youtube_url: Optional[str] = None
    audio_path: Optional[str] = None
    is_original: bool = False
    lyrics: Optional[str] = None
    clean_lyrics: Optional[str] = None
    word2vec_features: Optional[Dict] = None
    audio_features: Optional[Dict] = None
    sentiment_features: Optional[Dict] = None
    topic_features: Optional[List[float]] = None

class SongCreate(SongBase):
    pass

class Song(SongBase):
    id: int
    added_date: datetime

    class Config:
        from_attributes = True

class SongFeatures(BaseModel):
    word2vec_features: List[float]
    audio_features: Dict[str, float]
    sentiment_features: Dict[str, float]
    topic_features: List[float]

class RecommendationResponse(BaseModel):
    track_name: str
    track_artist: str
    track_album_name: Optional[str] = None
    playlist_genre: Optional[str] = None
    similarity_score: float
    component_scores: Dict[str, float]
    external_links: Optional[Dict[str, str]] = None

class Recommendation(BaseModel):
    input_song: Optional[Dict] = None
    input_features: Optional[Dict] = None
    recommendations: List[RecommendationResponse]
    similarity_scores: List[float]

class SongUpdate(BaseModel):
    track_name: Optional[str] = None
    track_artist: Optional[str] = None
    track_album_name: Optional[str] = None
    playlist_genre: Optional[str] = None
    lyrics: Optional[str] = None
    clean_lyrics: Optional[str] = None
    word2vec_features: Optional[Dict[str, Any]] = None
    audio_features: Optional[Dict[str, Any]] = None
    sentiment_features: Optional[Dict[str, Any]] = None
    topic_features: Optional[Dict[str, Any]] = None
    youtube_url: Optional[str] = None
    audio_path: Optional[str] = None
    is_original: Optional[bool] = None

    class Config:
        from_attributes = True
