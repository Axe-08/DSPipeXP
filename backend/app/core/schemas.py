from typing import Optional, List, Dict, Any
from pydantic import BaseModel

class SongBase(BaseModel):
    name: str
    artist: str
    youtube_url: Optional[str] = None
    
class SongCreate(SongBase):
    features: Optional[List[float]] = None
    audio_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
class Song(SongBase):
    id: int
    features: Optional[List[float]] = None
    audio_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        orm_mode = True
        
class SongSearch(BaseModel):
    query: Optional[str] = None
    artist: Optional[str] = None
    genre: Optional[str] = None
    limit: int = 10
    
class SongRecommendation(BaseModel):
    song_id: int
    similarity_score: float
    
class SongRecommendations(BaseModel):
    recommendations: List[SongRecommendation]
    source_song_id: int 