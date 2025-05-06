from typing import List, Dict, Optional
from pydantic import BaseModel

class RecommendationResponse(BaseModel):
    track_name: str
    track_artist: str
    track_album_name: Optional[str] = None
    playlist_genre: Optional[str] = None
    similarity_score: float
    component_scores: Dict[str, float]
    external_links: Optional[Dict[str, str]] = None

    class Config:
        from_attributes = True

class Recommendation(BaseModel):
    input_song: Optional[Dict] = None
    input_features: Optional[Dict] = None
    recommendations: List[Dict]
    similarity_scores: List[float]

    class Config:
        from_attributes = True 