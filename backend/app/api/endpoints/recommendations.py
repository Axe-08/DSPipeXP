from fastapi import APIRouter, HTTPException, Depends
from typing import List
from app.core.recommender import MusicRecommender
from app.models.song import Song
from app.models.recommendation import Recommendation

router = APIRouter()
recommender = MusicRecommender()

@router.get("/{song_id}", response_model=Recommendation)
async def get_recommendations(song_id: int, limit: int = 10):
    """Get song recommendations based on a song ID."""
    try:
        recommendations = await recommender.get_recommendations(song_id, limit=limit)
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/by-features", response_model=Recommendation)
async def get_recommendations_by_features(features: dict, limit: int = 10):
    """Get song recommendations based on audio features."""
    try:
        recommendations = await recommender.get_recommendations_by_features(features, limit=limit)
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) 