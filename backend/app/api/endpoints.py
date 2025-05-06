from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from typing import List, Optional
from sqlalchemy.orm import Session
from ..core.database import db_manager
from ..core.recommender import recommender
from ..core.models import SongCreate, SongInDB, RecommendationResponse
from .endpoints import health, songs, youtube, search, recommendations, monitoring
import tempfile
import os
import json
import logging

logger = logging.getLogger(__name__)

# Create main router
router = APIRouter()

# Include all sub-routers
logger.debug("Including health endpoints...")
router.include_router(health.router, prefix="/health", tags=["health"])

logger.debug("Including monitoring endpoints...")
router.include_router(monitoring.router, prefix="/monitoring", tags=["monitoring"])

logger.debug("Including song endpoints...")
router.include_router(songs.router, prefix="/songs", tags=["songs"])

logger.debug("Including YouTube endpoints...")
router.include_router(youtube.router, prefix="/youtube", tags=["youtube"])

logger.debug("Including search endpoints...")
router.include_router(search.router, prefix="/search", tags=["search"])

logger.debug("Including recommendation endpoints...")
router.include_router(recommendations.router, prefix="/recommendations", tags=["recommendations"])

def get_db():
    db = next(db_manager.get_db())
    try:
        yield db
    finally:
        db.close()

@router.post("/songs/", response_model=SongInDB)
async def create_song(
    audio_file: UploadFile = File(...),
    track_name: str = Form(...),
    track_artist: str = Form(...),
    track_album_name: Optional[str] = Form(None),
    playlist_genre: Optional[str] = Form(None),
    lyrics: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Create a new song entry with audio file and metadata"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename)[1]) as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        try:
            # Extract audio features
            audio_features = recommender.extract_audio_features(temp_path)
            
            # Create song data
            song_data = {
                "track_name": track_name,
                "track_artist": track_artist,
                "track_album_name": track_album_name,
                "playlist_genre": playlist_genre,
                "lyrics": lyrics,
                "audio_features": json.dumps(audio_features),
                # Initialize other features as empty for now
                "word2vec_features": json.dumps([0.0] * 100),  # Placeholder
                "sentiment_features": json.dumps({"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 0.0}),
                "topic_features": json.dumps([0.0] * 10)  # Placeholder
            }

            # Add song to database
            song = await db_manager.add_song(song_data)
            return song

        finally:
            # Clean up temporary file
            os.unlink(temp_path)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/songs/{song_id}", response_model=SongInDB)
async def get_song(song_id: int, db: Session = Depends(get_db)):
    """Get song details by ID"""
    song = db.query(SongInDB).filter(SongInDB.id == song_id).first()
    if not song:
        raise HTTPException(status_code=404, detail="Song not found")
    return song

@router.get("/songs/search/", response_model=List[SongInDB])
async def search_songs(
    query: str,
    artist: Optional[str] = None,
    genre: Optional[str] = None,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Search songs by name, artist, or genre"""
    search_query = db.query(SongInDB)
    
    if query:
        search_query = search_query.filter(SongInDB.track_name.ilike(f"%{query}%"))
    if artist:
        search_query = search_query.filter(SongInDB.track_artist.ilike(f"%{artist}%"))
    if genre:
        search_query = search_query.filter(SongInDB.playlist_genre.ilike(f"%{genre}%"))
    
    return search_query.limit(limit).all()

@router.get("/recommendations/similar/{song_id}", response_model=List[RecommendationResponse])
async def get_similar_songs(
    song_id: int,
    k: int = 5,
    feature_weights: Optional[dict] = None,
    db: Session = Depends(get_db)
):
    """Get song recommendations similar to a given song"""
    try:
        recommendations = recommender.recommend_similar_songs(
            song_id=song_id,
            k=k,
            weights=feature_weights
        )
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/recommendations/mood/{mood}", response_model=List[RecommendationResponse])
async def get_mood_recommendations(
    mood: str,
    k: int = 5,
    exclude_songs: Optional[List[int]] = None,
    db: Session = Depends(get_db)
):
    """Get song recommendations based on mood"""
    try:
        recommendations = recommender.recommend_by_mood(
            mood=mood,
            k=k,
            exclude_songs=exclude_songs
        )
        return [
            RecommendationResponse(
                track_name=song.track_name,
                track_artist=song.track_artist,
                track_album_name=song.track_album_name,
                playlist_genre=song.playlist_genre,
                similarity_score=score,
                component_scores={"audio": score},
                external_links={"spotify": f"spotify:track:{song.id}"}
            )
            for song, score in recommendations
        ]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/recommendations/features/", response_model=List[RecommendationResponse])
async def get_feature_recommendations(
    features: dict,
    feature_type: str,
    k: int = 5,
    exclude_songs: Optional[List[int]] = None,
    db: Session = Depends(get_db)
):
    """Get song recommendations based on specific features"""
    try:
        recommendations = recommender.recommend_by_features(
            features=features,
            feature_type=feature_type,
            k=k,
            exclude_songs=exclude_songs
        )
        return [
            RecommendationResponse(
                track_name=song.track_name,
                track_artist=song.track_artist,
                track_album_name=song.track_album_name,
                playlist_genre=song.playlist_genre,
                similarity_score=score,
                component_scores={feature_type: score},
                external_links={"spotify": f"spotify:track:{song.id}"}
            )
            for song, score in recommendations
        ]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) 