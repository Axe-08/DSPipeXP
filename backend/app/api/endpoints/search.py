from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form
from typing import List, Optional, Dict, Any
from ...models.models import Song, SongCreate
from ...core.database import DatabaseManager, get_db, db_manager
from . import youtube
from .songs import SongProcessor
from ...core.recommender import recommender
import json
import os
from pydantic import BaseModel, Field
from ...services.lyrics_service import lyrics_service
import logging

router = APIRouter()

def parse_json_fields(song):
    """Parse JSON string fields into dictionaries"""
    if song.audio_features and isinstance(song.audio_features, str):
        song.audio_features = json.loads(song.audio_features)
    if song.word2vec_features and isinstance(song.word2vec_features, str):
        song.word2vec_features = json.loads(song.word2vec_features)
    if song.sentiment_features and isinstance(song.sentiment_features, str):
        song.sentiment_features = json.loads(song.sentiment_features)
    if song.topic_features and isinstance(song.topic_features, str):
        song.topic_features = json.loads(song.topic_features)
    return song

class SearchResponse(BaseModel):
    song: Song
    recommendations: List[Song]

class FeatureRecommendationRequest(BaseModel):
    song_id: int = Field(..., description="ID of the song to get recommendations for")
    feature_type: str = Field(..., description="Type of features to use for recommendations", pattern="^(audio|sentiment|topic)$")
    limit: int = Field(10, ge=1, le=50, description="Number of recommendations to return")

@router.get("/", response_model=List[Song])
async def search_songs(
    query: Optional[str] = None,
    artist: Optional[str] = None,
    genre: Optional[str] = None,
    mood: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100)
):
    """
    Universal search endpoint that can search by any combination of parameters
    """
    try:
        # Validate at least one search parameter is provided
        if not any([query, artist, genre, mood]):
            raise HTTPException(
                status_code=400,
                detail="At least one search parameter (query, artist, genre, or mood) is required"
            )

        # Validate input lengths
        for param_name, param_value in [
            ("query", query),
            ("artist", artist),
            ("genre", genre),
            ("mood", mood)
        ]:
            if param_value and len(param_value) < 2:
                raise HTTPException(
                    status_code=400,
                    detail=f"{param_name} must be at least 2 characters long"
                )

        songs = await db_manager.search_songs(
            query=query,
            artist=artist,
            genre=genre,
            mood=mood,
            skip=skip,
            limit=limit
        )
        
        if not songs:
            raise HTTPException(
                status_code=404,
                detail="No songs found matching the search criteria"
            )
            
        return [parse_json_fields(song) for song in songs]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in search_songs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while searching for songs"
        )

@router.get("/by-name/{name}", response_model=SearchResponse)
async def search_by_name(name: str, rec_limit: int = 5):
    """Search for a specific song by name and get recommendations"""
    try:
        # First try to find in database
        songs = await db_manager.search_songs(query=name, limit=1)
        song = None
        
        if songs:
            song = parse_json_fields(songs[0])
            # Get recommendations for the found song
            recommendations = await recommender.get_recommendations(song.id, limit=rec_limit)
            
            # Process each recommendation to ensure it has lyrics and YouTube URL
            for rec in recommendations.recommendations:
                rec_song = await db_manager.get_song(rec['id'])
                
                # Get lyrics if not present
                if not rec_song.lyrics and rec_song.track_name and rec_song.track_artist:
                    lyrics_data = await lyrics_service.get_lyrics(rec_song.track_name, rec_song.track_artist)
                    if lyrics_data:
                        await db_manager.update_song(rec_song.id, {
                            'lyrics': lyrics_data['lyrics'],
                            'clean_lyrics': lyrics_data['clean_lyrics'],
                            'sentiment_features': lyrics_data['sentiment'],
                            'topic_features': lyrics_data['topics']
                        })
                        rec_song = await db_manager.get_song(rec_song.id)
                
                # Get YouTube URL if not present
                if not rec_song.youtube_url:
                    query = f"{rec_song.track_name} {rec_song.track_artist} official audio"
                    rec_song.youtube_url = youtube.search(query)
                    if rec_song.youtube_url:
                        await db_manager.update_song(rec_song.id, {
                            'youtube_url': rec_song.youtube_url
                        })
                
                # Update recommendation with full song data
                rec.update({
                    'lyrics': rec_song.lyrics,
                    'youtube_url': rec_song.youtube_url,
                    'audio_features': rec_song.audio_features,
                    'sentiment_features': rec_song.sentiment_features,
                    'topic_features': rec_song.topic_features
                })
            
            return SearchResponse(song=song, recommendations=recommendations.recommendations)
            
        # If not found, search YouTube
        video_url = youtube.search(name)
        if not video_url:
            raise HTTPException(status_code=404, detail=f"Song '{name}' not found")
            
        # Download audio and extract features
        audio_path, metadata = youtube.download_audio(video_url)
        if not audio_path:
            raise HTTPException(status_code=500, detail="Failed to download audio")
            
        # Process audio features
        processor = SongProcessor()
        
        # Get the actual file name from the directory
        actual_file = None
        for file in os.listdir(os.path.dirname(audio_path)):
            if file.lower().replace('[', '(').replace(']', ')') == os.path.basename(audio_path).lower().replace('[', '(').replace(']', ')'):
                actual_file = os.path.join(os.path.dirname(audio_path), file)
                break
        
        if not actual_file:
            raise HTTPException(status_code=500, detail=f"File not found: {audio_path}")
            
        features = processor.process_file(actual_file)
            
        # Get lyrics
        lyrics_data = await lyrics_service.get_lyrics(metadata['title'], metadata['channel'])
            
        # Add song to database
        song_data = {
            'track_name': metadata['title'],
            'track_artist': metadata['channel'],
            'youtube_url': video_url,
            'audio_path': actual_file.replace('/app/', ''),  # Convert to relative path
            'audio_features': features,
            'lyrics': lyrics_data['lyrics'] if lyrics_data else None,
            'clean_lyrics': lyrics_data['clean_lyrics'] if lyrics_data else None,
            'sentiment_features': lyrics_data['sentiment'] if lyrics_data else None,
            'topic_features': lyrics_data['topics'] if lyrics_data else None,
            'is_original': False
        }
        
        song = await db_manager.create_song(song_data)
        song = parse_json_fields(song)
        
        # Get recommendations for the new song
        recommendations = await recommender.get_recommendations(song.id, limit=rec_limit)
        
        # Process recommendations same as above
        for rec in recommendations.recommendations:
            rec_song = await db_manager.get_song(rec['id'])
            
            # Get lyrics if not present
            if not rec_song.lyrics and rec_song.track_name and rec_song.track_artist:
                lyrics_data = await lyrics_service.get_lyrics(rec_song.track_name, rec_song.track_artist)
                if lyrics_data:
                    await db_manager.update_song(rec_song.id, {
                        'lyrics': lyrics_data['lyrics'],
                        'clean_lyrics': lyrics_data['clean_lyrics'],
                        'sentiment_features': lyrics_data['sentiment'],
                        'topic_features': lyrics_data['topics']
                    })
                    rec_song = await db_manager.get_song(rec_song.id)
            
            # Get YouTube URL if not present
            if not rec_song.youtube_url:
                query = f"{rec_song.track_name} {rec_song.track_artist} official audio"
                rec_song.youtube_url = youtube.search(query)
                if rec_song.youtube_url:
                    await db_manager.update_song(rec_song.id, {
                        'youtube_url': rec_song.youtube_url
                    })
            
            # Update recommendation with full song data
            rec.update({
                'lyrics': rec_song.lyrics,
                'youtube_url': rec_song.youtube_url,
                'audio_features': rec_song.audio_features,
                'sentiment_features': rec_song.sentiment_features,
                'topic_features': rec_song.topic_features
            })
        
        return SearchResponse(song=song, recommendations=recommendations.recommendations)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/by-mood/{mood}", response_model=List[SearchResponse])
async def search_by_mood(
    mood: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    rec_limit: int = 5
):
    """
    Search for songs by mood and get recommendations for each
    """
    try:
        songs = await db_manager.search_songs(mood=mood, skip=skip, limit=limit)
        if not songs:
            raise HTTPException(status_code=404, detail=f"No songs found with mood '{mood}'")
            
        results = []
        for song in songs:
            song = parse_json_fields(song)
            recommendations = await recommender.get_recommendations(song.id, limit=rec_limit)
            results.append(SearchResponse(song=song, recommendations=recommendations.recommendations))
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/by-genre/{genre}", response_model=List[SearchResponse])
async def search_by_genre(
    genre: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    rec_limit: int = 5
):
    """
    Search for songs by genre and get recommendations for each
    """
    try:
        songs = await db_manager.search_songs(genre=genre, skip=skip, limit=limit)
        if not songs:
            raise HTTPException(status_code=404, detail=f"No songs found with genre '{genre}'")
            
        results = []
        for song in songs:
            song = parse_json_fields(song)
            recommendations = await recommender.get_recommendations(song.id, limit=rec_limit)
            results.append(SearchResponse(song=song, recommendations=recommendations.recommendations))
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recommendations/features")
async def get_feature_recommendations(request: FeatureRecommendationRequest):
    """Get song recommendations based on feature similarity"""
    try:
        recommendations = await db_manager.get_feature_recommendations(
            request.song_id,
            request.feature_type,
            request.limit
        )
        if not recommendations:
            raise HTTPException(status_code=404, detail="No recommendations found")
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 