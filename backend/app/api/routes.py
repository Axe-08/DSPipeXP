from fastapi import APIRouter
from . import endpoints

# Create routers
songs_router = APIRouter(prefix="/songs", tags=["songs"])
recommendations_router = APIRouter(prefix="/recommendations", tags=["recommendations"])

# Song routes
songs_router.add_api_route("/", endpoints.create_song, methods=["POST"])
songs_router.add_api_route("/", endpoints.get_songs, methods=["GET"])
songs_router.add_api_route("/{song_id}", endpoints.get_song, methods=["GET"])
songs_router.add_api_route("/{song_id}", endpoints.update_song, methods=["PUT"])
songs_router.add_api_route("/{song_id}", endpoints.delete_song, methods=["DELETE"])
songs_router.add_api_route("/search/", endpoints.search_songs, methods=["GET"])
songs_router.add_api_route("/youtube/", endpoints.add_youtube_song, methods=["POST"])
songs_router.add_api_route("/batch/", endpoints.batch_upload_songs, methods=["POST"])

# Recommendation routes
recommendations_router.add_api_route("/similar/{song_id}", endpoints.get_similar_songs, methods=["GET"])
recommendations_router.add_api_route("/mood/{mood}", endpoints.get_mood_recommendations, methods=["GET"])
recommendations_router.add_api_route("/features/", endpoints.get_feature_recommendations, methods=["POST"])
recommendations_router.add_api_route("/analyze/{song_id}", endpoints.analyze_song, methods=["GET"])
