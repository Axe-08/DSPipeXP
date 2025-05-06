# DSPipeXP Music Recommendation System API Documentation

## Base URL
```
http://localhost:8000/api/v1
```

## Error Handling
All endpoints follow a consistent error response format:
```json
{
    "detail": "Error message describing what went wrong"
}
```

Common HTTP status codes:
- 200: Success
- 400: Bad Request
- 404: Not Found
- 500: Internal Server Error

## Endpoints

### Songs

#### Create Song
```http
POST /songs/
```

Request Body (multipart/form-data):
- `audio_file`: Audio file (required)
- `track_name`: Song title (required)
- `track_artist`: Artist name (required)
- `track_album_name`: Album name (optional)
- `playlist_genre`: Genre (optional)
- `lyrics`: Song lyrics (optional)

Response:
```json
{
    "id": 1,
    "track_name": "Shape of You",
    "track_artist": "Ed Sheeran",
    "track_album_name": "÷",
    "playlist_genre": "pop",
    "lyrics": "The club isn't the best place to find a lover...",
    "audio_features": {
        "danceability": 0.825,
        "energy": 0.652,
        "key": 1,
        "loudness": -3.183,
        "mode": 0,
        "speechiness": 0.0802,
        "acousticness": 0.581,
        "instrumentalness": 0,
        "liveness": 0.0931,
        "valence": 0.931,
        "tempo": 95.977
    }
}
```

#### Get Songs
```http
GET /songs/
```

Response:
```json
[
    {
        "id": 1,
        "track_name": "Shape of You",
        "track_artist": "Ed Sheeran",
        "track_album_name": "÷",
        "playlist_genre": "pop",
        "audio_features": {...}
    }
]
```

#### Get Song by ID
```http
GET /songs/{song_id}
```

Response:
```json
{
    "id": 1,
    "track_name": "Shape of You",
    "track_artist": "Ed Sheeran",
    "track_album_name": "÷",
    "playlist_genre": "pop",
    "lyrics": "The club isn't the best place to find a lover...",
    "audio_features": {...},
    "word2vec_features": [...],
    "sentiment_features": {...},
    "topic_features": [...]
}
```

#### Update Song
```http
PUT /songs/{song_id}
```

Request Body:
```json
{
    "track_name": "Updated Title",
    "track_artist": "Updated Artist",
    "track_album_name": "Updated Album",
    "playlist_genre": "Updated Genre"
}
```

#### Delete Song
```http
DELETE /songs/{song_id}
```

#### Search Songs
```http
GET /songs/search
```

Query Parameters:
- `query`: Search term for song title (required)
- `artist`: Filter by artist name (optional)
- `genre`: Filter by genre (optional)
- `limit`: Number of results to return (default: 10)

Response:
```json
[
    {
        "id": 1,
        "track_name": "Shape of You",
        "track_artist": "Ed Sheeran",
        "track_album_name": "÷",
        "playlist_genre": "pop",
        "audio_features": {...}
    }
]
```

#### Add YouTube Song
```http
POST /songs/youtube
```

Request Body:
```json
{
    "url": "https://www.youtube.com/watch?v=...",
    "track_name": "Song Title",
    "track_artist": "Artist Name"
}
```

#### Batch Upload Songs
```http
POST /songs/batch
```

Request Body (multipart/form-data):
- `files`: List of audio files
- `metadata`: JSON string containing metadata for each file

### Recommendations

#### Get Similar Songs
```http
GET /recommendations/similar/{song_id}
```

Query Parameters:
- `k`: Number of recommendations (default: 5)
- `feature_weights`: Optional JSON object with feature weights

Response:
```json
[
    {
        "track_name": "Castle on the Hill",
        "track_artist": "Ed Sheeran",
        "track_album_name": "÷",
        "playlist_genre": "pop",
        "similarity_score": 0.92,
        "component_scores": {
            "audio": 0.92
        },
        "external_links": {
            "spotify": "spotify:track:2..."
        }
    }
]
```

#### Get Mood Recommendations
```http
GET /recommendations/mood/{mood}
```

Query Parameters:
- `k`: Number of recommendations (default: 5)
- `exclude_songs`: Optional list of song IDs to exclude

Response: Same format as similar songs.

#### Get Feature Recommendations
```http
POST /recommendations/features
```

Request Body:
```json
{
    "features": {
        "danceability": 0.8,
        "energy": 0.6,
        "valence": 0.9
    },
    "feature_type": "audio",
    "k": 5,
    "exclude_songs": [1, 2, 3]
}
```

Response: Same format as similar songs.

#### Analyze Song
```http
GET /recommendations/analyze/{song_id}
```

Response:
```json
{
    "audio_features": {...},
    "sentiment_analysis": {...},
    "topic_analysis": {...},
    "mood_prediction": "energetic"
}
```

## Development and Deployment

### Environment Variables
Required environment variables:
```
DATABASE_URL=postgresql+asyncpg://user:password@localhost/dbname
REDIS_URL=redis://localhost
GENIUS_API_KEY=your-genius-api-key
YOUTUBE_API_KEY=your-youtube-api-key
```

### Running with Docker
```bash
docker-compose up -d
```

### Running Tests
```bash
pytest backend/tests
``` 