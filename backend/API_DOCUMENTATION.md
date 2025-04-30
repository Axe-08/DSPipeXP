# DSPipeXP API Documentation

## Overview
DSPipeXP is a music processing and recommendation API that provides endpoints for audio processing, YouTube integration, song management, search functionality, and system monitoring.

## Base URL
```
/api/v1
```

## Authentication
Authentication is not implemented in the current version.

## API Endpoints

### Health Check

#### GET `/health`
Check if the API is running.

**Response**
```json
{
  "status": "healthy",
  "message": "API is running"
}
```

### Songs

#### POST `/songs/process/file`
Process an uploaded audio file and extract features.

**Request**
- Form Data:
  - `file`: Audio file (required)
  - `title`: Song title (optional)
  - `artist`: Artist name (optional)

**Response**
- `200`: SongResponse object with extracted features
- `400`: Bad request if file processing fails

#### POST `/songs/process/youtube`
Process a YouTube URL and extract features.

**Request Body**
```json
{
  "url": "https://youtube.com/watch?v=..."
}
```

**Response**
- `200`: SongResponse object with extracted features
- `400`: Bad request if URL processing fails

#### GET `/songs`
Get a list of songs with pagination.

**Query Parameters**
- `skip`: Number of records to skip (default: 0)
- `limit`: Number of records to return (default: 10, max: 100)

**Response**
- `200`: Array of SongBase objects
- `404`: If no songs found

#### GET `/songs/{song_id}`
Get details of a specific song.

**Path Parameters**
- `song_id`: ID of the song

**Response**
- `200`: SongResponse object
- `404`: If song not found

#### POST `/songs`
Create a new song entry.

**Request Body**
- SongCreate object

**Response**
- `200`: Created Song object
- `400`: Bad request if creation fails

#### PUT `/songs/{song_id}`
Update an existing song.

**Path Parameters**
- `song_id`: ID of the song

**Request Body**
- SongUpdate object

**Response**
- `200`: Updated Song object
- `404`: If song not found

#### DELETE `/songs/{song_id}`
Delete a song.

**Path Parameters**
- `song_id`: ID of the song

**Response**
- `200`: Success message
- `404`: If song not found

### YouTube Integration

#### POST `/youtube/process`
Process a YouTube URL to download audio and extract metadata.

**Request Body**
```json
{
  "url": "https://youtube.com/watch?v=..."
}
```

**Response**
```json
{
  "file_path": "path/to/downloaded/file.mp3",
  "metadata": {
    "title": "Video Title",
    "duration": 180,
    "view_count": 1000000,
    "channel": "Channel Name",
    "video_id": "video_id"
  }
}
```

#### GET `/youtube/metadata`
Extract metadata from a YouTube URL without downloading.

**Query Parameters**
- `url`: YouTube video URL

**Response**
```json
{
  "title": "Video Title",
  "duration": 180,
  "view_count": 1000000,
  "channel": "Channel Name",
  "video_id": "video_id"
}
```

#### GET `/youtube/search`
Search YouTube for a video.

**Query Parameters**
- `query`: Search query

**Response**
```json
{
  "url": "https://youtube.com/watch?v=..."
}
```

### Search

#### GET `/search`
Universal search endpoint for songs.

**Query Parameters**
- `query`: Search query (optional)
- `artist`: Artist name (optional)
- `genre`: Genre (optional)
- `mood`: Mood (optional)
- `skip`: Number of records to skip (default: 0)
- `limit`: Number of records to return (default: 10, max: 100)

**Response**
- `200`: Array of Song objects
- `404`: If no songs found

#### GET `/search/by-name/{name}`
Search for a specific song by name.

**Path Parameters**
- `name`: Song name

**Response**
- `200`: Song object
- `404`: If song not found

#### GET `/search/by-mood/{mood}`
Search for songs by mood.

**Path Parameters**
- `mood`: Mood to search for

**Query Parameters**
- `skip`: Number of records to skip (default: 0)
- `limit`: Number of records to return (default: 10, max: 100)

**Response**
- `200`: Array of Song objects
- `404`: If no songs found

#### GET `/search/by-genre/{genre}`
Search for songs by genre.

**Path Parameters**
- `genre`: Genre to search for

**Query Parameters**
- `skip`: Number of records to skip (default: 0)
- `limit`: Number of records to return (default: 10, max: 100)

**Response**
- `200`: Array of Song objects
- `404`: If no songs found

### Recommendations

#### GET `/recommendations/{song_id}`
Get song recommendations based on a song ID.

**Path Parameters**
- `song_id`: ID of the reference song

**Query Parameters**
- `limit`: Number of recommendations to return (default: 10)

**Response**
- `200`: Recommendation object
- `404`: If reference song not found

#### GET `/recommendations/by-features`
Get song recommendations based on audio features.

**Query Parameters**
- `features`: Dictionary of audio features
- `limit`: Number of recommendations to return (default: 10)

**Response**
- `200`: Recommendation object
- `400`: If features are invalid

### Monitoring

#### GET `/monitoring/stats`
Get system statistics including cache, storage, and performance metrics.

**Response**
```json
{
  "cache": {},
  "storage": {},
  "database": {},
  "system": {
    "cpu_percent": 0.0,
    "memory_percent": 0.0,
    "memory_used_gb": 0.0,
    "disk_percent": 0.0,
    "disk_used_gb": 0.0
  }
}
```

#### POST `/monitoring/cleanup`
Manually trigger file cleanup.

**Response**
```json
{
  "status": "success",
  "cleaned_files": [],
  "count": 0
}
```

## Data Models

### SongBase
```python
class SongBase:
    track_name: str
    track_artist: Optional[str]
    youtube_url: Optional[str]
```

### SongResponse (extends SongBase)
```python
class SongResponse(SongBase):
    id: int
    added_date: str
    audio_features: Optional[AudioFeatures]
```

### AudioFeatures
```python
class AudioFeatures:
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
```

### YouTubeRequest
```python
class YouTubeRequest:
    url: HttpUrl
```

### YouTubeResponse
```python
class YouTubeResponse:
    file_path: Optional[str]
    metadata: Optional[Dict]
```

## Error Handling
The API uses standard HTTP status codes:
- `200`: Success
- `400`: Bad Request
- `404`: Not Found
- `500`: Internal Server Error

Error responses include a detail message:
```json
{
  "detail": "Error message"
}
```

## Rate Limiting
Rate limiting is not implemented in the current version.

## Dependencies
- Python 3.10+
- FastAPI
- pytube
- librosa
- FFmpeg (for audio conversion)
- PostgreSQL (for database)
- Redis (for caching)

## Development Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables in `.env`
4. Run migrations: `alembic upgrade head`
5. Start the server: `uvicorn run:app --reload`

## Docker Setup
1. Build the image: `docker build -t dspipexp .`
2. Run with docker-compose: `docker-compose up`

For production deployment, use: `docker-compose -f docker-compose.prod.yml up` 