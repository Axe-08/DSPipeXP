# Music Recommendation System - Project Plan

## 1. System Architecture

### 1.1 Backend Components
- **Database Manager** (`backend/app/core/database.py`)
  - PostgreSQL integration (replacing SQLite)
  - FAISS vector store for efficient similarity search
  - Database initialization and migrations
  - Data loading and management
  - Efficient feature matrix updates
  - Persistent storage between deployments

- **YouTube Service** (`backend/app/core/youtube_service.py`)
  - Video search and metadata extraction
  - Audio download and conversion
  - Efficient file handling
  - Error management and retries
  - Metadata caching

- **Lyrics Service** (`backend/app/core/lyrics_service.py`)
  - Genius API integration for lyrics fetching
  - Sentiment analysis of lyrics
  - Caching of lyrics and analysis results
  - Error handling and rate limiting
  - Requires free Genius API access token (GENIUS_ACCESS_TOKEN)

- **Recommendation Engine** (`backend/app/core/recommender.py`)
  - Optimized feature extraction (audio, text, sentiment)
  - Multiple recommendation strategies
  - Hybrid recommendation system
  - Vectorized similarity computation
  - FAISS-based fast similarity search
  - Caching layer for frequent queries

- **API Layer** (`backend/app/api/endpoints.py`)
  - RESTful endpoints
  - File upload handling
  - Error handling
  - Response models
  - Rate limiting
  - Redis caching

### 1.2 Data Models
- **Song Model**
  - Basic metadata (title, artist, album)
  - Pre-computed audio features (float32)
  - Pre-computed text features (float32)
  - Genre and playlist information
  - YouTube metadata (if applicable)

- **Feature Vectors**
  - Word2Vec embeddings (100 dimensions, float32)
  - Audio features matrix (optimized numpy array)
  - Sentiment features (float32)
  - Topic features (10 dimensions, float32)
  - FAISS index for fast similarity search

### 1.3 Optimization Strategy
1. **Feature Processing**
   - Pre-compute all features during song upload
   - Store raw features in PostgreSQL
   - Load into optimized numpy arrays on startup
   - Use float32 instead of float64 where possible
   - Incremental FAISS index updates

2. **Memory Management**
   - Load only necessary features into memory
   - Efficient numpy array operations
   - Proper cleanup of temporary files
   - Streaming processing for large files

3. **Caching Strategy**
   - Redis cache for frequent recommendations
   - In-memory LRU cache for feature vectors
   - Cached FAISS indices
   - Periodic cache invalidation

## 2. Features and Functionality

### 2.1 Core Features
1. **Song Management**
   - Upload new songs with metadata
   - Extract audio features
   - Store and manage song data
   - Search and retrieve songs

2. **Audio Processing**
   - Format conversion and normalization
   - Feature extraction using librosa
   - Spectral and temporal analysis
   - Audio fingerprinting

3. **Recommendation System**
   - Similar song recommendations
   - Mood-based recommendations
   - Feature-based filtering
   - Hybrid recommendations

### 2.2 Advanced Features
1. **Audio Analysis**
   - Tempo detection
   - Key detection
   - Genre classification
   - Mood analysis

2. **Text Analysis**
   - Lyrics processing
   - Sentiment analysis
   - Topic modeling
   - Word embeddings

3. **Recommendation Strategies**
   - Content-based filtering
   - Collaborative filtering
   - Hybrid approaches
   - Contextual recommendations

## 3. Implementation Plan

### 3.1 Phase 1: Core Infrastructure ✓
1. Set up project structure
2. Implement PostgreSQL database models
3. Create optimized database manager
4. Basic API endpoints
5. FAISS integration

### 3.2 Phase 2: Feature Extraction ✓
1. Optimized audio feature extraction
2. Efficient text feature extraction
3. Vectorized sentiment analysis
4. Topic modeling with caching
5. Feature compression and optimization

### 3.3 Phase 3: Recommendation Engine ✓
1. FAISS-based similarity computation
2. Multiple recommendation strategies
3. Hybrid recommendation system
4. Redis caching layer
5. Performance optimization

### 3.4 Phase 4: API Development ✓
1. RESTful endpoints
2. File upload handling
3. Error handling
4. Response models

### 3.5 Phase 5: Testing and Optimization (In Progress)
1. Unit tests
2. Integration tests
3. Performance testing
4. Security auditing

### 3.6 Phase 6: Frontend Development (Pending)
1. User interface design
2. React/Vue.js implementation
3. API integration
4. User experience optimization

### 3.7 Phase 7: Deployment (In Progress)
1. **Environment Setup**
   - Configure environment variables
   - Set up PostgreSQL on Render
   - Configure Redis cache (if needed)
   - Set up S3/cloud storage for audio files

2. **Deployment Steps**
   - Create Render web service
   - Configure build and start commands
   - Set up environment variables:
     - DATABASE_URL
     - VECTOR_STORE_PATH
     - GENIUS_ACCESS_TOKEN (if using lyrics service)
     - Other API keys as needed

3. **Post-Deployment**
   - Monitor application logs
   - Set up error tracking
   - Configure backup strategy
   - Implement health checks

## 4. Technical Stack

### 4.1 Backend
- **Framework**: FastAPI
- **Database**: PostgreSQL (on Render/Railway)
- **ORM**: SQLAlchemy
- **Feature Extraction**: librosa, numpy
- **ML/AI**: scikit-learn, gensim, FAISS
- **Audio Processing**: ffmpeg, soundfile
- **Lyrics**: Genius API (free), vaderSentiment
- **Caching**: Redis
- **Storage**: Cloud (S3/similar)

### 4.2 Frontend (Planned)
- **Framework**: React/Vue.js
- **State Management**: Redux/Vuex
- **UI Components**: Material-UI/Vuetify
- **API Client**: Axios

### 4.3 Infrastructure
- **Version Control**: Git
- **CI/CD**: GitHub Actions
- **Containerization**: Docker
- **Cache**: Redis (optional)

## 5. API Endpoints

### 5.1 Song Management
```
POST /api/v1/songs/                    # Upload and process a new song
GET /api/v1/songs/{song_id}           # Get song details
GET /api/v1/songs/search              # Search songs by name
POST /api/v1/songs/process/file       # Process uploaded audio file
POST /api/v1/songs/process/youtube    # Process YouTube URL
```

### 5.2 Search and Discovery
```
GET /api/v1/songs/search/by-name      # Search song by name with auto-YouTube fallback
GET /api/v1/songs/search/by-mood      # Search songs by mood
GET /api/v1/songs/search/by-genre     # Search songs by genre
```

### 5.3 Recommendations
```
GET /api/v1/recommendations/similar/{song_id}  # Get similar songs
GET /api/v1/recommendations/mood/{mood}        # Get mood-based recommendations
GET /api/v1/recommendations/genre/{genre}      # Get genre-based recommendations
```

## 6. Data Flow

1. **Song Upload**
   ```
   Client -> API -> Feature Extraction -> Database
   ```

2. **Recommendation Request**
   ```
   Client -> API -> Recommendation Engine -> Database -> Response
   ```

3. **Search Flow**
   ```
   Client -> API -> Database -> Vector Store -> Response
   ```

## 7. Testing Strategy

### 7.1 Unit Tests
- Database operations
- Feature extraction
- Recommendation algorithms
- API endpoints

### 7.2 Integration Tests
- End-to-end workflows
- API integration
- Database integration
- Feature extraction pipeline

### 7.3 Performance Tests
- Load testing
- Stress testing
- Scalability testing
- Response time benchmarking

## 8. Deployment

### 8.1 Development
- Local development setup
- Environment configuration
- Database setup
- Development server

### 8.2 Production
- Server provisioning
- Database setup
- Environment configuration
- Monitoring setup
- Backup strategy

## 9. Security Considerations

1. **API Security**
   - Rate limiting
   - Input validation
   - Authentication/Authorization
   - CORS configuration

2. **Data Security**
   - Secure file uploads
   - Data encryption
   - Backup strategy
   - Access control

3. **Infrastructure Security**
   - Firewall configuration
   - SSL/TLS setup
   - Regular security updates
   - Monitoring and logging

## 10. Future Enhancements

1. **Feature Enhancements**
   - Real-time recommendations
   - User feedback integration
   - Playlist generation
   - Social features

2. **Technical Improvements**
   - Caching optimization
   - Distributed processing
   - Machine learning improvements
   - Mobile app integration

3. **User Experience**
   - Personalization
   - Advanced search
   - Visualization features
   - Social sharing

## 11. Maintenance Plan

1. **Regular Tasks**
   - Database backups
   - Log rotation
   - Performance monitoring
   - Security updates

2. **Periodic Reviews**
   - Code quality
   - Performance metrics
   - Security audits
   - User feedback

3. **Documentation**
   - API documentation
   - System architecture
   - Deployment guides
   - User guides

## 12. Performance Optimization

### 12.1 Database Optimization
- Efficient indexing strategies
- Proper data types (float32)
- Batch operations
- Connection pooling
- Query optimization

### 12.2 Feature Processing
- Vectorized operations
- Parallel processing where applicable
- Memory-efficient algorithms
- Incremental updates
- Compression techniques

### 12.3 Similarity Search
- FAISS indexing
- Approximate nearest neighbors
- Dimension reduction where applicable
- Index partitioning
- GPU acceleration (if available)

### 12.4 Caching Strategy
- Multi-level caching
- Cache warming
- Intelligent invalidation
- Memory monitoring
- Cache hit ratio optimization

## 13. Deployment Architecture

### 13.1 Backend Services
```
┌─────────────────┐     ┌─────────────────┐
│   FastAPI App   │────▶│  Redis Cache    │
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│   PostgreSQL    │◀────│  FAISS Index    │
└─────────────────┘     └─────────────────┘
```

### 13.2 Data Flow
```
1. Feature Computation:
   Raw Data → Preprocessing → Feature Extraction → PostgreSQL

2. Recommendation Flow:
   Query → Cache Check → FAISS Search → Results → Cache Update

3. Update Flow:
   New Data → Feature Extraction → DB Update → Index Update
```

### 13.3 Scaling Strategy
- Horizontal scaling of API servers
- Read replicas for PostgreSQL
- Distributed Redis cluster
- Sharded FAISS indices
- Load balancing

## 14. Monitoring and Maintenance

### 14.1 Performance Metrics
- Response times
- Cache hit rates
- Memory usage
- CPU utilization
- Database performance
- FAISS query times

### 14.2 Alerts and Logging
- Error rate monitoring
- Resource usage alerts
- Performance degradation detection
- Security incident alerts
- Backup verification

## 10. External API Integration

### 10.1 Genius API Setup
1. Create free account on genius.com
2. Visit genius.com/api-clients
3. Create new API client:
   - App Name: "DSPipeXP Music Analysis"
   - App Website URL: [Your repository URL]
4. Generate and store access token as GENIUS_ACCESS_TOKEN
5. Features available:
   - Lyrics fetching
   - Song metadata
   - Artist information
   - Rate limits: Standard free tier

### 10.2 YouTube API Integration
[Details to be added] 

## 15. User Scenarios and Interaction Flows

### 15.1 Song Discovery and Management

1. **Song Upload Flow**
   ```
   User Action: Upload new song
   → System checks file format
   → Extracts audio features
   → Downloads YouTube audio (if URL)
   → Fetches lyrics from Genius
   → Analyzes sentiment and features
   → Stores in database
   → Returns success/failure
   ```

2. **Song Search Flow**
   ```
   User Action: Search for songs
   → Enter search query (title/artist)
   → System searches database
   → Returns matching songs
   → Optional: Get recommendations
   ```

3. **Song Details Flow**
   ```
   User Action: View song details
   → System fetches:
     - Basic metadata
     - Audio features
     - Lyrics (if available)
     - Sentiment analysis
     - Similar songs
   ```

### 15.2 Recommendation Scenarios

1. **Similar Songs Flow**
   ```
   User Action: Get similar songs
   → Select reference song
   → System computes:
     - Audio similarity
     - Lyrical similarity
     - Mood similarity
   → Returns ranked recommendations
   ```

2. **Mood-Based Flow**
   ```
   User Action: Get songs by mood
   → Select mood category
   → System analyzes:
     - Audio features
     - Lyrical sentiment
     - Tempo and energy
   → Returns matching songs
   ```

3. **Feature-Based Flow**
   ```
   User Action: Custom feature search
   → Specify features:
     - Tempo range
     - Energy level
     - Valence
     - Genre
   → System filters database
   → Returns matching songs
   ```

### 15.3 Advanced User Scenarios

1. **Playlist Generation**
   ```
   User Action: Generate playlist
   → Choose generation method:
     - Mood-based
     - Artist-based
     - Feature-based
   → System generates sequence
   → User can refine/adjust
   ```

2. **Song Analysis**
   ```
   User Action: Analyze song
   → Upload/select song
   → System provides:
     - Audio analysis
     - Lyrical analysis
     - Feature breakdown
     - Similar songs
   ```

3. **Batch Processing**
   ```
   User Action: Batch upload
   → Upload multiple songs
   → System processes in background
   → Sends completion notification
   → Updates recommendation index
   ```

### 15.4 Error Scenarios

1. **Upload Failures**
   ```
   Error: Invalid file format
   → System notifies user
   → Suggests supported formats
   → Logs error for tracking
   ```

2. **API Limitations**
   ```
   Error: Rate limit reached
   → System implements backoff
   → Queues requests
   → Notifies user of delay
   ```

3. **Search Failures**
   ```
   Error: No results found
   → Suggest alternative queries
   → Show popular searches
   → Log failed searches
   ```

### 15.5 Performance Considerations

1. **High Load Scenarios**
   ```
   Multiple concurrent uploads
   → Queue processing
   → Load balancing
   → Cache frequent requests
   ```

2. **Large Dataset Handling**
   ```
   Extensive song library
   → Paginated results
   → Optimized search
   → Cached recommendations
   ```

3. **Real-time Requirements**
   ```
   Quick recommendations
   → Pre-computed features
   → Cached results
   → Approximate search
   ```

### 15.6 Integration Points

1. **YouTube Integration**
   ```
   User Action: Add YouTube song
   → Paste YouTube URL
   → System extracts audio
   → Processes metadata
   → Stores for recommendations
   ```

2. **Genius Integration**
   ```
   User Action: View lyrics
   → System fetches lyrics
   → Analyzes sentiment
   → Caches results
   → Updates features
   ```

3. **External API Fallbacks**
   ```
   API Failure scenario
   → Use cached data
   → Alternate data sources
   → Graceful degradation
   ```

## 6. Current Status

### 6.1 Completed Features ✓
- Database Manager with SQLAlchemy and vector store
- YouTube Service for audio downloads
- Basic recommendation engine
- API endpoints for song management
- Pydantic schemas for request/response validation

### 6.2 In Progress
- Testing and optimization
- Deployment configuration
- Documentation updates

### 6.3 Pending
- Frontend development
- Advanced recommendation features
- User authentication
- Analytics and monitoring

## 7. Deployment Guide

### 7.1 Prerequisites
- Render account
- PostgreSQL database (Render or external)
- Environment variables configured
- Docker setup (optional)

### 7.2 Configuration Files
- `requirements.txt`: All Python dependencies
- `.env.example`: Template for environment variables
- `Dockerfile`: Container configuration
- `docker-compose.yml`: Local development setup

### 7.3 Environment Variables
```
DATABASE_URL=postgresql://user:password@host:port/dbname
VECTOR_STORE_PATH=/path/to/vector/store
GENIUS_ACCESS_TOKEN=your_genius_token
REDIS_URL=redis://localhost:6379 (optional)
```

### 7.4 Build and Start Commands
```bash
# Build Command
pip install -r requirements.txt

# Start Command
uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT
``` 

## 16. API Documentation

### 16.1 Song Search and Processing

#### Search Song by Name
```
GET /api/v1/songs/search/by-name

Query Parameters:
- name: string (required) - Name of the song to search
- artist: string (optional) - Artist name to refine search

Response:
{
    "found": boolean,
    "song": {
        "id": string,
        "title": string,
        "artist": string,
        "features": object,
        "metadata": object
    },
    "recommendations": [
        {
            "id": string,
            "title": string,
            "artist": string,
            "similarity_score": float
        }
    ]
}

Notes:
- If song not found in database, automatically searches YouTube
- Downloads, processes, and adds to database if found on YouTube
- Returns recommendations based on processed song
```

#### Process Audio File
```
POST /api/v1/songs/process/file

Form Data:
- file: File (required) - Audio file to process
- title: string (optional) - Song title
- artist: string (optional) - Artist name

Response:
{
    "features": {
        "mfcc": array,
        "spectral_contrast": array,
        "chroma": array,
        "tempo": float,
        "beats": array,
        "harmonic": float,
        "percussive": float,
        "energy": float,
        "harmonicity": float,
        "rhythm_strength": float,
        "complexity": float
    },
    "metadata": object,
    "lyrics": object,
    "recommendations": array
}
```

#### Process YouTube URL
```
POST /api/v1/songs/process/youtube

Request Body:
{
    "url": string (required) - YouTube URL
}

Response:
{
    "features": object,
    "metadata": {
        "title": string,
        "duration": integer,
        "view_count": integer,
        "like_count": integer,
        "channel": string,
        "upload_date": string
    },
    "lyrics": {
        "lyrics": string,
        "analysis": {
            "word_count": integer,
            "line_count": integer,
            "char_count": integer
        }
    },
    "recommendations": array
}
```

#### Search by Mood
```
GET /api/v1/songs/search/by-mood

Query Parameters:
- mood: string (required) - Mood category (e.g., "happy", "sad", "energetic")
- limit: integer (optional) - Number of results to return

Response:
{
    "songs": [
        {
            "id": string,
            "title": string,
            "artist": string,
            "mood_score": float,
            "features": object
        }
    ]
}
```

#### Search by Genre
```
GET /api/v1/songs/search/by-genre

Query Parameters:
- genre: string (required) - Genre category
- limit: integer (optional) - Number of results to return

Response:
{
    "songs": [
        {
            "id": string,
            "title": string,
            "artist": string,
            "genre_confidence": float,
            "features": object
        }
    ]
}
```

## 17. Implementation Priorities

### 17.1 Phase 1: Core Search (In Progress)
1. ✓ YouTube URL processing
2. ✓ File upload processing
3. ⚠ Song name search with YouTube fallback
4. ⚠ Database integration for persistent storage

### 17.2 Phase 2: Advanced Features (Pending)
1. Mood classification model
2. Genre classification model
3. Feature vector optimization
4. Recommendation algorithm refinement

### 17.3 Phase 3: User Experience (Pending)
1. Search result caching
2. Batch processing
3. Progress tracking
4. Error handling improvements 