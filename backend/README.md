# DSPipeXP Backend API

A FastAPI-based backend service for music recommendation system with audio processing, lyrics analysis, and YouTube integration.

## Features

- 🎵 Audio feature extraction and analysis
- 📝 Lyrics fetching and sentiment analysis
- 🎥 YouTube video search and audio download
- 🔍 Advanced music recommendation system
- 🔒 Secure API with authentication
- 📊 Real-time health monitoring
- 🚀 Production-ready with Docker

## Tech Stack

- **Framework**: FastAPI
- **Database**: PostgreSQL with SQLAlchemy
- **Cache**: Redis
- **Audio Processing**: librosa
- **ML/AI**: scikit-learn, FAISS
- **External Services**: YouTube, Genius Lyrics
- **Containerization**: Docker
- **Deployment**: Render

## Project Structure

```
backend/
├── app/
│   ├── api/
│   │   └── endpoints/         # API route handlers
│   ├── core/                  # Core functionality
│   │   ├── config.py         # Configuration settings
│   │   ├── database.py       # Database setup
│   │   └── logging.py        # Logging configuration
│   ├── models/               # SQLAlchemy models
│   ├── services/             # Business logic
│   └── utils/                # Utility functions
├── data/
│   ├── audio/                # Audio file storage
│   └── cache/                # Cache storage
├── migrations/               # Alembic migrations
├── tests/                    # Test suite
├── .dockerignore            # Docker ignore rules
├── .env.example             # Example environment variables
├── alembic.ini              # Alembic configuration
├── docker-compose.yml       # Local development setup
├── docker-compose.prod.yml  # Production setup
├── Dockerfile              # Docker build instructions
├── render.yaml             # Render deployment config
├── requirements.txt        # Python dependencies
└── run.py                 # Application entry point
```

## Prerequisites

- Python 3.10+
- PostgreSQL 14+
- Redis 7+
- FFmpeg

## Local Development Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd backend
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate  # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. Start services with Docker Compose:
   ```bash
   docker-compose up -d
   ```

6. Run database migrations:
   ```bash
   alembic upgrade head
   ```

7. Start the development server:
   ```bash
   python run.py
   ```

## Docker Setup

Build and run the application using Docker:

```bash
# Build the image
docker build -t dspipexp-backend .

# Run the container
docker run -p 8000:8000 dspipexp-backend
```

Or using Docker Compose:

```bash
# Development
docker-compose up -d

# Production
docker-compose -f docker-compose.prod.yml up -d
```

## API Documentation

Once the server is running, access the API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Environment Variables

Required environment variables:

```
DATABASE_URL=postgresql://user:pass@localhost:5432/dbname
REDIS_URL=redis://localhost:6379
SECRET_KEY=your-secret-key
ENVIRONMENT=development
DEBUG=true
ALLOWED_ORIGINS=http://localhost:3000
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test file
pytest tests/test_youtube.py
```

## Deployment

The application is configured for deployment on Render:

1. Push your code to GitHub
2. Create a new Web Service on Render
3. Connect your repository
4. Render will automatically:
   - Create PostgreSQL database
   - Set up Redis instance
   - Deploy the application
   - Configure environment variables
   - Set up health checks

## Health Checks

The application provides a health check endpoint at `/api/v1/health` that monitors:
- API status
- Database connectivity
- Redis connection
- Overall system health

## Logging

Structured JSON logging is configured for production with:
- Request ID tracking
- Error tracking
- Performance monitoring
- Environment-based log levels

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 