from typing import List, Optional
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Try to load .env file if it exists
env_path = Path(__file__).parent.parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(str(env_path))

# Base directory is the backend directory
BASE_DIR: Path = Path(__file__).parent.parent.parent

class Settings(BaseSettings):
    PROJECT_NAME: str = "DSPipeXP"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = True
    ENVIRONMENT: str = "development"
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Database Settings
    DATABASE_URL: str
    
    # Redis
    REDIS_URL: Optional[str] = None
    REDIS_HOST: Optional[str] = None
    REDIS_PORT: Optional[int] = None
    REDIS_PASSWORD: Optional[str] = None
    
    # CORS Settings
    _CORS_ORIGINS: str = "*"  # Allow all origins in development
    ALLOWED_ORIGINS: str | None = None  # Alternative env var name
    _CORS_METHODS: str = "GET,POST,PUT,DELETE,OPTIONS"
    _CORS_HEADERS: str = "Content-Type,Authorization,X-API-Key"
    
    # Directory Settings
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    AUDIO_DIR: Path = DATA_DIR / "audio"
    CACHE_DIR: Path = DATA_DIR / "cache"
    
    # Storage Settings
    VECTOR_STORE_PATH: str = str(CACHE_DIR / "vector_store.pkl")
    AUDIO_STORAGE_PATH: str = os.getenv("AUDIO_STORAGE_PATH", "/tmp/audio")
    CACHE_STORAGE_PATH: str = os.getenv("CACHE_STORAGE_PATH", "/tmp/cache")
    
    # Feature Extraction Settings
    WORD2VEC_DIMENSION: int = 100
    TOPIC_DIMENSION: int = 10
    
    # File Upload Settings
    MAX_UPLOAD_SIZE: int = 20 * 1024 * 1024  # 20MB
    
    # Security Settings
    SECRET_KEY: str = "your-secret-key"
    API_KEY_HEADER: str = "X-API-Key"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # External API Settings
    GENIUS_ACCESS_TOKEN: Optional[str] = None
    
    @property
    def CORS_ORIGINS(self) -> List[str]:
        # First check ALLOWED_ORIGINS, then fall back to _CORS_ORIGINS
        origins = self.ALLOWED_ORIGINS if self.ALLOWED_ORIGINS is not None else self._CORS_ORIGINS
        return parse_comma_separated_list(origins)

    @property
    def CORS_METHODS(self) -> List[str]:
        return parse_comma_separated_list(self._CORS_METHODS)

    @property
    def CORS_HEADERS(self) -> List[str]:
        return parse_comma_separated_list(self._CORS_HEADERS)
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        json_schema_extra = {
            "title": "API Settings",
            "description": "Configuration settings for the Music Recommendation API"
        }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Parse Redis URL if provided
        if self.REDIS_URL and not any([self.REDIS_HOST, self.REDIS_PORT, self.REDIS_PASSWORD]):
            try:
                from urllib.parse import urlparse
                parsed = urlparse(self.REDIS_URL)
                self.REDIS_HOST = parsed.hostname
                self.REDIS_PORT = parsed.port or 6379
                self.REDIS_PASSWORD = parsed.password
            except Exception as e:
                print(f"Failed to parse REDIS_URL: {e}")

def parse_comma_separated_list(value: str | List[str] | None) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if value == "*":
        return ["*"]
    return [item.strip() for item in value.split(",") if item.strip()]

# Create global settings object
settings = Settings()

# Export constants
PROJECT_NAME = settings.PROJECT_NAME
API_V1_STR = settings.API_V1_STR
DEBUG = settings.DEBUG
ENVIRONMENT = settings.ENVIRONMENT

# Ensure required directories exist
os.makedirs(settings.AUDIO_STORAGE_PATH, exist_ok=True)
os.makedirs(settings.CACHE_STORAGE_PATH, exist_ok=True)
os.makedirs(os.path.dirname(settings.VECTOR_STORE_PATH), exist_ok=True) 