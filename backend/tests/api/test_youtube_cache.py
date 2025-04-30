import pytest
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession

from app.utils.cache import YouTubeCache
from app.models.song import Song
from app.core.config import settings

@pytest.fixture
def youtube_cache(db_session: AsyncSession):
    return YouTubeCache(db_session)

@pytest.fixture
async def test_song(db_session: AsyncSession):
    song = Song(
        title="Test Song",
        artist="Test Artist",
        album="Test Album",
        duration=180,
        release_date=datetime.utcnow()
    )
    db_session.add(song)
    await db_session.commit()
    await db_session.refresh(song)
    return song

async def test_cache_url(youtube_cache: YouTubeCache, test_song: Song):
    # Test setting URL
    test_url = "https://www.youtube.com/watch?v=test123"
    success = await youtube_cache.set_url(test_song.id, test_url)
    assert success is True
    
    # Test getting URL
    cached_url = await youtube_cache.get_url(test_song.id)
    assert cached_url == test_url
    
    # Test URL expiration
    test_song.youtube_url_updated_at = datetime.utcnow() - timedelta(days=settings.YOUTUBE_CACHE_DAYS + 1)
    await youtube_cache.db.commit()
    
    expired_url = await youtube_cache.get_url(test_song.id)
    assert expired_url is None

async def test_cache_metadata(youtube_cache: YouTubeCache, test_song: Song):
    # Test setting metadata
    test_metadata = {
        "title": "Test Video",
        "channel": "Test Channel",
        "duration": 180,
        "views": 1000
    }
    success = await youtube_cache.set_metadata(test_song.id, test_metadata)
    assert success is True
    
    # Test getting metadata
    cached_metadata = await youtube_cache.get_metadata(test_song.id)
    assert cached_metadata == test_metadata
    
    # Test metadata expiration
    test_song.youtube_metadata_updated_at = datetime.utcnow() - timedelta(days=settings.YOUTUBE_CACHE_DAYS + 1)
    await youtube_cache.db.commit()
    
    expired_metadata = await youtube_cache.get_metadata(test_song.id)
    assert expired_metadata is None

async def test_invalid_song_id(youtube_cache: YouTubeCache):
    # Test operations with invalid song ID
    invalid_id = 999999
    
    assert await youtube_cache.get_url(invalid_id) is None
    assert await youtube_cache.get_metadata(invalid_id) is None
    
    # Test setting operations with invalid ID
    assert await youtube_cache.set_url(invalid_id, "https://youtube.com/test") is False
    assert await youtube_cache.set_metadata(invalid_id, {"test": "data"}) is False

async def test_cache_invalidation(youtube_cache: YouTubeCache, test_song: Song):
    # Set up test data
    test_url = "https://www.youtube.com/watch?v=test123"
    test_metadata = {"title": "Test Video"}
    
    await youtube_cache.set_url(test_song.id, test_url)
    await youtube_cache.set_metadata(test_song.id, test_metadata)
    
    # Test invalidation
    success = await youtube_cache.invalidate(test_song.id)
    assert success is True
    
    # Verify data is cleared
    assert await youtube_cache.get_url(test_song.id) is None
    assert await youtube_cache.get_metadata(test_song.id) is None
    
    # Verify database fields are cleared
    await youtube_cache.db.refresh(test_song)
    assert test_song.youtube_url is None
    assert test_song.youtube_url_updated_at is None
    assert test_song.youtube_metadata is None
    assert test_song.youtube_metadata_updated_at is None

async def test_invalid_metadata_json(youtube_cache: YouTubeCache, test_song: Song):
    # Set invalid JSON in metadata
    test_song.youtube_metadata = "invalid json"
    test_song.youtube_metadata_updated_at = datetime.utcnow()
    await youtube_cache.db.commit()
    
    # Test getting invalid metadata
    assert await youtube_cache.get_metadata(test_song.id) is None 