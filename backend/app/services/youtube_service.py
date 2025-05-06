import os
import requests
from bs4 import BeautifulSoup
import logging
from typing import Dict, Optional, Tuple, List
import re
import json
from urllib.parse import quote, unquote
import yt_dlp
from datetime import datetime, timedelta
from fastapi import HTTPException
from ..utils.rate_limiter import RateLimiter
from ..core.database import db_manager
from youtubesearchpython import VideosSearch
import glob

logger = logging.getLogger(__name__)

RATE_LIMITS = {
    'search': 300,    # 300 searches per hour (increased as requested)
    'download': 50    # 50 downloads per hour
}

FALLBACK_SOURCES = [
    'yt-dlp',
    'youtube-search-python',
    'direct-url'
]

class EnhancedYouTubeService:
    def __init__(self, download_path: str = None):
        """Initialize the enhanced YouTube service."""
        from app.core.config import settings
        self.download_path = download_path or settings.AUDIO_STORAGE_PATH
        os.makedirs(self.download_path, exist_ok=True)
        
        self.rate_limiter = RateLimiter(RATE_LIMITS)
        self.cache_ttl = 7 * 24 * 60 * 60  # 7 days in seconds
        
        # Base yt-dlp options
        self.base_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'nocheckcertificate': True,
            'socket_timeout': 30,
            'retries': 3,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-us,en;q=0.5',
                'Sec-Fetch-Mode': 'navigate',
            }
        }

    async def _get_cached_url(self, song_name: str, artist: str) -> Optional[str]:
        """Get cached YouTube URL from database."""
        songs = await db_manager.search_songs(query=song_name, artist=artist, limit=1)
        if not songs:
            return None
            
        song = songs[0]
        if not song.youtube_url or not song.youtube_url_updated_at:
            return None
            
        # Check if cache is expired
        if datetime.utcnow() - song.youtube_url_updated_at > timedelta(seconds=self.cache_ttl):
            return None
            
        if song.youtube_url_status == 'valid':
            return song.youtube_url
            
        return None

    async def _cache_url(self, song_name: str, artist: str, url: str, status: str = 'valid', error: str = None):
        """Cache YouTube URL in database."""
        songs = await db_manager.search_songs(query=song_name, artist=artist, limit=1)
        if not songs:
            return
            
        await db_manager.update_song(songs[0].id, {
            'youtube_url': url,
            'youtube_url_updated_at': datetime.utcnow(),
            'youtube_url_status': status,
            'youtube_url_error': error
        })

    async def _search_with_ytdlp(self, query: str) -> Optional[str]:
        """Search using yt-dlp."""
        try:
            with yt_dlp.YoutubeDL(self.base_opts) as ydl:
                info = ydl.extract_info(f"ytsearch1:{query} official audio", download=False)
                if not info or 'entries' not in info or not info['entries']:
                    return None
                return f"https://www.youtube.com/watch?v={info['entries'][0]['id']}"
        except Exception as e:
            logger.error(f"yt-dlp search error: {e}")
            return None

    async def _search_with_youtube_search_python(self, query: str) -> Optional[str]:
        """Search using youtube-search-python."""
        try:
            videos_search = VideosSearch(query, limit=1)
            results = await videos_search.next()
            if not results or not results['result']:
                return None
            return results['result'][0]['link']
        except Exception as e:
            logger.error(f"youtube-search-python error: {e}")
            return None

    async def _search_with_direct_url(self, query: str) -> Optional[str]:
        """Try to construct direct YouTube URL if video ID is present."""
        try:
            # Check if query contains a YouTube video ID
            video_id_match = re.search(r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})', query)
            if video_id_match:
                return f"https://www.youtube.com/watch?v={video_id_match.group(1)}"
            return None
        except Exception as e:
            logger.error(f"Direct URL construction error: {e}")
            return None

    async def get_video_url(self, song_name: str, artist: str) -> Optional[str]:
        """Get YouTube video URL with caching and fallbacks."""
        # 1. Check cache first
        cached_url = await self._get_cached_url(song_name, artist)
        if cached_url:
            return cached_url

        # 2. Rate limit check
        if not self.rate_limiter.can_proceed('search'):
            raise HTTPException(status_code=429, detail="YouTube search rate limit exceeded")

        query = f"{song_name} {artist} official audio"
        
        # 3. Try each source with fallbacks
        for source in FALLBACK_SOURCES:
            try:
                url = None
                if source == 'yt-dlp':
                    url = await self._search_with_ytdlp(query)
                elif source == 'youtube-search-python':
                    url = await self._search_with_youtube_search_python(query)
                elif source == 'direct-url':
                    url = await self._search_with_direct_url(query)
                
                if url:
                    # Cache the successful result
                    await self._cache_url(song_name, artist, url)
                    return url
                    
            except Exception as e:
                logger.error(f"Error with source {source}: {e}")
                continue

        # Cache the failure
        await self._cache_url(song_name, artist, None, 'invalid', 'All sources failed')
        return None

    async def get_cache_stats(self) -> Dict:
        """Get YouTube URL cache statistics."""
        try:
            async with db_manager.SessionLocal() as session:
                total = await session.execute("SELECT COUNT(*) FROM songs")
                with_url = await session.execute("SELECT COUNT(*) FROM songs WHERE youtube_url IS NOT NULL")
                valid = await session.execute("SELECT COUNT(*) FROM songs WHERE youtube_url_status = 'valid'")
                invalid = await session.execute("SELECT COUNT(*) FROM songs WHERE youtube_url_status = 'invalid'")
                
                return {
                    'total_songs': total.scalar(),
                    'with_url': with_url.scalar(),
                    'valid_urls': valid.scalar(),
                    'invalid_urls': invalid.scalar(),
                    'rate_limits': {
                        'search_remaining': self.rate_limiter.get_remaining('search'),
                        'download_remaining': self.rate_limiter.get_remaining('download')
                    }
                }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}

    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL."""
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
            r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def search(self, query: str) -> Optional[Dict]:
        """Search YouTube using web scraping."""
        try:
            encoded_query = quote(query)
            response = requests.get(f"https://www.youtube.com/results?search_query={encoded_query}", headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find initial data script
            for script in soup.find_all('script'):
                if 'ytInitialData' in str(script):
                    data_str = str(script).split('ytInitialData = ')[1].split(';</script>')[0]
                    data = json.loads(data_str)
                    
                    # Extract video information from the data
                    contents = data.get('contents', {}).get('twoColumnSearchResultsRenderer', {}).get('primaryContents', {}).get('sectionListRenderer', {}).get('contents', [])
                    
                    for content in contents:
                        if 'itemSectionRenderer' in content:
                            items = content['itemSectionRenderer'].get('contents', [])
                            for item in items:
                                if 'videoRenderer' in item:
                                    video = item['videoRenderer']
                                    return {
                                        'id': video.get('videoId'),
                                        'title': video.get('title', {}).get('runs', [{}])[0].get('text', ''),
                                        'channel': video.get('ownerText', {}).get('runs', [{}])[0].get('text', ''),
                                        'url': f"https://www.youtube.com/watch?v={video.get('videoId')}"
                                    }
            return None
            
        except Exception as e:
            logger.error(f"Error searching YouTube for {query}: {str(e)}")
            return None

    def download_audio(self, url: str) -> Tuple[Optional[str], Optional[Dict]]:
        """Download audio from YouTube URL using yt-dlp."""
        try:
            video_id = self._extract_video_id(url)
            if not video_id:
                raise ValueError(f"Invalid YouTube URL: {url}")

            with yt_dlp.YoutubeDL(self.base_opts) as ydl:
                # Extract video info first
                info = ydl.extract_info(url, download=False)
                
                # Download audio
                ydl.download([url])
                
                # Get the output path
                output_path = os.path.join(
                    self.download_path,
                    f"{info['title']}.mp3"
                )
                
                metadata = {
                    'title': info.get('title', ''),
                    'duration': info.get('duration', 0),
                    'view_count': info.get('view_count', 0),
                    'channel': info.get('channel', ''),
                    'video_id': video_id
                }
                
                return output_path, metadata
                
        except Exception as e:
            logger.error(f"Error downloading YouTube video {url}: {str(e)}")
            return None, None

    def get_video_metadata(self, url: str) -> Optional[Dict]:
        """Get video metadata using web scraping."""
        try:
            video_id = self._extract_video_id(url)
            if not video_id:
                return None

            response = requests.get(f"https://www.youtube.com/watch?v={video_id}", headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find initial data script
            for script in soup.find_all('script'):
                if 'ytInitialData' in str(script):
                    data_str = str(script).split('ytInitialData = ')[1].split(';</script>')[0]
                    data = json.loads(data_str)
                    
                    video_data = data.get('videoDetails', {})
                    return {
                        'title': video_data.get('title', ''),
                        'channel': video_data.get('author', ''),
                        'video_id': video_id,
                        'url': url
                    }
                    
            return None
            
        except Exception as e:
            logger.error(f"Error getting metadata for {url}: {str(e)}")
            return None

# Create global instance
youtube_service = EnhancedYouTubeService() 