import os
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl
from app.core.config import settings
import yt_dlp
from youtubesearchpython import VideosSearch
import re

router = APIRouter()

logger = logging.getLogger(__name__)

class YouTubeRequest(BaseModel):
    url: HttpUrl
    
class YouTubeResponse(BaseModel):
    file_path: Optional[str]
    metadata: Optional[Dict]

class YouTubeService:
    def __init__(self, download_path: str = None):
        """Initialize the YouTube service with a download path.
        
        Args:
            download_path (str): Directory where downloaded audio files will be stored
        """
        self.download_path = download_path or settings.AUDIO_STORAGE_PATH
        os.makedirs(self.download_path, exist_ok=True)
        
        # Configure yt-dlp options
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(self.download_path, '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'nocheckcertificate': True,
            # Custom headers to avoid detection
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-us,en;q=0.5',
                'Sec-Fetch-Mode': 'navigate',
            }
        }

    def search_and_download(self, query: str, max_results: int = 5) -> Tuple[Optional[str], Optional[Dict]]:
        """Search for a video and download its audio.
        
        Args:
            query (str): Search query (typically artist + track name)
            max_results (int): Maximum number of search results to try
            
        Returns:
            Tuple[Optional[str], Optional[Dict]]: Tuple of (file path, video metadata)
        """
        logger.info(f"Searching YouTube for: {query}")
        
        try:
            # Search using youtube-search-python
            search = VideosSearch(query + " official audio", limit=max_results)
            results = search.result()['result']
            
            if not results:
                logger.warning(f"No results found for '{query}'")
                return None, None
            
            # Try to download from each result until successful
            for video in results:
                try:
                    video_url = f"https://www.youtube.com/watch?v={video['id']}"
                    logger.info(f"Attempting to download: {video['title']} ({video_url})")
                    
                    # Clean title for filename
                    title = re.sub(r'[^\w\s-]', '', video['title'])
                    mp3_file = os.path.join(self.download_path, f"{title}.mp3")
                    
                    # Try downloading with yt-dlp
                    with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                        info = ydl.extract_info(video_url, download=True)
                        
                    if not os.path.exists(mp3_file):
                        logger.error(f"Downloaded file not found at expected path: {mp3_file}")
                        continue
                    
                    metadata = {
                        'title': video['title'],
                        'duration': video['duration'],
                        'view_count': video.get('viewCount', {}).get('text', '0').replace(',', ''),
                        'channel': video['channel']['name'],
                        'video_id': video['id']
                    }
                    
                    logger.info(f"Successfully downloaded: {mp3_file}")
                    return mp3_file, metadata
                    
                except Exception as e:
                    logger.error(f"Error downloading {video['title']}: {str(e)}")
                    continue
            
            logger.warning(f"Could not download any audio for '{query}'")
            return None, None
            
        except Exception as e:
            logger.error(f"Error searching for '{query}': {str(e)}")
            return None, None

    def search(self, query: str) -> Optional[str]:
        """Search for a YouTube video.
        
        Args:
            query (str): Search query
            
        Returns:
            Optional[str]: YouTube URL if found, None otherwise
        """
        try:
            search = VideosSearch(query + " official audio", limit=1)
            results = search.result()['result']
            if not results:
                return None
            video = results[0]
            return f"https://www.youtube.com/watch?v={video['id']}"
        except Exception as e:
            logger.error(f"Error searching YouTube for {query}: {str(e)}")
            return None

    def extract_metadata(self, url: str) -> Optional[Dict]:
        """Extract metadata from a YouTube video without downloading.
        
        Args:
            url (str): YouTube video URL
            
        Returns:
            Optional[Dict]: Video metadata if successful, None otherwise
        """
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    'title': info['title'],
                    'duration': info['duration'],
                    'view_count': info['view_count'],
                    'channel': info['uploader'],
                    'video_id': info['id']
                }
        except Exception as e:
            logger.error(f"Error extracting metadata from {url}: {str(e)}")
            return None

# Create a global instance
youtube_service = YouTubeService()

@router.post("/youtube/process", response_model=YouTubeResponse)
async def process_youtube_url(request: YouTubeRequest):
    """Process a YouTube URL to download audio and extract metadata.
    
    Args:
        request (YouTubeRequest): Request containing YouTube URL
        
    Returns:
        YouTubeResponse: Downloaded file path and metadata
    """
    file_path, metadata = youtube_service.search_and_download(str(request.url))
    if not file_path:
        raise HTTPException(
            status_code=400,
            detail="Failed to process YouTube URL. Please check the URL and try again."
        )
    return YouTubeResponse(file_path=file_path, metadata=metadata)

@router.get("/youtube/metadata")
async def get_youtube_metadata(url: HttpUrl):
    """Extract metadata from a YouTube URL without downloading.
    
    Args:
        url (HttpUrl): YouTube video URL
        
    Returns:
        Dict: Video metadata
    """
    metadata = youtube_service.extract_metadata(str(url))
    if not metadata:
        raise HTTPException(
            status_code=400,
            detail="Failed to extract metadata. Please check the URL and try again."
        )
    return metadata

@router.get("/youtube/search")
async def search_youtube(query: str):
    """Search YouTube for a video.
    
    Args:
        query (str): Search query
        
    Returns:
        Dict: YouTube URL if found
    """
    url = youtube_service.search(query)
    if not url:
        raise HTTPException(
            status_code=404,
            detail="No videos found matching the query."
        )
    return {"url": url}

# Expose these functions for other modules to use
def search(query: str) -> Optional[str]:
    """Search for a YouTube video.
    
    Args:
        query (str): Search query
        
    Returns:
        Optional[str]: YouTube URL if found, None otherwise
    """
    return youtube_service.search(query)

def download_audio(url: str) -> Tuple[Optional[str], Optional[Dict]]:
    """Download audio from a YouTube URL.
    
    Args:
        url (str): YouTube URL
        
    Returns:
        Tuple[Optional[str], Optional[Dict]]: Tuple of (file path, metadata)
    """
    try:
        with yt_dlp.YoutubeDL(youtube_service.ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = re.sub(r'[^\w\s-]', '', info['title'])
            mp3_file = os.path.join(youtube_service.download_path, f"{title}.mp3")
            
            if not os.path.exists(mp3_file):
                logger.error(f"Downloaded file not found at expected path: {mp3_file}")
                return None, None
            
            metadata = {
                'title': info['title'],
                'duration': info['duration'],
                'view_count': info['view_count'],
                'channel': info['uploader'],
                'video_id': info['id']
            }
            
            return mp3_file, metadata
    except Exception as e:
        logger.error(f"Error downloading audio from {url}: {str(e)}")
        return None, None 