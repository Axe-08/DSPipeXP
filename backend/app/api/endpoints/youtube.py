import os
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
from fastapi import APIRouter, HTTPException, Query, status, Request
from pydantic import BaseModel, HttpUrl, Field, validator, constr
from app.core.config import settings
import yt_dlp
import re
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
import glob

router = APIRouter()

logger = logging.getLogger(__name__)

class YouTubeError(BaseModel):
    detail: str
    error_code: str = Field(..., description="Internal error code for tracking")
    status_code: int = Field(..., description="HTTP status code")

class YouTubeRequest(BaseModel):
    url: HttpUrl = Field(..., description="YouTube video URL to process")
    quality: str = Field("192", pattern="^(64|96|128|192|256|320)$", description="Audio quality in kbps")
    
    @validator('url')
    def validate_youtube_url(cls, v):
        if not re.match(r'^https?://(www\.)?(youtube\.com|youtu\.be)', str(v)):
            raise ValueError("Invalid YouTube URL. Must be from youtube.com or youtu.be")
        return v

class YouTubeSearchRequest(BaseModel):
    query: constr(min_length=1, max_length=200) = Field(..., description="Search query")
    limit: int = Field(5, ge=1, le=10, description="Number of results to return")

class YouTubeResponse(BaseModel):
    file_path: Optional[str] = Field(None, description="Path to downloaded audio file")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Video metadata")
    error: Optional[YouTubeError] = None

    class Config:
        schema_extra = {
            "example": {
                "file_path": "/path/to/audio.mp3",
                "metadata": {
                    "title": "Video Title",
                    "duration": 180,
                    "channel": "Channel Name"
                }
            }
        }

class YouTubeService:
    def __init__(self, download_path: str = None):
        """Initialize the YouTube service with a download path.
        
        Args:
            download_path (str): Directory where downloaded audio files will be stored
        """
        self.download_path = download_path or settings.AUDIO_STORAGE_PATH
        os.makedirs(self.download_path, exist_ok=True)
        
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

    def get_ydl_opts(self, quality: str = "192", download: bool = True) -> dict:
        """Get yt-dlp options with specified quality."""
        opts = self.base_opts.copy()
        
        # Add common options to avoid bot detection
        opts.update({
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True if not download else False,
            'nocheckcertificate': True,
            'socket_timeout': 30,
            'retries': 5,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-us,en;q=0.5',
                'Sec-Fetch-Mode': 'navigate',
            },
            'ignoreerrors': True,
            'no_check_certificate': True,
            'no_cache_dir': True,  # Disable cache to avoid permission issues
        })
        
        if download:
            opts.update({
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': quality,
                }],
                'outtmpl': os.path.join(self.download_path, '%(title)s.%(ext)s'),
                'restrictfilenames': True,
                'windowsfilenames': True,
                'nooverwrites': True,
                'fragment_retries': 10,
                'skip_unavailable_fragments': True,
            })
        return opts

    def format_video_metadata(self, info: dict) -> dict:
        """Format video metadata consistently."""
        return {
            'title': info.get('title', ''),
            'duration': info.get('duration', 0),
            'view_count': info.get('view_count', 0),
            'channel': info.get('uploader', ''),
            'video_id': info.get('id', ''),
            'url': f"https://www.youtube.com/watch?v={info.get('id', '')}",
            'thumbnail': info.get('thumbnail', ''),
            'description': info.get('description', '')[:500] if info.get('description') else ''
        }

    def find_downloaded_file(self, title: str) -> Optional[str]:
        """Find a downloaded file by title pattern."""
        pattern = re.sub(r'[^\w\s-]', '', title)
        pattern = pattern.replace(' ', '_')  # yt-dlp replaces spaces with underscores
        files = glob.glob(os.path.join(self.download_path, f"*{pattern}*.mp3"))
        return files[0] if files else None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def search_videos(self, query: str, limit: int = 5) -> List[dict]:
        """Search for videos with retries."""
        try:
            search_url = f"ytsearch{limit}:{query} official audio"
            with yt_dlp.YoutubeDL(self.get_ydl_opts(download=False)) as ydl:
                info = ydl.extract_info(search_url, download=False)
                if not info or 'entries' not in info:
                    return []
                return [self.format_video_metadata(entry) for entry in info['entries']]
        except Exception as e:
            logger.error(f"Error searching for videos: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def search_and_download(self, query: str, max_results: int = 5, quality: str = "192") -> Tuple[Optional[str], Optional[Dict]]:
        """Search for a video and download its audio with retries."""
        logger.info(f"Searching YouTube for: {query}")
        
        try:
            results = self.search_videos(query, limit=max_results)
            if not results:
                logger.warning(f"No results found for '{query}'")
                return None, None
            
            for video in results:
                try:
                    video_url = video['url']
                    logger.info(f"Attempting to download: {video['title']} ({video_url})")
                    
                    # Check if file already exists
                    existing_file = self.find_downloaded_file(video['title'])
                    if existing_file:
                        logger.info(f"File already exists: {existing_file}")
                        return existing_file, video
                    
                    # Download with specified quality
                    with yt_dlp.YoutubeDL(self.get_ydl_opts(quality)) as ydl:
                        info = ydl.extract_info(video_url, download=True)
                        
                    # Find the downloaded file
                    downloaded_file = self.find_downloaded_file(video['title'])
                    if not downloaded_file:
                        logger.error(f"Downloaded file not found for: {video['title']}")
                        continue
                    
                    logger.info(f"Successfully downloaded: {downloaded_file}")
                    return downloaded_file, video
                    
                except Exception as e:
                    logger.error(f"Error downloading {video['title']}: {str(e)}")
                    continue
            
            raise Exception(f"Could not download any audio for '{query}'")
            
        except Exception as e:
            logger.error(f"Error in search_and_download for '{query}': {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def extract_metadata(self, url: str) -> Optional[Dict]:
        """Extract metadata from a YouTube video with retries."""
        try:
            with yt_dlp.YoutubeDL(self.get_ydl_opts(download=False)) as ydl:
                info = ydl.extract_info(url, download=False)
                return self.format_video_metadata(info)
        except Exception as e:
            logger.error(f"Error extracting metadata from {url}: {str(e)}")
            raise

    def search(self, query: str) -> Optional[str]:
        """Search for a YouTube video and return its URL."""
        try:
            results = self.search_videos(query, limit=1)
            if not results:
                return None
            return results[0]['url']
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            return None

# Create a global instance
youtube_service = YouTubeService()

@router.post("/process", response_model=YouTubeResponse)
async def process_youtube_url(request: YouTubeRequest):
    """Process a YouTube URL to download audio and extract metadata."""
    try:
        with yt_dlp.YoutubeDL(youtube_service.get_ydl_opts(request.quality)) as ydl:
            info = ydl.extract_info(str(request.url), download=True)
            
            downloaded_file = youtube_service.find_downloaded_file(info['title'])
            if not downloaded_file:
                error = YouTubeError(
                    detail="Failed to download audio file",
                    error_code="DOWNLOAD_FAILED",
                    status_code=500
                )
                return YouTubeResponse(error=error)
            
            metadata = youtube_service.format_video_metadata(info)
            metadata['quality'] = request.quality
                
            return YouTubeResponse(file_path=downloaded_file, metadata=metadata)
    except yt_dlp.utils.DownloadError as e:
        error = YouTubeError(
            detail=f"Download failed: {str(e)}",
            error_code="YDL_DOWNLOAD_ERROR",
            status_code=400
        )
        return YouTubeResponse(error=error)
    except Exception as e:
        logger.error(f"Error processing YouTube URL {request.url}: {str(e)}")
        error = YouTubeError(
            detail=f"Internal server error: {str(e)}",
            error_code="INTERNAL_ERROR",
            status_code=500
        )
        return YouTubeResponse(error=error)

@router.get("/metadata", response_model=YouTubeResponse)
async def get_youtube_metadata(url: HttpUrl):
    """Extract metadata from a YouTube URL without downloading."""
    try:
        if not re.match(r'^https?://(www\.)?(youtube\.com|youtu\.be)', str(url)):
            error = YouTubeError(
                detail="Invalid YouTube URL",
                error_code="INVALID_URL",
                status_code=400
            )
            return YouTubeResponse(error=error)

        metadata = youtube_service.extract_metadata(str(url))
        if not metadata:
            error = YouTubeError(
                detail="Failed to extract metadata",
                error_code="METADATA_EXTRACTION_FAILED",
                status_code=400
            )
            return YouTubeResponse(error=error)
        
        return YouTubeResponse(metadata=metadata)
    except Exception as e:
        error = YouTubeError(
            detail=f"Failed to extract metadata: {str(e)}",
            error_code="METADATA_ERROR",
            status_code=500
        )
        return YouTubeResponse(error=error)

@router.post("/search", response_model=Dict[str, List[Dict[str, Any]]])
async def search_youtube(request: YouTubeSearchRequest):
    """Search YouTube for videos."""
    try:
        results = youtube_service.search_videos(request.query, limit=request.limit)
        if not results:
            error = YouTubeError(
                detail="No results found",
                error_code="NO_RESULTS",
                status_code=404
            )
            return {"results": [], "error": error.dict()}
            
        return {'results': results}
    except RetryError:
        error = YouTubeError(
            detail="Search failed after multiple retries",
            error_code="RETRY_EXHAUSTED",
            status_code=500
        )
        return {"results": [], "error": error.dict()}
    except Exception as e:
        logger.error(f"Error searching YouTube: {str(e)}")
        error = YouTubeError(
            detail=f"Search failed: {str(e)}",
            error_code="SEARCH_ERROR",
            status_code=500
        )
        return {"results": [], "error": error.dict()}

@router.post("/download", response_model=YouTubeResponse)
async def download_youtube_video(request: YouTubeRequest):
    """Download a YouTube video and convert it to audio"""
    try:
        with yt_dlp.YoutubeDL(youtube_service.get_ydl_opts(request.quality)) as ydl:
            info = ydl.extract_info(str(request.url), download=True)
            
            downloaded_file = youtube_service.find_downloaded_file(info['title'])
            if not downloaded_file:
                error = YouTubeError(
                    detail="Failed to download audio file",
                    error_code="DOWNLOAD_FAILED",
                    status_code=500
                )
                return YouTubeResponse(error=error)
            
            metadata = youtube_service.format_video_metadata(info)
            metadata['quality'] = request.quality
            
            return YouTubeResponse(file_path=downloaded_file, metadata=metadata)
    except yt_dlp.utils.DownloadError as e:
        error = YouTubeError(
            detail=f"Download failed: {str(e)}",
            error_code="YDL_DOWNLOAD_ERROR",
            status_code=400
        )
        return YouTubeResponse(error=error)
    except Exception as e:
        logger.error(f"Error downloading from {request.url}: {str(e)}")
        error = YouTubeError(
            detail=f"Download failed: {str(e)}",
            error_code="DOWNLOAD_ERROR",
            status_code=500
        )
        return YouTubeResponse(error=error)

# Helper functions for other modules
def search(query: str) -> Optional[str]:
    """Search for a YouTube video."""
    try:
        results = youtube_service.search_videos(query, limit=1)
        if not results:
            return None
        return results[0]['url']
    except Exception as e:
        logger.error(f"Error in search helper: {str(e)}")
        return None

def download_audio(url: str, quality: str = "192") -> Tuple[Optional[str], Optional[Dict]]:
    """Download audio from a YouTube URL."""
    try:
        with yt_dlp.YoutubeDL(youtube_service.get_ydl_opts(quality)) as ydl:
            info = ydl.extract_info(url, download=True)
            
            # Find the downloaded file
            downloaded_file = youtube_service.find_downloaded_file(info['title'])
            if not downloaded_file:
                logger.error(f"Downloaded file not found for: {info['title']}")
                return None, None
            
            metadata = youtube_service.format_video_metadata(info)
            metadata['quality'] = quality
            
            return downloaded_file, metadata
    except Exception as e:
        logger.error(f"Error in download_audio helper: {str(e)}")
        return None, None 