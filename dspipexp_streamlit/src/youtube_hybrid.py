# dspipexp_streamlit/src/youtube_hybrid.py
import os
import re
import logging
import tempfile
import time
import shutil
import requests
import json
from bs4 import BeautifulSoup
import yt_dlp
import importlib
from typing import Dict, Optional, List, Tuple, Any, Union
from datetime import timedelta

# Configure logging
logger = logging.getLogger(__name__)

# Define user agent for requests
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

# Constants
YOUTUBE_API_AVAILABLE = False
INNERTUBE_AVAILABLE = False
AIOTUBE_AVAILABLE = False

# Try to import optional dependencies
try:
    import innertube
    INNERTUBE_AVAILABLE = True
    logger.info("InnerTube library is available")
except ImportError:
    logger.info("InnerTube library not found")

try:
    import aiotube
    AIOTUBE_AVAILABLE = True
    logger.info("Aiotube library is available")
except ImportError:
    logger.info("Aiotube library not found")

try:
    from googleapiclient.discovery import build
    from google.oauth2.credentials import Credentials
    YOUTUBE_API_AVAILABLE = True
    logger.info("YouTube API library is available")
except ImportError:
    logger.info("YouTube API library not found")


def extract_video_id(youtube_url: str) -> Optional[str]:
    """Extract video ID from various YouTube URL formats"""
    if not youtube_url:
        return None
    
    # Extract video ID from various YouTube URL formats
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # Standard YouTube URLs
        r'(?:embed\/)([0-9A-Za-z_-]{11})',   # Embed URLs
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'  # Short URLs
    ]
    
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    
    return None


def get_video_info_with_innertube(youtube_url: str) -> Optional[Dict]:
    """Try to get video info using InnerTube API"""
    if not INNERTUBE_AVAILABLE:
        return None
    
    video_id = extract_video_id(youtube_url)
    if not video_id:
        return None
    
    try:
        # Create an InnerTube client for WEB platform
        client = innertube.InnerTube("WEB")
        
        # Get video details using player endpoint
        response = client.player(video_id=video_id)
        
        if not response or 'videoDetails' not in response:
            return None
            
        video_details = response['videoDetails']
        
        # Format the response into our standard format
        info = {
            'title': video_details.get('title', 'Unknown'),
            'uploader': video_details.get('author', 'Unknown'),
            'duration': int(video_details.get('lengthSeconds', 0)),
            'view_count': int(video_details.get('viewCount', 0)),
            'description': video_details.get('shortDescription', ''),
            'thumbnail': video_details.get('thumbnail', {}).get('thumbnails', [{}])[-1].get('url', '')
        }
        
        # Try to extract song and artist info from description
        description = info['description']
        
        # Look for patterns like "Song: xxx" or "Artist: xxx" in description
        song_match = re.search(r'Song:?\s*([^\n]+)', description, re.IGNORECASE)
        if song_match:
            info['parsed_song'] = song_match.group(1).strip()
        
        artist_match = re.search(r'Artist:?\s*([^\n]+)', description, re.IGNORECASE)
        if artist_match:
            info['parsed_artist'] = artist_match.group(1).strip()
        
        # Look for patterns like "Music: Artist - Song"
        music_match = re.search(r'Music:?\s*([^-\n]+)-([^\n]+)', description, re.IGNORECASE)
        if music_match:
            info['parsed_artist'] = music_match.group(1).strip()
            info['parsed_song'] = music_match.group(2).strip()
            
        return info
        
    except Exception as e:
        logger.warning(f"InnerTube error: {str(e)}")
        return None


def get_video_info_with_aiotube(youtube_url: str) -> Optional[Dict]:
    """Try to get video info using Aiotube library"""
    if not AIOTUBE_AVAILABLE:
        return None
    
    video_id = extract_video_id(youtube_url)
    if not video_id:
        return None
    
    try:
        # Create a Video object
        video = aiotube.Video(video_id)
        
        # Get metadata
        metadata = video.metadata
        
        if not metadata:
            return None
            
        # Format response
        info = {
            'title': metadata.get('title', 'Unknown'),
            'uploader': metadata.get('author', {}).get('name', 'Unknown'),
            'duration': metadata.get('lengthSeconds', 0),
            'view_count': metadata.get('viewCount', 0),
            'description': metadata.get('description', ''),
            'thumbnail': metadata.get('thumbnail', [{}])[-1].get('url', '')
        }
        
        # Try to extract song and artist info from description
        description = info['description']
        
        # Look for patterns in description
        song_match = re.search(r'Song:?\s*([^\n]+)', description, re.IGNORECASE)
        if song_match:
            info['parsed_song'] = song_match.group(1).strip()
        
        artist_match = re.search(r'Artist:?\s*([^\n]+)', description, re.IGNORECASE)
        if artist_match:
            info['parsed_artist'] = artist_match.group(1).strip()
        
        # Look for patterns like "Music: Artist - Song"
        music_match = re.search(r'Music:?\s*([^-\n]+)-([^\n]+)', description, re.IGNORECASE)
        if music_match:
            info['parsed_artist'] = music_match.group(1).strip()
            info['parsed_song'] = music_match.group(2).strip()
            
        return info
        
    except Exception as e:
        logger.warning(f"Aiotube error: {str(e)}")
        return None


def get_video_info_with_api(youtube_url: str, api_key: Optional[str] = None) -> Optional[Dict]:
    """Try to get video info using YouTube Data API"""
    if not YOUTUBE_API_AVAILABLE or not api_key:
        return None
    
    video_id = extract_video_id(youtube_url)
    if not video_id:
        return None
    
    try:
        # Create a YouTube service object
        youtube = build("youtube", "v3", developerKey=api_key)
        
        # Get video details
        response = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=video_id
        ).execute()
        
        if not response.get('items', []):
            return None
            
        item = response['items'][0]
        snippet = item.get('snippet', {})
        statistics = item.get('statistics', {})
        
        # Format response
        info = {
            'title': snippet.get('title', 'Unknown'),
            'uploader': snippet.get('channelTitle', 'Unknown'),
            'duration': 0,  # Need to parse contentDetails.duration (PT#M#S format)
            'upload_date': snippet.get('publishedAt', ''),
            'view_count': int(statistics.get('viewCount', 0)),
            'like_count': int(statistics.get('likeCount', 0)),
            'description': snippet.get('description', ''),
            'tags': snippet.get('tags', []),
            'thumbnail': snippet.get('thumbnails', {}).get('high', {}).get('url', '')
        }
        
        # Try to extract song and artist info from description
        description = info['description']
        
        # Look for patterns in description
        song_match = re.search(r'Song:?\s*([^\n]+)', description, re.IGNORECASE)
        if song_match:
            info['parsed_song'] = song_match.group(1).strip()
        
        artist_match = re.search(r'Artist:?\s*([^\n]+)', description, re.IGNORECASE)
        if artist_match:
            info['parsed_artist'] = artist_match.group(1).strip()
        
        # Look for patterns like "Music: Artist - Song"
        music_match = re.search(r'Music:?\s*([^-\n]+)-([^\n]+)', description, re.IGNORECASE)
        if music_match:
            info['parsed_artist'] = music_match.group(1).strip()
            info['parsed_song'] = music_match.group(2).strip()
            
        return info
        
    except Exception as e:
        logger.warning(f"YouTube API error: {str(e)}")
        return None


def get_video_info_with_yt_dlp(youtube_url: str) -> Optional[Dict]:
    """Get detailed information about a YouTube video using yt-dlp"""
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'writeinfojson': True,
        'noplaylist': True,
        'no_check_certificate': True,
        'ignoreerrors': True,
        'nocheckcertificate': True,
        'http_headers': {
            'User-Agent': USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-us,en;q=0.5',
            'Sec-Fetch-Mode': 'navigate',
        },
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        ydl_opts['outtmpl'] = os.path.join(temp_dir, 'info')
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                
                if not info:
                    return None
                
                # Extract more metadata from description and tags
                enhanced_info = {
                    'title': info.get('title', 'Unknown'),
                    'uploader': info.get('uploader', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'upload_date': info.get('upload_date', ''),
                    'view_count': info.get('view_count', 0),
                    'like_count': info.get('like_count', 0),
                    'description': info.get('description', ''),
                    'tags': info.get('tags', []),
                    'categories': info.get('categories', []),
                    'thumbnail': info.get('thumbnail', '')
                }
                
                # Try to extract song and artist info from description
                description = enhanced_info['description']
                
                # Look for patterns like "Song: xxx" or "Artist: xxx" in description
                song_match = re.search(r'Song:?\s*([^\n]+)', description, re.IGNORECASE)
                if song_match:
                    enhanced_info['parsed_song'] = song_match.group(1).strip()
                
                artist_match = re.search(r'Artist:?\s*([^\n]+)', description, re.IGNORECASE)
                if artist_match:
                    enhanced_info['parsed_artist'] = artist_match.group(1).strip()
                
                # Look for patterns like "Music: Artist - Song"
                music_match = re.search(r'Music:?\s*([^-\n]+)-([^\n]+)', description, re.IGNORECASE)
                if music_match:
                    enhanced_info['parsed_artist'] = music_match.group(1).strip()
                    enhanced_info['parsed_song'] = music_match.group(2).strip()
                
                return enhanced_info
                
        except Exception as e:
            logger.error(f"yt-dlp error: {e}")
            return None


def get_video_info_with_scraping(youtube_url: str) -> Optional[Dict]:
    """Get video information by scraping YouTube page"""
    video_id = extract_video_id(youtube_url)
    if not video_id:
        return None
    
    headers = {
        'User-Agent': USER_AGENT,
        'Accept-Language': 'en-US,en;q=0.9',
    }
    
    try:
        response = requests.get(f"https://www.youtube.com/watch?v={video_id}", headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try to find and extract JSON data from the page
        for script in soup.find_all('script'):
            script_text = str(script)
            
            # Look for initial player response
            if 'ytInitialPlayerResponse' in script_text:
                json_str = script_text.split('ytInitialPlayerResponse = ')[1].split(';</script>')[0]
                try:
                    data = json.loads(json_str)
                    video_details = data.get('videoDetails', {})
                    
                    info = {
                        'title': video_details.get('title', 'Unknown'),
                        'uploader': video_details.get('author', 'Unknown'),
                        'description': video_details.get('shortDescription', ''),
                        'view_count': int(video_details.get('viewCount', 0)),
                        'duration': int(video_details.get('lengthSeconds', 0)),
                        'thumbnail': f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
                    }
                    
                    # Try to extract song and artist info from description
                    description = info['description']
                    
                    # Look for patterns in description
                    song_match = re.search(r'Song:?\s*([^\n]+)', description, re.IGNORECASE)
                    if song_match:
                        info['parsed_song'] = song_match.group(1).strip()
                    
                    artist_match = re.search(r'Artist:?\s*([^\n]+)', description, re.IGNORECASE)
                    if artist_match:
                        info['parsed_artist'] = artist_match.group(1).strip()
                    
                    # Look for patterns like "Music: Artist - Song"
                    music_match = re.search(r'Music:?\s*([^-\n]+)-([^\n]+)', description, re.IGNORECASE)
                    if music_match:
                        info['parsed_artist'] = music_match.group(1).strip()
                        info['parsed_song'] = music_match.group(2).strip()
                        
                    return info
                except json.JSONDecodeError:
                    pass
        
        return None
        
    except Exception as e:
        logger.warning(f"Scraping error: {str(e)}")
        return None


def get_video_info_hybrid(youtube_url: str, api_key: Optional[str] = None) -> Optional[Dict]:
    """
    Try multiple methods to get video information, falling back to less reliable 
    methods if more reliable ones fail.
    
    Order:
    1. InnerTube API (if available)
    2. Aiotube (if available)
    3. YouTube Data API (if available and API key provided)
    4. yt-dlp
    5. Web scraping (last resort)
    
    Returns:
        Dict with video information or None if all methods fail
    """
    # Try each method in sequence
    info = None
    
    # Try InnerTube first (most reliable and lightweight)
    if INNERTUBE_AVAILABLE:
        logger.info("Trying InnerTube API...")
        info = get_video_info_with_innertube(youtube_url)
        if info:
            info['source'] = 'innertube'
            return info
    
    # Try Aiotube next (good alternative)
    if AIOTUBE_AVAILABLE:
        logger.info("Trying Aiotube...")
        info = get_video_info_with_aiotube(youtube_url)
        if info:
            info['source'] = 'aiotube'
            return info
    
    # Try YouTube Data API if available and key provided
    if YOUTUBE_API_AVAILABLE and api_key:
        logger.info("Trying YouTube Data API...")
        info = get_video_info_with_api(youtube_url, api_key)
        if info:
            info['source'] = 'youtube_api'
            return info
    
    # Try yt-dlp (might be blocked by YouTube on cloud servers)
    logger.info("Trying yt-dlp...")
    info = get_video_info_with_yt_dlp(youtube_url)
    if info:
        info['source'] = 'yt-dlp'
        return info
    
    # Last resort: web scraping
    logger.info("Trying web scraping...")
    info = get_video_info_with_scraping(youtube_url)
    if info:
        info['source'] = 'web_scraping'
        return info
    
    logger.error("All methods failed to get video info")
    return None


# YouTube processing functions with our hybrid approach
def download_youtube_audio(youtube_url, output_dir, progress_callback=None):
    """Download YouTube audio using yt-dlp with fallback options"""
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    if not ffmpeg_path or not ffprobe_path:
        return None, "ffmpeg or ffprobe not found in PATH. Please install them."
    
    # Use a generic output template to capture any extension
    output_template = os.path.join(output_dir, "yt_audio.%(ext)s")
    
    class ProgressHook:
        def __init__(self):
            self.start_time = time.time()
            self.previous_time = self.start_time
            self.downloaded_bytes = 0
            self.total_bytes = 0
            self.status = "downloading"
            self.eta = None
        
        def __call__(self, d):
            if d['status'] == 'downloading':
                self.status = "downloading"
                if 'downloaded_bytes' in d:
                    self.downloaded_bytes = d['downloaded_bytes']
                if 'total_bytes' in d:
                    self.total_bytes = d['total_bytes']
                
                # Calculate ETA and speed
                current_time = time.time()
                elapsed = current_time - self.start_time
                
                # Only update UI every 0.5 seconds to avoid excessive refreshes
                if current_time - self.previous_time >= 0.5 and progress_callback:
                    self.previous_time = current_time
                    
                    if self.total_bytes and self.downloaded_bytes:
                        progress = self.downloaded_bytes / self.total_bytes
                        if elapsed > 0 and progress > 0:
                            time_remaining = (elapsed / progress) - elapsed
                            speed = self.downloaded_bytes / elapsed / 1024  # KB/s
                            
                            self.eta = str(timedelta(seconds=int(time_remaining)))
                            progress_callback(progress, speed, self.eta)
            
            elif d['status'] == 'finished':
                self.status = "processing"
                if progress_callback:
                    progress_callback(1.0, 0, "Converting...")
    
    progress_hook = ProgressHook()
    
    # Enhanced yt-dlp options to avoid blocking
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'quiet': True,
        'noplaylist': True,
        'progress_hooks': [progress_hook],
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'nocheckcertificate': True,
        'http_headers': {
            'User-Agent': USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-us,en;q=0.5',
            'Sec-Fetch-Mode': 'navigate',
        },
        'geo_bypass': True,
        'geo_bypass_country': 'US',
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        # Look for the mp3 file
        mp3_path = os.path.join(output_dir, "yt_audio.mp3")
        if os.path.exists(mp3_path):
            return mp3_path, None
        # If not found, fall through to fallback
    except Exception as e:
        logger.warning(f"First yt-dlp attempt failed: {e}")
        pass  # Will try fallback below
    
    # Fallback: try downloading without postprocessing
    try:
        ydl_opts.pop('postprocessors', None)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        # Find the downloaded file (could be .webm, .m4a, etc.)
        files = glob.glob(os.path.join(output_dir, "yt_audio.*"))
        if not files:
            return None, "yt-dlp fallback failed: no audio file found."
        return files[0], "Warning: Could not convert to mp3. Downloaded raw audio."
    except Exception as e2:
        return None, f"yt-dlp failed: {e2}"


def process_youtube_url_hybrid(youtube_url, progress_callback=None, api_key=None):
    """
    Process a YouTube URL to extract audio features with improved success rate
    using our hybrid approach
    
    Args:
        youtube_url: URL of the YouTube video
        progress_callback: Function to call with progress updates
        api_key: Optional YouTube API key
    
    Returns:
        Tuple of (features_dict, error_message)
    """
    from src.audio import extract_audio_features
    
    if progress_callback:
        progress_callback(0.0, 0, "Starting YouTube processing...")
    
    start_time = time.time()
    
    # Step 1: Get detailed video info (30%)
    if progress_callback:
        progress_callback(0.05, 0, "Fetching video information...")
    
    # Use our hybrid function instead of the original
    video_info = get_video_info_hybrid(youtube_url, api_key)
    
    if progress_callback:
        progress_callback(0.1, 0, 
            f"Video information retrieved using {video_info.get('source', 'unknown') if video_info else 'N/A'}")
    
    # Step 2: Download and process audio (40%)
    with tempfile.TemporaryDirectory() as temp_dir:
        if progress_callback:
            progress_callback(0.1, 0, "Downloading audio...")
        
        # Custom progress callback for download
        def download_progress(prog, speed, eta):
            # Map download progress (0-1) to overall progress (0.1-0.4)
            overall_progress = 0.1 + (prog * 0.3)
            progress_callback(overall_progress, speed, f"Downloading: {eta}")
        
        audio_path, error = download_youtube_audio(
            youtube_url, temp_dir, 
            progress_callback=download_progress if progress_callback else None
        )
        
        if not audio_path or not os.path.exists(audio_path):
            return None, error or "Audio file was not downloaded."
        
        if progress_callback:
            progress_callback(0.4, 0, "Audio downloaded successfully")
        
        # Step 3: Convert audio format if needed (20%)
        ext = os.path.splitext(audio_path)[1].lower()
        if ext not in [".mp3", ".wav"]:
            if progress_callback:
                progress_callback(0.45, 0, "Converting audio format...")
            
            # Convert to wav for librosa compatibility
            from src.youtube import convert_to_wav  # Import here to avoid circular imports
            wav_path, conv_err = convert_to_wav(audio_path)
            if not wav_path or not os.path.exists(wav_path):
                return None, conv_err or "Audio conversion failed."
            audio_path = wav_path
            
            if progress_callback:
                progress_callback(0.5, 0, "Audio conversion complete")
        
        # Step 4: Extract audio features (30%)
        if progress_callback:
            progress_callback(0.6, 0, "Extracting audio features...")
        
        try:
            # This is where most of the processing time is spent
            features = extract_audio_features(audio_path)
            
            if progress_callback:
                progress_callback(0.9, 0, "Audio features extracted")
            
            # Step 5: Combine all information
            # Add source metadata as a separate field rather than mixing with audio features
            features_with_meta = {
                "audio_features": features,
                "source": "youtube",
                "extraction_method": video_info.get('source', 'unknown') if video_info else 'unknown',
                "processed_with": "youtube_hybrid_extraction",
                "video_info": video_info
            }
            
            elapsed_time = time.time() - start_time
            
            if progress_callback:
                progress_callback(1.0, 0, f"Processing complete in {elapsed_time:.1f}s")
            
            return features_with_meta, error  # error may be a warning
        except Exception as e:
            return None, f"Audio feature extraction failed: {e}"


def youtube_search_hybrid(query, max_results=5, api_key=None):
    """
    Search YouTube using the best available method
    
    Order:
    1. YouTube Data API (if available and API key provided)
    2. InnerTube API (if available)
    3. Aiotube (if available)
    4. yt-dlp
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        api_key: Optional YouTube API key
    
    Returns:
        List of video results
    """
    results = []
    
    # Try YouTube Data API first if available and key provided
    if YOUTUBE_API_AVAILABLE and api_key:
        try:
            logger.info("Searching with YouTube Data API...")
            youtube = build("youtube", "v3", developerKey=api_key)
            
            # Execute the search request
            search_response = youtube.search().list(
                q=query,
                part="id,snippet",
                maxResults=max_results,
                type="video"
            ).execute()
            
            # Process the search results
            for item in search_response.get("items", []):
                if item["id"]["kind"] == "youtube#video":
                    video_id = item["id"]["videoId"]
                    
                    # Optionally get more details about the video
                    video_response = youtube.videos().list(
                        part="contentDetails,statistics",
                        id=video_id
                    ).execute()
                    
                    video_details = {}
                    if video_response.get("items"):
                        video_details = video_response["items"][0]
                    
                    # Extract duration in seconds
                    duration = 0
                    
                    results.append({
                        'video_id': video_id,
                        'title': item["snippet"]["title"],
                        'channel': item["snippet"]["channelTitle"],
                        'duration': duration,
                        'url': f"https://www.youtube.com/watch?v={video_id}"
                    })
            
            if results:
                return results
                
        except Exception as e:
            logger.warning(f"YouTube Data API search error: {e}")
    
    # Continue with existing methods as fallbacks
    # Try InnerTube first
    if INNERTUBE_AVAILABLE and not results:
        try:
            logger.info("Searching with InnerTube API...")
            client = innertube.InnerTube("WEB")
            response = client.search(query=query)
            
            if response and 'contents' in response:
                # Extract search results from response
                items = []
                search_contents = response.get('contents', {}).get('twoColumnSearchResultsRenderer', {}).get(
                    'primaryContents', {}).get('sectionListRenderer', {}).get('contents', [])
                
                for content in search_contents:
                    if 'itemSectionRenderer' in content:
                        section_items = content['itemSectionRenderer'].get('contents', [])
                        for item in section_items:
                            if 'videoRenderer' in item:
                                video_renderer = item['videoRenderer']
                                video_id = video_renderer.get('videoId')
                                if video_id:
                                    items.append({
                                        'video_id': video_id,
                                        'title': video_renderer.get('title', {}).get('runs', [{}])[0].get('text', ''),
                                        'channel': video_renderer.get('ownerText', {}).get('runs', [{}])[0].get('text', ''),
                                        'duration': video_renderer.get('lengthText', {}).get('simpleText', ''),
                                        'url': f"https://www.youtube.com/watch?v={video_id}"
                                    })
                                    if len(items) >= max_results:
                                        break
                
                if items:
                    for item in items[:max_results]:
                        results.append(item)
                    return results
        except Exception as e:
            logger.warning(f"InnerTube search error: {e}")
    
    # Rest of the existing function...
    # [No changes to the existing Aiotube and yt-dlp parts]
    
    return results


def youtube_search_and_get_url_hybrid(query, api_key=None):
    """Search YouTube and return the URL of the first result using hybrid approach"""
    # Try to keep both versions working with or without api_key
    try:
        results = youtube_search_hybrid(query, max_results=1, api_key=api_key)
    except TypeError:
        # Fall back to version without api_key
        results = youtube_search_hybrid(query, max_results=1)
        
    if results and len(results) > 0:
        return results[0]['url']
    return None

# Export the main functions to use as replacements for the original ones
__all__ = [
    'get_video_info_hybrid',
    'process_youtube_url_hybrid',
    'youtube_search_hybrid',
    'youtube_search_and_get_url_hybrid',
    'extract_video_id'
]