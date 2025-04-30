import os
import requests
from bs4 import BeautifulSoup
import logging
from typing import Dict, Optional, Tuple, List
import re
import json
from urllib.parse import quote, unquote
import yt_dlp

logger = logging.getLogger(__name__)

class YouTubeService:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.base_url = "https://www.youtube.com"
        self.search_url = "https://www.youtube.com/results?search_query="
        
        # yt-dlp configuration
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': 'data/audio/%(title)s.%(ext)s',
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True
        }

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
            response = requests.get(f"{self.search_url}{encoded_query}", headers=self.headers)
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

            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                # Extract video info first
                info = ydl.extract_info(url, download=False)
                
                # Download audio
                ydl.download([url])
                
                # Get the output path
                output_path = os.path.join(
                    'data/audio',
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

            response = requests.get(f"{self.base_url}/watch?v={video_id}", headers=self.headers)
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
youtube_service = YouTubeService() 