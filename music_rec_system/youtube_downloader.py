#!/usr/bin/env python3
import os
import logging
from pathlib import Path
from pytube import YouTube, Search
import tempfile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YouTubeDownloader:
    """Class to handle downloading songs from YouTube."""
    
    def __init__(self, output_dir=None):
        """
        Initialize the downloader.
        
        Args:
            output_dir: Directory to save downloaded songs. Defaults to current directory.
        """
        self.output_dir = output_dir or os.getcwd()
        os.makedirs(self.output_dir, exist_ok=True)
    
    def search_and_download(self, query, max_results=5):
        """
        Search for a song on YouTube and download the audio.
        
        Args:
            query: Song name or artist to search for
            max_results: Maximum number of search results to consider
            
        Returns:
            Path to the downloaded audio file or None if download fails
        """
        logger.info(f"Searching YouTube for: {query}")
        search_query = f"{query} official audio"
        
        try:
            search_results = Search(search_query).results
            
            if not search_results:
                logger.warning(f"No results found for '{query}'")
                return None
            
            # Get top results
            results = search_results[:max_results]
            
            # Try to download from each result until successful
            for video in results:
                try:
                    logger.info(f"Attempting to download: {video.title}")
                    # Create a YouTube object
                    yt = YouTube(f"https://www.youtube.com/watch?v={video.video_id}")
                    
                    # Get the audio stream with highest quality
                    audio_stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()
                    
                    if not audio_stream:
                        logger.warning(f"No audio stream found for {video.title}")
                        continue
                    
                    # Sanitize filename
                    safe_title = "".join([c for c in video.title if c.isalpha() or c.isdigit() or c in ' ._-']).rstrip()
                    
                    # Download to the output directory with mp3 extension
                    temp_file = audio_stream.download(
                        output_path=self.output_dir,
                        filename=f"{safe_title}.mp4"
                    )
                    
                    # Convert to mp3
                    mp3_file = os.path.join(self.output_dir, f"{safe_title}.mp3")
                    self._convert_to_mp3(temp_file, mp3_file)
                    
                    logger.info(f"Successfully downloaded: {mp3_file}")
                    return mp3_file
                    
                except Exception as e:
                    logger.error(f"Error downloading {video.title}: {str(e)}")
                    continue
            
            logger.warning(f"Could not download any audio for '{query}'")
            return None
            
        except Exception as e:
            logger.error(f"Error searching for '{query}': {str(e)}")
            return None
    
    def _convert_to_mp3(self, input_file, output_file):
        """
        Convert audio file to MP3 format using FFmpeg.
        
        Args:
            input_file: Path to input audio file
            output_file: Path for the output MP3 file
        """
        try:
            # Try to use ffmpeg if available
            import subprocess
            command = [
                'ffmpeg', '-i', input_file, 
                '-vn', '-ab', '128k', '-ar', '44100', '-y', 
                output_file
            ]
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Remove the temporary file
            if os.path.exists(input_file):
                os.remove(input_file)
                
        except Exception as e:
            logger.warning(f"FFmpeg conversion failed: {str(e)}")
            logger.info("Using simple file rename as fallback")
            
            # Simple rename as fallback if ffmpeg isn't available
            os.rename(input_file, output_file)


# Example usage
if __name__ == "__main__":
    downloader = YouTubeDownloader()
    file_path = downloader.search_and_download("Foolmuse Peter Cat Recording Co")
    print(f"Downloaded file: {file_path}") 