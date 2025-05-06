# YouTube audio download utilities
import yt_dlp
import tempfile
import os
from src.audio import extract_audio_features
from sqlalchemy import text
import shutil
import glob
import subprocess
import time
import logging
import json
from datetime import datetime, timedelta
import re

# Configure logging
logger = logging.getLogger(__name__)

def download_youtube_audio(youtube_url, output_dir, progress_callback=None):
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

def convert_to_wav(input_path, progress_callback=None):
    output_path = os.path.splitext(input_path)[0] + ".wav"
    try:
        if progress_callback:
            progress_callback(0.0, 0, "Converting audio format...")
        
        result = subprocess.run([
            "ffmpeg", "-y", "-i", input_path, output_path
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if progress_callback:
            progress_callback(1.0, 0, "Conversion complete")
        
        return output_path, None
    except subprocess.CalledProcessError as e:
        return None, f"ffmpeg conversion failed: {e.stderr.decode()}"

def get_video_info(youtube_url):
    """Get detailed information about a YouTube video"""
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'writeinfojson': True,
        'noplaylist': True,
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        ydl_opts['outtmpl'] = os.path.join(temp_dir, 'info')
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                
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
            logger.error(f"Error getting video info: {e}")
            return None

def process_youtube_url(youtube_url, progress_callback=None):
    """
    Process a YouTube URL to extract audio features with detailed progress reporting
    
    Args:
        youtube_url: URL of the YouTube video
        progress_callback: Function to call with progress updates (progress, speed, eta)
    
    Returns:
        Tuple of (features_dict, error_message)
    """
    if progress_callback:
        progress_callback(0.0, 0, "Starting YouTube processing...")
    
    start_time = time.time()
    
    # Step 1: Get detailed video info (30%)
    if progress_callback:
        progress_callback(0.05, 0, "Fetching video information...")
    
    video_info = get_video_info(youtube_url)
    
    if progress_callback:
        progress_callback(0.1, 0, "Video information retrieved")
    
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
                "processed_with": "youtube_enhanced_similarity",
                "video_info": video_info
            }
            
            elapsed_time = time.time() - start_time
            
            if progress_callback:
                progress_callback(1.0, 0, f"Processing complete in {elapsed_time:.1f}s")
            
            return features_with_meta, error  # error may be a warning
        except Exception as e:
            return None, f"Audio feature extraction failed: {e}"

def youtube_search_and_get_url(query):
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'extract_flat': True,
        'default_search': 'ytsearch',
        'noplaylist': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            result = ydl.extract_info(query, download=False)
            if 'entries' in result and result['entries']:
                video = result['entries'][0]
                return f"https://www.youtube.com/watch?v={video['id']}"
        except Exception:
            return None
    return None

def update_song_youtube_url(engine, song_id, youtube_url):
    sql = text("UPDATE songs SET youtube_url = :youtube_url WHERE id = :id")
    with engine.begin() as conn:
        conn.execute(sql, {"youtube_url": youtube_url, "id": song_id})

def youtube_search(query, max_results=5):
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
        'skip_download': True,
        'default_search': 'ytsearch',
        'noplaylist': True,
    }
    results = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        search_results = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
        for entry in search_results.get('entries', [])[:max_results]:
            results.append({
                'video_id': entry.get('id'),
                'title': entry.get('title'),
                'channel': entry.get('uploader'),
                'duration': entry.get('duration'),
                'url': f"https://www.youtube.com/watch?v={entry.get('id')}"
            })
    return results 