# YouTube audio download utilities
import yt_dlp
import tempfile
import os
from src.audio import extract_audio_features
from sqlalchemy import text
import shutil
import glob
import subprocess

def download_youtube_audio(youtube_url, output_dir):
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    if not ffmpeg_path or not ffprobe_path:
        return None, "ffmpeg or ffprobe not found in PATH. Please install them."
    # Use a generic output template to capture any extension
    output_template = os.path.join(output_dir, "yt_audio.%(ext)s")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'quiet': True,
        'noplaylist': True,
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

def convert_to_wav(input_path):
    output_path = os.path.splitext(input_path)[0] + ".wav"
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path, output_path
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_path, None
    except subprocess.CalledProcessError as e:
        return None, f"ffmpeg conversion failed: {e.stderr.decode()}"

def process_youtube_url(youtube_url):
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path, error = download_youtube_audio(youtube_url, temp_dir)
        if not audio_path or not os.path.exists(audio_path):
            return None, error or "Audio file was not downloaded."
        ext = os.path.splitext(audio_path)[1].lower()
        if ext not in [".mp3", ".wav"]:
            # Convert to wav for librosa compatibility
            wav_path, conv_err = convert_to_wav(audio_path)
            if not wav_path or not os.path.exists(wav_path):
                return None, conv_err or "Audio conversion failed."
            audio_path = wav_path
        try:
            features = extract_audio_features(audio_path)
            return features, error  # error may be a warning
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