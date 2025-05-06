# Music metadata fetching utilities
import requests
import logging
import time
import re
from urllib.parse import quote_plus
import musicbrainzngs
import streamlit as st

# Configure logging
logger = logging.getLogger(__name__)

# Initialize MusicBrainz client
musicbrainzngs.set_useragent("DSPipeXP", "1.0", "https://github.com/yourusername/dspipexp")

def fetch_song_metadata(track_name, artist_name, max_retries=3):
    """
    Attempt to fetch comprehensive song metadata from multiple sources
    Returns a dictionary with all available metadata
    """
    metadata = {
        "track_name": track_name,
        "artist_name": artist_name,
        "album": None,
        "release_date": None,
        "genre": None,
        "isrc": None,
        "duration": None,
        "album_art_url": None,
        "source": None,
        "confidence": 0.0
    }
    
    # Try multiple sources in order of reliability/completeness
    sources = [
        fetch_metadata_from_genius,    # Start with Genius since we already have API access
        fetch_metadata_from_lastfm,    # Then Last.fm which is also already integrated
        fetch_metadata_from_musicbrainz
    ]
    
    # Try each source until we get good results
    for source_fn in sources:
        try:
            source_metadata = source_fn(track_name, artist_name)
            if source_metadata and source_metadata["confidence"] > metadata["confidence"]:
                # Update our metadata with the better source
                metadata.update(source_metadata)
                
                # If we have a high confidence match, we can stop
                if metadata["confidence"] >= 0.8:
                    break
        except Exception as e:
            logger.warning(f"Error fetching metadata from {source_fn.__name__}: {e}")
            continue
    
    return metadata

def fetch_metadata_from_genius(track_name, artist_name, max_retries=3):
    """
    Fetch metadata from Genius API - reusing the existing API access
    from the lyrics.py module
    """
    # Get API key from streamlit secrets
    api_key = st.secrets.get("genius_api_key", None)
    if not api_key:
        logger.warning("Genius API key not set.")
        return None
    
    base_url = "https://api.genius.com/search"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"q": f"{track_name} {artist_name}"}
    
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = requests.get(base_url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 429:  # Rate limited
                retry_count += 1
                time.sleep(2 * retry_count)  # Exponential backoff
                continue
                
            if response.status_code != 200:
                logger.warning(f"Genius API returned status code {response.status_code}.")
                return None
                
            hits = response.json().get("response", {}).get("hits", [])
            if not hits:
                logger.warning("No results found via Genius API.")
                return None
                
            # Try the top result
            song_info = hits[0]["result"]
            
            # Match confidence check - compare titles and artists
            hit_title = song_info.get('title', '').lower()
            hit_artist = song_info.get('primary_artist', {}).get('name', '').lower()
            track_name_lower = track_name.lower()
            artist_name_lower = artist_name.lower()
            
            confidence = 0.5  # Base confidence
            
            # Title match
            if track_name_lower == hit_title:
                confidence += 0.3
            elif track_name_lower in hit_title or hit_title in track_name_lower:
                confidence += 0.2
            
            # Artist match
            if artist_name_lower == hit_artist:
                confidence += 0.2
            elif artist_name_lower in hit_artist or hit_artist in artist_name_lower:
                confidence += 0.1
            
            # Get additional song data
            # We need to make a separate request to get full song details
            song_id = song_info.get('id')
            if song_id:
                try:
                    song_url = f"https://api.genius.com/songs/{song_id}"
                    song_response = requests.get(song_url, headers=headers, timeout=10)
                    
                    if song_response.status_code == 200:
                        song_data = song_response.json().get("response", {}).get("song", {})
                        
                        # Extract album info
                        album_name = song_data.get("album", {}).get("name") if song_data.get("album") else None
                        
                        # Extract release date
                        release_date = song_data.get("release_date")
                        
                        # Extract album art URL
                        album_art_url = song_data.get("song_art_image_url")
                        
                        metadata = {
                            "track_name": song_info["title"],
                            "artist_name": song_info.get("primary_artist", {}).get("name"),
                            "album": album_name,
                            "release_date": release_date,
                            "album_art_url": album_art_url,
                            "source": "genius",
                            "confidence": confidence
                        }
                        
                        return metadata
                except Exception as e:
                    logger.warning(f"Error fetching full song details from Genius: {e}")
            
            # Fallback to basic metadata
            return {
                "track_name": song_info["title"],
                "artist_name": song_info.get("primary_artist", {}).get("name"),
                "source": "genius",
                "confidence": confidence
            }
            
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                logger.warning(f"Error fetching from Genius API: {e}")
                return None
            time.sleep(1)
    
    return None

def fetch_metadata_from_lastfm(track_name, artist_name, max_retries=3):
    """Fetch metadata from Last.fm API"""
    # Get API key from streamlit secrets
    api_key = st.secrets.get("lastfm_api_key", None)
    if not api_key:
        logger.warning("Last.fm API key not set.")
        return None
    
    base_url = "http://ws.audioscrobbler.com/2.0/"
    params = {
        "method": "track.getInfo",
        "api_key": api_key,
        "artist": artist_name,
        "track": track_name,
        "format": "json",
        "autocorrect": 1  # Use Last.fm correction
    }
    
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = requests.get(base_url, params=params, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"Last.fm API returned status code {response.status_code}.")
                return None
            
            data = response.json()
            
            # Check if we got track info
            if "track" not in data:
                logger.warning("No track data returned from Last.fm.")
                return None
            
            track_data = data["track"]
            
            # Calculate confidence
            confidence = 0.6  # Last.fm has autocorrect, so base confidence is higher
            
            # Extract metadata
            album_name = track_data.get("album", {}).get("title") if "album" in track_data else None
            album_art_url = None
            
            # Get largest album art if available
            if "album" in track_data and "image" in track_data["album"]:
                for image in track_data["album"]["image"]:
                    if image["size"] == "extralarge" and image["#text"]:
                        album_art_url = image["#text"]
                        break
            
            # Try to get genre tags
            genres = []
            if "toptags" in track_data and "tag" in track_data["toptags"]:
                tags = track_data["toptags"]["tag"]
                if isinstance(tags, list):
                    for tag in tags:
                        if "name" in tag:
                            genres.append(tag["name"])
            
            metadata = {
                "track_name": track_data.get("name", track_name),
                "artist_name": track_data.get("artist", {}).get("name", artist_name),
                "album": album_name,
                "album_art_url": album_art_url,
                "genre": genres[0] if genres else None,  # Take the most popular tag as genre
                "duration": int(track_data.get("duration", 0)),
                "source": "lastfm",
                "confidence": confidence
            }
            
            return metadata
            
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                logger.warning(f"Error fetching from Last.fm API: {e}")
                return None
            time.sleep(1)
    
    return None

def fetch_metadata_from_musicbrainz(track_name, artist_name):
    """Fetch metadata from MusicBrainz API"""
    try:
        # Search for the recording
        results = musicbrainzngs.search_recordings(
            query=f'recording:"{track_name}" AND artist:"{artist_name}"',
            limit=5
        )
        
        if not results or 'recording-list' not in results or not results['recording-list']:
            return None
        
        # Get the best match
        best_match = results['recording-list'][0]
        
        # Calculate confidence based on position and score
        confidence = 0.5  # Base confidence
        
        # Check if title matches
        if track_name.lower() in best_match['title'].lower() or best_match['title'].lower() in track_name.lower():
            confidence += 0.3
            
        # Check if artist matches
        artist_found = False
        if 'artist-credit' in best_match:
            for artist_credit in best_match['artist-credit']:
                if isinstance(artist_credit, dict) and 'artist' in artist_credit:
                    if artist_name.lower() in artist_credit['artist']['name'].lower() or \
                       artist_credit['artist']['name'].lower() in artist_name.lower():
                        artist_found = True
                        break
        
        if artist_found:
            confidence += 0.2
        else:
            confidence -= 0.3  # Penalize if artist doesn't match
        
        # Extract metadata
        metadata = {
            "track_name": best_match['title'],
            "artist_name": artist_name,  # Keep the original artist name
            "album": best_match.get('release-list', [{}])[0].get('title') if 'release-list' in best_match else None,
            "release_date": best_match.get('release-list', [{}])[0].get('date') if 'release-list' in best_match else None,
            "genre": None,  # MusicBrainz doesn't provide genre directly
            "isrc": best_match.get('isrc-list', [None])[0] if 'isrc-list' in best_match else None,
            "duration": best_match.get('length', None),
            "album_art_url": None,  # Need to fetch separately
            "source": "musicbrainz",
            "confidence": max(0.0, min(1.0, confidence))
        }
        
        return metadata
    except Exception as e:
        logger.error(f"MusicBrainz API error: {e}")
        return None

def merge_song_metadata(youtube_info, fetched_metadata):
    """Merge YouTube info with fetched metadata, prioritizing the most reliable source"""
    # Start with YouTube info as base
    merged = {
        "track_name": youtube_info.get("title", "Unknown"),
        "track_artist": youtube_info.get("uploader", "Unknown"),
        "album": None,
        "genre": None,
        "release_date": None,
        "isrc": None,
        "duration": youtube_info.get("duration", 0),
        "source": "youtube"
    }
    
    # Extract artist and title from YouTube if possible
    if " - " in merged["track_name"]:
        artist, title = merged["track_name"].split(" - ", 1)
        merged["track_artist"] = artist.strip()
        merged["track_name"] = title.strip()
    
    # If we have fetched metadata with good confidence, use it to override YouTube info
    if fetched_metadata and fetched_metadata.get("confidence", 0) > 0.5:
        # Override with fetched metadata, preserving YouTube data where fetched is None
        for key, value in fetched_metadata.items():
            if value is not None and key in merged:
                merged[key] = value
        
        # Special case for track_name and artist_name
        if "track_name" in fetched_metadata and fetched_metadata["track_name"]:
            merged["track_name"] = fetched_metadata["track_name"]
        if "artist_name" in fetched_metadata and fetched_metadata["artist_name"]:
            merged["track_artist"] = fetched_metadata["artist_name"]
    
    return merged 