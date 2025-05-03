# Lyrics fetching and sentiment analysis utilities
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import streamlit as st
from bs4 import BeautifulSoup
import re
import time
from urllib.parse import quote_plus

def fetch_lyrics(track_name, artist_name, max_retries=3):
    """Fetch lyrics with multiple retries and fallback methods"""
    # Try Genius API first
    lyrics, error = fetch_lyrics_from_genius(track_name, artist_name, max_retries)
    
    # If Genius API fails, try Google search fallback
    if not lyrics and error:
        lyrics, fallback_error = fetch_lyrics_via_google(track_name, artist_name)
        if lyrics:
            return lyrics, None
        
        # If both methods fail, return the original error
        return None, f"All lyrics sources failed. {error}"
    
    return lyrics, error

def fetch_lyrics_from_genius(track_name, artist_name, max_retries=3):
    """Fetch lyrics from Genius API with retries"""
    api_key = st.secrets.get("genius_api_key", None)
    if not api_key:
        return None, "Genius API key not set."
    
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
                return None, f"Genius API returned status code {response.status_code}."
                
            hits = response.json().get("response", {}).get("hits", [])
            if not hits:
                return None, "No lyrics found via Genius API."
                
            # Try the top 3 results if available
            for hit in hits[:3]:
                song_info = hit["result"]
                url = song_info['url']
                
                # Match confidence check - skip if title/artist don't seem to match
                hit_title = song_info.get('title', '').lower()
                hit_artist = song_info.get('primary_artist', {}).get('name', '').lower()
                track_name_lower = track_name.lower()
                artist_name_lower = artist_name.lower()
                
                # Skip if neither title nor artist matches
                if (track_name_lower not in hit_title and hit_title not in track_name_lower) and \
                   (artist_name_lower not in hit_artist and hit_artist not in artist_name_lower):
                    continue
                
                # Try to scrape lyrics from the Genius page
                lyrics, scrape_error = scrape_lyrics_from_genius(url, max_retries=2)
                if lyrics:
                    return lyrics, None
            
            # If we get here, none of the top results worked
            return None, "Could not find matching lyrics on Genius."
            
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                return None, f"Error fetching lyrics after {max_retries} retries: {e}"
            time.sleep(1)  # Wait before retrying
    
    return None, "Max retries exceeded while connecting to Genius API."

def scrape_lyrics_from_genius(url, max_retries=2):
    """Scrape lyrics from Genius page with improved parsing for different page structures"""
    retry_count = 0
    while retry_count < max_retries:
        try:
            # Add a desktop user agent to avoid blocks
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            page = requests.get(url, headers=headers, timeout=15)
            if page.status_code != 200:
                retry_count += 1
                time.sleep(1)
                continue
                
            soup = BeautifulSoup(page.text, "html.parser")
            
            # Try multiple selectors for lyrics containers
            # 1. Modern Genius container
            lyrics_divs = soup.find_all("div", attrs={"data-lyrics-container": "true"})
            if lyrics_divs:
                lyrics = "\n".join([div.get_text(separator="\n") for div in lyrics_divs])
                lyrics = re.sub(r'\n+', '\n', lyrics).strip()
                return clean_lyrics(lyrics), None
            
            # 2. Legacy class
            lyrics_box = soup.find("div", class_="lyrics")
            if lyrics_box:
                lyrics = lyrics_box.get_text(separator="\n").strip()
                return clean_lyrics(lyrics), None
            
            # 3. Try another common container class
            lyrics_box = soup.find("div", class_="Lyrics__Container-sc-1ynbvzw-6")
            if lyrics_box:
                lyrics = lyrics_box.get_text(separator="\n").strip()
                return clean_lyrics(lyrics), None
                
            # 4. Desperate approach: look for any div with "lyrics" in class name
            for div in soup.find_all("div"):
                class_attr = div.get("class", [])
                if class_attr and any("lyric" in c.lower() for c in class_attr):
                    lyrics = div.get_text(separator="\n").strip()
                    if len(lyrics) > 100:  # Assume it's valid if reasonably long
                        return clean_lyrics(lyrics), None
            
            retry_count += 1
            time.sleep(1)
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                return None, f"Error scraping Genius: {e}"
            time.sleep(1)
    
    return None, "Failed to extract lyrics from Genius page after multiple attempts."

def fetch_lyrics_via_google(track_name, artist_name):
    """Fallback method: Try to find lyrics via Google search"""
    try:
        # Format the search query
        query = f"{artist_name} {track_name} lyrics genius"
        encoded_query = quote_plus(query)
        
        # Use the Google search API
        search_url = f"https://www.google.com/search?q={encoded_query}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(search_url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None, f"Google search failed with status code {response.status_code}"
            
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Look for links to Genius
        for link in soup.find_all("a"):
            href = link.get("href", "")
            if "genius.com" in href and "/lyrics/" in href:
                # Extract the actual URL from Google's redirect URL
                match = re.search(r"(?:url\?q=)(https://genius.com[^&]+)", href)
                if match:
                    genius_url = match.group(1)
                    # Now try to scrape this Genius page
                    lyrics, error = scrape_lyrics_from_genius(genius_url)
                    if lyrics:
                        return lyrics, None
        
        return None, "Could not find lyrics via Google search"
    except Exception as e:
        return None, f"Error in Google search fallback: {e}"

def clean_lyrics(lyrics):
    """Clean up lyrics text by removing annotations and extra whitespace"""
    # Remove [Verse], [Chorus], etc.
    lyrics = re.sub(r'\[.*?\]', '', lyrics)
    # Remove extra whitespace and newlines
    lyrics = re.sub(r'\s+', ' ', lyrics)
    lyrics = re.sub(r'\n\s*\n', '\n\n', lyrics)
    return lyrics.strip()

def analyze_sentiment(lyrics):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(lyrics)

def fetch_lyrics_and_sentiment(track_name, artist_name):
    lyrics, error = fetch_lyrics(track_name, artist_name)
    sentiment = None
    if lyrics and not error:
        sentiment = analyze_sentiment(lyrics)
    return {"lyrics": lyrics, "sentiment": sentiment, "error": error} 