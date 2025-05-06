# Lyrics fetching and sentiment analysis utilities
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import streamlit as st
from bs4 import BeautifulSoup
import re
import time
import logging
from urllib.parse import quote_plus

# Configure logging
logger = logging.getLogger(__name__)

def fetch_lyrics(track_name, artist_name, max_retries=3):
    """Fetch lyrics with multiple retries and fallback methods"""
    logger.info(f"Attempting to fetch lyrics for '{track_name}' by '{artist_name}'")
    
    # Try Genius API first
    lyrics, error = fetch_lyrics_from_genius(track_name, artist_name, max_retries)
    
    # If Genius API fails, try Google search fallback
    if not lyrics and error:
        logger.info(f"Genius API failed: {error}. Trying Google fallback...")
        lyrics, fallback_error = fetch_lyrics_via_google(track_name, artist_name)
        if lyrics:
            logger.info(f"Successfully found lyrics via Google fallback! Length: {len(lyrics)} chars")
            return lyrics, None
        
        # If both methods fail, return the original error
        logger.warning(f"All lyrics sources failed for '{track_name}' by '{artist_name}'. Genius error: {error}, Google error: {fallback_error}")
        return None, f"All lyrics sources failed. {error}"
    
    if lyrics:
        logger.info(f"Successfully found lyrics via Genius API! Length: {len(lyrics)} chars")
    
    return lyrics, error

def fetch_lyrics_from_genius(track_name, artist_name, max_retries=3):
    """Fetch lyrics from Genius API with retries"""
    logger.info(f"Attempting to fetch lyrics from Genius API for '{track_name}' by '{artist_name}'")
    
    api_key = st.secrets.get("genius_api_key", None)
    if not api_key:
        logger.error("Genius API key not set in Streamlit secrets.")
        return None, "Genius API key not set."
    
    base_url = "https://api.genius.com/search"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"q": f"{track_name} {artist_name}"}
    
    logger.debug(f"Searching Genius with query: {params['q']}")
    
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = requests.get(base_url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 429:  # Rate limited
                retry_count += 1
                logger.warning(f"Genius API rate limited. Retry {retry_count}/{max_retries} after {2 * retry_count}s delay")
                time.sleep(2 * retry_count)  # Exponential backoff
                continue
                
            if response.status_code != 200:
                logger.error(f"Genius API returned status code {response.status_code}")
                return None, f"Genius API returned status code {response.status_code}."
            
            # Debug: log the first part of the response
            response_json = response.json()
            logger.debug(f"Genius API search response meta: {response_json.get('meta', {})}")
            
            hits = response_json.get("response", {}).get("hits", [])
            logger.info(f"Found {len(hits)} potential matches on Genius")
            
            if not hits:
                logger.warning(f"No matches found on Genius for '{track_name}' by '{artist_name}'")
                return None, "No lyrics found via Genius API."
            
            # Log top hits for debugging
            for i, hit in enumerate(hits[:3]):
                result = hit["result"]
                logger.info(f"Hit #{i+1}: '{result.get('title')}' by '{result.get('primary_artist', {}).get('name')}'")
                
            # Try the top 3 results if available
            for hit in hits[:3]:
                song_info = hit["result"]
                url = song_info['url']
                
                # Match confidence check - skip if title/artist don't seem to match
                hit_title = song_info.get('title', '').lower()
                hit_artist = song_info.get('primary_artist', {}).get('name', '').lower()
                track_name_lower = track_name.lower()
                artist_name_lower = artist_name.lower()
                
                # Debug info about match
                logger.debug(f"Comparing titles: '{track_name_lower}' vs '{hit_title}'")
                logger.debug(f"Comparing artists: '{artist_name_lower}' vs '{hit_artist}'")
                
                # Skip if neither title nor artist matches
                if (track_name_lower not in hit_title and hit_title not in track_name_lower) and \
                   (artist_name_lower not in hit_artist and hit_artist not in artist_name_lower):
                    logger.debug(f"Skipping result due to low title/artist match confidence")
                    continue
                
                # Try to scrape lyrics from the Genius page
                logger.info(f"Attempting to scrape lyrics from {url}")
                lyrics, scrape_error = scrape_lyrics_from_genius(url, max_retries=2)
                if lyrics:
                    logger.info(f"Successfully scraped lyrics from Genius page! Length: {len(lyrics)} chars")
                    return lyrics, None
                else:
                    logger.warning(f"Failed to scrape lyrics from {url}: {scrape_error}")
            
            # If we get here, none of the top results worked
            logger.warning("None of the top results from Genius contained scrapable lyrics")
            return None, "Could not find matching lyrics on Genius."
            
        except Exception as e:
            retry_count += 1
            logger.error(f"Error fetching from Genius API (retry {retry_count}/{max_retries}): {e}")
            if retry_count >= max_retries:
                return None, f"Error fetching lyrics after {max_retries} retries: {e}"
            time.sleep(1)  # Wait before retrying
    
    return None, "Max retries exceeded while connecting to Genius API."

def scrape_lyrics_from_genius(url, max_retries=2):
    """Scrape lyrics from Genius page with improved parsing for different page structures"""
    logger.info(f"Attempting to scrape lyrics from Genius page: {url}")
    
    retry_count = 0
    while retry_count < max_retries:
        try:
            # Add a desktop user agent to avoid blocks
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            logger.debug(f"Requesting page (attempt {retry_count+1}/{max_retries})")
            page = requests.get(url, headers=headers, timeout=15)
            
            if page.status_code != 200:
                retry_count += 1
                logger.warning(f"Failed to fetch page with status {page.status_code}. Retrying...")
                time.sleep(1)
                continue
            
            logger.debug(f"Page fetched successfully, parsing HTML content")
            soup = BeautifulSoup(page.text, "html.parser")
            
            # Try multiple selectors for lyrics containers
            # 1. Modern Genius container
            lyrics_divs = soup.find_all("div", attrs={"data-lyrics-container": "true"})
            if lyrics_divs:
                logger.debug(f"Found {len(lyrics_divs)} modern lyrics containers")
                lyrics = "\n".join([div.get_text(separator="\n") for div in lyrics_divs])
                lyrics = re.sub(r'\n+', '\n', lyrics).strip()
                return clean_lyrics(lyrics), None
            else:
                logger.debug("No modern lyrics containers found")
            
            # 2. Legacy class
            lyrics_box = soup.find("div", class_="lyrics")
            if lyrics_box:
                logger.debug("Found legacy lyrics container")
                lyrics = lyrics_box.get_text(separator="\n").strip()
                return clean_lyrics(lyrics), None
            else:
                logger.debug("No legacy lyrics container found")
            
            # 3. Try another common container class
            lyrics_box = soup.find("div", class_="Lyrics__Container-sc-1ynbvzw-6")
            if lyrics_box:
                logger.debug("Found alternative lyrics container")
                lyrics = lyrics_box.get_text(separator="\n").strip()
                return clean_lyrics(lyrics), None
            else:
                logger.debug("No alternative lyrics container found")
                
            # 4. Desperate approach: look for any div with "lyrics" in class name
            lyric_candidates = []
            for div in soup.find_all("div"):
                class_attr = div.get("class", [])
                if class_attr and any("lyric" in c.lower() for c in class_attr):
                    lyric_candidates.append(div)
            
            if lyric_candidates:
                logger.debug(f"Found {len(lyric_candidates)} potential lyric containers via class name")
                for div in lyric_candidates:
                    lyrics = div.get_text(separator="\n").strip()
                    if len(lyrics) > 100:  # Assume it's valid if reasonably long
                        logger.debug(f"Found valid lyrics with length {len(lyrics)}")
                        return clean_lyrics(lyrics), None
            else:
                logger.debug("No lyrics containers found via class name search")
            
            # Debug info on page structure
            logger.debug("HTML structure analysis:")
            logger.debug(f"Page title: {soup.title.string if soup.title else 'No title'}")
            logger.debug(f"Total divs: {len(soup.find_all('div'))}")
            logger.debug(f"Text length: {len(page.text)}")
            logger.debug(f"Page might be protected or lyrics not present")
            
            retry_count += 1
            time.sleep(1)
        except Exception as e:
            retry_count += 1
            logger.error(f"Error scraping Genius page (retry {retry_count}/{max_retries}): {e}")
            if retry_count >= max_retries:
                return None, f"Error scraping Genius: {e}"
            time.sleep(1)
    
    return None, "Failed to extract lyrics from Genius page after multiple attempts."

def fetch_lyrics_via_google(track_name, artist_name):
    """Fallback method: Try to find lyrics via Google search"""
    logger.info(f"Attempting to find lyrics via Google search for '{track_name}' by '{artist_name}'")
    
    try:
        # Format the search query
        query = f"{artist_name} {track_name} lyrics genius"
        encoded_query = quote_plus(query)
        
        logger.debug(f"Google search query: {query}")
        
        # Use the Google search API
        search_url = f"https://www.google.com/search?q={encoded_query}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(search_url, headers=headers, timeout=10)
        if response.status_code != 200:
            logger.error(f"Google search failed with status code {response.status_code}")
            return None, f"Google search failed with status code {response.status_code}"
            
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Look for links to Genius
        genius_links = []
        for link in soup.find_all("a"):
            href = link.get("href", "")
            if "genius.com" in href and "/lyrics/" in href:
                # Extract the actual URL from Google's redirect URL
                match = re.search(r"(?:url\?q=)(https://genius.com[^&]+)", href)
                if match:
                    genius_url = match.group(1)
                    genius_links.append(genius_url)
        
        logger.info(f"Found {len(genius_links)} Genius links in Google search results")
        
        # Try each genius link
        for i, genius_url in enumerate(genius_links[:3]):  # Try top 3 at most
            logger.info(f"Trying Google result #{i+1}: {genius_url}")
            # Now try to scrape this Genius page
            lyrics, error = scrape_lyrics_from_genius(genius_url)
            if lyrics:
                logger.info(f"Successfully found lyrics via Google search! Length: {len(lyrics)} chars")
                return lyrics, None
        
        logger.warning(f"Could not find lyrics via Google search for '{track_name}' by '{artist_name}'")
        return None, "Could not find lyrics via Google search"
    except Exception as e:
        logger.error(f"Error in Google search fallback: {e}")
        return None, f"Error in Google search fallback: {e}"

def clean_lyrics(lyrics):
    """Clean up lyrics text by removing annotations and extra whitespace"""
    # Remove [Verse], [Chorus], etc.
    original_length = len(lyrics)
    lyrics = re.sub(r'\[.*?\]', '', lyrics)
    # Remove extra whitespace and newlines
    lyrics = re.sub(r'\s+', ' ', lyrics)
    lyrics = re.sub(r'\n\s*\n', '\n\n', lyrics)
    final_lyrics = lyrics.strip()
    
    logger.debug(f"Cleaned lyrics from {original_length} to {len(final_lyrics)} chars")
    return final_lyrics

def analyze_sentiment(lyrics):
    """Analyze sentiment of lyrics text"""
    logger.info(f"Analyzing sentiment of lyrics (length: {len(lyrics) if lyrics else 0} chars)")
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(lyrics)
    logger.debug(f"Sentiment results: {sentiment}")
    return sentiment

def fetch_lyrics_and_sentiment(track_name, artist_name):
    """Fetch lyrics and analyze sentiment, with detailed error reporting"""
    logger.info(f"=== Starting lyrics and sentiment fetch for '{track_name}' by '{artist_name}' ===")
    
    start_time = time.time()
    lyrics, error = fetch_lyrics(track_name, artist_name)
    
    sentiment = None
    if lyrics and not error:
        sentiment = analyze_sentiment(lyrics)
        
    elapsed_time = time.time() - start_time
    logger.info(f"=== Completed lyrics and sentiment fetch in {elapsed_time:.2f}s ===")
    
    if error:
        logger.warning(f"Failed to fetch lyrics: {error}")
    
    # Log detailed result stats
    result = {
        "lyrics": lyrics,
        "sentiment": sentiment,
        "error": error,
        "fetched": bool(lyrics),
        "fetch_time": elapsed_time
    }
    
    logger.info(f"Lyrics fetched: {bool(lyrics)}, Length: {len(lyrics) if lyrics else 0}, Error: {bool(error)}")
    
    return result 