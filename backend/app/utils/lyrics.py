import os
import logging
from typing import Optional, Dict
import lyricsgenius
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

class LyricsService:
    def __init__(self, genius_token: Optional[str] = None):
        """Initialize the Lyrics service with Genius API token.
        
        Args:
            genius_token (Optional[str]): Genius API token. If None, will try to get from environment
        """
        self.genius_token = genius_token or os.getenv('GENIUS_ACCESS_TOKEN')
        if not self.genius_token:
            logger.warning("No Genius API token provided. Lyrics functionality will be limited.")
            self.genius = None
        else:
            self.genius = lyricsgenius.Genius(
                self.genius_token,
                verbose=False,
                remove_section_headers=True,
                skip_non_songs=True
            )
        
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
    
    def get_lyrics(self, track_name: str, artist_name: str) -> Optional[Dict]:
        """Fetch lyrics and metadata for a song.
        
        Args:
            track_name (str): Name of the track
            artist_name (str): Name of the artist
            
        Returns:
            Optional[Dict]: Dictionary containing lyrics and metadata if found
        """
        if not self.genius:
            logger.error("Genius API client not initialized")
            return None
            
        try:
            # Search for the song
            song = self.genius.search_song(track_name, artist_name)
            if not song:
                logger.warning(f"No lyrics found for {track_name} by {artist_name}")
                return None
            
            # Get sentiment scores
            sentiment_scores = self.sentiment_analyzer.polarity_scores(song.lyrics)
            
            return {
                'lyrics': song.lyrics,
                'title': song.title,
                'artist': song.artist,
                'genius_url': song.url,
                'sentiment': {
                    'compound': sentiment_scores['compound'],
                    'positive': sentiment_scores['pos'],
                    'negative': sentiment_scores['neg'],
                    'neutral': sentiment_scores['neu']
                }
            }
            
        except Exception as e:
            logger.error(f"Error fetching lyrics for {track_name} by {artist_name}: {str(e)}")
            return None
    
    def analyze_lyrics(self, lyrics: str) -> Dict:
        """Analyze lyrics text and return sentiment scores.
        
        Args:
            lyrics (str): Raw lyrics text to analyze
            
        Returns:
            Dict: Dictionary containing sentiment scores
        """
        sentiment_scores = self.sentiment_analyzer.polarity_scores(lyrics)
        return {
            'compound': sentiment_scores['compound'],
            'positive': sentiment_scores['pos'],
            'negative': sentiment_scores['neg'],
            'neutral': sentiment_scores['neu']
        }

# Create a global instance
lyrics_service = LyricsService() 