import os
import re
import logging
from typing import Dict, Optional
from bs4 import BeautifulSoup
import requests
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import nltk
from ..core.config import settings
import urllib.parse

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

logger = logging.getLogger(__name__)

class LyricsService:
    def __init__(self):
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stop_words = set(stopwords.words('english'))
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
    def _clean_lyrics(self, lyrics: str) -> str:
        """Clean and preprocess lyrics text"""
        if not lyrics:
            return ""
            
        # Remove section headers like [Verse], [Chorus], etc.
        lyrics = re.sub(r'\[.*?\]', '', lyrics)
        
        # Remove empty lines and extra whitespace
        lyrics = '\n'.join(line.strip() for line in lyrics.split('\n') if line.strip())
        
        return lyrics
        
    def _tokenize_lyrics(self, text: str) -> list:
        """Tokenize lyrics into words"""
        if not text:
            return []
            
        # Convert to lowercase and tokenize
        tokens = self.tokenizer.tokenize(text.lower())
        
        # Remove stop words
        tokens = [token for token in tokens if token not in self.stop_words]
        
        return tokens
        
    def _analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of lyrics using VADER and TextBlob"""
        if not text:
            return {
                'compound': 0,
                'pos': 0,
                'neg': 0,
                'neu': 1,
                'polarity': 0,
                'subjectivity': 0.5
            }
            
        # VADER sentiment
        vader_scores = self.sentiment_analyzer.polarity_scores(text)
        
        # TextBlob sentiment
        blob = TextBlob(text)
        
        return {
            'compound': vader_scores['compound'],
            'pos': vader_scores['pos'],
            'neg': vader_scores['neg'],
            'neu': vader_scores['neu'],
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
        
    def _extract_topics(self, tokens: list, num_topics: int = 10) -> Dict:
        """Extract topics from lyrics using LDA"""
        if not tokens:
            return {'topics': [0] * num_topics}
            
        # Create dictionary and corpus
        dictionary = Dictionary([tokens])
        corpus = [dictionary.doc2bow(tokens)]
        
        # Train LDA model
        lda = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=10
        )
        
        # Get topic distribution
        topic_dist = [0] * num_topics
        for topic_id, prob in lda.get_document_topics(corpus[0]):
            topic_dist[topic_id] = prob
            
        return {'topics': topic_dist}
    
    def _scrape_lyrics(self, title: str, artist: str) -> Optional[str]:
        """Scrape lyrics from multiple sources"""
        # Try AZLyrics
        try:
            # Format artist and title for URL
            artist_url = re.sub(r'[^\w\s-]', '', artist.lower()).replace(' ', '')
            title_url = re.sub(r'[^\w\s-]', '', title.lower()).replace(' ', '')
            url = f'https://www.azlyrics.com/lyrics/{artist_url}/{title_url}.html'
            
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # AZLyrics keeps lyrics in a div without class/id, between comments
                lyrics_div = soup.find('div', {'class': None, 'id': None})
                if lyrics_div:
                    return lyrics_div.get_text().strip()
        except Exception as e:
            logger.warning(f"Failed to scrape AZLyrics: {str(e)}")
        
        # Try Genius web scraping as fallback
        try:
            search_url = f'https://genius.com/search?q={urllib.parse.quote(f"{title} {artist}")}'
            response = requests.get(search_url, headers=self.headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                song_link = soup.find('a', {'class': 'mini_card'})
                if song_link and 'href' in song_link.attrs:
                    lyrics_url = song_link['href']
                    response = requests.get(lyrics_url, headers=self.headers)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        lyrics_div = soup.find('div', {'class': 'Lyrics__Container-sc-1ynbvzw-6'})
                        if lyrics_div:
                            return lyrics_div.get_text().strip()
        except Exception as e:
            logger.warning(f"Failed to scrape Genius: {str(e)}")
        
        return None
        
    async def get_lyrics(self, title: str, artist: str) -> Optional[Dict]:
        """Get lyrics by web scraping and analyze them"""
        try:
            # Get lyrics through web scraping
            lyrics = self._scrape_lyrics(title, artist)
            if not lyrics:
                logger.warning(f"No lyrics found for {title} by {artist}")
                return None
                
            # Clean lyrics
            clean_lyrics = self._clean_lyrics(lyrics)
            
            # Tokenize
            tokens = self._tokenize_lyrics(clean_lyrics)
            
            # Analyze sentiment
            sentiment = self._analyze_sentiment(clean_lyrics)
            
            # Extract topics
            topics = self._extract_topics(tokens)
            
            return {
                'lyrics': lyrics,
                'clean_lyrics': clean_lyrics,
                'sentiment': sentiment,
                'topics': topics['topics']
            }
            
        except Exception as e:
            logger.error(f"Error getting lyrics for {title} by {artist}: {str(e)}")
            return None

# Create global instance
lyrics_service = LyricsService() 