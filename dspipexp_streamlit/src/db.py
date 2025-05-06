import streamlit as st
from sqlalchemy import create_engine, text, pool
import json
import difflib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@st.cache_resource
def get_engine():
    """
    Create SQLAlchemy engine with connection pooling and retries.
    Uses QueuePool with optimized settings for cloud DB connection.
    """
    logger.info("Creating database engine")
    db_url = st.secrets["db_url"]
    
    # Configure connection pool with settings optimized for cloud DB
    engine = create_engine(
        db_url,
        poolclass=pool.QueuePool,
        pool_size=5,  # Start with 5 connections
        max_overflow=10,  # Allow up to 10 additional connections
        pool_timeout=30,  # Wait up to 30 seconds for a connection
        pool_recycle=1800,  # Recycle connections after 30 minutes
        pool_pre_ping=True,  # Test connections with a ping before using
        connect_args={"connect_timeout": 10}  # 10 second connection timeout
    )
    
    # Test the connection
    test_connection(engine)
    
    return engine

def test_connection(engine):
    """Test the database connection with retries"""
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1")).fetchone()
                logger.info("✅ Database connection successful")
                return True
        except Exception as e:
            logger.warning(f"❌ Database connection attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"⏳ Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error(f"❌ Failed to connect to database after {max_retries} attempts")
                st.error(f"Database connection failed. Please try again later.")
    return False

def execute_with_retry(engine, query_func, max_retries=3):
    """
    Execute a database query with retry logic
    
    :param engine: SQLAlchemy engine
    :param query_func: Function that takes a connection and executes a query
    :param max_retries: Maximum number of retry attempts
    :return: Query result or None on failure
    """
    retry_count = 0
    retry_delay = 1  # Start with 1 second delay
    
    while retry_count < max_retries:
        try:
            logger.info(f"Executing database query (attempt {retry_count+1}/{max_retries})")
            with engine.connect() as conn:
                # Begin a transaction that will be automatically rolled back if an exception occurs
                with conn.begin():
                    result = query_func(conn)
                    logger.info("✅ Query executed successfully")
                    return result
        except Exception as e:
            retry_count += 1
            logger.warning(f"❌ Database query failed (attempt {retry_count}/{max_retries}): {e}")
            
            if retry_count >= max_retries:
                logger.error(f"❌ Query failed after {max_retries} attempts: {e}")
                raise
            
            # Exponential backoff
            logger.info(f"⏳ Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2
    
    return None

def search_songs(engine, query, artist=None, genre=None, limit=20):
    logger.info(f"Searching songs with query: '{query}', artist: '{artist}', genre: '{genre}', limit: {limit}")
    sql = """
        SELECT id, track_name, track_artist, track_album_name, playlist_genre, audio_features
        FROM songs
        WHERE (track_name ILIKE :query OR track_artist ILIKE :query)
    """
    params = {'query': f'%{query}%'}
    if artist:
        sql += " AND track_artist ILIKE :artist"
        params['artist'] = f'%{artist}%'
    if genre:
        sql += " AND playlist_genre ILIKE :genre"
        params['genre'] = f'%{genre}%'
    sql += " LIMIT :limit"
    params['limit'] = limit
    
    def query_func(conn):
        result = conn.execute(text(sql), params).fetchall()
        logger.info(f"✅ Found {len(result)} songs matching query")
        return result
    
    return execute_with_retry(engine, query_func)

def insert_song(engine, song_data):
    # song_data: dict with keys matching DB columns
    logger.info(f"Inserting/updating song: '{song_data.get('track_name')}' by '{song_data.get('track_artist')}'")
    sql = text("""
        INSERT INTO songs
        (track_name, track_artist, track_album_name, playlist_genre, lyrics, audio_features, word2vec_features, sentiment_features, topic_features, youtube_url, is_original)
        VALUES (:track_name, :track_artist, :track_album_name, :playlist_genre, :lyrics, :audio_features, :word2vec_features, :sentiment_features, :topic_features, :youtube_url, :is_original)
        ON CONFLICT (track_name, track_artist) DO UPDATE SET
            track_album_name = EXCLUDED.track_album_name,
            playlist_genre = EXCLUDED.playlist_genre,
            lyrics = EXCLUDED.lyrics,
            audio_features = EXCLUDED.audio_features,
            word2vec_features = EXCLUDED.word2vec_features,
            sentiment_features = EXCLUDED.sentiment_features,
            topic_features = EXCLUDED.topic_features,
            youtube_url = EXCLUDED.youtube_url,
            is_original = EXCLUDED.is_original
        RETURNING id
    """)
    # Ensure all dicts are JSON strings for DB
    for key in ["audio_features", "word2vec_features", "sentiment_features", "topic_features"]:
        if song_data.get(key) is not None and not isinstance(song_data[key], str):
            song_data[key] = json.dumps(song_data[key])
    
    def query_func(conn):
        result = conn.execute(sql, song_data)
        song_id = result.scalar()
        logger.info(f"✅ Song saved with ID: {song_id}")
        return song_id
    
    return execute_with_retry(engine, query_func)

def normalize_title(title):
    # Remove common video suffixes and lowercase
    title = title.lower()
    title = re.sub(r'\(.*?\)', '', title)  # Remove anything in parentheses
    title = re.sub(r'\[.*?\]', '', title)  # Remove anything in brackets
    title = re.sub(r'[^a-z0-9 ]', '', title)  # Remove non-alphanumeric
    title = re.sub(r'\s+', ' ', title).strip()
    # Remove common suffixes
    for suffix in [
        'official audio', 'official video', 'music video', 'lyric video', 'lyrics', 'audio', 'video', 'hd', 'remastered', 'explicit', 'clean', 'visualizer', 'feat', 'ft', 'featuring'
    ]:
        title = title.replace(suffix, '')
    return title.strip()

def check_duplicate_song(engine, track_name, track_artist, lyrics=None, title_threshold=0.8, lyrics_threshold=0.8):
    logger.info(f"Checking for duplicates: '{track_name}' by '{track_artist}'")
    norm_title = normalize_title(track_name)
    sql = text("""
        SELECT id, track_name, track_artist, lyrics
        FROM songs
        WHERE track_artist ILIKE :artist
    """)
    
    def query_func(conn):
        rows = conn.execute(sql, {"artist": f"%{track_artist}%"}).fetchall()
        logger.info(f"Found {len(rows)} songs by same artist to check for duplicates")
        possible_duplicates = []
        for row in rows:
            db_id = row.id
            db_title = row.track_name
            db_artist = row.track_artist
            db_lyrics = row.lyrics
            norm_db_title = normalize_title(db_title)
            # Fuzzy title match
            title_sim = difflib.SequenceMatcher(None, norm_title, norm_db_title).ratio()
            lyrics_sim = None
            if lyrics and db_lyrics:
                try:
                    tfidf = TfidfVectorizer().fit([lyrics, db_lyrics])
                    tfidf_matrix = tfidf.transform([lyrics, db_lyrics])
                    lyrics_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0,0]
                except Exception as e:
                    logger.warning(f"Failed to compute lyrics similarity: {e}")
                    lyrics_sim = None
            if title_sim >= title_threshold or (lyrics_sim is not None and lyrics_sim >= lyrics_threshold):
                possible_duplicates.append({
                    "id": db_id,
                    "track_name": db_title,
                    "track_artist": db_artist,
                    "title_similarity": title_sim,
                    "lyrics_similarity": lyrics_sim
                })
        logger.info(f"✅ Found {len(possible_duplicates)} possible duplicates")
        return possible_duplicates
    
    return execute_with_retry(engine, query_func)

# TODO: Add functions for song search, insert, update, delete, etc. 