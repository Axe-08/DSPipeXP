# Update imports
import streamlit as st

# Set page config FIRST before any other Streamlit commands
st.set_page_config(
    page_title="DSPipeXP Music Recommendation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# All other imports
import pandas as pd
from src.db import get_engine, search_songs, insert_song, check_duplicate_song
from src.audio import extract_audio_features
from src.utils import (
    save_uploaded_file,
    get_youtube_thumbnail,
    get_album_cover,
    get_default_album_art,
    load_custom_css,
    create_radar_chart,
    format_lyrics,
    format_duration,
    safe_image_load,
    explain_recommendation
)
from src.config import setup_youtube_api_keys, get_next_youtube_api_key
from src.lyrics import fetch_lyrics_and_sentiment
from src.recommender import get_similar_songs, get_similar_songs_for_features
from sqlalchemy import text
import json
import yt_dlp
import time
import io
from PIL import Image
import requests
import os
import re
import logging

# Set up a logger that will show which methods are being used
logger = logging.getLogger("youtube_methods")
logger.setLevel(logging.INFO)

# Configure Streamlit logging to show our YouTube API method selection
# Create a handler that will display logs in the Streamlit UI
class StreamlitLogHandler(logging.Handler):
    def emit(self, record):
        message = self.format(record)
        if record.levelno >= logging.ERROR:
            st.error(message)
        elif record.levelno >= logging.WARNING:
            st.warning(message)
        elif record.levelno >= logging.INFO:
            # Don't display all info messages, just ones about YouTube methods
            if "YouTube" in message or "youtube" in message:
                with st.sidebar:
                    st.info(message)

# Add our custom handler to the logger
streamlit_handler = StreamlitLogHandler()
logger.addHandler(streamlit_handler)

# Set up YouTube API keys
setup_youtube_api_keys()

# Create a sidebar section to display YouTube API method info
with st.sidebar:
    st.subheader("YouTube API Methods")
    youtube_method_expander = st.expander("View API Method Usage", expanded=False)
    with youtube_method_expander:
        st.write("This section will show which YouTube API methods are being used")
        if 'youtube_api_keys' in st.session_state:
            key_count = len(st.session_state.youtube_api_keys)
            if key_count > 0:
                st.success(f"🎉 Using hybrid YouTube extraction with {key_count} API key(s)")
            else:
                st.info("Using hybrid YouTube extraction (no API keys found)")
        else:
            st.warning("YouTube API keys not configured")

# Try to import hybrid implementation first, fall back to original if not available
try:
    # Import hybrid implementations
    from src.youtube_hybrid import process_youtube_url_hybrid, youtube_search_hybrid, youtube_search_and_get_url_hybrid
    from src.youtube import update_song_youtube_url  # Still use original for this function
    
    # Create wrapped functions that automatically handle API key rotation
    def process_youtube_url(youtube_url, progress_callback=None):
        api_key = get_next_youtube_api_key()
        logger.info(f"Processing YouTube URL using API key rotation")
        return process_youtube_url_hybrid(youtube_url, progress_callback, api_key)
    
    def youtube_search(query, max_results=5):
        api_key = get_next_youtube_api_key()
        logger.info(f"Searching YouTube using API key rotation")
        return youtube_search_hybrid(query, max_results, api_key)
    
    def youtube_search_and_get_url(query):
        api_key = get_next_youtube_api_key()
        logger.info(f"Getting YouTube URL using API key rotation")
        return youtube_search_and_get_url_hybrid(query, api_key)
    
    with st.sidebar:
        st.success("Using enhanced YouTube extraction")

except ImportError:
    # Fall back to original implementation
    from src.youtube import process_youtube_url, youtube_search_and_get_url, update_song_youtube_url, youtube_search
    with st.sidebar:
        st.info("Using standard YouTube extraction. For better reliability, install: innertube, aiotube, google-api-python-client")
# --- Theme and Dark Mode Settings ---


def initialize_theme_settings():
    # Initialize theme settings in session state if not already present
    if 'theme' not in st.session_state:
        # Default to light mode
        st.session_state.theme = "light"

    # Custom CSS for dark/light mode
    if st.session_state.theme == "dark":
        dark_mode_css = """
        /* Dark Mode CSS */
        [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background-color: #121212;
            color: #f1f1f1;
        }

        .song-card {
            background-color: #1e1e1e;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
        }

        .stTabs [data-baseweb="tab-list"] {
            background-color: #1e1e1e;
            border-radius: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            color: #ccc;
        }

        .stTabs [aria-selected="true"] {
            background-color: #2e2e2e !important;
            color: #fff !important;
        }

        .stButton button {
            background-color: #333;
            color: white;
            border: 1px solid #444;
        }

        .stButton button:hover {
            background-color: #444;
            border-color: #555;
        }

        .stTextInput input, .stTextArea textarea, .stNumberInput input {
            background-color: #252525;
            color: #eee;
            border-color: #444;
        }

        .stSelectbox [data-baseweb="select"] {
            background-color: #252525;
        }

        .stSelectbox [data-baseweb="select"] > div {
            background-color: #252525;
            color: #eee;
        }

        .similarity-score {
            color: #4CAF50;
            font-size: 0.85em;
            margin-left: 8px;
        }

        /* About Section Styling */
        .about-section {
            background-color: #1e1e1e;
            border-left: 1px solid #333;
            padding: 20px;
            height: 100vh;
            position: fixed;
            right: 0;
            top: 0;
            width: 300px;
            box-shadow: -2px 0 5px rgba(0,0,0,0.3);
            z-index: 1000;
            overflow-y: auto;
        }

        .close-btn {
            position: absolute;
            top: 10px;
            right: 10px;
            cursor: pointer;
            color: #ccc;
            font-size: 24px;
        }
        """
        st.markdown(f"<style>{dark_mode_css}</style>", unsafe_allow_html=True)
    else:
        light_mode_css = """
        /* Light Mode CSS */
        .song-card {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease-in-out;
        }

        .song-card:hover {
            transform: translateY(-3px);
        }

        .stTabs [data-baseweb="tab-list"] {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 5px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 5px;
            font-weight: 500;
        }

        .stTabs [aria-selected="true"] {
            background-color: #e6f3ff !important;
            color: #1e88e5 !important;
        }

        .stButton button {
            border-radius: 6px;
            font-weight: 500;
        }

        .similarity-score {
            color: #2e7d32;
            font-size: 0.85em;
            margin-left: 8px;
        }

        .album-cover {
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        }

        /* About Section Styling */
        .about-section {
            background-color: #ffffff;
            border-left: 1px solid #e0e0e0;
            padding: 20px;
            height: 100vh;
            position: fixed;
            right: 0;
            top: 0;
            width: 300px;
            box-shadow: -2px 0 5px rgba(0,0,0,0.1);
            z-index: 1000;
            overflow-y: auto;
        }

        .close-btn {
            position: absolute;
            top: 10px;
            right: 10px;
            cursor: pointer;
            color: #666;
            font-size: 24px;
        }

        /* Header styling */
        .header-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding: 10px 0;
        }

        .header-title {
            display: flex;
            align-items: center;
        }

        .header-actions {
            display: flex;
            gap: 10px;
        }
        """
        st.markdown(f"<style>{light_mode_css}</style>", unsafe_allow_html=True)


def toggle_theme():
    """Toggle between light and dark mode"""
    if st.session_state.theme == "light":
        st.session_state.theme = "dark"
    else:
        st.session_state.theme = "light"


def show_about_section():
    """Display the About Us section"""
    st.markdown(
        """
        <div class="about-section">
            <div class="close-btn" onclick="document.querySelector('.about-section').style.display='none';">×</div>
            <h2>About DSPipeXP</h2>
            <p>DSPipeXP Music Recommendation is a state-of-the-art platform that uses advanced audio processing and machine learning to help you discover music you'll love.</p>
            
            <h3>Our Technology</h3>
            <p>We analyze songs using:</p>
            <ul>
                <li>Audio feature extraction</li>
                <li>Natural language processing for lyrics analysis</li>
                <li>Sentiment analysis</li>
                <li>Hybrid recommendation algorithms</li>
            </ul>
            
            <h3>The Team</h3>
            <p>DSPipeXP was created by a team of three data science students:</p>
            <ul>
                <li><strong>Akshit S Bansal</strong> - Lead Developer</li>
                <li><strong>Kriti Chaturvedi</strong> - Data Scientist</li>
                <li><strong>Hussain Haidary</strong> - Machine Learning Engineer</li>
            </ul>
            
            <h3>Learn More</h3>
            <p>For more information about our project:</p>
            <ul>
                <li><a href="https://github.com/Heisenberg-Vader/DSPipeXP" target="_blank">GitHub Repository</a></li>
                <li><a href="https://medium.com/@heisenberg-vader/dspipexp" target="_blank">Medium Article</a></li>
            </ul>
            
            <h3>Our Mission</h3>
            <p>To create a personalized music discovery experience by combining the science of sound with the art of musical taste, offering recommendations based on audio features, lyrics content, and emotional tones.</p>
        </div>
        
        <script>
            // JavaScript to handle the close button
            const closeBtn = document.querySelector('.close-btn');
            if (closeBtn) {
                closeBtn.addEventListener('click', function() {
                    document.querySelector('.about-section').style.display = 'none';
                });
            }
        </script>
        """, 
        unsafe_allow_html=True
    )


# Set up page configuration
st.set_page_config(
    page_title="DSPipeXP Music Recommendation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize theme settings
initialize_theme_settings()

# Load custom CSS for better styling
load_custom_css()

# --- Helper Functions ---


def show_song_card(song, similarity=None, with_recommendations=False):
    """Display a song in a modern card layout with album art"""
    import json

    audio_features = song.get('audio_features', {})
    if isinstance(audio_features, str):
        try:
            audio_features = json.loads(audio_features)
        except BaseException:
            audio_features = {}

    sentiment = song.get('sentiment_features', None)
    if isinstance(sentiment, str):
        try:
            sentiment = json.loads(sentiment)
        except BaseException:
            sentiment = None

    # Card container
    with st.container():
        st.markdown('<div class="song-card">', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 3])
        img_url = get_youtube_thumbnail(
            song['youtube_url']) if song.get('youtube_url') else None
        img_url = img_url or get_album_cover(
            song.get(
                'track_name', ''), song.get(
                'track_artist', ''))

        with col1:
            safe_image_load(img_url, use_class="album-cover")

        with col2:
            # Title and artist
            if similarity is not None:
                # Add color-coded similarity score based on value
                score_color = "#ff0000"  # Red for low scores
                if similarity >= 0.7:
                    score_color = "#00aa00"  # Green for high scores
                elif similarity >= 0.4:
                    score_color = "#aaaa00"  # Yellow for medium scores

                st.markdown(f"""
                    ### {song.get('track_name', 'Unknown Track')}
                    <span class="similarity-score" style="color: {score_color};">{similarity:.2f} match</span>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"### {song.get('track_name', 'Unknown Track')}")
            st.markdown(
                f"**Artist:** {song.get('track_artist', 'Unknown Artist')}")

            # Metadata
            album = song.get('track_album_name', 'Unknown Album')
            genre = song.get('playlist_genre', 'Unknown')
            duration = format_duration(song.get('duration_ms', 0))

            st.markdown(f"**Album:** {album}")
            st.markdown(f"**Genre:** {genre}")
            st.markdown(f"**Duration:** {duration}")

            # Action buttons (inline instead of nested)
            if song.get('youtube_url'):
                st.link_button("▶️ Play on YouTube", song['youtube_url'])

            if not with_recommendations and 'id' in song:
                # Initialize the state if not present
                rec_key = f'show_recs_for_{song["id"]}'
                if rec_key not in st.session_state:
                    st.session_state[rec_key] = False

                if st.button("🔍 Get Recommendations",
                             key=f"rec_btn_{song['id']}"):
                    st.session_state[rec_key] = True

            if audio_features:
                # Initialize visualization state if not present
                viz_key = f"show_viz_for_{song.get('id', 'temp')}"
                if viz_key not in st.session_state:
                    st.session_state[viz_key] = False

                if st.button("📊 Audio Profile",
                             key=f"audio_viz_{song.get('id', 'temp')}"):
                    st.session_state[viz_key] = True

        # Expandable Lyrics - only use expander when not in a recommendation
        if song.get('lyrics'):
            if with_recommendations:
                # For recommendations, use a simple toggle button instead of
                # expander
                lyrics_key = f"show_lyrics_{song.get('id', 'temp')}"
                if lyrics_key not in st.session_state:
                    st.session_state[lyrics_key] = False

                if st.button("📝 View Lyrics", key=f"lyrics_btn_{song.get('id', 'temp')}"):
                    st.session_state[lyrics_key] = not st.session_state.get(lyrics_key, False)

                # Show lyrics if toggled on
                if st.session_state.get(lyrics_key, False):
                    st.markdown(format_lyrics(song['lyrics']), unsafe_allow_html=True)
            else:
                # For top-level songs, use expander as before
                with st.expander("📝 View Lyrics"):
                    st.markdown(
                        format_lyrics(
                            song['lyrics']),
                        unsafe_allow_html=True)

        # Audio Visualization
        viz_key = f'show_viz_for_{song.get("id", "temp")}'
        if st.session_state.get(viz_key):
            # Use a dedicated section instead of expander
            st.markdown("### Audio Feature Visualization")
            st.plotly_chart(
                create_radar_chart(audio_features),
                use_container_width=True)
            st.write("""
                This radar chart shows the audio characteristics of the song. Higher values mean stronger presence of that trait:
                - **Danceability**: How suitable the track is for dancing
                - **Energy**: Intensity and activity level
                - **Acousticness**: How acoustic (vs. electronic) the song is
                - **Instrumentalness**: Likelihood of having no vocals
                - **Liveness**: Presence of audience/live performance elements
                - **Speechiness**: Presence of spoken words
                - **Valence**: Musical positiveness/happiness
                """)
            if st.button("Hide Visualization",
                         key=f"hide_viz_{song.get('id', 'temp')}"):
                    st.session_state[viz_key] = False

        # Recommendations
        rec_key = f'show_recs_for_{song.get("id")}'
        if st.session_state.get(rec_key) and not with_recommendations:
            # Use a dedicated section instead of expander
            st.markdown("### Similar Songs")
            with st.container():  # Use container to prevent nesting issues
                with st.spinner("Finding similar songs..."):
                    show_recommendations_for_song(song.get('id'), k=5)
            if st.button("Hide Recommendations",
                         key=f"hide_recs_{song.get('id')}"):
                    st.session_state[rec_key] = False

        # Explanation for recommendation
        if similarity is not None and all(
                k in song for k in ['audio_score', 'lyrics_score', 'sentiment_score']):
            # Show explanation with button toggle when in a recommendation card
            explain_key = f"show_explain_{song.get('id', 'temp')}"
            if explain_key not in st.session_state:
                st.session_state[explain_key] = False

            if st.button(
    "ℹ️ Why this recommendation?",
    key=f"explain_btn_{
        song.get(
            'id',
             'temp')}"):
                st.session_state[explain_key] = not st.session_state.get(
                    explain_key, False)

            # Show explanation if toggled on
            if st.session_state.get(explain_key, False):
                with st.container():
                    # Detailed similarity visualization
                    components = [
                        {"name": "Sound Profile",
                         "score": song['audio_score'],
                            "weight": 0.7},
                        {"name": "Lyrics",
                         "score": song['lyrics_score'],
                            "weight": 0.2},
                        {"name": "Emotional Tone",
                         "score": song['sentiment_score'],
                            "weight": 0.1}
                    ]

                    # Display detailed scores
                    st.markdown("#### Similarity Components")
                    for comp in components:
                        # Calculate weighted score for context
                        weighted = comp["score"] * comp["weight"]
                        # Color-code based on component score
                        color = "#ff0000"  # Red for low scores
                        if comp["score"] >= 0.7:
                            color = "#00aa00"  # Green for high scores
                        elif comp["score"] >= 0.4:
                            color = "#aaaa00"  # Yellow for medium scores

                        st.markdown(f"""
                        **{comp['name']}**: <span style="color:{color};">{comp['score']:.2f}</span>
                        (contributes {weighted:.2f} to total score)
                        """, unsafe_allow_html=True)

                    # Explanation text
                st.info(explain_recommendation(
                    song['audio_score'],
                    song['lyrics_score'],
                    song['sentiment_score']
                ))

        st.markdown('</div>', unsafe_allow_html=True)


def show_recommendations_for_song(song_id, k=5):
    """Show recommendations for a song with detailed component scores"""
    engine = get_engine()

    # Initialize variables to avoid UnboundLocalError
    recs = []
    loading_placeholder = st.empty()

    # First, get song name for context - define this outside other blocks
    song_name = "this song"
    try:
        with engine.connect() as conn:
            song_row = conn.execute(
                text("SELECT track_name, track_artist FROM songs WHERE id = :id"), 
                {"id": song_id}).fetchone()
            if song_row:
                song_name = f"'{song_row.track_name}' by {song_row.track_artist}"
    except Exception as e:
        print(f"Error getting song name: {e}")
        # Just use the default name

    # Add a caching key to prevent duplicate processing
    cache_key = f"song_recs_{song_id}_{k}"

    # Add a processing flag to prevent double loading
    processing_key = f"processing_recs_{song_id}_{k}"

    # Add keys for progressive refinement
    quality_key = f"quality_recs_{song_id}_{k}"
    progress_key = f"progress_refinement_{song_id}_{k}"

    # Initialize processing flag if not present
    if processing_key not in st.session_state:
        st.session_state[processing_key] = False

    # Initialize progressive refinement flag
    if progress_key not in st.session_state:
        st.session_state[progress_key] = {
            "refining": False,
            "iteration": 0,
            "threshold": MIN_COMBINED_SIM if 'MIN_COMBINED_SIM' in globals() else 0.15,
            "started_at": None,
            "last_update": None
        }

    # Check if we already have cached recommendations
    if cache_key in st.session_state and st.session_state[cache_key]:
        recs = st.session_state[cache_key]

        # Check if we need to start background refinement
        if not st.session_state[progress_key]["refining"]:
            # Start progressive refinement in background
            st.session_state[progress_key]["refining"] = True
            st.session_state[progress_key]["started_at"] = time.time()
            st.session_state[progress_key]["iteration"] = 1

            # Create a placeholder for refinement status
            if "refinement_status" not in st.session_state:
                st.session_state.refinement_status = st.empty()

            # Show initial recommendations immediately while starting
            # background refinement
            _start_progressive_refinement(song_id, k, engine)

    # Only process if not already processing
    elif not st.session_state[processing_key]:
        # Set processing flag to prevent double execution
        st.session_state[processing_key] = True

        # Add a loading message with timer
        loading_placeholder = st.empty()

        start_time = time.time()
        loading_message_shown = False

        # Get recommendations using the hybrid engine
        with st.spinner("Finding similar songs..."):
            # Show loading warning if it takes more than 3 seconds
            while True:
                if time.time() - start_time > 3 and not loading_message_shown:
                    loading_placeholder.warning(
                        "Recommendations are taking longer than usual. Please wait a moment...")
                    loading_message_shown = True

                # Try to get recommendations
                try:
                    recs = get_similar_songs(engine, song_id, k=k, 
                                           audio_weight=0.7, lyrics_weight=0.2, sentiment_weight=0.1)

                    # Cache the recommendations
                    st.session_state[cache_key] = recs

                    # Set quality metric for first batch
                    if quality_key not in st.session_state:
                        if recs and len(recs) > 0:
                            # Use average similarity as quality metric
                            avg_sim = sum(rec[3] for rec in recs) / len(recs)
                            st.session_state[quality_key] = avg_sim
                        else:
                            st.session_state[quality_key] = 0.0

                    # Reset processing flag
                    st.session_state[processing_key] = False
                    break  # Exit the loop if successful
                except Exception as e:
                    # Error handling - wait briefly and try again
                    print(f"Error retrieving recommendation: {e}")
                    time.sleep(0.5)
                    if time.time() - start_time > 15:  # Timeout after 15 seconds
                        st.error(f"Error retrieving recommendation: {e}")
                        st.session_state[processing_key] = False
                        break

    # Clear the loading message
    loading_placeholder.empty()
    
    # Start progressive refinement in background
    st.session_state[progress_key]["refining"] = True
    st.session_state[progress_key]["started_at"] = time.time()
    st.session_state[progress_key]["iteration"] = 1
    _start_progressive_refinement(song_id, k, engine)

    if not recs:
        st.warning(
            f"No recommendations found that are similar enough to {song_name}.")
        st.info("This could be because the song has unique characteristics or there aren't enough similar songs in the database.")
        return
    
    # Show refinement status if in progress
    if st.session_state[progress_key]["refining"]:
        # Calculate time since last update
        last_update = st.session_state[progress_key].get("last_update")
        if last_update:
            time_since_update = time.time() - last_update
            if time_since_update > 10:  # If it's been more than 10 seconds
                refinement_status = st.container()
                with refinement_status:
                    st.info(f"Progressive refinement in progress (iteration {st.session_state[progress_key]['iteration']}). If better results are found, they will replace these automatically.")

    st.markdown(f"### Similar Songs You Might Enjoy")
    st.markdown(f"Songs similar to {song_name}, sorted by similarity:")

    # Use a status container to show loading progress for each recommendation
    status_container = st.empty()

    for i, rec in enumerate(recs):
        status_container.text(
            f"Loading recommendation {
                i +
                1} of {
                len(recs)}...")

        # Check if we have the component scores
        if len(rec) >= 7:  # New format with component scores
            rec_id, rec_name, rec_artist, sim, audio_score, lyrics_score, sentiment_score = rec
            has_component_scores = True
        else:  # Old format without component scores
            rec_id, rec_name, rec_artist, sim = rec
            has_component_scores = False
            # We'll set default scores later if needed

        try:
            # Fetch full song details
            with engine.connect() as conn:
                row = conn.execute(
                    text("SELECT * FROM songs WHERE id = :id"), {"id": rec_id}).fetchone()

            if row:
                song = dict(row._mapping)
                
                # Fetch YouTube URL if missing
                youtube_url = song.get('youtube_url')
                if not youtube_url:
                    query = f"{rec_name} {rec_artist} official audio"
                    youtube_url = youtube_search_and_get_url(query)
                    if youtube_url:
                        update_song_youtube_url(engine, rec_id, youtube_url)
                    song['youtube_url'] = youtube_url
                
                # Fetch lyrics if missing
                lyrics = song.get('lyrics')
                sentiment = song.get('sentiment_features')
                if not lyrics:
                    lyrics_result = fetch_lyrics_and_sentiment(rec_name, rec_artist)
                    lyrics = lyrics_result.get('lyrics')
                    sentiment = lyrics_result.get('sentiment')
                    song['lyrics'] = lyrics
                    song['sentiment_features'] = sentiment
                
                # Add component scores from recommendation engine or create reasonable estimates
                if has_component_scores:
                    song['audio_score'] = audio_score
                    song['lyrics_score'] = lyrics_score
                    song['sentiment_score'] = sentiment_score
                else:
                    # Fallback - simulate scores based on overall similarity
                    song['audio_score'] = sim * 0.8 + 0.1
                    song['lyrics_score'] = sim * 0.7 + 0.2
                    song['sentiment_score'] = sim * 0.6 + 0.3
                
                # Show recommendation card
                show_song_card(song, similarity=sim, with_recommendations=True)
            else:
                st.warning(f"Could not fetch details for {rec_name} by {rec_artist}")
        except Exception as e:
            st.error(f"Error loading recommendation: {e}")

    # Clear the status when done
    status_container.empty()

def _start_progressive_refinement(song_id, k, engine):
    """
    Start progressive refinement of recommendations in the background
    This will incrementally increase thresholds to find higher quality matches
    """
    # Use session state keys
    cache_key = f"song_recs_{song_id}_{k}"
    quality_key = f"quality_recs_{song_id}_{k}"
    progress_key = f"progress_refinement_{song_id}_{k}"
    
    # Get current thresholds from recommender module
    try:
        from src.recommender import MIN_AUDIO_SIM, MIN_COMBINED_SIM
        initial_audio_sim = MIN_AUDIO_SIM
        initial_combined_sim = MIN_COMBINED_SIM
    except ImportError:
        # Default values if can't import
        initial_audio_sim = 0.15
        initial_combined_sim = 0.15
    
    # Configure number of refinement iterations
    max_iterations = 3
    
    # Ensure the progress key exists in session state with proper values
    if progress_key not in st.session_state:
        st.session_state[progress_key] = {
            "refining": True,
            "iteration": 1,
            "threshold": initial_combined_sim,
            "started_at": time.time(),
            "last_update": time.time(),
            "quality": 0.0
        }
    
    # Get current iteration from session state
    current_iteration = st.session_state[progress_key]["iteration"]
    
    if current_iteration > max_iterations:
        # We've completed all refinement iterations
        st.session_state[progress_key]["refining"] = False
        return
    
    # Threshold increases for each iteration
    audio_threshold_increase = 0.05 * current_iteration
    combined_threshold_increase = 0.05 * current_iteration
    
    # Adjusted thresholds for this iteration
    adjusted_audio_sim = initial_audio_sim + audio_threshold_increase
    adjusted_combined_sim = initial_combined_sim + combined_threshold_increase
    
    # Update the threshold in session state for this iteration
    st.session_state[progress_key]["threshold"] = adjusted_combined_sim
    
    try:
        # Create a background task for refinement
        import threading
        import warnings
        
        def refinement_task():
            # Suppress missing ScriptRunContext warnings in background thread
            warnings.filterwarnings('ignore', message='.*missing ScriptRunContext.*')
            
            try:
                # Check if progress_key exists in session state before proceeding
                if progress_key not in st.session_state:
                    print(f"Warning: progress_key {progress_key} not found in session state, initializing it")
                    # Initialize the progress key with default values
                    st.session_state[progress_key] = {
                        "refining": True,
                        "iteration": 1,
                        "threshold": adjusted_combined_sim,
                        "started_at": time.time(),
                        "last_update": time.time(),
                        "quality": 0.0
                    }
                
                # Set temporary thresholds in recommender module
                import src.recommender
                original_audio_sim = src.recommender.MIN_AUDIO_SIM
                original_combined_sim = src.recommender.MIN_COMBINED_SIM
                
                # Apply stricter thresholds
                src.recommender.MIN_AUDIO_SIM = adjusted_audio_sim
                src.recommender.MIN_COMBINED_SIM = adjusted_combined_sim
                
                # Get refined recommendations with higher thresholds
                refined_recs = src.recommender.get_similar_songs(
                    engine, 
                    song_id, 
                    k=k,
                    audio_weight=0.7, 
                    lyrics_weight=0.2, 
                    sentiment_weight=0.1
                )
                
                # Restore original thresholds
                src.recommender.MIN_AUDIO_SIM = original_audio_sim
                src.recommender.MIN_COMBINED_SIM = original_combined_sim
                
                # Check if the new recommendations are better
                if refined_recs and len(refined_recs) > 0:
                    # Calculate average similarity as quality metric
                    avg_sim = sum(rec[3] for rec in refined_recs) / len(refined_recs)
                    current_quality = st.session_state.get(quality_key, 0.0)
                    
                    # If new recommendations are better, replace the old ones
                    if avg_sim > current_quality and len(refined_recs) >= k * 0.7:  # At least 70% of requested count
                        st.session_state[cache_key] = refined_recs
                        st.session_state[quality_key] = avg_sim
                        
                        # Update refinement status - Add check to make sure key exists
                        if progress_key in st.session_state:
                            st.session_state[progress_key]["last_update"] = time.time()
                            st.session_state[progress_key]["threshold"] = adjusted_combined_sim
                            st.session_state[progress_key]["quality"] = avg_sim
                
                # Schedule next iteration - Only if progress_key exists
                if progress_key in st.session_state:
                    st.session_state[progress_key]["iteration"] += 1
                
                    # Wait before starting next iteration
                    time.sleep(5)  # Wait 5 seconds between iterations
                    
                    # Continue refinement if not at max iterations
                    if st.session_state[progress_key]["iteration"] <= max_iterations:
                        _start_progressive_refinement(song_id, k, engine)
                    else:
                        st.session_state[progress_key]["refining"] = False
                else:
                    print(f"Warning: progress_key {progress_key} not found in session state, stopping refinement")
                    
            except Exception as e:
                print(f"Error in progressive refinement: {e}")
                # Stop refinement on error
                if progress_key in st.session_state:
                    st.session_state[progress_key]["refining"] = False
        
        # Start background thread
        refinement_thread = threading.Thread(target=refinement_task)
        refinement_thread.daemon = True
        refinement_thread.start()
    
    except Exception as e:
        print(f"Error starting progressive refinement: {e}")
        st.session_state[progress_key]["refining"] = False

# --- YouTube Cookies Helper ---


def youtube_cookies_sidebar():
    st.session_state.setdefault('show_cookies_sidebar', True)
    st.session_state.setdefault('youtube_cookies_file', None)
    if st.session_state['show_cookies_sidebar']:
        with st.sidebar:
            st.markdown("### Help Us Improve YouTube Access")
            st.markdown("""
            To provide the best experience and ensure reliable access to all YouTube songs (including those that may be restricted or require login), you can optionally upload your YouTube cookies.
            
            Your cookies are used only for this session and are never stored or shared.
            """)
            st.markdown(
                "[Learn how to export cookies](https://github.com/yt-dlp/yt-dlp/wiki/How-to-provide-your-own-cookies-to-yt-dlp)")
            cookies_file = st.file_uploader(
                "Upload YouTube cookies.txt (optional)",
                type=["txt"],
                key="sidebar_cookies")
            if cookies_file:
                st.session_state['youtube_cookies_file'] = cookies_file
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Skip", key="sidebar_no_cookies"):
                    st.session_state['show_cookies_sidebar'] = False
            with col2:
                if st.button("Continue", key="sidebar_yes_cookies"):
                    st.session_state['show_cookies_sidebar'] = False

# --- Header with Dark Mode Toggle and About Us Button ---


def render_header():
    # Initialize session state for About section
    if 'show_about' not in st.session_state:
        st.session_state.show_about = False

    # Custom HTML/CSS for header
    st.markdown("""
    <div class="header-container">
        <div class="header-title">
            <h1>🎵Music Recommendation</h1>
        </div>
        <div class="header-actions" id="header-actions">
            <!-- Placeholder for buttons (will be added via Streamlit) -->
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Add the buttons via Streamlit
    col1, col2 = st.columns([6, 1])
    
    with col2:
        # Row for buttons
        btn_col1, btn_col2 = st.columns(2)
        
        with btn_col1:
            # Dark mode toggle
            icon = "🌙" if st.session_state.theme == "light" else "☀️"
            if st.button(icon, help="Toggle dark/light mode"):
                toggle_theme()
                st.rerun()
        
        with btn_col2:
            # About us button
            if st.button("ℹ️", help="About Us"):
                st.session_state.show_about = not st.session_state.show_about
                st.rerun()
    
    # Show About section if active
    if st.session_state.show_about:
        show_about_section()


# Show sidebar for cookies if needed
if 'show_cookies_sidebar' not in st.session_state:
    st.session_state['show_cookies_sidebar'] = True

if st.session_state['show_cookies_sidebar']:
    youtube_cookies_sidebar()

# Render header with dark mode toggle and about us button
render_header()

st.write("Search for songs, get recommendations, or upload your own music!")

# Create centered tabs with better styling
tab1, tab2, tab3 = st.tabs(["🔍 Search", "⬆️ Upload", "🎥 YouTube"])


# --- Search Tab Implementation ---
def render_search_tab(tab):
    with tab:
        st.header("Song Search & Discovery")
        
        # Search inputs with better styling
        search_col1, search_col2 = st.columns([3, 1])
        
        with search_col1:
            search_query = st.text_input(
                "Enter a song or artist name", 
                placeholder="Try 'Bohemian Rhapsody' or 'Taylor Swift'",
                help="Search by song name, artist, or a combination of both"
            )
        
        with search_col2:
            search_limit = st.selectbox(
                "Results", 
                options=[5, 10, 20, 50],
                index=1,
                help="Maximum number of search results to show"
            )
        
        # Database connection
        engine = get_engine()
        
        if search_query:
            with st.spinner("Searching database..."):
                results = search_songs(
                    engine, search_query, limit=search_limit)
                
            if results:
                st.success(
                    f"Found {
                        len(results)} songs matching '{search_query}'")
                
                # Create a clean results display - either grid or list view
                view_type = st.radio(
                    "View results as:", 
                    options=["Detailed List", "Grid View"],
                    horizontal=True,
                    help="Choose how to display search results"
                )
                
                if view_type == "Grid View":
                    # Create a grid of cards (3 per row)
                    rows = [results[i:i + 3]
                            for i in range(0, len(results), 3)]
                    
                    for row in rows:
                        cols = st.columns(3)
                        for i, result in enumerate(row):
                            with cols[i]:
                                # Fetch full song details
                                song_id = result[0]
                                with engine.connect() as conn:
                                    song_row = conn.execute(
                                        text("SELECT * FROM songs WHERE id = :id"), 
                                        {"id": song_id}
                                    ).fetchone()
                                
                                if song_row:
                                    song = dict(song_row._mapping)
                                    
                                    # Mini card for grid view
                                    st.markdown(f"### {song['track_name']}")
                                    st.markdown(
                                        f"**Artist:** {song['track_artist']}")
                                    
                                    # Try to get thumbnail
                                    img_url = None
                                    if song.get('youtube_url'):
                                        img_url = get_youtube_thumbnail(
                                            song['youtube_url'])
                                    if not img_url:
                                        img_url = get_album_cover(
                                            song['track_name'], song['track_artist'])
                                    
                                    if img_url:
                                        try:
                                            st.image(
                                                img_url, use_container_width=True)
                                        except BaseException:
                                            pass
                                    
                                    # Show details button
                                    detail_btn_key = f"detail_grid_{song_id}"
                                    if st.button("View Details",
                                                 key=detail_btn_key):
                                        st.session_state['selected_song_id'] = song_id
                                        # No rerun needed - the UI will update on next render
                
                else:  # Detailed List
                    for result in results:
                        song_id = result[0]
                        # Fetch full song details
                        with engine.connect() as conn:
                            song_row = conn.execute(
                                text("SELECT * FROM songs WHERE id = :id"), 
                                {"id": song_id}
                            ).fetchone()
                        
                        if song_row:
                            song = dict(song_row._mapping)
                            
                            # If song has no lyrics, try to fetch them
                            if not song.get('lyrics'):
                                with st.spinner(f"Fetching lyrics for {song['track_name']}..."):
                                    lyrics_result = fetch_lyrics_and_sentiment(
                                        song['track_name'], 
                                        song['track_artist']
                                    )
                                    song['lyrics'] = lyrics_result.get(
                                        'lyrics')
                                    song['sentiment_features'] = lyrics_result.get(
                                        'sentiment')
                            
                            # Display song card
                            show_song_card(song)
                
                # Show selected song details if any
                if 'selected_song_id' in st.session_state:
                    selected_id = st.session_state['selected_song_id']
                    # Fetch full song details
                    with engine.connect() as conn:
                        song_row = conn.execute(
                            text("SELECT * FROM songs WHERE id = :id"), 
                            {"id": selected_id}
                        ).fetchone()
                    
                    if song_row:
                        song = dict(song_row._mapping)
                        st.subheader("Selected Song Details")
                        show_song_card(song)
                        
                        # Number of recommendations slider
                        num_recs = st.slider(
                            "Number of recommendations", 
                            min_value=3, 
                            max_value=15, 
                            value=5,
                            help="How many similar songs to find"
                        )
                        
                        # Get recommendations
                        if st.button("Get Recommendations",
                                     key=f"main_recs_{selected_id}"):
                            with st.spinner("Finding similar songs..."):
                                show_recommendations_for_song(
                                    selected_id, k=num_recs)
                        
                        # Clear selection button
                        if st.button("Back to Search Results"):
                            del st.session_state['selected_song_id']
                            # No rerun needed - the UI will update on next render
            
            else:
                st.info(f"No songs found matching '{search_query}'")
                
                # "Search YouTube Instead" option
                if st.button("Search YouTube Instead"):
                    st.session_state['youtube_search_query'] = search_query
                    st.session_state['show_youtube_tab'] = True
                    # This rerun is needed to switch tabs
                    st.rerun()
                
                # Suggest related searches
                st.markdown("**Try these popular searches:**")
                suggestion_cols = st.columns(4)
                suggestions = ["Ed Sheeran", "Billie Eilish", "Drake", "BTS"]
                
                for i, suggestion in enumerate(suggestions):
                    with suggestion_cols[i]:
                        if st.button(suggestion):
                            # Set search query without rerun
                            st.session_state['search_query'] = suggestion
                            # No rerun needed - the UI will update on next render


# Display search tab
render_search_tab(tab1)

# --- YouTube Tab Implementation ---


def render_youtube_tab(tab):
    with tab:
        st.header("Find Songs from YouTube")
        
        # Option to enter YouTube URL directly or search YouTube
        yt_method = st.radio(
            "How would you like to find a song?",
            options=["Enter YouTube URL", "Search YouTube"],
            horizontal=True
        )
        
        if yt_method == "Enter YouTube URL":
            youtube_url = st.text_input(
                "Enter YouTube URL", 
                placeholder="https://www.youtube.com/watch?v=...",
                help="Paste a YouTube URL of a song to process"
            )
            
            if youtube_url:
                process_youtube_url_with_ui(youtube_url)
                
        else:  # Search YouTube
            # Check if there's a search query from the search tab
            if 'youtube_search_query' in st.session_state:
                default_query = st.session_state['youtube_search_query']
                # Clear it after use
                del st.session_state['youtube_search_query']
            else:
                default_query = ""
                
            youtube_search_query = st.text_input(
                "Search YouTube", 
                value=default_query,
                placeholder="Enter song name, artist, or both",
                help="Search YouTube for songs to process"
            )
            
            max_results = st.slider(
                "Maximum results", 
                min_value=3, 
                max_value=10, 
                value=5,
                help="Maximum number of YouTube search results to show"
            )
            
            if youtube_search_query:
                search_youtube_with_ui(youtube_search_query, max_results)


def process_youtube_url_with_ui(youtube_url):
    """Process a YouTube URL with improved UI feedback"""
    # Initialize session state variables if not present
    if 'yt_processed_url' not in st.session_state:
        st.session_state.yt_processed_url = None
    if 'yt_features' not in st.session_state:
        st.session_state.yt_features = None
    if 'yt_info_dict' not in st.session_state:
        st.session_state.yt_info_dict = None
    if 'yt_lyrics' not in st.session_state:
        st.session_state.yt_lyrics = ''
    if 'yt_sentiment' not in st.session_state:
        st.session_state.yt_sentiment = None
    if 'yt_possible_duplicates' not in st.session_state:
        st.session_state.yt_possible_duplicates = []
    if 'yt_save_clicked' not in st.session_state:
        st.session_state.yt_save_clicked = False
    if 'yt_recommend_clicked' not in st.session_state:
        st.session_state.yt_recommend_clicked = False
    if 'yt_saved_song_id' not in st.session_state:
        st.session_state.yt_saved_song_id = None
    if 'yt_metadata' not in st.session_state:
        st.session_state.yt_metadata = {}
    if 'yt_processing_state' not in st.session_state:
        st.session_state.yt_processing_state = None
        
    # First check if we've already processed this URL to avoid redundant work
    if st.session_state.yt_processed_url != youtube_url:
        # Set up progress tracking containers
        progress_container = st.empty()
        progress_bar = progress_container.progress(0)
        status_container = st.empty()
        status_text = status_container.text("Initializing...")
        eta_container = st.empty()
        eta_text = eta_container.text("")
        
        def update_progress(progress, speed=None, message=None):
            """Update the progress UI elements"""
            try:
                progress_bar.progress(float(progress))
                if message:
                    status_text.text(message)
                if speed and speed > 0:
                    eta_text.text(f"Speed: {speed:.1f} KB/s")
                
                # Store progress in session state for resilience
                st.session_state.yt_processing_state = {
                    "progress": progress,
                    "message": message,
                    "speed": speed
                }
            except:
                # Handle any UI update errors gracefully
                pass
        
        # Reset the session state
        st.session_state.yt_processed_url = youtube_url
        st.session_state.yt_features = None
        st.session_state.yt_info_dict = None
        st.session_state.yt_lyrics = ''
        st.session_state.yt_sentiment = None
        st.session_state.yt_possible_duplicates = []
        st.session_state.yt_save_clicked = False
        st.session_state.yt_recommend_clicked = False
        st.session_state.yt_saved_song_id = None
        st.session_state.yt_metadata = {}

        # Process the new URL
        with st.spinner("Processing YouTube URL..."):
            # Step 1: Process YouTube URL with progress tracking
            update_progress(0.05, message="Starting YouTube processing...")
            
            try:
                features_result, error = process_youtube_url(youtube_url, progress_callback=update_progress)
                
                if error:
                    # Check for specific error messages to provide better user guidance
                    if "Video unavailable" in error:
                        status_container.error(f"This YouTube video is unavailable. It may have been removed, set to private, or doesn't exist. Please try another video URL.")
                        progress_container.empty()
                        eta_container.empty()
                        return
                    else:
                        status_text.warning(error)
                
                if not features_result:
                    progress_container.empty()
                    status_container.error("Failed to extract audio features. Please try a different YouTube URL.")
                    eta_container.empty()
                    return
            except Exception as e:
                progress_container.empty()
                status_container.error(f"Error processing YouTube URL: {str(e)}")
                eta_container.empty()
                return
            
            # Extract the audio features and video info
            if isinstance(features_result, dict):
                if 'audio_features' in features_result:
                    features = features_result['audio_features']
                    video_info = features_result.get('video_info', {})
                    st.session_state.yt_info_dict = video_info
                else:
                    # Backward compatibility
                    features = features_result
                    video_info = {}
            else:
                features = features_result
                video_info = {}
            
            st.session_state.yt_features = features
            
            # Step 2: Extract and verify metadata
            update_progress(0.70, message="Extracting metadata...")
            
            # Get the title and artist information
            title = video_info.get('title', 'Unknown title')
            uploader = video_info.get('uploader', 'Unknown uploader')
            
            # Try to get parsed song/artist from video description first
            track_name = video_info.get('parsed_song', None)
            track_artist = video_info.get('parsed_artist', None)
            
            # If not found in description, extract from title
            if not track_name or not track_artist:
                if ' - ' in title:
                    parsed_artist, parsed_title = title.split(' - ', 1)
                    track_artist = track_artist or parsed_artist.strip()
                    track_name = track_name or parsed_title.strip()
                else:
                    track_name = track_name or title
                    track_artist = track_artist or uploader
            
            # Step 3: Fetch enhanced metadata from MusicBrainz
            update_progress(0.75, message="Fetching additional song metadata...")
            
            try:
                from src.metadata import fetch_song_metadata, merge_song_metadata
                
                # Fetch detailed metadata
                enhanced_metadata = fetch_song_metadata(track_name, track_artist)
                
                # Merge with YouTube info
                merged_metadata = merge_song_metadata(video_info, enhanced_metadata)
                st.session_state.yt_metadata = merged_metadata
                
                # Use the enhanced metadata
                track_name = merged_metadata.get('track_name', track_name)
                track_artist = merged_metadata.get('track_artist', track_artist)
            except Exception as e:
                logger.warning(f"Error fetching enhanced metadata: {e}")
                # Continue with YouTube-derived metadata
                st.session_state.yt_metadata = {
                    'track_name': track_name,
                    'track_artist': track_artist,
                    'album': None,
                    'source': 'youtube'
                }
            
            # Step 4: Fetch lyrics
            update_progress(0.85, message="Fetching lyrics...")
        
            lyrics_result = fetch_lyrics_and_sentiment(track_name, track_artist)
            st.session_state.yt_lyrics = lyrics_result.get('lyrics', '')
            st.session_state.yt_sentiment = lyrics_result.get('sentiment', None)
        
            # Step 5: Check for duplicates
            update_progress(0.95, message="Checking for duplicate songs...")
        
            engine = get_engine()
            st.session_state.yt_possible_duplicates = check_duplicate_song(engine, track_name, track_artist, st.session_state.yt_lyrics)
            
            # Complete!
            update_progress(1.0, message="Processing complete!")
            time.sleep(0.5)  # Brief pause so user can see completion
            
            # Clear progress indicators
            progress_container.empty()
            status_container.empty()
            eta_container.empty()
    else:
        # If we've already processed this URL, use the cached values
        features = st.session_state.yt_features
        video_info = st.session_state.yt_info_dict
        
        if not features:
            st.error("Failed to process the YouTube URL. Please try again.")
            return

    # Show video info with thumbnail
    st.subheader("YouTube Video Information")

    # Two-column layout
    col1, col2 = st.columns([1, 2])

    with col1:
        # Video thumbnail
        thumbnail_url = video_info.get('thumbnail') if video_info else None
        if not thumbnail_url:
            thumbnail_url = get_youtube_thumbnail(youtube_url)
        
        if thumbnail_url:
            st.image(thumbnail_url, use_container_width=True)
        else:
            st.markdown("No thumbnail available")

    with col2:
        # Use metadata if available, otherwise fall back to video_info
        metadata = st.session_state.yt_metadata
        track_name = metadata.get('track_name', 'Unknown')
        track_artist = metadata.get('track_artist', 'Unknown')
        album = metadata.get('album', 'Unknown')
        
        # Display enhanced video info
        if video_info:
            st.markdown(f"**Title:** {video_info.get('title', 'Unknown')}")
            st.markdown(f"**Channel:** {video_info.get('uploader', 'Unknown')}")
            st.markdown(f"**Duration:** {video_info.get('duration', 0)} seconds")
            st.markdown(f"**Views:** {video_info.get('view_count', 'Unknown')}")
            
            # Show upload date in a nicer format if available
            upload_date = video_info.get('upload_date', '')
            if upload_date and len(upload_date) == 8:
                try:
                    formatted_date = f"{upload_date[0:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
                    st.markdown(f"**Upload Date:** {formatted_date}")
                except:
                    pass
        
        # Show song metadata
        st.subheader("Detected Song Information")
        
        # Show metadata source if available
        if metadata and metadata.get('source') and metadata['source'] != 'youtube':
            st.info(f"Enhanced metadata provided by {metadata['source']}")

    # Editable metadata with form - use callback handlers instead of form
    # submission
    with st.form("youtube_metadata_form", clear_on_submit=False):
        edited_track_name = st.text_input(
            "Track Name", value=track_name, key="yt_track_name")
        edited_track_artist = st.text_input(
            "Artist", value=track_artist, key="yt_track_artist")
        edited_album = st.text_input(
            "Album", value=metadata.get('album', ''), key="yt_album")
        edited_genre = st.text_input(
            "Genre", value=metadata.get('genre', ''), key="yt_genre")
        edited_lyrics = st.text_area(
            "Lyrics",
            value=st.session_state.yt_lyrics,
            height=200,
            key="yt_lyrics")
            
            # Audio feature visualization
    with st.expander("🔊 Audio Feature Visualization"):
            st.plotly_chart(create_radar_chart(features),use_container_width=True)
                
            # Form submission buttons
            col1, col2 = st.columns(2)
            
            with col1:
                save_button = st.form_submit_button(
                    "Save to Database & Get Recommendations")
                if save_button:
                    st.session_state.yt_save_clicked = True
            
            with col2:
                recommendations_only = st.form_submit_button(
                    "Get Recommendations Only (Don't Save)")
                if recommendations_only:
                    st.session_state.yt_recommend_clicked = True

    # Handle form submissions through session state
    engine = get_engine()

    # Save to database and show recommendations
    if st.session_state.yt_save_clicked:
        st.session_state.yt_save_clicked = False  # Reset for next time

            # Prepare song data
        song_data = {
                "track_name": edited_track_name,
                "track_artist": edited_track_artist,
                "track_album_name": edited_album,
                "playlist_genre": edited_genre,
                "lyrics": edited_lyrics,
                "audio_features": features,
            "sentiment_features": st.session_state.yt_sentiment,
                "word2vec_features": None,  # These would be computed separately
                "topic_features": None,     # These would be computed separately
                "youtube_url": youtube_url,
                "is_original": False
        }
            
            # Save to database
        with st.spinner("Saving song to database..."):
            try:
                song_id = insert_song(engine, song_data)
                st.session_state.yt_saved_song_id = song_id
                st.success(f"Song saved with ID {song_id}")
            
            # Now get recommendations
                st.subheader("Recommendations")
                with st.spinner("Finding similar songs..."):
                    show_recommendations_for_song(song_id, k=5)
            except Exception as e:
                st.error(f"Error saving song to database: {e}")
                
            # Just get recommendations without saving
    elif st.session_state.yt_recommend_clicked:
        st.session_state.yt_recommend_clicked = False  # Reset for next time

        st.subheader("Recommendations")
            
            # Use the features directly (not from DB)
        with st.spinner("Finding similar songs..."):
            from src.recommender import get_similar_songs_for_features

            # Add a unique session state key to prevent duplicate processing
            rec_key = f"yt_recs_{hash(str(features))}"

            # Check if we already have recommendations cached
            if rec_key in st.session_state and st.session_state[rec_key]:
                # Use cached recommendations
                recs = st.session_state[rec_key]
                st.success(
                    f"Found {
                        len(recs)} songs with similar sound profile")
            else:
                # Check if we have lyrics and sentiment data to use
                use_lyrics = bool(edited_lyrics and edited_lyrics.strip())
                use_sentiment = bool(st.session_state.yt_sentiment)

                if use_lyrics:
                    st.info(
                        "Using audio features, lyrics, and sentiment for better matching with enhanced YouTube-specific similarity calculations")
                    recs = get_similar_songs_for_features(
                    engine, 
                    features, 
                    k=5,
                    audio_weight=0.7, 
                    lyrics_weight=0.2, 
                    sentiment_weight=0.1, 
                    lyrics=edited_lyrics,
                        sentiment=st.session_state.yt_sentiment
                    )
                else:
                    st.info(
                        "Couldn't find lyrics. Using audio-only matching with lower threshold.")
                    # Note: We're using the lower threshold we set in
                    # recommender.py (0.15)
                    recs = get_similar_songs_for_features(engine,features,k=5,audio_weight=1.0,lyrics_weight=0.0,sentiment_weight=0.0)  

                # Cache the recommendations
                st.session_state[rec_key] = recs

                if not recs:
                    st.warning("No similar songs found")
                else:
                    st.success(
                        f"Found {
                            len(recs)} songs with similar sound profile")

            if recs:
                # Display loading progress - use one single progress area
                progress_text = st.empty()

                # Process each recommendation
                for i, rec in enumerate(recs):
                    progress_text.text(
                        f"Loading recommendation {
                            i +
                            1} of {
                            len(recs)}...")
                    
                    # Handle both old and new recommendation formats
                    if len(rec) >= 7:  # New format with component scores
                        rec_id, rec_name, rec_artist, sim, audio_score, lyrics_score, sentiment_score = rec
                    else:  # Old format for backward compatibility
                        rec_id, rec_name, rec_artist, sim = rec
                        audio_score = sim
                        lyrics_score = 0.0
                        sentiment_score = 0.0

                    try:
                    # Fetch full song details
                        with engine.connect() as conn:
                            row = conn.execute(
                                text("SELECT * FROM songs WHERE id = :id"), {"id": rec_id}).fetchone()
                    
                        if row:
                            song = dict(row._mapping)

                            # Set component scores from the actual returned values
                            song['audio_score'] = audio_score
                            song['lyrics_score'] = lyrics_score
                            song['sentiment_score'] = sentiment_score

                        # Show recommendation card
                            show_song_card(
                                song, similarity=sim, with_recommendations=False)
                        else:
                            st.warning(
                                f"Could not fetch details for {rec_name} by {rec_artist}")
                    except Exception as e:
                        st.error(f"Error retrieving recommendation: {e}")

                # Clear progress indicator
                progress_text.empty()
        
        # Display duplicates if any were found
    if hasattr(st.session_state,
               'yt_possible_duplicates') and st.session_state.yt_possible_duplicates:
        with st.expander("Possible Duplicate Songs", expanded=False):
            st.markdown("We found songs that might be duplicates of this one:")
            
            for i, dup in enumerate(st.session_state.yt_possible_duplicates):
                with st.container():
                    st.markdown(f"### Possible Match #{i + 1}")
                    
                    # Metrics for similarity scores
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        title_sim = dup.get('title_similarity', 0)
                        st.metric("Title Similarity", f"{title_sim:.2f}")
                        
                    with col2:
                        lyrics_sim = dup.get('lyrics_similarity')
                        if lyrics_sim is not None:
                            st.metric("Lyrics Similarity", f"{lyrics_sim:.2f}")
                        else:
                            st.metric("Lyrics Similarity", "N/A")
                    
                    # Fetch the duplicate song details
                    try:
                        with engine.connect() as conn:
                            row = conn.execute(
                                text("SELECT * FROM songs WHERE id = :id"), {"id": dup['id']}).fetchone()
                    
                        if row:
                            song = dict(row._mapping)
                        # Show song card
                            show_song_card(song)
                    
                    # View recommendations button
                            dup_rec_key = f"dup_recs_{dup['id']}"
                            if dup_rec_key not in st.session_state:
                                st.session_state[dup_rec_key] = False

                            if st.button(
                                    f"View Recommendations for This Song", key=f"dup_rec_btn_{dup['id']}"):
                                st.session_state[dup_rec_key] = True

                            if st.session_state[dup_rec_key]:
                                with st.spinner("Finding similar songs..."):
                                    show_recommendations_for_song(
                                        dup['id'], k=5)

                                if st.button(f"Hide Recommendations",
                                             key=f"hide_dup_recs_{dup['id']}"):
                                    st.session_state[dup_rec_key] = False
                    except Exception as e:
                        st.error(f"Error loading duplicate song: {e}")


def search_youtube_with_ui(query, max_results=5):
    """Search YouTube with improved UI"""
    with st.spinner(f"Searching YouTube for '{query}'..."):
        results = youtube_search(query, max_results=max_results)
        
    if not results:
        st.warning("No YouTube results found. Try a different search query.")
        return
    
    st.success(f"Found {len(results)} videos on YouTube")
    
    # Display results in a grid - 2 videos per row
    rows = [results[i:i + 2] for i in range(0, len(results), 2)]
    
    for row_idx, row in enumerate(rows):
        cols = st.columns(2)
        
        for col_idx, video in enumerate(row):
            with cols[col_idx]:
                # Video container with styling
                st.markdown('<div class="song-card">', unsafe_allow_html=True)
                
                # Video thumbnail
                thumbnail_url = get_youtube_thumbnail(video['url'])
                if thumbnail_url:
                    st.image(thumbnail_url, use_container_width=True)
                
                # Video details
                st.markdown(f"### {video['title']}")
                st.markdown(f"**Channel:** {video['channel']}")
                
                # Duration and video link
                video_info_cols = st.columns(2)
                with video_info_cols[0]:
                    duration = video.get('duration', 0)
                    if duration:
                        minutes = int(duration / 60)
                        seconds = int(duration % 60)
                        st.markdown(f"**Duration:** {minutes}:{seconds:02d}")
                    else:
                        st.markdown("**Duration:** Unknown")
                
                with video_info_cols[1]:
                    st.markdown(f"[Watch on YouTube]({video['url']})")
                
                # Process buttons
                action_cols = st.columns(2)

                with action_cols[0]:
                    process_btn_key = f"process_yt_{row_idx}_{col_idx}"
                    if st.button("Process This Song", key=process_btn_key):
                        # Store the selection in session state without forcing a rerun
                        st.session_state['selected_youtube_url'] = video['url']
                        # Don't rerun here - the main loop will detect this change

                with action_cols[1]:
                    recommend_btn_key = f"recommend_yt_{row_idx}_{col_idx}"
                    if st.button("Get Recommendations", key=recommend_btn_key):
                        # Use session state to keep track of which video to
                        # recommend
                        st.session_state[f'recommend_for_url_{row_idx}_{col_idx}'] = video['url']

                # Show recommendations if the button was clicked
                if f'recommend_for_url_{row_idx}_{col_idx}' in st.session_state:
                    url = st.session_state[f'recommend_for_url_{row_idx}_{col_idx}']
                    with st.spinner("Processing video to find recommendations..."):
                        # Process the video without saving to get features
                        features_result, error = process_youtube_url(url)
                        if error:
                            st.error(f"Error processing video: {error}")
                        elif not features_result:
                            st.error("Failed to extract audio features")
                        else:
                            # Extract actual features from the result structure
                            # New structure has metadata separated from audio
                            # features
                            if isinstance(
                                    features_result, dict) and 'audio_features' in features_result:
                                features = features_result['audio_features']
                                source = features_result.get(
                                    'source', 'unknown')
                                st.info(
                                    f"Processing audio from source: {source}")
                            else:
                                # Backward compatibility for old format
                                features = features_result

                            # Get recommendations based on audio features only
                            # (faster)
                            st.markdown("### Similar Songs")
                            engine = get_engine()
                            from src.recommender import get_similar_songs_for_features

                            # Use a try-except block to handle any issues
                            try:
                                with st.spinner("Finding similar songs..."):
                                    # Try to get title and artist for lyrics
                                    video_title = video.get('title', '')
                                    video_artist = video.get('channel', '')

                                    # Try to extract artist from title if in
                                    # format "Artist - Song"
                                    if ' - ' in video_title:
                                        parts = video_title.split(' - ', 1)
                                        video_artist = parts[0].strip()
                                        video_title = parts[1].strip()

                                    # Try to fetch lyrics
                                    lyrics_data = None
                                    with st.spinner("Fetching lyrics..."):
                                        try:
                                            lyrics_data = fetch_lyrics_and_sentiment(
                                                video_title, video_artist)
                                            if lyrics_data and lyrics_data.get(
                                                    'lyrics'):
                                                st.success(
                                                    f"Found lyrics for '{video_title}' by {video_artist}")
                                            else:
                                                st.warning(
                                                    "Couldn't find lyrics. Using audio-only matching with lower threshold.")
                                        except Exception as e:
                                            st.warning(
                                                f"Error fetching lyrics: {e}")

                                    # Determine if we can use lyrics
                                    if lyrics_data and lyrics_data.get(
                                            'lyrics'):
                                        # Get recommendations with hybrid
                                        # approach
                                        st.info(
                                            "Using audio features, lyrics, and sentiment for better matching with enhanced YouTube-specific similarity calculations")
                                        recs = get_similar_songs_for_features(
                                            engine,
                                            features,
                                            k=5,
                                            audio_weight=0.7,
                                            lyrics_weight=0.2,
                                            sentiment_weight=0.1,
                                            lyrics=lyrics_data.get(
                                                'lyrics', ''),
                                            sentiment=lyrics_data.get(
                                                'sentiment')
                                        )
                                    else:
                                        # No lyrics - use audio only with lower
                                        # threshold
                                        st.info(
                                            "Using audio features only with enhanced YouTube-specific similarity measures and reduced threshold (0.15)")
                                        # Note: We're using the lower threshold
                                        # we set in recommender.py (0.15)
                                        recs = get_similar_songs_for_features(
                                            engine,
                                            features,
                                            k=5,
                                            audio_weight=1.0,
                                            lyrics_weight=0.0,
                                            sentiment_weight=0.0
                                        )

                                    if not recs:
                                        st.warning(
                                            "No songs found that are similar enough to this video")
                                        st.info(
                                            "This could be because the audio has unique characteristics or there aren't enough similar songs in our database.")
                                    else:
                                        st.success(
                                            f"Found {
                                                len(recs)} songs with similar sound profile")

                                        # Display loading progress
                                        progress_text = st.empty()

                                        # Process each recommendation
                                        for i, rec in enumerate(recs):
                                            progress_text.text(
                                                f"Loading recommendation {
                                                    i +
                                                    1} of {
                                                    len(recs)}...")
                                            
                                            # Handle both old and new recommendation formats
                                            if len(rec) >= 7:  # New format with component scores
                                                rec_id, rec_name, rec_artist, sim, audio_score, lyrics_score, sentiment_score = rec
                                            else:  # Old format for backward compatibility
                                                rec_id, rec_name, rec_artist, sim = rec
                                                audio_score = sim
                                                lyrics_score = 0.0
                                                sentiment_score = 0.0

                                            try:
                                                # Fetch full song details
                                                with engine.connect() as conn:
                                                    row = conn.execute(
                                                        text("SELECT * FROM songs WHERE id = :id"), {"id": rec_id}).fetchone()

                                                if row:
                                                    song = dict(row._mapping)

                                                    # Set component scores from the actual returned values
                                                    song['audio_score'] = audio_score
                                                    song['lyrics_score'] = lyrics_score
                                                    song['sentiment_score'] = sentiment_score

                                                    # Show recommendation card
                                                    show_song_card(
                                                        song, similarity=sim, with_recommendations=False)
                                                else:
                                                    st.warning(
                                                        f"Could not fetch details for {rec_name} by {rec_artist}")
                                            except Exception as e:
                                                st.error(
                                                    f"Error retrieving recommendation: {e}")

                                        # Clear progress indicator
                                        progress_text.empty()

                                        # If no recommendations were found that
                                        # meet threshold
                                        if len(recs) == 0:
                                            st.warning(
                                                "No songs were similar enough to recommend")
                            except Exception as e:
                                st.error(f"Error finding recommendations: {e}")
                                st.info(
                                    "Please try again or select a different video with a clearer audio track")
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Process the selected video if any
    if 'selected_youtube_url' in st.session_state:
        url = st.session_state['selected_youtube_url']
        
        # Clear the selection
        del st.session_state['selected_youtube_url']
        
        # Add a divider
        st.markdown("---")
        st.subheader("Processing Selected Video")
        
        # Process the URL
        process_youtube_url_with_ui(url)


# Display YouTube tab
render_youtube_tab(tab3)

# --- Upload Tab Implementation ---


def render_upload_tab(tab):
    with tab:
        st.header("Upload Your Own Music")
        
        # Introduction
        st.markdown("""
        Upload your own audio files to:
        - Extract audio features
        - Find similar songs in our database
        - Add your songs to the database (optional)
        
        Supported formats: MP3, WAV, FLAC, OGG
        """)
        
        # File uploader with clear prompt
        uploaded_file = st.file_uploader(
            "Upload an audio file", 
            type=["mp3", "wav", "flac", "ogg"],
            help="Upload a song file to analyze and find similar music"
        )
        
        if uploaded_file:
            # Show processing steps with meaningful feedback
            st.subheader("Processing Audio")
            
            # Set up progress tracking
            progress = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Save the file
            status_text.markdown("⏳ Saving uploaded file...")
            progress.progress(10)
            
            file_path = save_uploaded_file(uploaded_file)
            
            # Step 2: Extract audio features
            status_text.markdown("⏳ Analyzing audio features...")
            progress.progress(30)
            
            with st.spinner("Extracting audio features..."):
                try:
                    features = extract_audio_features(file_path)
                    progress.progress(60)
                except Exception as e:
                    st.error(f"Failed to extract audio features: {e}")
                    return
            
            # Step 3: Show analysis results
            status_text.markdown("✅ Audio analysis complete!")
            progress.progress(100)
            time.sleep(0.5)
            
            # Clear progress indicators
            progress.empty()
            status_text.empty()
            
            # Show detected features with visualization
            st.subheader("Audio Analysis Results")
            
            # Audio characteristics visualization
            st.markdown("### Audio Characteristics")
            st.plotly_chart(
                create_radar_chart(features),
                use_container_width=True)
            
            # Feature explanation
            with st.expander("What do these features mean?"):
                st.markdown("""
                - **Danceability**: How suitable the track is for dancing (0-1)
                - **Energy**: Intensity and activity level (0-1)
                - **Acousticness**: How acoustic (vs. electronic) the song is (0-1)
                - **Instrumentalness**: Likelihood of having no vocals (0-1)
                - **Liveness**: Presence of audience/live performance elements (0-1)
                - **Speechiness**: Presence of spoken words (0-1)
                - **Valence**: Musical positiveness/happiness (0-1)
                """)
            
            # Get metadata from user
            st.subheader("Song Information")
            st.markdown("Please provide information about this song:")
            
            with st.form("upload_metadata_form"):
                # Try to extract filename as initial track name
                initial_track_name = os.path.splitext(uploaded_file.name)[0]
                # Remove numbers and underscores at beginning
                initial_track_name = re.sub(
                    r'^[\d_\s-]+', '', initial_track_name)
                
                # Metadata form fields
                col1, col2 = st.columns(2)
                
                with col1:
                    track_name = st.text_input(
                        "Track Name", value=initial_track_name)
                    track_artist = st.text_input("Artist")
                
                with col2:
                    track_album = st.text_input("Album (optional)")
                    track_genre = st.text_input("Genre (optional)")
                
                # Lyrics (optional)
                lyrics = st.text_area("Lyrics (optional)", height=200)
                
                # Get lyrics button
                fetch_lyrics = st.checkbox(
                    "Try to fetch lyrics automatically", value=True)
                
                # Options for recommendations
                st.markdown("### Recommendations")
                num_recs = st.slider("Number of Recommendations", 1, 20, 5)
                
                # Save options
                save_to_db = st.checkbox(
                    "Save this song to the database", value=True)
                
                # Submit buttons
                submit = st.form_submit_button("Process")
            
            # Handle form submission
            if submit:
                engine = get_engine()
                
                # Try to fetch lyrics if requested and not provided
                if fetch_lyrics and not lyrics and track_name and track_artist:
                    with st.spinner("Fetching lyrics..."):
                        lyrics_result = fetch_lyrics_and_sentiment(
                            track_name, track_artist)
                        if lyrics_result.get('lyrics'):
                            lyrics = lyrics_result.get('lyrics')
                            sentiment = lyrics_result.get('sentiment')
                            st.success("Successfully fetched lyrics!")
                        else:
                            st.warning("Could not fetch lyrics automatically. Error: " + 
                                   lyrics_result.get('error', 'Unknown error'))
                            sentiment = None
                else:
                    # Analyze sentiment if lyrics were provided manually
                    sentiment = analyze_sentiment(lyrics) if lyrics else None
                
                # Check for duplicates
                if track_name and track_artist:
                    with st.spinner("Checking for duplicate songs..."):
                        possible_duplicates = check_duplicate_song(
                            engine, track_name, track_artist, lyrics)
                        
                    if possible_duplicates:
                        st.warning(
                            f"Found {
                                len(possible_duplicates)} possible duplicate(s) in the database")
                        
                        with st.expander("View Possible Duplicates"):
                            for i, dup in enumerate(possible_duplicates):
                                st.markdown(f"### Possible Match #{i + 1}")
                                
                                # Similarity scores
                                cols = st.columns(2)
                                with cols[0]:
                                    st.metric(
                                        "Title Similarity", f"{
                                            dup['title_similarity']: .2f}")
                                with cols[1]:
                                    if dup['lyrics_similarity'] is not None:
                                        st.metric(
                                            "Lyrics Similarity", f"{
                                                dup['lyrics_similarity']: .2f}")
                                    else:
                                        st.metric("Lyrics Similarity", "N/A")
                                
                                # Fetch details of the duplicate
                                with engine.connect() as conn:
                                    dup_row = conn.execute(
                                        text("SELECT * FROM songs WHERE id = :id"), 
                                        {"id": dup['id']}
                                    ).fetchone()
                                
                                if dup_row:
                                    dup_song = dict(dup_row._mapping)
                                    # Show the duplicate
                                    show_song_card(dup_song)
                                    
                                    # Recommendations button
                                    dup_key = f"dup_recs_btn_{dup['id']}"
                                    if st.button(
                                            "Show Recommendations", key=dup_key):
                                        with st.spinner("Finding similar songs..."):
                                            show_recommendations_for_song(
                                                dup['id'], k=num_recs)
                
                # Save to database if requested
                if save_to_db:
                    if not track_name or not track_artist:
                        st.error(
                            "Track name and artist are required to save to the database")
                    else:
                        song_data = {
                            "track_name": track_name,
                            "track_artist": track_artist,
                            "track_album_name": track_album,
                            "playlist_genre": track_genre,
                            "lyrics": lyrics,
                            "audio_features": features,
                            "sentiment_features": sentiment,
                            "word2vec_features": None,
                            "topic_features": None,
                            "youtube_url": None,
                            "is_original": True
                        }
                        
                        with st.spinner("Saving to database..."):
                            try:
                                song_id = insert_song(engine, song_data)
                                st.success(f"Song saved with ID: {song_id}")
                                
                                # Get recommendations for the saved song
                                st.subheader(
                                    "Recommendations Based on Your Song")
                                with st.spinner("Finding similar songs..."):
                                    show_recommendations_for_song(
                                        song_id, k=num_recs)
                                    
                            except Exception as e:
                                st.error(f"Failed to save song: {e}")
                                
                                # Still show recommendations based on features
                                st.subheader(
                                    "Recommendations Based on Your Song")
                                st.info(
                                    "Finding similar songs based on audio features only (without saving)")
                                
                                with st.spinner("Finding similar songs..."):
                                    # Fallback to feature-based recommendations
                                    recs = get_similar_songs_for_features(
                                        engine, 
                                        features, 
                                        k=num_recs,
                                        audio_weight=0.7,
                                        lyrics_weight=0.2 if lyrics else 0,
                                        sentiment_weight=0.1 if sentiment else 0,
                                        lyrics=lyrics,
                                        sentiment=sentiment
                                    )
                                    
                                    if not recs:
                                        st.warning("No similar songs found")
                                    else:
                                        for rec in recs:
                                            rec_id, rec_name, rec_artist, sim = rec
                                            # Fetch full song details
                                            with engine.connect() as conn:
                                                row = conn.execute(
                                                    text(
                                                        "SELECT * FROM songs WHERE id = :id"),
                                                    {"id": rec_id}
                                                ).fetchone()
                                            
                                            if row:
                                                song = dict(row._mapping)
                                                # Add simulated component
                                                # scores
                                                song['audio_score'] = sim * 0.9
                                                song['lyrics_score'] = sim * \
                                                    0.5 if lyrics else 0
                                                song['sentiment_score'] = sim * \
                                                    0.3 if sentiment else 0
                                                # Show recommendation card
                                                show_song_card(
                                                    song, similarity=sim, with_recommendations=True)
                
                # If not saving but still want recommendations
                elif not save_to_db:
                    st.subheader("Recommendations Based on Your Song")
                    st.info(
                        "Finding similar songs based on audio features only (without saving)")
                    
                    with st.spinner("Finding similar songs..."):
                        # Feature-based recommendations
                        recs = get_similar_songs_for_features(
                            engine, 
                            features, 
                            k=num_recs,
                            audio_weight=0.7,
                            lyrics_weight=0.2 if lyrics else 0,
                            sentiment_weight=0.1 if sentiment else 0,
                            lyrics=lyrics,
                            sentiment=sentiment
                        )
                        
                        if not recs:
                            st.warning("No similar songs found")
                        else:
                            for rec in recs:
                                rec_id, rec_name, rec_artist, sim = rec
                                # Fetch full song details
                                with engine.connect() as conn:
                                    row = conn.execute(
                                        text("SELECT * FROM songs WHERE id = :id"), 
                                        {"id": rec_id}
                                    ).fetchone()
                                
                                if row:
                                    song = dict(row._mapping)
                                    # Add simulated component scores
                                    song['audio_score'] = sim * 0.9
                                    song['lyrics_score'] = sim * \
                                        0.5 if lyrics else 0
                                    song['sentiment_score'] = sim * \
                                        0.3 if sentiment else 0
                                    # Show recommendation card
                                    show_song_card(
                                        song, similarity=sim, with_recommendations=True)


# Display Upload tab
render_upload_tab(tab2)

# Main function to run the app
if __name__ == "__main__":
    pass  # All rendering is done through tab functions 
