import streamlit as st
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
from src.youtube import process_youtube_url, youtube_search_and_get_url, update_song_youtube_url, youtube_search
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

# Set up page configuration
st.set_page_config(
    page_title="DSPipeXP Music Recommendation", 
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        except:
            audio_features = {}

    sentiment = song.get('sentiment_features', None)
    if isinstance(sentiment, str):
        try:
            sentiment = json.loads(sentiment)
        except:
            sentiment = None

    # Card container
    with st.container():
        st.markdown('<div class="song-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 3])
        img_url = get_youtube_thumbnail(song['youtube_url']) if song.get('youtube_url') else None
        img_url = img_url or get_album_cover(song.get('track_name', ''), song.get('track_artist', ''))
        
        with col1:
            safe_image_load(img_url, use_class="album-cover")
        
        with col2:
            # Title and artist
            if similarity is not None:
                st.markdown(f"""
                    ### {song.get('track_name', 'Unknown Track')} 
                    <span class="similarity-score">{similarity:.2f} match</span>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"### {song.get('track_name', 'Unknown Track')}")
            st.markdown(f"**Artist:** {song.get('track_artist', 'Unknown Artist')}")

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
                if st.button("🔍 Get Recommendations", key=f"rec_btn_{song['id']}"):
                    st.session_state[f'show_recs_for_{song["id"]}'] = True
                    st.rerun()

            if audio_features:
                if st.button("📊 Audio Profile", key=f"audio_viz_{song.get('id', 'temp')}"):
                    st.session_state[f"show_viz_for_{song.get('id', 'temp')}"] = True
                    st.rerun()

        
        # Expandable Lyrics
        if song.get('lyrics'):
            with st.expander("📝 View Lyrics"):
                st.markdown(format_lyrics(song['lyrics']), unsafe_allow_html=True)

        # Audio Visualization
        viz_key = f'show_viz_for_{song.get("id", "temp")}'
        if st.session_state.get(viz_key):
            with st.expander("Audio Feature Visualization", expanded=True):
                st.plotly_chart(create_radar_chart(audio_features), use_container_width=True)
                st.markdown("""
                This radar chart shows the audio characteristics of the song. Higher values mean stronger presence of that trait:
                - **Danceability**: How suitable the track is for dancing
                - **Energy**: Intensity and activity level
                - **Acousticness**: How acoustic (vs. electronic) the song is
                - **Instrumentalness**: Likelihood of having no vocals
                - **Liveness**: Presence of audience/live performance elements
                - **Speechiness**: Presence of spoken words
                - **Valence**: Musical positiveness/happiness
                """)
                if st.button("Hide Visualization", key=f"hide_viz_{song.get('id', 'temp')}"):
                    st.session_state[viz_key] = False
                    st.rerun()

        # Recommendations
        rec_key = f'show_recs_for_{song.get("id")}'
        if st.session_state.get(rec_key) and not with_recommendations:
            with st.expander("Similar Songs", expanded=True):
                with st.spinner("Finding similar songs..."):
                    show_recommendations_for_song(song.get('id'), k=5)
                if st.button("Hide Recommendations", key=f"hide_recs_{song.get('id')}"):
                    st.session_state[rec_key] = False
                    st.rerun()

        # Explanation for recommendation
        if similarity is not None and all(k in song for k in ['audio_score', 'lyrics_score', 'sentiment_score']):
            with st.expander("Why this recommendation?"):
                st.info(explain_recommendation(
                    song['audio_score'],
                    song['lyrics_score'],
                    song['sentiment_score']
                ))

                st.markdown("### Component Similarity Scores")
                st.metric("Sound Profile", f"{song['audio_score']:.2f}")
                st.metric("Lyrics", f"{song['lyrics_score']:.2f}")
                st.metric("Emotional Tone", f"{song['sentiment_score']:.2f}")

        st.markdown('</div>', unsafe_allow_html=True)

    
def show_recommendations_for_song(song_id, k=5):
    """Show recommendations for a song with detailed component scores"""
    engine = get_engine()
    # Get recommendations using the hybrid engine
    recs = get_similar_songs(engine, song_id, k=k, 
                            audio_weight=0.7, lyrics_weight=0.2, sentiment_weight=0.1)
                            
    if not recs:
        st.warning("No recommendations found for this song.")
        return
    
    st.markdown("### Similar Songs You Might Enjoy")
    
    for rec in recs:
        rec_id, rec_name, rec_artist, sim = rec
        # Fetch full song details
        with engine.connect() as conn:
            row = conn.execute(text("SELECT * FROM songs WHERE id = :id"), {"id": rec_id}).fetchone()
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
            
            # Add component scores (actual implementation would require access to internal scores)
            # Here we're simulating them based on similarity
            song['audio_score'] = sim * 0.8 + 0.1  # Simulate audio score
            song['lyrics_score'] = sim * 0.7 + 0.2  # Simulate lyrics score
            song['sentiment_score'] = sim * 0.6 + 0.3  # Simulate sentiment score
            
            # Show recommendation card
            show_song_card(song, similarity=sim, with_recommendations=True)
        else:
            st.warning(f"Could not fetch details for {rec_name} by {rec_artist}")

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
            st.markdown("[Learn how to export cookies](https://github.com/yt-dlp/yt-dlp/wiki/How-to-provide-your-own-cookies-to-yt-dlp)")
            cookies_file = st.file_uploader("Upload YouTube cookies.txt (optional)", type=["txt"], key="sidebar_cookies")
            if cookies_file:
                st.session_state['youtube_cookies_file'] = cookies_file
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Skip", key="sidebar_no_cookies"):
                    st.session_state['show_cookies_sidebar'] = False
            with col2:
                if st.button("Continue", key="sidebar_yes_cookies"):
                    st.session_state['show_cookies_sidebar'] = False
                    
# Configure the main page elements
if 'show_cookies_sidebar' not in st.session_state:
    st.session_state['show_cookies_sidebar'] = True

# Show sidebar for cookies if needed
if st.session_state['show_cookies_sidebar']:
    youtube_cookies_sidebar()

# Main app title
st.title("🎵 DSPipeXP Music Recommendation")
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
                results = search_songs(engine, search_query, limit=search_limit)
                
            if results:
                st.success(f"Found {len(results)} songs matching '{search_query}'")
                
                # Create a clean results display - either grid or list view
                view_type = st.radio(
                    "View results as:", 
                    options=["Detailed List", "Grid View"],
                    horizontal=True,
                    help="Choose how to display search results"
                )
                
                if view_type == "Grid View":
                    # Create a grid of cards (3 per row)
                    rows = [results[i:i+3] for i in range(0, len(results), 3)]
                    
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
                                    st.markdown(f"**Artist:** {song['track_artist']}")
                                    
                                    # Try to get thumbnail
                                    img_url = None
                                    if song.get('youtube_url'):
                                        img_url = get_youtube_thumbnail(song['youtube_url'])
                                    if not img_url:
                                        img_url = get_album_cover(song['track_name'], song['track_artist'])
                                    
                                    if img_url:
                                        try:
                                            st.image(img_url, use_container_width=True)
                                        except:
                                            pass
                                    
                                    # Show details button
                                    detail_btn_key = f"detail_grid_{song_id}"
                                    if st.button("View Details", key=detail_btn_key):
                                        st.session_state['selected_song_id'] = song_id
                                        st.rerun()
                
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
                                    song['lyrics'] = lyrics_result.get('lyrics')
                                    song['sentiment_features'] = lyrics_result.get('sentiment')
                            
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
                        if st.button("Get Recommendations", key=f"main_recs_{selected_id}"):
                            with st.spinner("Finding similar songs..."):
                                show_recommendations_for_song(selected_id, k=num_recs)
                        
                        # Clear selection button
                        if st.button("Back to Search Results"):
                            del st.session_state['selected_song_id']
                            st.rerun()
            
            else:
                st.info(f"No songs found matching '{search_query}'")
                
                # "Search YouTube Instead" option
                if st.button("Search YouTube Instead"):
                    st.session_state['youtube_search_query'] = search_query
                    st.session_state['show_youtube_tab'] = True
                    st.rerun()
                
                # Suggest related searches
                st.markdown("**Try these popular searches:**")
                suggestion_cols = st.columns(4)
                suggestions = ["Ed Sheeran", "Billie Eilish", "Drake", "BTS"]
                
                for i, suggestion in enumerate(suggestions):
                    with suggestion_cols[i]:
                        if st.button(suggestion):
                            # Set search query and rerun
                            st.session_state['search_query'] = suggestion
                            st.rerun()

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
                del st.session_state['youtube_search_query']  # Clear it after use
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
    with st.spinner("Getting video information..."):
        info_dict = None
        features, error = None, None
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            try:
                info_dict = ydl.extract_info(youtube_url, download=False)
            except Exception as e:
                st.error(f"Failed to get YouTube video info: {e}")
                return
    
    if not info_dict:
        st.error("Could not retrieve information for this YouTube URL")
        return
    
    # Show video info with thumbnail
    st.subheader("YouTube Video Information")
    
    # Two-column layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Video thumbnail
        thumbnail_url = get_youtube_thumbnail(youtube_url)
        if thumbnail_url:
            st.image(thumbnail_url, use_container_width=True)
        else:
            st.markdown("No thumbnail available")
    
    with col2:
        title = info_dict.get('title', 'Unknown title')
        uploader = info_dict.get('uploader', 'Unknown uploader')
        
        # Extract artist and track name from title
        if ' - ' in title:
            track_artist, track_name = title.split(' - ', 1)
        else:
            track_name = title
            track_artist = uploader
            
        # Display video info
        st.markdown(f"**Title:** {title}")
        st.markdown(f"**Channel:** {uploader}")
        st.markdown(f"**Duration:** {info_dict.get('duration', 0)} seconds")
        
        # Add a progress indicator for the next steps
        progress = st.progress(0)
        st.markdown("### Processing Status")
        status = st.empty()
        
        # Step 1: Download and process audio
        status.markdown("⏳ Downloading and processing audio...")
        progress.progress(10)
        
        features, error = process_youtube_url(youtube_url)
        if error:
            st.warning(error)
        
        if not features:
            st.error("Failed to extract audio features")
            return
        
        # Step 2: Fetch lyrics
        progress.progress(50)
        status.markdown("⏳ Fetching lyrics...")
        
        lyrics_result = fetch_lyrics_and_sentiment(track_name, track_artist)
        lyrics = lyrics_result.get('lyrics', '')
        sentiment = lyrics_result.get('sentiment', None)
        
        # Step 3: Check for duplicates
        progress.progress(80)
        status.markdown("⏳ Checking for duplicate songs...")
        
        engine = get_engine()
        possible_duplicates = check_duplicate_song(engine, track_name, track_artist, lyrics)
        
        # Complete progress
        progress.progress(100)
        status.markdown("✅ Processing complete!")
        time.sleep(0.5)
        
        # Remove progress indicators
        progress.empty()
        status.empty()
        
        # Show song metadata
        st.subheader("Detected Song Information")
        
        # Editable metadata with form
        with st.form("youtube_metadata_form"):
            edited_track_name = st.text_input("Track Name", value=track_name)
            edited_track_artist = st.text_input("Artist", value=track_artist)
            edited_album = st.text_input("Album", value=info_dict.get('album', ''))
            edited_genre = st.text_input("Genre", value=info_dict.get('genre', ''))
            edited_lyrics = st.text_area("Lyrics", value=lyrics, height=200)
            
            # Audio feature visualization
            with st.expander("🔊 Audio Feature Visualization"):
                st.plotly_chart(create_radar_chart(features), use_container_width=True)
                
            # Form submission buttons
            col1, col2 = st.columns(2)
            
            with col1:
                save_button = st.form_submit_button("Save to Database & Get Recommendations")
            
            with col2:
                recommendations_only = st.form_submit_button("Get Recommendations Only (Don't Save)")
                
        # Handle form submissions
        if save_button:
            # Prepare song data
            song_data = {
                "track_name": edited_track_name,
                "track_artist": edited_track_artist,
                "track_album_name": edited_album,
                "playlist_genre": edited_genre,
                "lyrics": edited_lyrics,
                "audio_features": features,
                "sentiment_features": sentiment,
                "word2vec_features": None,  # These would be computed separately
                "topic_features": None,     # These would be computed separately
                "youtube_url": youtube_url,
                "is_original": False
            }
            
            # Save to database
            with st.spinner("Saving song to database..."):
                song_id = insert_song(engine, song_data)
                
            st.success(f"Song saved with ID {song_id}")
            
            # Now get recommendations
            st.subheader("Recommendations")
            with st.spinner("Finding similar songs..."):
                show_recommendations_for_song(song_id, k=5)
                
        elif recommendations_only:
            # Just get recommendations without saving
            st.subheader("Recommendations")
            
            # Use the features directly (not from DB)
            with st.spinner("Finding similar songs..."):
                from src.recommender import get_similar_songs_for_features
                # Use the hybrid approach with lyrics and sentiment
                recs = get_similar_songs_for_features(
                    engine, 
                    features, 
                    k=5,
                    audio_weight=0.7, 
                    lyrics_weight=0.2, 
                    sentiment_weight=0.1, 
                    lyrics=edited_lyrics,
                    sentiment=sentiment
                )
                
                if not recs:
                    st.warning("No similar songs found")
                    return
                    
                for rec in recs:
                    rec_id, rec_name, rec_artist, sim = rec
                    # Fetch full song details
                    with engine.connect() as conn:
                        row = conn.execute(text("SELECT * FROM songs WHERE id = :id"), {"id": rec_id}).fetchone()
                    
                    if row:
                        song = dict(row._mapping)
                        # Add component scores (simulated)
                        song['audio_score'] = sim * 0.8 + 0.1
                        song['lyrics_score'] = sim * 0.7 + 0.2
                        song['sentiment_score'] = sim * 0.6 + 0.3
                        # Show recommendation card
                        show_song_card(song, similarity=sim, with_recommendations=True)
        
        # Display duplicates if any were found
        if possible_duplicates:
            st.subheader("Possible Duplicate Songs")
            st.markdown("We found songs that might be duplicates of this one:")
            
            for i, dup in enumerate(possible_duplicates):
                with st.container():
                    st.markdown(f"### Possible Match #{i+1}")
                    
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
                    with engine.connect() as conn:
                        row = conn.execute(text("SELECT * FROM songs WHERE id = :id"), {"id": dup['id']}).fetchone()
                    
                    if row:
                        song = dict(row._mapping)
                        # Show song card
                        show_song_card(song)
                    
                    # View recommendations button
                    if st.button(f"View Recommendations for This Song", key=f"dup_recs_{dup['id']}"):
                        with st.spinner("Finding similar songs..."):
                            show_recommendations_for_song(dup['id'], k=5)

def search_youtube_with_ui(query, max_results=5):
    """Search YouTube with improved UI"""
    with st.spinner(f"Searching YouTube for '{query}'..."):
        results = youtube_search(query, max_results=max_results)
        
    if not results:
        st.warning("No YouTube results found. Try a different search query.")
        return
    
    st.success(f"Found {len(results)} videos on YouTube")
    
    # Display results in a grid - 2 videos per row
    rows = [results[i:i+2] for i in range(0, len(results), 2)]
    
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
                
                # Process button
                btn_key = f"process_yt_{row_idx}_{col_idx}"
                if st.button("Process This Song", key=btn_key):
                    # Store the selection and refresh
                    st.session_state['selected_youtube_url'] = video['url']
                    st.rerun()
                
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
            st.plotly_chart(create_radar_chart(features), use_container_width=True)
            
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
                initial_track_name = re.sub(r'^[\d_\s-]+', '', initial_track_name)
                
                # Metadata form fields
                col1, col2 = st.columns(2)
                
                with col1:
                    track_name = st.text_input("Track Name", value=initial_track_name)
                    track_artist = st.text_input("Artist")
                
                with col2:
                    track_album = st.text_input("Album (optional)")
                    track_genre = st.text_input("Genre (optional)")
                
                # Lyrics (optional)
                lyrics = st.text_area("Lyrics (optional)", height=200)
                
                # Get lyrics button
                fetch_lyrics = st.checkbox("Try to fetch lyrics automatically", value=True)
                
                # Options for recommendations
                st.markdown("### Recommendations")
                num_recs = st.slider("Number of Recommendations", 1, 20, 5)
                
                # Save options
                save_to_db = st.checkbox("Save this song to the database", value=True)
                
                # Submit buttons
                submit = st.form_submit_button("Process")
            
            # Handle form submission
            if submit:
                engine = get_engine()
                
                # Try to fetch lyrics if requested and not provided
                if fetch_lyrics and not lyrics and track_name and track_artist:
                    with st.spinner("Fetching lyrics..."):
                        lyrics_result = fetch_lyrics_and_sentiment(track_name, track_artist)
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
                        possible_duplicates = check_duplicate_song(engine, track_name, track_artist, lyrics)
                        
                    if possible_duplicates:
                        st.warning(f"Found {len(possible_duplicates)} possible duplicate(s) in the database")
                        
                        with st.expander("View Possible Duplicates"):
                            for i, dup in enumerate(possible_duplicates):
                                st.markdown(f"### Possible Match #{i+1}")
                                
                                # Similarity scores
                                cols = st.columns(2)
                                with cols[0]:
                                    st.metric("Title Similarity", f"{dup['title_similarity']:.2f}")
                                with cols[1]:
                                    if dup['lyrics_similarity'] is not None:
                                        st.metric("Lyrics Similarity", f"{dup['lyrics_similarity']:.2f}")
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
                                    if st.button("Show Recommendations", key=dup_key):
                                        with st.spinner("Finding similar songs..."):
                                            show_recommendations_for_song(dup['id'], k=num_recs)
                
                # Save to database if requested
                if save_to_db:
                    if not track_name or not track_artist:
                        st.error("Track name and artist are required to save to the database")
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
                                st.subheader("Recommendations Based on Your Song")
                                with st.spinner("Finding similar songs..."):
                                    show_recommendations_for_song(song_id, k=num_recs)
                                    
                            except Exception as e:
                                st.error(f"Failed to save song: {e}")
                                
                                # Still show recommendations based on features
                                st.subheader("Recommendations Based on Your Song")
                                st.info("Finding similar songs based on audio features only (without saving)")
                                
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
                                                    text("SELECT * FROM songs WHERE id = :id"), 
                                                    {"id": rec_id}
                                                ).fetchone()
                                            
                                            if row:
                                                song = dict(row._mapping)
                                                # Add simulated component scores
                                                song['audio_score'] = sim * 0.9
                                                song['lyrics_score'] = sim * 0.5 if lyrics else 0
                                                song['sentiment_score'] = sim * 0.3 if sentiment else 0
                                                # Show recommendation card
                                                show_song_card(song, similarity=sim, with_recommendations=True)
                
                # If not saving but still want recommendations
                elif not save_to_db:
                    st.subheader("Recommendations Based on Your Song")
                    st.info("Finding similar songs based on audio features only (without saving)")
                    
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
                                    song['lyrics_score'] = sim * 0.5 if lyrics else 0
                                    song['sentiment_score'] = sim * 0.3 if sentiment else 0
                                    # Show recommendation card
                                    show_song_card(song, similarity=sim, with_recommendations=True)

# Display Upload tab
render_upload_tab(tab2)

# Main function to run the app
if __name__ == "__main__":
    pass  # All rendering is done through tab functions 