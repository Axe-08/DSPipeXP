# Updated utility functions for DSPipeXP Streamlit app
import os
import tempfile
import re
import requests
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
import base64

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temp directory and return the path"""
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def get_youtube_thumbnail(youtube_url):
    """Extract thumbnail URL from YouTube video URL"""
    if not youtube_url:
        return None
    
    # Extract video ID from various YouTube URL formats
    video_id = None
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # Standard YouTube URLs
        r'(?:embed\/)([0-9A-Za-z_-]{11})',   # Embed URLs
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'  # Short URLs
    ]
    
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            video_id = match.group(1)
            break
    
    if not video_id:
        return None
    
    # Return the highest quality thumbnail
    return f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"

def get_album_cover(track_name, artist_name):
    """Fetch album cover from Last.fm API or other sources"""
    try:
        # Try Last.fm API first
        last_fm_api_key = st.secrets.get("lastfm_api_key", None)
        if last_fm_api_key:
            url = f"http://ws.audioscrobbler.com/2.0/?method=track.getInfo&api_key={last_fm_api_key}&artist={artist_name}&track={track_name}&format=json"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if 'track' in data and 'album' in data['track']:
                    for img in data['track']['album']['image']:
                        if img['size'] == 'large' or img['size'] == 'extralarge':
                            return img['#text']
        
        # Fallback to MusicBrainz/CoverArtArchive
        url = f"https://musicbrainz.org/ws/2/recording/?query=recording:{track_name} AND artist:{artist_name}&fmt=json"
        response = requests.get(url, headers={'User-Agent': 'DSPipeXP/1.0'}, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'recordings' in data and len(data['recordings']) > 0:
                for recording in data['recordings']:
                    if 'releases' in recording and len(recording['releases']) > 0:
                        release_id = recording['releases'][0]['id']
                        cover_url = f"https://coverartarchive.org/release/{release_id}/front"
                        # Check if cover exists
                        cover_response = requests.head(cover_url, timeout=5)
                        if cover_response.status_code == 200:
                            return cover_url
        
        return None
    except Exception as e:
        st.warning(f"Failed to fetch album cover: {e}")
        return None

def get_default_album_art():
    """Return a default album art placeholder image"""
    try:
        # Use a static placeholder from a reliable source
        response = requests.get("https://via.placeholder.com/200x200.png?text=No+Cover", timeout=5)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        # Fallback to a simple colored image
        img = Image.new('RGB', (200, 200), color=(200, 200, 200))
        return img
    except Exception as e:
        # Create a blank image if all else fails
        img = Image.new('RGB', (200, 200), color=(200, 200, 200))
        return img

def load_custom_css():
    """Add custom CSS for better UI styling"""
    st.markdown("""
    <style>
        /* Improved card styling */
        .stCard {
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
            background-color: white;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .stCard:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        }
        
        /* Better typography */
        h1, h2, h3 {
            font-weight: 600;
            margin-bottom: 0.5em;
        }
        
        /* Round corners for images */
        img {
            border-radius: 8px;
        }

        /* Image sizing for album covers */
        .album-cover {
            width: 200px;
            height: 200px;
            object-fit: cover;
            display: block;
            margin: 0 auto;
        }
        
        /* Center and style tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 16px;
            justify-content: center;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
            border-radius: 10px 10px 0 0;
            height: 60px;
            font-weight: 600;
            background-color: rgba(238, 238, 238, 0.3);
        }
        .stTabs [aria-selected="true"] {
            background-color: rgba(144, 238, 144, 0.2);
            box-shadow: 0px -2px 8px rgba(0, 0, 0, 0.05);
        }
        
        /* Better buttons */
        button {
            border-radius: 10px !important;
            font-weight: 500 !important;
            padding: 3px 5px;
            transition: all 0.2s ease;
        }
        
        /* Format lyrics with proper line breaks */
        .lyrics {
            white-space: pre-wrap !important;
            font-style: italic;
            line-height: 1.7;
            background-color: rgba(240, 240, 240, 0.4);
            padding: 16px;
            border-radius: 8px;
            max-height: 300px;
            overflow-y: auto;
            font-family: inherit;
        }
        
        /* Song cards */
        .song-card {
            padding: 16px;
            border-radius: 12px;
            background-color: #f8f9fa;
            margin-bottom: 16px;
            border-left: 4px solid #4CAF50;
        }
        
        /* Style for expandable sections */
        .streamlit-expanderHeader {
            border-radius: 8px;
            background-color: #f1f3f4;
            font-weight: 500;
        }
        
        /* Similarity score */
        .similarity-score {
            font-size: 12px;
            color: #1E88E5;
            padding: 2px 8px;
            background-color: rgba(30, 136, 229, 0.1);
            border-radius: 16px;
            margin-left: 8px;
        }
        
        /* Info tooltip */
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
    </style>
    """, unsafe_allow_html=True)

def create_radar_chart(audio_features):
    """Create a radar chart from audio features"""
    # Define which features to include in the chart
    features_to_plot = [
        'danceability', 'energy', 'acousticness', 
        'instrumentalness', 'liveness', 'speechiness', 'valence'
    ]
    
    # Extract values and ensure they're between 0 and 1
    feature_values = []
    for feature in features_to_plot:
        value = audio_features.get(feature, 0)
        # Ensure the value is between 0 and 1
        if isinstance(value, (int, float)):
            if feature == 'loudness':  # Special case for loudness (-60 to 0)
                value = (value + 60) / 60  # Normalize to 0-1
            feature_values.append(max(0, min(1, value)))
        else:
            feature_values.append(0)
    
    # Create radar chart
    fig = go.Figure()
    
    # Add trace
    fig.add_trace(go.Scatterpolar(
        r=feature_values,
        theta=features_to_plot,
        fill='toself',
        name='Audio Features',
        line_color='rgba(75, 192, 192, 0.8)',
        fillcolor='rgba(75, 192, 192, 0.2)'
    ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        height=400,
        margin=dict(l=80, r=80, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def format_lyrics(lyrics):
    """Format lyrics with proper styling and line breaks"""
    if not lyrics:
        return "<div class='lyrics'>No lyrics available</div>"
    
    # Remove redundant whitespace but preserve intended line breaks
    formatted = re.sub(r'\n\s*\n', '\n\n', lyrics)
    
    # Remove [verse], [chorus] etc. markers
    formatted = re.sub(r'\[(.*?)\]', '', formatted)
    
    # Convert newlines to HTML breaks to ensure proper rendering
    formatted = formatted.replace('\n', '<br>')
    
    # Wrap in a div with proper styling
    return f'<div class="lyrics">{formatted}</div>'

def format_duration(ms):
    """Format milliseconds duration to mm:ss format"""
    seconds = int((ms / 1000) % 60)
    minutes = int((ms / (1000 * 60)) % 60)
    
    return f"{minutes}:{seconds:02d}"

def explain_recommendation(audio_score, lyrics_score, sentiment_score):
    """Generate an explanation for why a song was recommended"""
    reasons = []
    
    if audio_score > 0.7:
        reasons.append("very similar sound profile")
    elif audio_score > 0.5:
        reasons.append("similar sound profile")
    
    if lyrics_score > 0.7:
        reasons.append("very similar lyrical themes")
    elif lyrics_score > 0.5:
        reasons.append("similar lyrical themes")
    
    if sentiment_score > 0.7:
        reasons.append("matching emotional tone")
    elif sentiment_score > 0.5:
        reasons.append("similar emotional tone")
    
    if not reasons:
        reasons.append("overall similarity")
    
    return f"Recommended because of {' and '.join(reasons)}"

# Add function to safely load images
def safe_image_load(url_or_image, use_class=None):
    """Safely load and display an image with fallback to default"""
    try:
        def display_image(img_obj):
            if use_class:
                buffered = BytesIO()
                img_obj.save(buffered, format="PNG")
                encoded = base64.b64encode(buffered.getvalue()).decode()
                img_tag = f'<img src="data:image/png;base64,{encoded}" class="{use_class}" width="200"/>'
                st.markdown(img_tag, unsafe_allow_html=True)
            else:
                st.image(img_obj, use_container_width=True, output_format="PNG", caption="", width=200)

        # Load default if needed
        if url_or_image is None or (
            isinstance(url_or_image, str) and not url_or_image.startswith(('http://', 'https://'))
        ):
            img = get_default_album_art()
            return display_image(img)

        # Handle direct Image object
        if isinstance(url_or_image, Image.Image):
            return display_image(url_or_image)

        # Handle URL string
        if isinstance(url_or_image, str):
            response = requests.get(url_or_image, timeout=5)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                return display_image(img)

        img = get_default_album_art()
        return display_image(img)

    except Exception as e:
        st.warning(f"Error displaying image: {e}")
        img = get_default_album_art()
        return display_image(img)
