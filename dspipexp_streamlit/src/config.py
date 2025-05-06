# Config and secrets loading for DSPipeXP Streamlit app
import streamlit as st
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Example: Accessing secrets
# db_url = st.secrets["db_url"]
# genius_api_key = st.secrets["genius_api_key"]
# youtube_api_key = st.secrets["youtube_api_key"]

# TODO: Add config helpers as needed 

# YouTube API key rotation system
def setup_youtube_api_keys():
    """
    Set up YouTube API key rotation. This pulls keys from Streamlit secrets.
    If no keys are found, we'll use an empty list for graceful degradation.
    """
    if 'youtube_api_keys' not in st.session_state:
        # Try to get keys from secrets
        api_keys = []
        
        # Look for keys in the format youtube_api_key_1, youtube_api_key_2, etc.
        for i in range(1, 10):  # Look for up to 9 keys
            key_name = f"youtube_api_key_{i}"
            if key_name in st.secrets:
                api_keys.append(st.secrets[key_name])
        
        # Also check for legacy youtube_api_key format
        if "youtube_api_key" in st.secrets and st.secrets["youtube_api_key"] not in api_keys:
            api_keys.append(st.secrets["youtube_api_key"])
        
        # Set up the state
        st.session_state.youtube_api_keys = api_keys
        st.session_state.current_api_key_index = 0
        st.session_state.api_key_usage_count = {}
        
        # Log the number of keys found (not the actual keys for security)
        logger.info(f"Found {len(api_keys)} YouTube API key(s)")
        
        # Initialize usage counters
        for key in api_keys:
            st.session_state.api_key_usage_count[key] = 0


def get_next_youtube_api_key():
    """
    Get the next YouTube API key in rotation.
    If no keys are available, returns None.
    """
    # Make sure keys are set up
    if 'youtube_api_keys' not in st.session_state:
        setup_youtube_api_keys()
    
    keys = st.session_state.youtube_api_keys
    if not keys:
        return None
    
    # Get the next key in rotation
    key = keys[st.session_state.current_api_key_index]
    
    # Rotate to next key
    st.session_state.current_api_key_index = (st.session_state.current_api_key_index + 1) % len(keys)
    
    # Track usage
    st.session_state.api_key_usage_count[key] = st.session_state.api_key_usage_count.get(key, 0) + 1
    
    return key


def get_secret(key, default=None):
    """Get a value from Streamlit secrets with a default fallback"""
    return st.secrets.get(key, default) 