# Config and secrets loading for DSPipeXP Streamlit app
import streamlit as st

# Example: Accessing secrets
# db_url = st.secrets["db_url"]
# genius_api_key = st.secrets["genius_api_key"]
# youtube_api_key = st.secrets["youtube_api_key"]

# TODO: Add config helpers as needed 

def get_secret(key, default=None):
    return st.secrets.get(key, default) 