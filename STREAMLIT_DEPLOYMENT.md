# DSPipeXP Streamlit Cloud Deployment Guide

This guide provides instructions for deploying the DSPipeXP Music Recommendation System to Streamlit Cloud.

## Prerequisites

- A GitHub account
- A PostgreSQL database (Render, Railway, or Supabase recommended)
- Genius API key (for lyrics fetching)

## Deployment Steps

1. **Prepare your repository**:
   - Ensure your code is committed to GitHub
   - Make sure `dspipexp_streamlit/requirements.txt` includes all dependencies

2. **Set up a PostgreSQL database**:
   - Create a PostgreSQL database on your preferred provider (Render, Railway, etc.)
   - Note down your connection details (host, port, database name, username, password)

3. **Get API keys**:
   - Get a Genius API key from [genius.com/api-clients](https://genius.com/api-clients)
   - (Optional) Get a LastFM API key from [last.fm/api](https://www.last.fm/api)

4. **Deploy on Streamlit Cloud**:
   - Visit [streamlit.io/cloud](https://streamlit.io/cloud) and sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set the main file path to `dspipexp_streamlit/app.py`
   - Set Python version to 3.8 or higher
   - Click "Deploy!"

5. **Configure secrets**:
   - In your Streamlit Cloud app settings, navigate to the "Secrets" section
   - Add your database credentials and API keys in TOML format:
     ```toml
     [postgres]
     host = "your-postgres-host.example.com"
     port = 5432
     database = "yourdbname"
     user = "yourdbuser"
     password = "yourdbpass"

     [api_keys]
     genius_access_token = "your_genius_access_token"
     lastfm_api_key = "your_lastfm_api_key"
     lastfm_shared_secret = "your_lastfm_shared_secret"
     ```
   - Save your secrets

6. **Install ffmpeg** (important for YouTube processing):
   - In your Streamlit Cloud app settings, go to the "Advanced" section
   - Add the following packages to install in the "Packages" field:
     ```
     ffmpeg
     ```
   - This ensures that ffmpeg is installed on the server, which is required for YouTube audio processing

7. **Reboot your app**:
   - After adding secrets and packages, reboot your app from the Streamlit Cloud dashboard

Your app should now be up and running on Streamlit Cloud! The URL will be something like `https://username-app-name.streamlit.app`

## Troubleshooting

- If the app fails to deploy, check your `requirements.txt` file for any missing dependencies
- If database connections fail, verify your connection details and ensure your database allows connections from Streamlit Cloud
- If you see API errors, check your API keys and rate limits
- If YouTube processing fails with "ffmpeg not found" errors, make sure you've added ffmpeg to the packages list in your app's Advanced settings

## Known Limitations in Cloud Deployment

### YouTube Functionality Limitations

YouTube integration has significant limitations when running in cloud environments like Streamlit Cloud:

1. **IP-based Blocking**: YouTube actively blocks requests from cloud provider IP ranges, resulting in 403 Forbidden errors
2. **Rate Limiting**: YouTube imposes strict rate limits on requests from shared IP addresses
3. **Connection Timeouts**: Requests to YouTube often time out in cloud environments
4. **API Quota Restrictions**: The YouTube API has limited free quotas that can be quickly exhausted

Despite adding ffmpeg to packages and configuring YouTube API keys, these limitations are largely unavoidable in cloud deployments due to YouTube's anti-scraping measures.

### Recommendation for Users

For users who need reliable YouTube functionality, we recommend:

1. Include clear messaging in your app's YouTube section about these limitations
2. Suggest users run the application locally for full YouTube functionality
3. Provide a link to the GitHub repository and installation instructions
4. Consider implementing fallback behaviors when YouTube operations fail

For local installation instructions, refer to the main [README.md](README.md#-youtube-functionality-important).

## Updating Your App

When you push changes to your GitHub repository, Streamlit Cloud will automatically detect the changes and update your app. No manual redeployment is needed! 