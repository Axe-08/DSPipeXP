# DSPipeXP Music Recommendation System

A state-of-the-art music recommendation system that combines audio feature extraction, lyrics analysis, sentiment analysis, and machine learning to provide personalized song recommendations.

## 🎵 Features

- **Advanced Audio Analysis**: Extract and analyze audio features using librosa and numpy
- **Lyrics Processing**: Fetch and analyze lyrics with natural language processing techniques
- **Sentiment Analysis**: Understand the emotional tone of songs using VADER and TextBlob
- **Hybrid Recommendation Engine**: Get personalized recommendations based on sound profile, lyrics content, and emotional tone
- **YouTube Integration**: Search, download, and process songs directly from YouTube
- **Upload Your Own Music**: Analyze and get recommendations for your personal music collection
- **Beautiful Interactive UI**: Easy-to-use Streamlit interface with dark mode support

## 🚀 Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Heisenberg-Vader/DSPipeXP.git
   cd DSPipeXP
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the database**:
   Follow the instructions in [DATABASE_SETUP.md](DATABASE_SETUP.md) to set up a local PostgreSQL database with initial music data.

5. **Run the Streamlit app**:
   ```bash
   cd dspipexp_streamlit
   streamlit run app.py
   ```

## 🎥 YouTube Functionality (Important)

### Cloud Limitations

When running in cloud environments like Streamlit Cloud, the YouTube functionality faces significant limitations:

- **IP-based Blocking**: YouTube actively blocks requests from known cloud providers and data centers
- **Rate Limiting**: YouTube imposes strict rate limits, especially for automated requests
- **API Quota Restrictions**: YouTube API has limited free quotas
- **Connection Timeouts**: Requests often time out when processed through cloud servers

### Running Locally for Full YouTube Features

For reliable YouTube functionality, we **strongly recommend running the app locally** on your computer:

1. **Install FFmpeg** (required for audio processing):
   
   - **Windows**:
     ```
     winget install FFmpeg
     ```
     or download from [FFmpeg.org](https://ffmpeg.org/download.html)
   
   - **macOS**:
     ```
     brew install ffmpeg
     ```
   
   - **Linux**:
     ```
     sudo apt update && sudo apt install ffmpeg
     ```

2. **Install Python Dependencies with Extra YouTube Features**:
   ```bash
   pip install -r dspipexp_streamlit/requirements.txt
   pip install innertube aiotube google-api-python-client
   ```

3. **Set up YouTube API keys** (optional but recommended):
   - Create a project in [Google Developer Console](https://console.developers.google.com/)
   - Enable YouTube Data API v3
   - Create API keys and add to `.streamlit/secrets.toml`:
     ```
     YOUTUBE_API_KEY_1 = "your_api_key_1"
     YOUTUBE_API_KEY_2 = "your_api_key_2"
     YOUTUBE_API_KEY_3 = "your_api_key_3"
     ```

4. **Run the app locally**:
   ```bash
   cd dspipexp_streamlit
   streamlit run app.py
   ```

5. **Troubleshooting YouTube Issues**:
   - If you see "ffmpeg not found" errors, ensure FFmpeg is properly installed and in your PATH
   - If YouTube search fails, try using direct YouTube URLs instead
   - Consider using a VPN if your ISP blocks YouTube API access

## 🎥 Project Structure

```
DSPipeXP/
├── dspipexp_streamlit/    # Main Streamlit application
│   ├── app.py             # Main application entry point
│   ├── src/               # Core functionality
│   │   ├── audio.py       # Audio feature extraction
│   │   ├── db.py          # Database operations
│   │   ├── lyrics.py      # Lyrics fetching and analysis
│   │   ├── recommender.py # Recommendation algorithms
│   │   ├── utils.py       # Utility functions
│   │   └── youtube.py     # YouTube integration
│   └── requirements.txt   # App-specific dependencies
├── scripts/               # Utility scripts
├── src/                   # Additional source code
├── lyric_dataset/         # Dataset files
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## 💻 Usage Guide

### Song Search

1. Enter a song name or artist in the search tab
2. Select a song from the results
3. Click "Get Recommendations" to see similar songs
4. Explore audio features and lyrics for each recommendation

### YouTube Integration

1. Enter a YouTube URL or search for songs on YouTube
2. The system will extract audio features and lyrics
3. View recommendations based on the YouTube song
4. Optionally save the song to the database

### Upload Your Own Music

1. Upload an MP3, WAV, FLAC, or OGG file
2. The system will analyze the audio
3. Enter song metadata (title, artist, etc.)
4. Get recommendations based on your uploaded song

## 🔧 Technical Details

### Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python, FastAPI
- **Database**: PostgreSQL
- **Audio Processing**: librosa, ffmpeg
- **NLP & ML**: scikit-learn, gensim, FAISS
- **APIs**: YouTube, Genius Lyrics

### Recommendation Engine

Our recommendation system uses a hybrid approach combining:

1. **Audio Feature Similarity**: Analyzes sound characteristics like tempo, energy, danceability
2. **Lyrics Similarity**: Compares the semantic content of lyrics using NLP techniques
3. **Sentiment Analysis**: Matches songs with similar emotional tones
4. **Progressive Refinement**: Continuously improves recommendations in the background

## 👥 Team

DSPipeXP was developed by a team of data science students:

- **Akshit S Bansal** - Lead Developer
- **Kriti Chaturvedi** - Data Scientist
- **Hussain Haidary** - Machine Learning Engineer

## 📚 Learn More

- [GitHub Repository](https://github.com/Axe-08/DSPipeXP)
- [Medium Article](https://medium.com/@23ucs625/lars-lyric-aware-recommendation-system-4aac512098b7)
- [Project Documentation](dspipexp_streamlit/README.md)

## 🚀 Deployment

This project is optimized for deployment on Streamlit Cloud. For detailed deployment instructions, see [STREAMLIT_DEPLOYMENT.md](STREAMLIT_DEPLOYMENT.md).

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.