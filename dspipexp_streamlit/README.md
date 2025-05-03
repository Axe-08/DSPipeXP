# DSPipeXP Streamlit App

A unified Streamlit-based music recommendation system with audio processing, lyrics analysis, YouTube integration, and advanced recommendations—all in one app.

## Features

- 🎵 Audio feature extraction and analysis (librosa, numpy)
- 📝 Lyrics fetching and sentiment analysis (Genius API, vaderSentiment)
- 🎥 YouTube video search and audio download (yt-dlp/pytube)
- 🔍 Advanced music recommendation system (FAISS, scikit-learn)
- 📊 Interactive UI for search, upload, and recommendations
- 🚀 Deployable on Streamlit Community Cloud

## User Scenarios & Flows

### 1. Song Name or Artist Search
- User enters a song name or artist.
- App searches the database and shows results (song name, artist, duration, etc.).
- User selects the correct song and number of recommendations.
- App fetches and displays recommendations (song name, artist, duration, similarity score, lyrics link, YouTube link).
- **If not found:**
  - App prompts user to correct query or search the web.
  - If searching the web, app shows top YouTube results.
  - User selects a result; app downloads, extracts features, fetches lyrics, and recommends similar songs.
  - New song is added to the database.

### 2. YouTube URL
- User enters a YouTube URL.
- App downloads audio, extracts features, fetches lyrics, and recommends similar songs.
- If the song is new, it is added to the database.

### 3. Audio File Upload
- User uploads a song.
- App extracts features, tries to match with database.
- If a match is found, app asks user to confirm details.
- If not found or details are incorrect, user can provide/correct metadata.
- New song is added to the database if needed.
- Recommendations are shown.

## Recommendation Engine
- Combines audio features, lyrics similarity, mood, and sentiment analysis to recommend songs.
- Uses vector similarity (cosine/FAISS) and NLP for lyrics/mood.

## Database Update Logic
- Any new song (from YouTube or upload) is analyzed and added to the database with all extracted features and user-supplied/corrected metadata.

## Tech Stack

- **App/UI/Backend:** Streamlit (Python)
- **Database:** Neon (PostgreSQL, cloud)
- **Audio Processing:** librosa, ffmpeg, numpy
- **ML/AI:** scikit-learn, FAISS, gensim
- **External APIs:** YouTube, Genius Lyrics

## Project Structure

```
dspipexp_streamlit/
├── app.py                  # Main Streamlit app (UI + logic)
├── requirements.txt
├── src/
│   ├── db.py               # DB connection, queries
│   ├── audio.py            # Audio feature extraction
│   ├── lyrics.py           # Genius API integration
│   ├── youtube.py          # YouTube download/processing
│   ├── recommender.py      # Recommendation logic (FAISS, ML)
│   ├── utils.py            # Misc utilities
│   └── config.py           # Config, secrets loading
├── data/
│   └── (optional local cache, temp files)
├── tests/
│   └── (unit/integration tests)
├── .streamlit/
│   └── secrets.toml        # For Streamlit Cloud secrets
└── README.md
```

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd dspipexp_streamlit
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate  # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure secrets:**
   - Add your Neon DB credentials and API keys to `.streamlit/secrets.toml` (see Streamlit docs).

5. **Run the app locally:**
   ```bash
   streamlit run app.py
   ```

## Deployment

- Push your code to GitHub.
- Deploy on [Streamlit Community Cloud](https://streamlit.io/cloud):
  - Connect your repo
  - Set up secrets in the dashboard
  - Deploy and test

## Notes

- No separate API endpoints: all logic/UI in Streamlit.
- All features (search, upload, recommend, etc.) are accessible via the web UI.
- For large-scale or production use, consider adding caching, authentication, and Dockerization.

## Advanced Features & Enhancements

### Progress Indicators
Show progress bars or spinners for long operations (e.g., YouTube download, feature extraction) to keep users informed.

### Friendly Error Feedback
Provide clear, actionable error messages throughout the app (e.g., for failed searches, missing lyrics, or upload issues).

### Song Details on Hover/Click
Allow users to view detailed song information (album art, genre, lyrics preview, etc.) in a modal or sidebar when hovering or clicking on a song in search/results.

### Hybrid Recommendation Weighting
Let users adjust the importance of audio, lyrics, mood, and sentiment in recommendations using sliders or controls, for more personalized results.

### User Feedback on Recommendations
Enable users to rate or give feedback on recommendations (e.g., thumbs up/down, star rating) to improve the system over time.

### Playlist Generation
Allow users to generate playlists based on a seed song, mood, or genre, using the recommendation engine.

### Async Processing
Use asynchronous/background processing for heavy tasks (YouTube download, feature extraction) to keep the UI responsive.

### Caching
Cache frequent queries and recommendation results to improve speed and reduce load on the database and external APIs.

### Monitoring & Analytics
Provide an admin dashboard to monitor app usage, errors, popular searches, and system health.

### Mood/Emotion Visualization
Visualize the mood or sentiment of a song or playlist using radar/spider charts or similar visual tools.

### Explainable Recommendations
Show users why a song was recommended (e.g., "Similar tempo and positive mood"), increasing trust and transparency in the system.

## Duplicate Song Prevention

To avoid adding near-duplicate songs (e.g., different YouTube uploads, lyric vs. music video, minor metadata differences):
- **Fuzzy Metadata Matching:** On new song insert, normalize the title (remove terms like '(lyric video)', '(official audio)', etc.) and search for songs by the same artist with a similar normalized title using fuzzy string matching. If a match above a threshold (e.g., 80% similarity) is found, the user is warned and shown both entries for manual review.
- **Lyrics Similarity:** If lyrics are available, cosine similarity is computed between the new song's lyrics and existing songs by the same artist. If similarity is high, the user is warned.
- **Manual Review:** If a possible duplicate is detected, the user can confirm or override the insert.
- **Future:** Audio fingerprinting and admin tools for merging/marking duplicates may be added.

## Recommendation Engine Feature Extraction

- **Word2Vec Features:** Lyrics are tokenized and embedded using Word2Vec; the average vector is used for each song.
- **Sentiment Analysis:** Sentiment scores (polarity, subjectivity, VADER) are extracted from lyrics using TextBlob and VADER.
- **Topic Features:** Topic distribution for each song's lyrics is computed using LDA.
- **Integration:** On song upload, YouTube, or new entry, all features are extracted and cached, stored in the database, and used in the recommendation engine. Users can adjust the importance of lyrics, audio, sentiment, and topic in recommendations.
- **Reference:** Feature extraction logic is adapted from `faster_sound_rec.py`.

## Possible Future Enhancement: Full Audio Feature Recommendations

Currently, the system uses only the 12 standard Spotify features (danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms) for all songs. This ensures compatibility between songs imported from Spotify CSV and those processed from audio files or YouTube.

**Potential Enhancement:**
- In the future, the system can be upgraded to use a richer set of audio features, including MFCCs, spectral features, and more, for all songs.
- This would enable more nuanced and accurate recommendations based on timbre, production style, and other audio characteristics.
- To enable this, all songs in the database would need to have their audio features re-extracted using the latest feature extraction code and the actual audio files.

If you want to enable this in the future, plan for a database migration and batch audio processing step. 