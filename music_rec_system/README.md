# Music Recommendation System

An audio content-based music recommendation system that analyzes music features and suggests similar songs.

## Setup

### Download dataset from Kaggle
```
python3 kaggle_dataset.py
```

### Copy the path on your terminal and then paste it in the command below to get the dataset to your directory:
```
cp -r ${COPIED PATH} ~/music_rec_system/spotify_dataset
```

### Install dependencies
```
pip install kagglehub pandas numpy librosa scikit-learn pytube ffmpeg-python
```

### Optional FFmpeg Installation
For better audio conversion from YouTube downloads, install FFmpeg:
```
sudo apt-get install ffmpeg    # Debian/Ubuntu
# or
brew install ffmpeg            # macOS with Homebrew
```

## Usage

The system provides multiple ways to get song recommendations:

### Using the command-line interface

```bash
# Get recommendations for a song file using the hybrid method (default)
python3 recommend_songs.py --song "path/to/your/song.mp3" --method hybrid --count 5

# Search for a song in the dataset
python3 recommend_songs.py --search "song name or artist"

# Different recommendation methods
python3 recommend_songs.py --song "path/to/your/song.mp3" --method knn
python3 recommend_songs.py --song "path/to/your/song.mp3" --method cosine

# Auto-download song from YouTube if not found locally
python3 recommend_songs.py --song "song name not in dataset" --download

# Disable auto-download from YouTube
python3 recommend_songs.py --song "song name" --no-download
```

### Using the original script (legacy)
```
python3 music_rec_system.py
```

## Features

### Audio Analysis
The system extracts various audio features from songs including:

- **Tempo**: Speed or pace of a track
- **Energy**: Intensity and activity 
- **Acousticness**: Presence of acoustic elements
- **Danceability**: How suitable for dancing
- **Speechiness**: Presence of spoken words
- **Instrumentalness**: Lack of vocal content
- **Liveness**: Presence of audience in the recording
- **Valence**: Musical positiveness
- **Loudness**: Overall volume of the track

### YouTube Integration
- Automatic song download from YouTube when a requested song isn't found locally
- Audio extraction and conversion to MP3 format
- Feature extraction from downloaded audio

## Recommendation Methods

1. **K-Nearest Neighbors (KNN)**: Finds songs with the most similar audio features
2. **Cosine Similarity**: Measures angle between audio feature vectors
3. **Hybrid Method**: Combines KNN and cosine similarity for better recommendations

## How It Works

1. Audio features are extracted from songs using the librosa library
2. If a song isn't found locally, it's automatically downloaded from YouTube
3. Features are normalized using StandardScaler
4. Recommendation algorithms find songs with similar features
5. Results are ranked and returned based on similarity scores

## Contributing

Feel free to open issues or submit pull requests with improvements.
