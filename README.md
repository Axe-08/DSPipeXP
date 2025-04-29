# Music Recommendation System

A comprehensive music recommendation system that uses lyrics, audio features, sentiment analysis, and topic modeling to provide personalized song recommendations.

## Features

- Lyric-based similarity using Word2Vec embeddings
- Audio feature analysis
- Sentiment analysis using VADER and TextBlob
- Topic modeling using LDA
- Cross-genre recommendations
- Mood-based song suggestions

## Project Structure

```
.
├── src/                    # Source code
│   └── sentiment_analysis_music_rec.py
├── data/                   # Dataset files
│   └── spotify_songs.csv
├── scripts/               # Utility scripts
│   └── lyric_dataset_download.py
├── tests/                 # Test files
├── backup/                # Backup of unused files
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
```bash
python scripts/lyric_dataset_download.py
```

## Usage

The main recommendation system can be used in several ways:

1. Get recommendations based on a song:
```python
recommendations = recommender.get_recommendations_by_name(
    song_name="Your Song Name",
    artist_name="Artist Name",
    top_n=5
)
```

2. Get mood-based recommendations:
```python
happy_songs = recommender.get_emotional_recommendations(
    mood='happy',
    top_n=5
)
```

3. Find similar songs from different genres:
```python
cross_genre = recommender.find_similar_lyrics_different_genre(
    song_idx=0,
    top_n=5
)
```

## Customization

You can adjust the weights of different features using:
```python
recommender.adjust_weights(
    lyrics_weight=0.4,
    audio_weight=0.3,
    sentiment_weight=0.2,
    topic_weight=0.1
)
```

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License

MIT License
