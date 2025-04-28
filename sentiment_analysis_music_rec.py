import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from gensim.models import Word2Vec, LdaModel
from gensim.corpora import Dictionary
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load your dataset
df = pd.read_csv('your_spotify_data.csv')

# Preprocessing function for lyrics
def preprocess_lyrics(text):
    if not isinstance(text, str) or text.strip() == '':
        return ''
    
    # Convert to lowercase and remove special characters
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply cleaning function
df['clean_lyrics'] = df['lyrics'].apply(preprocess_lyrics)

# Tokenize lyrics
stop_words = set(stopwords.words('english'))
df['tokenized_lyrics'] = df['clean_lyrics'].apply(
    lambda x: [word for word in word_tokenize(x) if word not in stop_words] if isinstance(x, str) and x.strip() != '' else []
)

# 1. SENTIMENT ANALYSIS
# Function to extract sentiment features from lyrics
def extract_sentiment_features(lyrics_list):
    # Initialize sentiment analyzer
    sid = SentimentIntensityAnalyzer()
    textblob_sentiments = []
    vader_sentiments = []
    
    for lyrics in lyrics_list:
        if not isinstance(lyrics, str) or lyrics.strip() == '':
            # Default neutral values for empty lyrics
            textblob_sentiments.append({'polarity': 0, 'subjectivity': 0.5})
            vader_sentiments.append({
                'compound': 0, 'neg': 0, 'neu': 1, 'pos': 0
            })
            continue
            
        # TextBlob sentiment analysis
        blob = TextBlob(lyrics)
        textblob_sentiments.append({
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        })
        
        # VADER sentiment analysis
        vader_sentiment = sid.polarity_scores(lyrics)
        vader_sentiments.append(vader_sentiment)
    
    # Create sentiment dataframes
    tb_df = pd.DataFrame(textblob_sentiments)
    vader_df = pd.DataFrame(vader_sentiments)
    
    # Rename columns to avoid confusion
    tb_df.columns = ['tb_' + col for col in tb_df.columns]
    vader_df.columns = ['vader_' + col for col in vader_df.columns]
    
    # Combine all sentiment features
    sentiment_df = pd.concat([tb_df, vader_df], axis=1)
    return sentiment_df

# Extract sentiment features
sentiment_features = extract_sentiment_features(df['clean_lyrics'])
df = pd.concat([df, sentiment_features], axis=1)

# 2. TOPIC MODELING
# Function to perform topic modeling on lyrics
def perform_topic_modeling(tokenized_lyrics, num_topics=10):
    # Create dictionary
    dictionary = Dictionary(tokenized_lyrics)
    
    # Filter extreme values
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    
    # Create corpus
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_lyrics]
    
    # Train LDA model
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=10,
        alpha='auto',
        random_state=42
    )
    
    # Extract topic distributions for each document
    topic_distributions = []
    for doc in corpus:
        topics = lda_model.get_document_topics(doc, minimum_probability=0)
        # Convert to dense representation
        topic_dist = np.zeros(num_topics)
        for topic_id, prob in topics:
            topic_dist[topic_id] = prob
        topic_distributions.append(topic_dist)
    
    # Create topic distribution dataframe
    topic_df = pd.DataFrame(
        topic_distributions,
        columns=[f'topic_{i}' for i in range(num_topics)]
    )
    
    # Get topic keywords for interpretation
    topic_keywords = {}
    for topic_id in range(num_topics):
        keywords = lda_model.show_topic(topic_id, topn=10)
        topic_keywords[f'topic_{topic_id}'] = [word for word, _ in keywords]
    
    return topic_df, topic_keywords, lda_model, dictionary

# Filter out rows with empty tokenized lyrics for topic modeling
valid_lyrics_mask = df['tokenized_lyrics'].apply(lambda x: len(x) > 0)
valid_tokenized_lyrics = df.loc[valid_lyrics_mask, 'tokenized_lyrics'].tolist()

# Perform topic modeling if there are enough valid lyrics
if len(valid_tokenized_lyrics) > 10:  # Arbitrary threshold
    topic_df, topic_keywords, lda_model, dictionary = perform_topic_modeling(
        valid_tokenized_lyrics, num_topics=10
    )
    
    # Add topic distributions to the original dataframe
    # First, create a dataframe of the same length as df with zeros
    all_topic_df = pd.DataFrame(
        0, 
        index=df.index, 
        columns=[f'topic_{i}' for i in range(10)]
    )
    
    # Then fill in the values for rows with valid lyrics
    all_topic_df.loc[valid_lyrics_mask] = topic_df.values
    
    # Concatenate with the main dataframe
    df = pd.concat([df, all_topic_df], axis=1)
    
    # Print topic keywords for interpretation
    print("Topic Keywords:")
    for topic, keywords in topic_keywords.items():
        print(f"{topic}: {', '.join(keywords)}")
else:
    print("Not enough valid lyrics for topic modeling")

# 3. WORD2VEC EMBEDDINGS
# Function to create Word2Vec embeddings
def get_word2vec_embeddings(tokenized_lyrics, vector_size=100):
    # Filter out empty lists
    valid_lyrics = [tokens for tokens in tokenized_lyrics if tokens]
    
    if not valid_lyrics:
        return np.zeros((len(tokenized_lyrics), vector_size))
    
    # Train Word2Vec model
    model = Word2Vec(
        sentences=valid_lyrics,
        vector_size=vector_size,
        window=5,
        min_count=1,
        workers=4,
        sg=1  # Skip-gram model
    )
    
    # Create document vectors by averaging word vectors for each song
    doc_vectors = []
    for tokens in tokenized_lyrics:
        if tokens:
            vectors = [model.wv[word] for word in tokens if word in model.wv]
            if vectors:
                doc_vector = np.mean(vectors, axis=0)
            else:
                doc_vector = np.zeros(vector_size)
        else:
            doc_vector = np.zeros(vector_size)
        doc_vectors.append(doc_vector)
    
    return np.array(doc_vectors)

# Get word embeddings
word2vec_embeddings = get_word2vec_embeddings(df['tokenized_lyrics'].tolist())

# 4. AUDIO FEATURES PROCESSING
# Extract audio features
audio_features = df[[
    'danceability', 'energy', 'key', 'loudness', 'mode', 
    'speechiness', 'acousticness', 'instrumentalness', 
    'liveness', 'valence', 'tempo', 'duration_ms'
]].values

# Normalize audio features
scaler = StandardScaler()
audio_features_scaled = scaler.fit_transform(audio_features)

# 5. CREATING COMPREHENSIVE FEATURE SET
# Extract sentiment features as numpy array
sentiment_cols = sentiment_features.columns.tolist()
sentiment_array = df[sentiment_cols].values

# Extract topic features if they exist
if 'topic_0' in df.columns:
    topic_cols = [col for col in df.columns if col.startswith('topic_')]
    topic_array = df[topic_cols].values
else:
    topic_array = np.zeros((len(df), 1))  # Placeholder

# Create a comprehensive song recommender class
class ComprehensiveSongRecommender:
    def __init__(self, df, word2vec_embeddings, audio_features, sentiment_features, topic_features=None, 
                 lyrics_weight=0.4, audio_weight=0.3, sentiment_weight=0.2, topic_weight=0.1):
        self.df = df
        self.word2vec_embeddings = word2vec_embeddings
        self.audio_features = audio_features
        self.sentiment_features = sentiment_features
        self.topic_features = topic_features if topic_features is not None else np.zeros((len(df), 1))
        
        # Store weights
        self.lyrics_weight = lyrics_weight
        self.audio_weight = audio_weight
        self.sentiment_weight = sentiment_weight
        self.topic_weight = topic_weight
        
        # Normalize and standardize all feature sets
        self._prepare_features()
        
    def _prepare_features(self):
        # Standardize embeddings
        self.word2vec_scaled = StandardScaler().fit_transform(self.word2vec_embeddings)
        
        # Audio features are already scaled
        self.audio_scaled = self.audio_features
        
        # Standardize sentiment features
        self.sentiment_scaled = StandardScaler().fit_transform(self.sentiment_features)
        
        # Standardize topic features if they exist
        if self.topic_features.shape[1] > 1:
            self.topic_scaled = StandardScaler().fit_transform(self.topic_features)
        else:
            self.topic_scaled = self.topic_features
            
        # Optional: Apply dimensionality reduction to word embeddings if needed
        if self.word2vec_scaled.shape[1] > 50:
            pca = PCA(n_components=50)
            self.word2vec_scaled = pca.fit_transform(self.word2vec_scaled)
            
        # Combine all features with respective weights
        self.combined_features = np.hstack([
            self.lyrics_weight * self.word2vec_scaled,
            self.audio_weight * self.audio_scaled,
            self.sentiment_weight * self.sentiment_scaled,
            self.topic_weight * self.topic_scaled
        ])
    
    def get_recommendations(self, song_idx, top_n=5, include_explanation=False):
        """Get song recommendations based on a seed song"""
        # Calculate similarity
        song_vector = self.combined_features[song_idx].reshape(1, -1)
        similarity_scores = cosine_similarity(song_vector, self.combined_features)[0]
        
        # Get top similar songs (excluding the seed song itself)
        similar_indices = similarity_scores.argsort()[::-1][1:top_n+1]
        
        # Create recommendations dataframe
        recommendations = self.df.iloc[similar_indices][
            ['track_name', 'track_artist', 'track_album_name', 'playlist_genre']
        ].copy()
        
        recommendations['similarity_score'] = similarity_scores[similar_indices]
        
        if include_explanation:
            # Calculate component-wise similarity
            lyrics_sim = cosine_similarity(
                self.word2vec_scaled[song_idx].reshape(1, -1), 
                self.word2vec_scaled[similar_indices]
            )[0]
            
            audio_sim = cosine_similarity(
                self.audio_scaled[song_idx].reshape(1, -1), 
                self.audio_scaled[similar_indices]
            )[0]
            
            sentiment_sim = cosine_similarity(
                self.sentiment_scaled[song_idx].reshape(1, -1), 
                self.sentiment_scaled[similar_indices]
            )[0]
            
            topic_sim = cosine_similarity(
                self.topic_scaled[song_idx].reshape(1, -1), 
                self.topic_scaled[similar_indices]
            )[0]
            
            # Add component similarities to recommendations
            recommendations['lyrics_similarity'] = lyrics_sim
            recommendations['audio_similarity'] = audio_sim
            recommendations['sentiment_similarity'] = sentiment_sim
            recommendations['topic_similarity'] = topic_sim
            
            # Add seed song info for reference
            seed_song = self.df.iloc[song_idx][['track_name', 'track_artist']]
            seed_sentiment = {
                'polarity': self.df.iloc[song_idx].get('tb_polarity', 0),
                'valence': self.df.iloc[song_idx].get('valence', 0),
                'energy': self.df.iloc[song_idx].get('energy', 0)
            }
            
            return recommendations, seed_song, seed_sentiment
        
        return recommendations
    
    def get_recommendations_by_name(self, song_name, artist_name=None, top_n=5, include_explanation=False):
        """Get recommendations by song name and optional artist name"""
        # Find the song in the dataframe
        if artist_name:
            mask = (self.df['track_name'].str.lower() == song_name.lower()) & \
                   (self.df['track_artist'].str.lower() == artist_name.lower())
        else:
            mask = self.df['track_name'].str.lower() == song_name.lower()
        
        if not mask.any():
            print(f"Song '{song_name}' not found in the dataset.")
            return None
        
        song_idx = mask.idxmax()
        return self.get_recommendations(song_idx, top_n, include_explanation)
    
    def adjust_weights(self, lyrics_weight=None, audio_weight=None, sentiment_weight=None, topic_weight=None):
        """Adjust feature weights and recalculate combined features"""
        # Update weights if provided
        if lyrics_weight is not None:
            self.lyrics_weight = lyrics_weight
        if audio_weight is not None:
            self.audio_weight = audio_weight
        if sentiment_weight is not None:
            self.sentiment_weight = sentiment_weight
        if topic_weight is not None:
            self.topic_weight = topic_weight
            
        # Normalize weights to sum to 1
        total_weight = self.lyrics_weight + self.audio_weight + self.sentiment_weight + self.topic_weight
        self.lyrics_weight /= total_weight
        self.audio_weight /= total_weight
        self.sentiment_weight /= total_weight
        self.topic_weight /= total_weight
        
        # Recalculate combined features
        self._prepare_features()
        
        print(f"Weights adjusted to: Lyrics={self.lyrics_weight:.2f}, Audio={self.audio_weight:.2f}, "
              f"Sentiment={self.sentiment_weight:.2f}, Topic={self.topic_weight:.2f}")
        
    def get_emotional_recommendations(self, mood='happy', top_n=10):
        """Get recommendations based on emotional/mood criteria"""
        # Define mood profiles (simplified version)
        mood_profiles = {
            'happy': {'valence': 0.8, 'energy': 0.7, 'vader_pos': 0.6},
            'sad': {'valence': 0.3, 'energy': 0.4, 'vader_neg': 0.5},
            'energetic': {'energy': 0.8, 'tempo': 0.7, 'valence': 0.6},
            'calm': {'energy': 0.3, 'acousticness': 0.7, 'speechiness': 0.3},
            'angry': {'energy': 0.8, 'valence': 0.3, 'vader_neg': 0.6}
        }
        
        if mood not in mood_profiles:
            print(f"Mood '{mood}' not recognized. Available moods: {list(mood_profiles.keys())}")
            return None
        
        # Create a mood vector
        profile = mood_profiles[mood]
        mood_scores = []
        
        for idx in range(len(self.df)):
            score = 0
            total_weight = 0
            
            for feature, target_value in profile.items():
                if feature in self.df.columns:
                    # Normalize the feature value to [0,1] range if needed
                    value = self.df.iloc[idx][feature]
                    if feature == 'tempo':  # Special handling for tempo
                        value = min(value / 200, 1)  # Assuming 200 BPM is the upper limit
                    
                    # Calculate proximity to target value (inverted distance)
                    proximity = 1 - abs(value - target_value)
                    score += proximity
                    total_weight += 1
            
            if total_weight > 0:
                mood_scores.append(score / total_weight)
            else:
                mood_scores.append(0)
        
        # Get top matching songs
        top_indices = np.argsort(mood_scores)[::-1][:top_n]
        
        # Create recommendations dataframe
        recommendations = self.df.iloc[top_indices][
            ['track_name', 'track_artist', 'track_album_name', 'playlist_genre']
        ].copy()
        
        recommendations['mood_match_score'] = [mood_scores[i] for i in top_indices]
        
        return recommendations
    
    def find_similar_lyrics_different_genre(self, song_idx, top_n=5):
        """Find songs with similar lyrics but from different genres"""
        # Get the genre of the seed song
        seed_genre = self.df.iloc[song_idx]['playlist_genre']
        
        # Calculate lyrical similarity
        song_vector = self.word2vec_scaled[song_idx].reshape(1, -1)
        lyrics_similarity = cosine_similarity(song_vector, self.word2vec_scaled)[0]
        
        # Create a mask for songs from different genres
        different_genre_mask = self.df['playlist_genre'] != seed_genre
        
        # Apply mask and get top similar songs
        lyrics_similarity_filtered = lyrics_similarity.copy()
        lyrics_similarity_filtered[~different_genre_mask] = -1  # Set same genre songs to low similarity
        
        # Get top indices (excluding the seed song)
        top_indices = lyrics_similarity_filtered.argsort()[::-1][:top_n]
        
        # Create recommendations dataframe
        recommendations = self.df.iloc[top_indices][
            ['track_name', 'track_artist', 'track_album_name', 'playlist_genre']
        ].copy()
        
        recommendations['lyrics_similarity'] = lyrics_similarity[top_indices]
        
        return recommendations

# Initialize the comprehensive recommender
recommender = ComprehensiveSongRecommender(
    df=df,
    word2vec_embeddings=word2vec_embeddings,
    audio_features=audio_features_scaled,
    sentiment_features=sentiment_array,
    topic_features=topic_array,
    lyrics_weight=0.4,
    audio_weight=0.3,
    sentiment_weight=0.2,
    topic_weight=0.1
)

# Example usage
example_song_idx = 0  # Replace with a valid index from your dataset
recommendations, seed_song, seed_sentiment = recommender.get_recommendations(
    example_song_idx, top_n=5, include_explanation=True
)

print(f"Recommendations for '{seed_song['track_name']}' by {seed_song['track_artist']}:")
print(recommendations[['track_name', 'track_artist', 'similarity_score', 'lyrics_similarity', 'sentiment_similarity']])

# Get mood-based recommendations
happy_songs = recommender.get_emotional_recommendations(mood='happy', top_n=5)
print("\nHappy Songs:")
print(happy_songs[['track_name', 'track_artist', 'mood_match_score']])

# Find songs with similar lyrics from different genres
cross_genre = recommender.find_similar_lyrics_different_genre(example_song_idx, top_n=5)
print("\nSongs with similar lyrics from different genres:")
print(cross_genre[['track_name', 'track_artist', 'playlist_genre', 'lyrics_similarity']])

# Save recommendations to CSV
recommendations.to_csv('enhanced_song_recommendations.csv', index=False)
