import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
from gensim.models import Word2Vec, LdaModel
from gensim.corpora import Dictionary
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pickle
import gc
import os
import time
from scipy.sparse import csr_matrix, save_npz, load_npz

os.makedirs('cache', exist_ok=True)

nltk_data_path = os.path.expanduser('~/nltk_data')
if not os.path.exists(os.path.join(nltk_data_path, 'tokenizers/punkt')):
    nltk.download('punkt')
if not os.path.exists(os.path.join(nltk_data_path, 'corpora/stopwords')):
    nltk.download('stopwords')

# Initialize tokenizer and stop words
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))

def preprocess_lyrics(text):
    if not isinstance(text, str) or not text.strip():
        return ''
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)    #removes punctuation with spaces
    text = re.sub(r'\d+', ' ', text) #replaces number sequences with spaces
    return ' '.join(tokenizer.tokenize(text))

def fast_tokenize(text):
    if not isinstance(text, str) or not text.strip():
        return []
    return [w for w in tokenizer.tokenize(text.lower()) if w not in stop_words]

def batch_preprocess_lyrics(lyrics_list, batch_size=1000):
    results = []
    for i in range(0, len(lyrics_list), batch_size):
        batch = lyrics_list[i:i+batch_size]
        results.extend([preprocess_lyrics(text) for text in batch])
        gc.collect()  # Force garbage collection
    return results

def batch_tokenize(lyrics_list, batch_size=1000):
    results = []
    for i in range(0, len(lyrics_list), batch_size):
        batch = lyrics_list[i:i+batch_size]
        results.extend([fast_tokenize(text) for text in batch])
        gc.collect()  # Force garbage collection
    return results

def extract_sentiment_features(lyrics_list, cache_file='cache/sentiment_cache.pkl'):
    if os.path.exists(cache_file):
        print("Loading sentiment features from cache...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print("Extracting sentiment features (this may take a while)...")
    sid = SentimentIntensityAnalyzer()
    results = []

    for lyrics in tqdm(lyrics_list, desc="Processing sentiment"):
        if not isinstance(lyrics, str) or not lyrics.strip():
            results.append({
                'polarity': 0, 'subjectivity': 0.5,
                'compound': 0, 'neg': 0, 'neu': 1, 'pos': 0
            })
            continue
        
        blob = TextBlob(lyrics)
        vader = sid.polarity_scores(lyrics)
        results.append({
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity,
            **vader
        })

    sentiment_df = pd.DataFrame(results)
    
    with open(cache_file, 'wb') as f:
        pickle.dump(sentiment_df, f)
    
    return sentiment_df

def perform_topic_modeling(tokenized_lyrics, num_topics=10, cache_file='cache/lda_cache.pkl'):
    try:
        with open(cache_file, 'rb') as f:
            print("Loading topic modeling results from cache...")
            return pickle.load(f)
    except (FileNotFoundError, EOFError):
        print("Performing topic modeling...")
        dictionary = Dictionary(tokenized_lyrics)
        dictionary.filter_extremes(no_below=10, no_above=0.4)  # More aggressive filtering
        corpus = [dictionary.doc2bow(doc) for doc in tokenized_lyrics]
        
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,  
            passes=10, 
            alpha='auto',
            random_state=42
        )
        
        topic_distributions = []
        for doc in tqdm(corpus, desc="Extracting topics"):
            topics = lda_model.get_document_topics(doc, minimum_probability=0)
            topic_dist = np.zeros(num_topics)
            for topic_id, prob in topics:
                topic_dist[topic_id] = prob
            topic_distributions.append(topic_dist)
        
        topic_df = pd.DataFrame(
            topic_distributions,
            columns=[f'topic_{i}' for i in range(num_topics)]
        )
        
        topic_keywords = {
            f'topic_{topic_id}': [word for word, _ in lda_model.show_topic(topic_id, topn=10)]
            for topic_id in range(num_topics)
        }
        
        results = (topic_df, topic_keywords, lda_model, dictionary)
        with open(cache_file, 'wb') as f:
            pickle.dump(results, f)
        
        return results

def get_word2vec_embeddings(tokenized_lyrics, vector_size=100, cache_path='cache/word2vec_cache.npy'):
    try:
        print("Loading word embeddings from cache...")
        return np.load(cache_path)
    except (FileNotFoundError, EOFError):
        print("Generating word embeddings...")
        valid_lyrics = [tokens for tokens in tokenized_lyrics if tokens]
        
        if not valid_lyrics:
            return np.zeros((len(tokenized_lyrics), vector_size))
        
        model = Word2Vec(
            sentences=valid_lyrics,
            vector_size=vector_size, 
            window=5,
            min_count=3,  
            workers=min(8, os.cpu_count() or 4),  # Use more workers if available
            sg=1,
            epochs=5
        )
        
        doc_vectors = np.zeros((len(tokenized_lyrics), vector_size))
        for idx, tokens in enumerate(tqdm(tokenized_lyrics, desc="Creating document vectors")):
            if tokens:
                vectors = [model.wv[word] for word in tokens if word in model.wv]
                if vectors:
                    doc_vectors[idx] = np.mean(vectors, axis=0)
        
        np.save(cache_path, doc_vectors)
        return doc_vectors
    
class OptimizedSongRecommender:
    def __init__(self, df, word2vec_embeddings, audio_features, sentiment_features, topic_features=None, 
                 lyrics_weight=0.4, audio_weight=0.3, sentiment_weight=0.2, topic_weight=0.1):
        self.df = df
        self.word2vec_embeddings = word2vec_embeddings
        self.audio_features = audio_features
        self.sentiment_features = sentiment_features
        self.topic_features = topic_features if topic_features is not None else np.zeros((len(df), 1))
        
        self.lyrics_weight = lyrics_weight
        self.audio_weight = audio_weight
        self.sentiment_weight = sentiment_weight
        self.topic_weight = topic_weight
        
        self._prepare_features()
        
    def _prepare_features(self):
        self.word2vec_scaled = StandardScaler().fit_transform(self.word2vec_embeddings)
        
        self.audio_scaled = self.audio_features
        
        self.sentiment_scaled = StandardScaler().fit_transform(self.sentiment_features)
        
        if self.topic_features.shape[1] > 1:
            self.topic_scaled = StandardScaler().fit_transform(self.topic_features)
        else:
            self.topic_scaled = self.topic_features
            
        self.combined_features = np.hstack([
            self.lyrics_weight * self.word2vec_scaled,
            self.audio_weight * self.audio_scaled,
            self.sentiment_weight * self.sentiment_scaled,
            self.topic_weight * self.topic_scaled
        ])
        
        self.nn_model = NearestNeighbors(
            n_neighbors=50, 
            algorithm='auto', 
            metric='cosine'
        )
        self.nn_model.fit(self.combined_features)
    
    def get_recommendations(self, song_idx, top_n=5, include_explanation=True):
        distances, indices = self.nn_model.kneighbors(
            self.combined_features[song_idx].reshape(1, -1), 
            n_neighbors=top_n+1
        )
        
        similarities = 1 - distances[0]
        
        similar_indices = indices[0][1:]
        similarity_scores = similarities[1:]
        
        recommendations = self.df.iloc[similar_indices][
            ['track_name', 'track_artist', 'track_album_name', 'playlist_genre']
        ].copy()
        
        recommendations['similarity_score'] = similarity_scores
        
        nn_lyrics = NearestNeighbors(n_neighbors=top_n+1, algorithm='auto', metric='cosine')
        nn_lyrics.fit(self.word2vec_scaled)
        
        nn_audio = NearestNeighbors(n_neighbors=top_n+1, algorithm='auto', metric='cosine')
        nn_audio.fit(self.audio_scaled)
        
        lyrics_distances, _ = nn_lyrics.kneighbors(
            self.word2vec_scaled[song_idx].reshape(1, -1), n_neighbors=len(similar_indices)+1
        )
        audio_distances, _ = nn_audio.kneighbors(
            self.audio_scaled[song_idx].reshape(1, -1), n_neighbors=len(similar_indices)+1
        )
        
        lyrics_similarities = 1 - lyrics_distances[0]
        audio_similarities = 1 - audio_distances[0]
        
        lyrics_sim = np.array([
            lyrics_similarities[np.where(indices[0] == idx)[0][0]] if idx in indices[0] else 0
            for idx in similar_indices
        ])
        
        audio_sim = np.array([
            audio_similarities[np.where(indices[0] == idx)[0][0]] if idx in indices[0] else 0
            for idx in similar_indices
        ])
            
        recommendations['lyrics_similarity'] = lyrics_sim
        recommendations['audio_similarity'] = audio_sim
        
        if include_explanation:
            sentiment_vector = self.sentiment_scaled[song_idx].reshape(1, -1)
            topic_vector = self.topic_scaled[song_idx].reshape(1, -1)
            
            sentiment_sim = 1 - np.array([
                np.linalg.norm(sentiment_vector - self.sentiment_scaled[idx].reshape(1, -1))
                for idx in similar_indices
            ])
            
            topic_sim = 1 - np.array([
                np.linalg.norm(topic_vector - self.topic_scaled[idx].reshape(1, -1))
                for idx in similar_indices
            ])
            
            sentiment_sim = (sentiment_sim - np.min(sentiment_sim)) / (np.max(sentiment_sim) - np.min(sentiment_sim) + 1e-10)
            topic_sim = (topic_sim - np.min(topic_sim)) / (np.max(topic_sim) - np.min(topic_sim) + 1e-10)
            
            recommendations['sentiment_similarity'] = sentiment_sim
            recommendations['topic_similarity'] = topic_sim
            
            seed_song = self.df.iloc[song_idx][['track_name', 'track_artist']]
            seed_sentiment = {
                'polarity': self.df.iloc[song_idx].get('polarity', 0),
                'valence': self.df.iloc[song_idx].get('valence', 0),
                'energy': self.df.iloc[song_idx].get('energy', 0)
            }
            
            return recommendations, seed_song, seed_sentiment
        
        return recommendations
    
    def get_recommendations_by_name(self, song_name, artist_name=None, top_n=5):
        if artist_name:
            mask = (self.df['track_name'].str.lower() == song_name.lower()) & \
                   (self.df['track_artist'].str.lower() == artist_name.lower())
        else:
            mask = self.df['track_name'].str.lower() == song_name.lower()
        
        if not mask.any():
            print(f"Song '{song_name}' not found in the dataset.")
            return None
        
        song_idx = mask.idxmax()
        return self.get_recommendations(song_idx, top_n)
 
print("Starting song recommender optimization...")
start_time = time.time()
    
print("Loading dataset...")
df = pd.read_csv('./lyric_dataset/spotify_songs.csv')
    
print(f"Dataset loaded with {len(df)} songs. Processing...")
    
if os.path.exists('cache/preprocessed_df.pkl'):
    print("Loading preprocessed data from cache...")
    df = pd.read_pickle('cache/preprocessed_df.pkl')
else:
    print("Preprocessing lyrics...")
    df['clean_lyrics'] = batch_preprocess_lyrics(df['lyrics'].tolist())
    df['tokenized_lyrics'] = batch_tokenize(df['clean_lyrics'].tolist())
        
    df.to_pickle('cache/preprocessed_df.pkl')
    
sentiment_features = extract_sentiment_features(df['clean_lyrics'])
sentiment_array = sentiment_features.values
    
for col in sentiment_features.columns:
    df[col] = sentiment_features[col]
    
valid_lyrics_mask = df['tokenized_lyrics'].apply(len) > 0
valid_tokenized_lyrics = df.loc[valid_lyrics_mask, 'tokenized_lyrics'].tolist()
    
topic_array = np.zeros((len(df), 10))
    
if len(valid_tokenized_lyrics) > 10:
    topic_df, topic_keywords, _, _ = perform_topic_modeling(valid_tokenized_lyrics, num_topics=10)
        
    all_topic_df = pd.DataFrame(
        0.0,
        index=df.index,
        columns=[f'topic_{i}' for i in range(10)],
        dtype=np.float64
    )
        
    all_topic_df.loc[valid_lyrics_mask] = topic_df.values
    topic_array = all_topic_df.values
        
    for col in all_topic_df.columns:
        df[col] = all_topic_df[col]
    
word2vec_embeddings = get_word2vec_embeddings(df['tokenized_lyrics'].tolist())
    
print("Processing audio features...")
audio_features = df[[
        'danceability', 'energy', 'key', 'loudness', 'mode',
        'speechiness', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'tempo', 'duration_ms'
]].values
    
scaler = StandardScaler()
audio_features_scaled = scaler.fit_transform(audio_features)
    
print("Creating optimized recommender...")
recommender = OptimizedSongRecommender(
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
    
# example_song_idx = 767  
# recommendations, seed_song, seed_sentiment = recommender.get_recommendations(example_song_idx, top_n=5, include_explanation=True)
song_name = input("Enter song you want to look for recommendations for: ")
recommendations, seed_song, seed_sentiment = recommender.get_recommendations_by_name(song_name, top_n=5)

print(f"Recommendations for '{seed_song['track_name']}' by {seed_song['track_artist']}:")
print(recommendations[['track_name', 'track_artist', 'similarity_score', 'lyrics_similarity', 'audio_similarity']])
    
with open('cache/optimized_recommender.pkl', 'wb') as f:
    pickle.dump(recommender, f)
    
print(f"Optimization completed in {time.time() - start_time:.2f} seconds.")
    