# Recommendation logic (FAISS, ML)
# Hybrid recommendation engine combining audio features, lyrics, and sentiment

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import text
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

def get_similar_songs(engine, song_id, k=5, audio_weight=0.7, lyrics_weight=0.2, sentiment_weight=0.1):
    """
    Get similar songs using a hybrid approach with customizable weights.
    By default, uses 70% audio features, 20% lyrics, and 10% sentiment.
    """
    # Backward compatibility: if weights aren't provided, use only audio
    if audio_weight == 1.0 and lyrics_weight == 0.0 and sentiment_weight == 0.0:
        return get_similar_songs_audio_only(engine, song_id, k)
        
    # Fetch the reference song's features
    with engine.connect() as conn:
        ref_row = conn.execute(text("""
            SELECT id, audio_features, lyrics, sentiment_features
            FROM songs WHERE id = :id
        """), {"id": song_id}).fetchone()
        
        if not ref_row:
            print("[DEBUG] Reference song not found in DB.")
            return []
            
        # Get audio features
        audio_dict = ref_row.audio_features
        if isinstance(audio_dict, str):
            audio_dict = json.loads(audio_dict)
            
        # Get lyric features
        ref_lyrics = ref_row.lyrics or ""
        
        # Get sentiment features
        ref_sentiment = ref_row.sentiment_features
        if ref_sentiment and isinstance(ref_sentiment, str):
            ref_sentiment = json.loads(ref_sentiment)
        elif not ref_sentiment:
            # If no sentiment stored, calculate it now
            ref_sentiment = analyze_sentiment(ref_lyrics) if ref_lyrics else {}
            
        # Fetch all other songs' features
        rows = conn.execute(text("""
            SELECT id, track_name, track_artist, audio_features, lyrics, sentiment_features
            FROM songs WHERE id != :id
        """), {"id": song_id}).fetchall()
        
        if not rows:
            print("[DEBUG] No other songs in DB.")
            return []
            
        # Prepare audio features (flatten as before)
        ref_audio_vec = flatten_features(audio_dict)
        ref_audio_features = np.array(ref_audio_vec, dtype=np.float32).reshape(1, -1)
        ref_audio_len = ref_audio_features.shape[1]
        
        # Process candidates
        ids, meta, audio_scores, lyrics_scores, sentiment_scores = [], [], [], [], []
        skipped = 0
        
        for row in rows:
            try:
                # Process audio features
                audio_dict = row.audio_features
                if isinstance(audio_dict, str):
                    audio_dict = json.loads(audio_dict)
                audio_vec = flatten_features(audio_dict)
                
                # Skip if audio feature length doesn't match
                if len(audio_vec) != ref_audio_len:
                    print(f"[DEBUG] Skipping song ID {row.id} due to feature mismatch")
                    skipped += 1
                    continue
                
                # Process lyrics
                row_lyrics = row.lyrics or ""
                
                # Process sentiment
                row_sentiment = row.sentiment_features
                if row_sentiment and isinstance(row_sentiment, str):
                    row_sentiment = json.loads(row_sentiment)
                elif not row_sentiment:
                    # If no sentiment stored, calculate it now
                    row_sentiment = analyze_sentiment(row_lyrics) if row_lyrics else {}
                
                # Calculate similarity scores
                
                # 1. Audio similarity
                audio_feat = np.array(audio_vec, dtype=np.float32).reshape(1, -1)
                audio_sim = float(cosine_similarity(ref_audio_features, audio_feat)[0][0])
                
                # 2. Lyrics similarity
                lyrics_sim = 0.0
                if ref_lyrics and row_lyrics:
                    lyrics_sim = compute_lyrics_similarity(ref_lyrics, row_lyrics)
                
                # 3. Sentiment similarity
                sentiment_sim = 0.0
                if ref_sentiment and row_sentiment:
                    sentiment_sim = compute_sentiment_similarity(ref_sentiment, row_sentiment)
                
                # Store data for final ranking
                ids.append(row.id)
                meta.append((row.track_name, row.track_artist))
                audio_scores.append(audio_sim)
                lyrics_scores.append(lyrics_sim)
                sentiment_scores.append(sentiment_sim)
                
            except Exception as e:
                print(f"[DEBUG] Error processing song ID {row.id}: {e}")
                continue
        
        print(f"[DEBUG] Candidates after filtering: {len(ids)} (skipped {skipped})")
        if not ids:
            print("[DEBUG] No candidates with matching features.")
            return []
            
        # Calculate weighted combined score
        weighted_scores = []
        for i in range(len(ids)):
            combined_score = (
                audio_weight * audio_scores[i] +
                lyrics_weight * lyrics_scores[i] +
                sentiment_weight * sentiment_scores[i]
            )
            weighted_scores.append(combined_score)
        
        # Get top k
        top_k_idx = np.argsort(weighted_scores)[-k:][::-1]
        
        # Return results with component scores for transparency
        results = []
        for i in top_k_idx:
            results.append({
                "id": ids[i],
                "track_name": meta[i][0],
                "track_artist": meta[i][1],
                "score": weighted_scores[i],
                "audio_score": audio_scores[i],
                "lyrics_score": lyrics_scores[i],
                "sentiment_score": sentiment_scores[i]
            })
        
        print(f"[DEBUG] Returning {len(results)} recommendations.")
        # Format for backward compatibility
        return [(r["id"], r["track_name"], r["track_artist"], r["score"]) for r in results]

def get_similar_songs_audio_only(engine, song_id, k=5):
    """Original audio-only recommendation function (for backward compatibility)"""
    # Fetch the reference song's audio features
    with engine.connect() as conn:
        ref_row = conn.execute(text("SELECT id, audio_features FROM songs WHERE id = :id"), {"id": song_id}).fetchone()
        if not ref_row:
            print("[DEBUG] Reference song not found in DB.")
            return []
        audio_dict = ref_row.audio_features
        if isinstance(audio_dict, str):
            audio_dict = json.loads(audio_dict)
        # Flatten features
        ref_vec = flatten_features(audio_dict)
        ref_features = np.array(ref_vec, dtype=np.float32).reshape(1, -1)
        ref_len = ref_features.shape[1]
        print(f"[DEBUG] Reference song feature vector length: {ref_len}")
        # Fetch all other songs' ids and features
        rows = conn.execute(text("SELECT id, track_name, track_artist, audio_features FROM songs WHERE id != :id"), {"id": song_id}).fetchall()
        if not rows:
            print("[DEBUG] No other songs in DB.")
            return []
        ids = []
        features = []
        meta = []
        skipped = 0
        for row in rows:
            try:
                audio_dict = row.audio_features
                if isinstance(audio_dict, str):
                    audio_dict = json.loads(audio_dict)
                vec = flatten_features(audio_dict)
                if len(vec) != ref_len:
                    print(f"[DEBUG] Skipping song ID {row.id} ({row.track_name}) due to feature length mismatch: {len(vec)} vs {ref_len}")
                    skipped += 1
                    continue  # skip songs with mismatched feature length
                features.append(vec)
                ids.append(row.id)
                meta.append((row.track_name, row.track_artist))
            except Exception as e:
                print(f"[DEBUG] Error processing song ID {row.id}: {e}")
                continue
        print(f"[DEBUG] Candidates after filtering: {len(features)} (skipped {skipped})")
        if not features:
            print("[DEBUG] No candidates with matching feature length.")
            return []
        features = np.stack(features)
        sims = cosine_similarity(ref_features, features)[0]
        top_k_idx = np.argsort(sims)[-k:][::-1]
        print(f"[DEBUG] Returning {len(top_k_idx)} recommendations.")
        return [(ids[i], meta[i][0], meta[i][1], float(sims[i])) for i in top_k_idx]

def get_similar_songs_for_features(engine, features, k=5, audio_weight=0.7, lyrics_weight=0.2, sentiment_weight=0.1, lyrics=None, sentiment=None):
    """
    Get similar songs for external features (e.g., from YouTube) using hybrid approach
    """
    # If no lyrics or sentiment provided, fall back to audio-only
    if (lyrics_weight > 0 and not lyrics) or (sentiment_weight > 0 and not sentiment):
        # Revert to audio-only if we don't have all the required data
        audio_weight = 1.0
        lyrics_weight = 0.0
        sentiment_weight = 0.0
    
    # If audio only, use the original function
    if audio_weight == 1.0 and lyrics_weight == 0.0 and sentiment_weight == 0.0:
        return get_similar_songs_for_features_audio_only(engine, features, k)
    
    # Process input features
    ref_vec = flatten_features(features)
    ref_features = np.array(ref_vec, dtype=np.float32).reshape(1, -1)
    ref_len = ref_features.shape[1]
    
    # Calculate sentiment if provided
    ref_sentiment = analyze_sentiment(lyrics) if lyrics and sentiment_weight > 0 else {}
    
    # Fetch all songs from database
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, track_name, track_artist, audio_features, lyrics, sentiment_features
            FROM songs
        """)).fetchall()
    
    # Process candidates
    ids, names, artists, audio_scores, lyrics_scores, sentiment_scores = [], [], [], [], [], []
    skipped = 0
    
    for row in rows:
        try:
            # Process audio features
            audio_dict = row.audio_features
            if isinstance(audio_dict, str):
                audio_dict = json.loads(audio_dict)
            vec = flatten_features(audio_dict)
            
            # Skip if audio feature length doesn't match
            if len(vec) != ref_len:
                print(f"[DEBUG] Skipping song ID {row.id} due to feature mismatch")
                skipped += 1
                continue
                
            # Process lyrics
            row_lyrics = row.lyrics or ""
            
            # Process sentiment
            row_sentiment = row.sentiment_features
            if row_sentiment and isinstance(row_sentiment, str):
                row_sentiment = json.loads(row_sentiment)
            elif not row_sentiment:
                # If no sentiment stored, calculate it now (if we have lyrics)
                row_sentiment = analyze_sentiment(row_lyrics) if row_lyrics else {}
            
            # Calculate similarity scores
            
            # 1. Audio similarity
            audio_feat = np.array(vec, dtype=np.float32).reshape(1, -1)
            audio_sim = float(cosine_similarity(ref_features, audio_feat)[0][0])
            
            # 2. Lyrics similarity
            lyrics_sim = 0.0
            if lyrics and row_lyrics and lyrics_weight > 0:
                lyrics_sim = compute_lyrics_similarity(lyrics, row_lyrics)
            
            # 3. Sentiment similarity
            sentiment_sim = 0.0
            if ref_sentiment and row_sentiment and sentiment_weight > 0:
                sentiment_sim = compute_sentiment_similarity(ref_sentiment, row_sentiment)
            
            # Store all data
            ids.append(row.id)
            names.append(row.track_name)
            artists.append(row.track_artist)
            audio_scores.append(audio_sim)
            lyrics_scores.append(lyrics_sim)
            sentiment_scores.append(sentiment_sim)
            
        except Exception as e:
            print(f"[DEBUG] Error processing song ID {row.id}: {e}")
            continue
    
    print(f"[DEBUG] Candidates after filtering: {len(ids)} (skipped {skipped})")
    if not ids:
        print("[DEBUG] No candidates with matching feature length.")
        return []
    
    # Calculate weighted scores
    weighted_scores = []
    for i in range(len(ids)):
        combined_score = (
            audio_weight * audio_scores[i] +
            lyrics_weight * lyrics_scores[i] +
            sentiment_weight * sentiment_scores[i]
        )
        weighted_scores.append(combined_score)
    
    # Get top k
    top_idx = np.argsort(weighted_scores)[::-1][:k]
    
    # Format results for backward compatibility
    return [(ids[i], names[i], artists[i], weighted_scores[i]) for i in top_idx]

def get_similar_songs_for_features_audio_only(engine, features, k=5):
    """Original audio-only version for backward compatibility"""
    def flatten_features(d):
        vec = []
        for v in d.values():
            if isinstance(v, list):
                vec.extend(v)
            else:
                vec.append(v)
        return vec
    ref_vec = flatten_features(features)
    ref_features = np.array(ref_vec, dtype=np.float32).reshape(1, -1)
    ref_len = ref_features.shape[1]
    print(f"[DEBUG] Reference (YouTube) song feature vector length: {ref_len}")
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT id, track_name, track_artist, audio_features FROM songs")).fetchall()
    ids, names, artists, feats = [], [], [], []
    skipped = 0
    for row in rows:
        try:
            audio_dict = row.audio_features
            if isinstance(audio_dict, str):
                audio_dict = json.loads(audio_dict)
            vec = flatten_features(audio_dict)
            if len(vec) != ref_len:
                print(f"[DEBUG] Skipping song ID {row.id} ({row.track_name}) due to feature length mismatch: {len(vec)} vs {ref_len}")
                skipped += 1
                continue  # skip songs with mismatched feature length
            feats.append(vec)
            ids.append(row.id)
            names.append(row.track_name)
            artists.append(row.track_artist)
        except Exception as e:
            print(f"[DEBUG] Error processing song ID {row.id}: {e}")
            continue
    print(f"[DEBUG] Candidates after filtering: {len(feats)} (skipped {skipped})")
    if not feats:
        print("[DEBUG] No candidates with matching feature length.")
        return []
    feats = np.stack(feats)
    sims = cosine_similarity(ref_features, feats)[0]
    top_idx = np.argsort(sims)[::-1][:k]
    print(f"[DEBUG] Returning {len(top_idx)} recommendations.")
    return [(ids[i], names[i], artists[i], sims[i]) for i in top_idx]

def flatten_features(d):
    """Helper function to flatten feature dictionaries into vectors"""
    vec = []
    for v in d.values():
        if isinstance(v, list):
            vec.extend(v)
        else:
            vec.append(v)
    return vec

def compute_lyrics_similarity(lyrics1, lyrics2):
    """Compute similarity between two lyrics texts"""
    if not lyrics1 or not lyrics2:
        return 0.0
    
    # Clean and preprocess lyrics
    def preprocess_lyrics(text):
        text = text.lower()
        text = re.sub(r'\[.*?\]', '', text)  # Remove [Verse], [Chorus], etc.
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text
    
    lyrics1_clean = preprocess_lyrics(lyrics1)
    lyrics2_clean = preprocess_lyrics(lyrics2)
    
    # Convert to bag of words vectors using cosine similarity
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([lyrics1_clean, lyrics2_clean])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity)
    except Exception as e:
        print(f"[DEBUG] Error computing lyrics similarity: {e}")
        return 0.0

def compute_sentiment_similarity(sentiment1, sentiment2):
    """Compute similarity between two sentiment dictionaries"""
    if not sentiment1 or not sentiment2:
        return 0.0
    
    try:
        # Extract common keys
        common_keys = set(sentiment1.keys()) & set(sentiment2.keys())
        if not common_keys:
            return 0.0
            
        # Create vectors from the common keys
        vec1 = [sentiment1[k] for k in common_keys]
        vec2 = [sentiment2[k] for k in common_keys]
        
        # Calculate cosine similarity
        a = np.array(vec1).reshape(1, -1)
        b = np.array(vec2).reshape(1, -1)
        return float(cosine_similarity(a, b)[0][0])
    except Exception as e:
        print(f"[DEBUG] Error computing sentiment similarity: {e}")
        return 0.0

def analyze_sentiment(lyrics):
    """Analyze sentiment of lyrics text"""
    if not lyrics:
        return {}
        
    try:
        analyzer = SentimentIntensityAnalyzer()
        sentiment = analyzer.polarity_scores(lyrics)
        return sentiment
    except Exception as e:
        print(f"[DEBUG] Error analyzing sentiment: {e}")
        return {} 