# Recommendation logic (FAISS, ML)
# Hybrid recommendation engine combining audio features, lyrics, and sentiment

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import text
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import logging
import time
import math
from functools import lru_cache

# Configure logging
logger = logging.getLogger(__name__)

# Add minimum thresholds for similarity scores
# Minimum audio similarity threshold - main filter (LOWERED FOR TESTING)
MIN_AUDIO_SIM = 0.15
# YouTube-specific audio similarity threshold (lower to account for
# YouTube audio quality)
MIN_YT_AUDIO_SIM = 0.12
MIN_LYRICS_SIM = 0.1  # Minimum lyrics similarity threshold
MIN_SENTIMENT_SIM = 0.1  # Minimum sentiment similarity threshold
# Minimum combined similarity score (LOWERED FOR TESTING)
MIN_COMBINED_SIM = 0.15


def safe_compare(a, b, operator="<"):
    """
    Safely compare two values which might be of different types.
    Handles common type mismatches like tuple vs float.

    Args:
        a: First value to compare
        b: Second value to compare
        operator: String representing the comparison operator ("<", "<=", ">", ">=", "==")

    Returns:
        Boolean result of the comparison
    """
    # Extract first element if tuple
    if isinstance(a, tuple) and len(a) > 0:
        a = a[0]
    if isinstance(b, tuple) and len(b) > 0:
        b = b[0]

    # Try to convert to float
    try:
        a_val = float(a)
        b_val = float(b)

        if operator == "<":
            return a_val < b_val
        elif operator == "<=":
            return a_val <= b_val
        elif operator == ">":
            return a_val > b_val
        elif operator == ">=":
            return a_val >= b_val
        elif operator == "==":
            return a_val == b_val
        else:
            raise ValueError(f"Unsupported operator: {operator}")
    except (ValueError, TypeError):
        # If conversion fails, convert to string and compare lexicographically
        a_str = str(a)
        b_str = str(b)

        if operator == "<":
            return a_str < b_str
        elif operator == "<=":
            return a_str <= b_str
        elif operator == ">":
            return a_str > b_str
        elif operator == ">=":
            return a_str >= b_str
        elif operator == "==":
            return a_str == b_str
        else:
            raise ValueError(f"Unsupported operator: {operator}")


def safe_float(value, default=0.0):
    """
    Safely convert a value to float, handling various types:
    - If value is a tuple, list or any sequence, use the first element
    - If conversion fails, return the default

    Args:
        value: Value to convert to float
        default: Default value if conversion fails

    Returns:
        Float value
    """
    try:
        # Handle tuple, list or any other sequence type
        if hasattr(
    value,
    '__len__') and hasattr(
        value,
         '__getitem__') and len(value) > 0:
            # Get first item if it's a sequence
            try:
                return float(value[0])
            except (IndexError, TypeError, ValueError):
                pass

        # Try direct conversion
        return float(value)
    except (ValueError, TypeError, IndexError):
        return default


def get_similar_songs(
    engine, song_id, k=5, audio_weight=0.7, lyrics_weight=0.2, sentiment_weight=0.1
):
    """
    Get similar songs using a hybrid approach with customizable weights.
    By default, uses 70% audio features, 20% lyrics, and 10% sentiment.
    """
    logger.info(
        f"Getting similar songs for song_id: {song_id}, k={k}, weights: audio={audio_weight}, lyrics={lyrics_weight}, sentiment={sentiment_weight}"
    )

    # Define this as False for regular searches (not YouTube)
    youtube_source = False

    # Backward compatibility: if weights aren't provided, use only audio
    if audio_weight == 1.0 and lyrics_weight == 0.0 and sentiment_weight == 0.0:
        logger.info("Using audio-only recommendation approach")
        return get_similar_songs_audio_only(engine, song_id, k)

    # Implement retry logic
    max_retries = 3
    retry_delay = 1  # seconds
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Fetch the reference song's features
            logger.info(
                f"Connecting to database (attempt {
    retry_count + 1}/{max_retries})"
            )
            with engine.connect() as conn:
                # Start transaction explicitly
                with conn.begin():
                    logger.info(f"Fetching reference song (id={song_id})")
                    ref_row = conn.execute(
                        text(
                            """
                        SELECT id, track_name, track_artist, audio_features, lyrics, sentiment_features
                        FROM songs WHERE id = :id
                        """
                        ),
                        {"id": song_id},
                    ).fetchone()

                    if not ref_row:
                        logger.warning(
    f"❌ Reference song not found in DB (id={song_id})")
                        return []

                    logger.info(
                        f"✅ Reference song fetched: '{
    ref_row.track_name}' by '{
        ref_row.track_artist}'"
                    )

                    # Get audio features
                    ref_audio_dict = ref_row.audio_features
                    if isinstance(ref_audio_dict, str):
                        ref_audio_dict = json.loads(ref_audio_dict)

                    # Get lyric features
                    ref_lyrics = ref_row.lyrics or ""

                    # Get sentiment features
                    ref_sentiment = ref_row.sentiment_features
                    if ref_sentiment and isinstance(ref_sentiment, str):
                        ref_sentiment = json.loads(ref_sentiment)
                    elif not ref_sentiment:
                        # If no sentiment stored, calculate it now
                        ref_sentiment = analyze_sentiment(
                            ref_lyrics) if ref_lyrics else {}

                    # Prepare audio features (flatten as before)
                    logger.info("Processing reference song features")
                    ref_audio_vec = flatten_features(ref_audio_dict)
                    ref_audio_features = np.array(
                        ref_audio_vec, dtype=np.float32
                    ).reshape(1, -1)
                    ref_audio_len = ref_audio_features.shape[1]

                    # Debug information about reference song
                    logger.info(
                        f"Reference song: '{
    ref_row.track_name}' by '{
        ref_row.track_artist}'"
                    )
                    logger.info(
    f"Reference audio vector length: {ref_audio_len}")
                    logger.info(f"Reference has lyrics: {bool(ref_lyrics)}")
                    logger.info(
                        f"Reference has sentiment: {bool(ref_sentiment)}"
                    )

                    # Get total song count for batching
                    total_songs = conn.execute(
                        text("SELECT COUNT(*) FROM songs WHERE id != :id"),
                        {"id": song_id},
                    ).scalar()
                    logger.info(f"Total songs to compare: {total_songs}")

                    # Process songs in batches to reduce memory usage
                    BATCH_SIZE = 500  # Reduced from 1000 to process data in smaller chunks
                    all_candidates = []
                    batch_count = (total_songs + BATCH_SIZE - 1) // BATCH_SIZE
                    skipped = 0

                    # Define min thresholds to match the ones in other
                    # functions
                    min_audio_sim = MIN_AUDIO_SIM
                    min_combined_sim = MIN_COMBINED_SIM

                    # Target number of candidates before sorting (to allow early termination)
                    # Get at least 3x needed or 100 candidates
                    target_candidates = min(k * 3, 100)

                    # Dynamic thresholds - start with lower thresholds and
                    # increase as we find matches
                    current_min_audio_sim = MIN_AUDIO_SIM * \
                        0.8  # Start with 80% of normal threshold
                    current_min_combined_sim = MIN_COMBINED_SIM * 0.8
                    # Increase threshold by 5% each batch if we have enough
                    # candidates
                    threshold_increase_factor = 1.05

                    # Adaptive threshold control
                    # Need at least 2x k candidates to start increasing
                    # threshold
                    min_candidates_for_threshold_increase = k * 2

                    for batch_idx in range(batch_count):
                        # Skip if we already have enough candidates
                        if len(all_candidates) >= target_candidates:
                            logger.info(
    f"Early termination: Already found {
        len(all_candidates)} candidates, which is enough")
                            break

                        batch_start = batch_idx * BATCH_SIZE
                        logger.info(
                            f"Processing batch {
    batch_idx + 1}/{batch_count} (songs {
        batch_start + 1}-{
            min(
                batch_start + BATCH_SIZE,
                 total_songs)})"
                        )

                        # Dynamically adjust thresholds based on candidates
                        # found so far
                        if len(
                            all_candidates) >= min_candidates_for_threshold_increase:
                            # Increase thresholds to get better quality matches
                            old_audio_sim = current_min_audio_sim
                            old_combined_sim = current_min_combined_sim

                            current_min_audio_sim = min(
    MIN_AUDIO_SIM * 1.2, current_min_audio_sim * threshold_increase_factor)
                            current_min_combined_sim = min(
    MIN_COMBINED_SIM * 1.2, current_min_combined_sim * threshold_increase_factor)

                            logger.info(f"Increasing thresholds: audio_sim {old_audio_sim:.3f} -> {current_min_audio_sim:.3f}, " +
                                        f"combined_sim {old_combined_sim:.3f} -> {current_min_combined_sim:.3f}")

                        try:
                            with engine.connect() as conn:
                                with conn.begin():
                                    # More efficient SQL filtering with improved indexing
                                    # Add optimization hints for better query
                                    # performance
                                    batch_rows = conn.execute(
                                        text(
                                            """
                                        -- Optimization hint for batch processing
                                        SELECT /*+ INDEX(songs) */
                                            id, track_name, track_artist, audio_features, lyrics, sentiment_features
                                        FROM songs WHERE id != :id
                                        ORDER BY id
                                        LIMIT :limit OFFSET :offset
                                    """
                                        ),
                                        {
                                            "id": song_id,
                                            "limit": BATCH_SIZE,
                                            "offset": batch_start,
                                        },
                                    ).fetchall()

                            # Process this batch
                            batch_candidates = []

                            for row in batch_rows:
                                try:
                                    # Process audio features
                                    audio_dict = row.audio_features
                                    if isinstance(audio_dict, str):
                                        audio_dict = json.loads(audio_dict)
                                    audio_vec = flatten_features(audio_dict)

                                    # Skip if audio feature length doesn't
                                    # match
                                    if len(audio_vec) != ref_audio_len:
                                        logger.debug(
                                            f"Skipping song ID {
    row.id} due to feature mismatch: {
        len(audio_vec)} vs {ref_audio_len}"
                                        )
                                        skipped += 1
                                        continue

                                    # Calculate similarity using our improved
                                    # function
                                    try:
                                        audio_sim, _ = calculate_similarity(
                                            ref_audio_dict,
                                            audio_dict,
                                            youtube_source=False,
                                            detailed=True,
                                        )
                                    except Exception as e:
                                        # Fall back to cosine similarity if our
                                        # function fails
                                        logger.warning(
                                            f"Error using calculate_similarity for song ID {
    row.id}: {e}, falling back to cosine"
                                        )
                                        audio_feat = np.array(
                                            audio_vec, dtype=np.float32
                                        ).reshape(1, -1)
                                        audio_sim = float(
                                            cosine_similarity(
                                                ref_audio_features, audio_feat
                                            )[0][0]
                                        )

                                    # Apply adaptive thresholds
                                    if safe_compare(
    audio_sim, current_min_audio_sim, "<") and audio_weight > 0:
                                        continue  # Skip low-quality audio matches

                                    # Process lyrics
                                    row_lyrics = row.lyrics or ""

                                    # Calculate lyrics similarity
                                    lyrics_sim = 0.0
                                    if ref_lyrics and row_lyrics and lyrics_weight > 0:
                                        lyrics_sim = compute_lyrics_similarity(
                                            ref_lyrics, row_lyrics
                                        )
                                        if safe_compare(
    lyrics_sim, MIN_LYRICS_SIM, "<") and lyrics_weight > 0.3:
                                            continue

                                    # Process sentiment
                                    row_sentiment = row.sentiment_features
                                    if row_sentiment and isinstance(
                                        row_sentiment, str):
                                        row_sentiment = json.loads(
                                            row_sentiment)
                                    elif (
                                        not row_sentiment
                                        and row_lyrics
                                        and sentiment_weight > 0
                                    ):
                                        # Calculate sentiment from lyrics if
                                        # available
                                        row_sentiment = analyze_sentiment(
                                            row_lyrics)

                                    # Calculate sentiment similarity
                                    sentiment_sim = 0.0
                                    if (
                                        ref_sentiment
                                        and row_sentiment
                                        and sentiment_weight > 0
                                    ):
                                        sentiment_sim = compute_sentiment_similarity(
                                            ref_sentiment, row_sentiment
                                        )
                                        if safe_compare(
    sentiment_sim, MIN_SENTIMENT_SIM, "<") and sentiment_weight > 0.2:
                                            continue

                                    # Calculate combined score
                                    combined_sim = (
                                        safe_float(audio_sim) * audio_weight +
                                        safe_float(lyrics_sim) * lyrics_weight +
                                        safe_float(sentiment_sim) *
                                                   sentiment_weight
                                    )

                                    # Apply adaptive threshold for combined
                                    # score
                                    if safe_compare(
    combined_sim, current_min_combined_sim, "<"):
                                        continue  # Skip low combined score

                                    # Store for final ranking
                                    batch_candidates.append(
                                        {
                                            "id": row.id,
                                            "title": row.track_name,
                                            "artist": row.track_artist,
                                            "audio_score": audio_sim,
                                            "lyrics_score": lyrics_sim,
                                            "sentiment_score": sentiment_sim,
                                            "combined_score": combined_sim,
                                        }
                                    )
                                except Exception as e:
                                    logger.warning(
                                        f"Error processing song ID {
    row.id}: {e}"
                                    )
                                    continue

                            # Add batch results to overall results
                            all_candidates.extend(batch_candidates)
                            logger.info(
                                f"Batch {
    batch_idx +
    1}: Found {
        len(batch_candidates)} candidates"
                            )

                        except Exception as batch_err:
                            logger.error(
                                f"Error processing batch {
    batch_idx + 1}: {batch_err}"
                            )

                    logger.info(
                        f"Processed all songs (found {
    len(all_candidates)}, skipped {skipped})"
                    )

                    if not all_candidates:
                        logger.warning(
                            "❌ No candidates with matching features that meet similarity thresholds."
                        )
                        return []

                    # Progressive refinement step - if we have many candidates,
                    # apply stricter thresholds to get higher quality results
                    if len(all_candidates) > k * 3:
                        logger.info(
    f"Found {
        len(all_candidates)} candidates, performing progressive refinement")

                        # Calculate statistics on the scores
                        audio_scores = [c["audio_score"]
                            for c in all_candidates]
                        combined_scores = [c["combined_score"]
                            for c in all_candidates]

                        if audio_scores and combined_scores:
                            try:
                                # Calculate median scores safely
                                median_audio = np.median(
                                    [safe_float(score) for score in audio_scores])
                                median_combined = np.median(
                                    [safe_float(score) for score in combined_scores])

                                # Also calculate standard deviations for more
                                # intelligent thresholds
                                std_audio = np.std(
                                    [safe_float(score) for score in audio_scores])
                                std_combined = np.std(
                                    [safe_float(score) for score in combined_scores])

                                # Apply stricter thresholds based on median scores
                                # Use 85% of median or 1 standard deviation below median, whichever is higher
                                # This adapts better to the actual score
                                # distribution
                                refined_threshold_audio = max(
                                    min_audio_sim,
                                    max(median_audio * 0.85,
                                        median_audio - std_audio)
                                )
                                refined_threshold_combined = max(
                                    min_combined_sim,
                                    max(median_combined * 0.85,
                                        median_combined - std_combined)
                                )
                            except Exception as e:
                                logger.warning(
    f"Error calculating refined thresholds: {e}")
                                # Fall back to simple thresholds if statistical
                                # calculation fails
                                refined_threshold_audio = min_audio_sim
                                refined_threshold_combined = min_combined_sim

                            logger.info(
                                f"Score statistics - Audio: median={median_audio:.3f}, std={std_audio:.3f}")
                            logger.info(
    f"Score statistics - Combined: median={
        median_combined:.3f}, std={
            std_combined:.3f}")
                            logger.info(
    f"Refining with stricter thresholds: audio >= {
        refined_threshold_audio:.3f}, combined >= {
            refined_threshold_combined:.3f}")

                            # Filter candidates with stricter thresholds
                            refined_candidates = [
                                c for c in all_candidates
                                if safe_compare(c["audio_score"], refined_threshold_audio, ">=") and safe_compare(c["combined_score"], refined_threshold_combined, ">=")
                            ]

                            # Only use refined list if we still have enough
                            # candidates
                            if len(refined_candidates) >= max(
    k, 3):  # Ensure we have at least 3 or k candidates
                                logger.info(
    f"After refinement: {
        len(refined_candidates)} candidates (from {
            len(all_candidates)})")
                                all_candidates = refined_candidates
                            else:
                                logger.info(
    f"Not enough candidates after refinement ({
        len(refined_candidates)}), using original list")

                    # Sort by combined score
                    all_candidates.sort(
    key=lambda x: x["combined_score"], reverse=True)

                    # Take top k
                    top_k = all_candidates[:k]

                    # Format results for return
                    results = []
                    for candidate in top_k:
                        results.append(
                            (
                                candidate["id"],
                                candidate["title"],
                                candidate["artist"],
                                candidate["combined_score"],
                                candidate["audio_score"],
                                candidate["lyrics_score"],
                                candidate["sentiment_score"],
                            )
                        )
                        logger.info(
                            f"Selected for recommendation: '{
    candidate['title']}' by '{
        candidate['artist']}' with score {
            candidate['combined_score']:.3f}"
                        )

                    logger.info(
                        f"✅ Found {
    len(results)} recommendations for song_id={song_id}"
                    )
                    return results

        except Exception as e:
            retry_count += 1
            logger.error(
                f"❌ Database connection error (attempt {retry_count}/{max_retries}): {e}"
            )

            if retry_count >= max_retries:
                logger.error(f"❌ Failed after {max_retries} attempts.")
                return []

            logger.info(f"⏳ Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff

    return []


def get_similar_songs_audio_only(engine, song_id, k=5):
    """Original audio-only recommendation function (for backward compatibility)"""
    logger.info(
    f"Getting audio-only recommendations for song_id: {song_id}, k={k}")

    # Implement retry logic
    max_retries = 3
    retry_delay = 1  # seconds
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Fetch the reference song's audio features
            with engine.connect() as conn:
                with conn.begin():
                    ref_row = conn.execute(
                        text(
                            """
                        SELECT id, track_name, track_artist, audio_features
                        FROM songs WHERE id = :id
                    """
                        ),
                        {"id": song_id},
                    ).fetchone()

                    if not ref_row:
                        logger.warning(
    f"❌ Reference song not found in DB (id={song_id})")
                        return []

                    logger.info(
    f"✅ Reference song fetched: '{
        ref_row.track_name}' by '{
            ref_row.track_artist}'")

                    audio_dict = ref_row.audio_features
                    if isinstance(audio_dict, str):
                        audio_dict = json.loads(audio_dict)

                    # Flatten features
                    ref_vec = flatten_features(audio_dict)
                    ref_features = np.array(
    ref_vec, dtype=np.float32).reshape(
        1, -1)
                    ref_len = ref_features.shape[1]
                    logger.info(
    f"Reference song feature vector length: {ref_len}")

                    # Fetch all other songs' ids and features
                    rows = conn.execute(
                        text(
                            """
                        SELECT id, track_name, track_artist, audio_features
                        FROM songs WHERE id != :id
                    """
                        ),
                        {"id": song_id},
                    ).fetchall()

                    if not rows:
                        logger.warning("❌ No other songs in DB.")
                        return []

                    # Process candidates outside the connection scope for
                    # better resource management
                    ids = []
                    features = []
                    meta = []
                    titles = []
                    artists = []
                    skipped = 0

                    for row in rows:
                        try:
                            audio_dict = row.audio_features
                            if isinstance(audio_dict, str):
                                audio_dict = json.loads(audio_dict)
                            vec = flatten_features(audio_dict)

                            if len(vec) != ref_len:
                                logger.debug(
                                    f"Skipping song ID {
    row.id} ({
        row.track_name}) due to feature length mismatch: {
            len(vec)} vs {ref_len}"
                                )
                                skipped += 1
                                continue  # skip songs with mismatched feature length

                            features.append(vec)
                            ids.append(row.id)
                            meta.append((row.track_name, row.track_artist))
                            titles.append(row.track_name)
                            artists.append(row.track_artist)
                        except Exception as e:
                            logger.warning(
    f"Error processing song ID {
        row.id}: {e}")
                            continue

                    logger.info(
    f"Processed {
        len(features)} songs successfully (skipped {skipped} with mismatched features)")

                    if not features:
                        logger.warning(
                            "❌ No candidates with matching feature length.")
                        return []

                    # Calculate cosine similarities
                    features_np = np.stack(features)
                    sims = cosine_similarity(ref_features, features_np)[0]

                    # Filter by minimum similarity threshold
                    filtered_indices = [
                        i for i, sim in enumerate(sims) if safe_compare(sim, MIN_AUDIO_SIM, ">=")
                    ]

                    if not filtered_indices:
                        logger.warning(
    f"❌ No songs with similarity score >= {MIN_AUDIO_SIM}")
                        return []

                    # Show all scores for debugging
                    logger.info(
                        "All audio recommendation candidates (sorted by score):")
                    for i in np.argsort(sims)[::-1]:
                        if safe_compare(sims[i], MIN_AUDIO_SIM, ">="):
                            logger.info(
    f"  {
        titles[i]} by {
            artists[i]}: Audio score={
                sims[i]:.3f}")

                    # Get top k from filtered indices
                    filtered_sims = [sims[i] for i in filtered_indices]
                    filtered_idx_order = np.argsort(filtered_sims)[-k:][::-1]
                    top_k_idx = [filtered_indices[i]
                        for i in filtered_idx_order]

                    logger.info(
    f"✅ Found {
        len(top_k_idx)} audio-only recommendations")

                    # Log final recommendations for debugging
                    for i in top_k_idx:
                        logger.info(
    f"Selected for recommendation: '{
        titles[i]}' by '{
            artists[i]}' with score {
                sims[i]:.3f}")

                    return [
                        (ids[i], meta[i][0], meta[i][1], float(sims[i]))
                        for i in top_k_idx
                    ]

        except Exception as e:
            retry_count += 1
            logger.error(
    f"❌ Database connection error (attempt {retry_count}/{max_retries}): {e}")

            if retry_count >= max_retries:
                logger.error(f"❌ Failed after {max_retries} attempts: {e}")
                return []

            logger.info(f"⏳ Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff

    return []


def get_similar_songs_for_features(
    engine,
    features,
    k=5,
    audio_weight=0.7,
    lyrics_weight=0.2,
    sentiment_weight=0.1,
    lyrics=None,
    sentiment=None,
):
    """
    Get similar songs for external features (e.g., from YouTube) using hybrid approach
    """
    logger.info(
        f"Getting similar songs for external features, k={k}, weights: audio={audio_weight}, lyrics={lyrics_weight}, sentiment={sentiment_weight}"
    )

    # Check if we received metadata structure
    youtube_source = False
    original_features = features

    # More robust YouTube source detection
    if isinstance(features, dict):
        # Check for explicit source marker
        if "source" in features and features["source"] == "youtube":
            youtube_source = True
        # Extract actual features if we got the metadata structure
        if "audio_features" in features:
            features = features["audio_features"]

        # Check for characteristic YouTube feature patterns by examining value
        # ranges
        if not youtube_source and isinstance(features, dict):
            # Look for any feature with unnormalized values (original YouTube
            # extraction)
            if ("danceability" in features and features["danceability"] > 1.0) or (
                "speechiness" in features and features["speechiness"] > 1.0
            ):
                youtube_source = True
                logger.info(
                    "Detected YouTube source based on feature value ranges")

    # FORCE YouTube mode for YouTube data - since we're called from youtube_search
    # The calling context knows these are YouTube features even if they're
    # normalized
    if not youtube_source:
        # Check if this is called from a YouTube search context
        import traceback

        stack = traceback.extract_stack()
        for frame in stack:
            if "youtube" in frame.name.lower():
                youtube_source = True
                logger.info(
                    "Forcing YouTube-optimized similarity for YouTube search function"
                )
                break

    # Debug: log the input features in detail
    logger.info(
    f"INPUT FEATURES FROM YOUTUBE: {
        json.dumps(
            features,
             indent=2)}")
    logger.info(f"Using YouTube-optimized similarity: {youtube_source}")

    # If no lyrics or sentiment provided, fall back to audio-only
    if not lyrics and not sentiment:
        lyrics_weight = 0
        sentiment_weight = 0
        audio_weight = 1.0
        logger.info(
            "No lyrics or sentiment provided, using audio-only matching")

    # Process input features
    logger.info("Processing input features")

    # Analyzing sentiment from lyrics if provided
    if lyrics and sentiment_weight > 0:
        logger.info("Analyzing sentiment from lyrics")
        sentiment = analyze_sentiment(lyrics)

    # Convert features to a vector for similarity calculation
    ref_vec = flatten_features(features)
    ref_features = np.array(ref_vec, dtype=np.float32).reshape(1, -1)
    ref_len = ref_features.shape[1]
    logger.info(f"Reference feature vector length: {ref_len}")

    # Get minimum similarity thresholds
    min_audio_sim = MIN_YT_AUDIO_SIM if youtube_source else MIN_AUDIO_SIM
    min_lyrics_sim = MIN_LYRICS_SIM
    min_sentiment_sim = MIN_SENTIMENT_SIM
    min_combined_sim = MIN_COMBINED_SIM

    # Implement retry logic
    max_retries = 3
    retry_delay = 1  # seconds
    retry_count = 0

    # Lists to store results
    all_candidates = []
    BATCH_SIZE = 500  # Reduced from 1000 to process data in smaller chunks

    while retry_count < max_retries:
        try:
            # First fetch all song IDs (lighter query)
            logger.info(
    f"Connecting to database (attempt {
        retry_count + 1}/{max_retries})")
            with engine.connect() as conn:
                # Get total count first
                count_result = conn.execute(
    text("SELECT COUNT(*) FROM songs")).fetchone()
                total_songs = count_result[0] if count_result else 0
                logger.info(f"Found {total_songs} total songs in database")

                # Get all song IDs, names and artists
                rows = conn.execute(
                    text("""
                        SELECT id, track_name, track_artist
                        FROM songs
                    """)
                ).fetchall()

                if not rows:
                    logger.warning("No songs found in database")
                    return []

                logger.info(f"Retrieved {len(rows)} song IDs from database")

                # Calculate number of batches
                batch_count = (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE

                # Adaptive thresholds - start with lower thresholds and
                # increase as we find matches
                current_min_audio_sim = min_audio_sim * \
                    0.8  # Start with 80% of normal threshold
                current_min_combined_sim = min_combined_sim * 0.8
                # Increase threshold by 5% when we have enough candidates
                threshold_increase_factor = 1.05

                # Minimum candidates needed to increase threshold
                min_candidates_for_threshold_increase = k * 2

                # Target candidates before final sorting
                target_candidates = min(k * 3, 100)

                # Track statistics
                skipped_audio = 0
                skipped_combined = 0
                skipped_other = 0
                processed = 0

                # Process in batches
                for batch_idx in range(batch_count):
                    # Check if we already have enough candidates for early
                    # termination
                    if len(all_candidates) >= target_candidates:
                        logger.info(
    f"Early termination: Already found {
        len(all_candidates)} candidates, which is enough")
                        break

                    # Dynamically adjust thresholds based on candidates found
                    # so far
                    if len(all_candidates) >= min_candidates_for_threshold_increase:
                        # Increase thresholds to get better quality matches
                        old_audio_sim = current_min_audio_sim
                        old_combined_sim = current_min_combined_sim

                        current_min_audio_sim = min(
    min_audio_sim * 1.2, current_min_audio_sim * threshold_increase_factor)
                        current_min_combined_sim = min(
    min_combined_sim * 1.2, current_min_combined_sim * threshold_increase_factor)

                        logger.info(f"Increasing thresholds: audio_sim {old_audio_sim:.3f} -> {current_min_audio_sim:.3f}, " +
                                   f"combined_sim {old_combined_sim:.3f} -> {current_min_combined_sim:.3f}")

                    batch_start = batch_idx * BATCH_SIZE
                    batch_end = min(batch_start + BATCH_SIZE, len(rows))
                    batch = rows[batch_start:batch_end]

                    logger.info(
                        f"Processing batch {batch_idx + 1}/{batch_count} (songs {batch_start + 1}-{batch_end})")

                    # Fetch full data for this batch
                    batch_ids = [row.id for row in batch]

                    # Create a map of id to name/artist for easy lookup
                    id_to_meta = {
    row.id: (
        row.track_name,
         row.track_artist) for row in batch}

                    with engine.connect() as conn:
                        # Get detailed data for this batch
                        batch_data = conn.execute(
                            text("""
                                SELECT id, audio_features, lyrics, sentiment_features
                                FROM songs
                                WHERE id IN :ids
                            """),
                            {
                                "ids": tuple(batch_ids) if len(batch_ids) > 1 else f"({batch_ids[0]})"
                            }
                        ).fetchall()

                    # Process each song in this batch
                    for row in batch_data:
                        processed += 1

                        try:
                            song_id = row.id
                            song_title, song_artist = id_to_meta.get(
                                song_id, ("Unknown", "Unknown"))

                            # Extract and parse features
                            audio_features = row.audio_features
                            if isinstance(audio_features, str):
                                try:
                                    audio_features = json.loads(audio_features)
                                except json.JSONDecodeError as e:
                                    logger.warning(
    f"Error parsing audio_features JSON for song {song_id} ({song_title}): {e}")
                                    continue

                            # Calculate audio similarity
                            try:
                                if youtube_source:
                                    audio_sim = compute_audio_similarity_yt(
                                        features, audio_features)
                                else:
                                    audio_sim = calculate_similarity(
                                        features, audio_features)
                            except Exception as e:
                                logger.warning(
    f"Error calculating similarity for song {song_id} ({song_title}): {e}")
                                continue

                            # Skip if audio similarity is too low (main filter)
                            if safe_compare(
    audio_sim, current_min_audio_sim, "<"):
                                skipped_audio += 1
                                continue

                            # Calculate lyrics similarity if needed
                            lyrics_sim = 0
                            if lyrics and lyrics_weight > 0 and row.lyrics:
                                try:
                                    lyrics_sim = compute_lyrics_similarity(
                                        lyrics, row.lyrics)
                                    if safe_compare(
    lyrics_sim, min_lyrics_sim, "<") and lyrics_weight > 0.3:
                                        skipped_other += 1
                                        continue
                                except Exception as e:
                                    logger.warning(
    f"Error calculating lyrics similarity for song {song_id} ({song_title}): {e}")
                                    # Continue with zero lyrics similarity
                                    # instead of skipping entirely

                            # Calculate sentiment similarity if needed
                            sentiment_sim = 0
                            if sentiment and sentiment_weight > 0 and row.sentiment_features:
                                try:
                                    # Parse sentiment if needed
                                    db_sentiment = row.sentiment_features
                                    if isinstance(db_sentiment, str):
                                        db_sentiment = json.loads(db_sentiment)
                                    sentiment_sim = compute_sentiment_similarity(
                                        sentiment, db_sentiment)
                                    if safe_compare(
    sentiment_sim, min_sentiment_sim, "<") and sentiment_weight > 0.2:
                                        skipped_other += 1
                                        continue
                                except Exception as e:
                                    logger.warning(
    f"Error calculating sentiment similarity for song {song_id} ({song_title}): {e}")
                                    # Continue with zero sentiment similarity
                                    # instead of skipping entirely

                            # Calculate combined similarity score with weights
                            combined_sim = (
                                safe_float(audio_sim) * audio_weight +
                                safe_float(lyrics_sim) * lyrics_weight +
                                safe_float(sentiment_sim) * sentiment_weight
                            )

                            # Skip if combined score is too low
                            if safe_compare(
    combined_sim, current_min_combined_sim, "<"):
                                skipped_combined += 1
                                continue

                            # Add to candidates
                            all_candidates.append({
                                "id": song_id,
                                "title": song_title,
                                "artist": song_artist,
                                "combined_score": combined_sim,
                                "audio_score": audio_sim,
                                "lyrics_score": lyrics_sim,
                                "sentiment_score": sentiment_sim
                            })
                        except Exception as e:
                            logger.warning(
    f"Error processing song {
        row.id}: {
            str(e)[
                :100]}")
                            continue

                logger.info(
    f"Processed {processed} songs, skipped {skipped_audio} (audio), {skipped_combined} (combined), {skipped_other} (other)")
                logger.info(
    f"Found {
        len(all_candidates)} potential candidates")

                # Progressive refinement step - if we have many candidates,
                # apply stricter thresholds to get higher quality results
                if len(all_candidates) > k * 3:
                    logger.info(
    f"Found {
        len(all_candidates)} candidates, performing progressive refinement")

                    # Calculate statistics on the scores
                    audio_scores = [c["audio_score"] for c in all_candidates]
                    combined_scores = [c["combined_score"]
                        for c in all_candidates]

                    if audio_scores and combined_scores:
                        try:
                            # Calculate median scores safely
                            median_audio = np.median(
                                [safe_float(score) for score in audio_scores])
                            median_combined = np.median(
                                [safe_float(score) for score in combined_scores])

                            # Also calculate standard deviations for more
                            # intelligent thresholds
                            std_audio = np.std([safe_float(score)
                                               for score in audio_scores])
                            std_combined = np.std(
                                [safe_float(score) for score in combined_scores])

                            # Apply stricter thresholds based on median scores
                            # Use 85% of median or 1 standard deviation below median, whichever is higher
                            # This adapts better to the actual score
                            # distribution
                            refined_threshold_audio = max(
                                min_audio_sim,
                                max(median_audio * 0.85,
                                    median_audio - std_audio)
                            )
                            refined_threshold_combined = max(
                                min_combined_sim,
                                max(median_combined * 0.85,
                                    median_combined - std_combined)
                            )
                        except Exception as e:
                            logger.warning(
    f"Error calculating refined thresholds: {e}")
                            # Fall back to simple thresholds if statistical
                            # calculation fails
                            refined_threshold_audio = min_audio_sim
                            refined_threshold_combined = min_combined_sim

                        logger.info(
                            f"Score statistics - Audio: median={median_audio:.3f}, std={std_audio:.3f}")
                        logger.info(
    f"Score statistics - Combined: median={
        median_combined:.3f}, std={
            std_combined:.3f}")
                        logger.info(
    f"Refining with stricter thresholds: audio >= {
        refined_threshold_audio:.3f}, combined >= {
            refined_threshold_combined:.3f}")

                        # Filter candidates with stricter thresholds
                        refined_candidates = [
                            c for c in all_candidates
                            if safe_compare(c["audio_score"], refined_threshold_audio, ">=") and safe_compare(c["combined_score"], refined_threshold_combined, ">=")
                        ]

                        # Only use refined list if we still have enough
                        # candidates
                        if len(refined_candidates) >= max(
    k, 3):  # Ensure we have at least 3 or k candidates
                            logger.info(
    f"After refinement: {
        len(refined_candidates)} candidates (from {
            len(all_candidates)})")
                            all_candidates = refined_candidates
                        else:
                            logger.info(
    f"Not enough candidates after refinement ({
        len(refined_candidates)}), using original list")

                # Sort by combined score
                all_candidates.sort(
    key=lambda x: x["combined_score"], reverse=True)

                # Take top k
                top_k = all_candidates[:k]

                # Format results for return
                results = []
                for candidate in top_k:
                    results.append(
                        (
                            candidate["id"],
                            candidate["title"],
                            candidate["artist"],
                            candidate["combined_score"],
                            candidate["audio_score"],
                            candidate["lyrics_score"],
                            candidate["sentiment_score"],
                        )
                    )
                    logger.info(
                        f"Selected for recommendation: '{
    candidate['title']}' by '{
        candidate['artist']}' with score {
            candidate['combined_score']:.3f}"
                    )

                logger.info(
    f"✅ Found {
        len(results)} recommendations for external features")
                return results

        except Exception as e:
            retry_count += 1
            logger.error(
    f"Error in recommendation process (attempt {retry_count}/{max_retries}): {e}")
            if retry_count < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error(
    f"Failed to get recommendations after {max_retries} attempts")
                return []

    # If we get here, all retries failed
    return []


def get_similar_songs_for_features_audio_only(engine, features, k=5):
    """
    Get similar songs to the given audio features using only audio features.
    Used for compatibility with older code paths.
    """
    logger.info(f"Getting audio-only recommendations for features, k={k}")

    # Determine if these are YouTube features
    youtube_source = True
    logger.info(f"Using YouTube-optimized similarity calculation")

    # Use YouTube thresholds if using YouTube features
    min_audio_sim = MIN_YT_AUDIO_SIM if youtube_source else MIN_AUDIO_SIM

    # Implement retry logic
    max_retries = 3
    retry_delay = 1  # seconds
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Convert features to a vector for cosine similarity
            ref_vec = flatten_features(features)
            ref_features = np.array(ref_vec, dtype=np.float32).reshape(1, -1)
            ref_len = ref_features.shape[1]
            logger.info(f"Reference feature vector length: {ref_len}")

            logger.info(f"Connecting to database (attempt {retry_count + 1}/{max_retries})")
            with engine.connect() as conn:
                with conn.begin():
                    # Basic song filtering using SQL for better performance
                    base_sql = """
                        SELECT id, track_name, track_artist, audio_features
                        FROM songs
                    """

                    # If we have reference values, use them for early filtering
                    filters = []
                    params = {}

                    # Only add filters if the reference has these attributes
                    if "tempo" in features:
                        # Filter songs with tempo within 50% range
                        try:
                            ref_tempo = float(features["tempo"])
                            # Create a temporary table with the reference features
                            # This is more efficient than processing all songs in Python
                            temp_table_name = f"temp_ref_features_{int(time.time())}"
                            conn.execute(
                                text(
                                    f"""
                                CREATE TEMPORARY TABLE {temp_table_name} (
                                    feature_idx INTEGER,
                                    feature_val FLOAT
                                )
                            """
                                )
                            )

                            # Insert reference features into temp table
                            for i, val in enumerate(ref_vec):
                                conn.execute(
                                    text(
                                        f"""
                                    INSERT INTO {temp_table_name} (feature_idx, feature_val)
                                    VALUES (:idx, :val)
                                """
                                    ),
                                    {"idx": i, "val": float(val)},
                                )

                            # Get all songs from database - IDs only first for efficiency
                            rows = conn.execute(
                                text(
                                    """
                                SELECT id, track_name, track_artist
                                FROM songs
                            """
                                )
                            ).fetchall()

                            logger.info(f"Retrieved {len(rows)} song IDs from database")
                        except Exception as e:
                            logger.warning(f"Error with reference tempo: {e}")
                            # Continue without filtering
                            rows = conn.execute(
                                text(
                                    """
                                SELECT id, track_name, track_artist
                                FROM songs
                            """
                                )
                            ).fetchall()

                # Process songs in smaller batches to reduce memory usage
                BATCH_SIZE = 500  # Reduced from 1000 to process data in smaller chunks
                all_matches = []
                total_songs = len(rows)

                # Variables to track similarity score distribution
                all_similarities = []
                above_01 = 0
                above_02 = 0
                above_03 = 0
                above_04 = 0
                above_05 = 0
                
                # For progressive streaming
                streaming_matches = []
                
                # Adaptive threshold for YouTube content - start lower, then increase
                current_min_similarity = min_audio_sim * 0.8  # Start with 80% of min threshold
                threshold_increase_factor = 1.05  # Increase threshold by 5% when we have enough candidates
                
                # Minimum candidates needed to start increasing threshold
                min_candidates_for_adjustment = k * 2

                # Track the top 30 songs with most details for debugging
                detailed_comparisons = []

                # Process in batches
                for batch_start in range(0, total_songs, BATCH_SIZE):
                    batch_end = min(batch_start + BATCH_SIZE, total_songs)
                    batch = rows[batch_start:batch_end]
                    
                    # Check if we already have enough good candidates
                    if len(streaming_matches) >= min_candidates_for_adjustment:
                        # We can increase our threshold for better quality matches
                        old_threshold = current_min_similarity
                        # Increase threshold up to 110% of original min_audio_sim
                        current_min_similarity = min(min_audio_sim * 1.1, current_min_similarity * threshold_increase_factor)
                        logger.info(f"Increasing similarity threshold: {old_threshold:.3f} -> {current_min_similarity:.3f}")

                    logger.info(f"Processing batch {batch_start // BATCH_SIZE + 1}/{(total_songs + BATCH_SIZE - 1) // BATCH_SIZE}")

                    # Fetch audio features for this batch
                    batch_ids = [row.id for row in batch]

                    with engine.connect() as conn:
                        # Get audio features for this batch
                        batch_features = conn.execute(
                            text(
                                """
                            SELECT id, audio_features
                            FROM songs
                            WHERE id IN :ids
                        """
                            ),
                            {
                                "ids": (
                                    tuple(batch_ids)
                                    if len(batch_ids) > 1
                                    else f"({batch_ids[0]})"
                                )
                            },
                        ).fetchall()

                    # Process this batch
                    batch_ids = []
                    batch_features_list = []
                    batch_meta = []
                    batch_titles = []
                    batch_artists = []
                    batch_features_dict = []  # Store raw feature dicts for YouTube optimized comparison

                    for row in batch_features:
                        try:
                            audio_dict = row.audio_features
                            if isinstance(audio_dict, str):
                                audio_dict = json.loads(audio_dict)
                            vec = flatten_features(audio_dict)

                            # Skip if feature length doesn't match
                            if len(vec) != ref_len:
                                continue

                            # Store features for batch processing
                            batch_features_list.append(vec)
                            # Store raw dict for YouTube comparison
                            batch_features_dict.append(audio_dict)
                            batch_ids.append(row.id)

                            # Find corresponding metadata
                            meta_row = next((r for r in batch if r.id == row.id), None)
                            if meta_row:
                                batch_meta.append((meta_row.track_name, meta_row.track_artist))
                                batch_titles.append(meta_row.track_name)
                                batch_artists.append(meta_row.track_artist)
                            else:
                                # Fallback if metadata not found
                                batch_meta.append(("Unknown", "Unknown"))
                                batch_titles.append("Unknown")
                                batch_artists.append("Unknown")
                        except Exception as e:
                            logger.debug(f"Error processing song ID {row.id}: {e}")
                            continue

                    # Calculate similarities for this batch
                    if batch_features_list:
                        batch_sims = []
                        detailed_comparisons_batch = []

                        for i, audio_dict in enumerate(batch_features_dict):
                            try:
                                # Use our improved similarity function
                                similarity, feature_details = calculate_similarity(
                                    features,
                                    audio_dict,
                                    youtube_source=youtube_source,
                                    detailed=True,
                                )
                                batch_sims.append(similarity)

                                # Store detailed comparison information
                                if similarity >= current_min_similarity:  # Use current_min_similarity for filtering
                                    comparison = {
                                        "id": batch_ids[i],
                                        "title": batch_titles[i],
                                        "artist": batch_artists[i],
                                        "similarity": similarity,
                                        "feature_details": feature_details,
                                    }
                                    detailed_comparisons_batch.append(comparison)
                            except Exception as e:
                                # If there's an error, use 0 similarity
                                batch_sims.append(0.0)
                                logger.warning(f"Error computing similarity: {e}")

                        # Add detailed comparisons to main list
                        detailed_comparisons.extend(detailed_comparisons_batch)

                        # Track similarity distribution
                        all_similarities.extend(batch_sims)

                        # Filter by threshold
                        for i, sim in enumerate(batch_sims):
                            # Track thresholds
                            if sim >= 0.1:
                                above_01 += 1
                            if sim >= 0.2:
                                above_02 += 1
                            if sim >= 0.3:
                                above_03 += 1
                            if sim >= 0.4:
                                above_04 += 1
                            if sim >= 0.5:
                                above_05 += 1

                            if sim >= current_min_similarity:
                                streaming_matches.append(
                                    (
                                        batch_ids[i],
                                        batch_meta[i][0],
                                        batch_meta[i][1],
                                        float(sim),
                                    )
                                )

                # Sort detailed comparisons by similarity
                detailed_comparisons.sort(key=lambda x: x["similarity"], reverse=True)

                # Log the top 30 comparisons in detail
                logger.info("========== DETAILED FEATURE COMPARISONS ==========")
                logger.info(f"Showing top {min(30, len(detailed_comparisons))} detailed comparisons")

                for i, comp in enumerate(detailed_comparisons[:30]):
                    logger.info(f"Comparison #{i + 1}: '{comp['title']}' by {comp['artist']} - Similarity: {comp['similarity']:.3f}")
                    for feat, details in comp["feature_details"].items():
                        logger.info(f"  {feat}: YouTube={details['youtube']:.3f}, DB={details['db']:.3f}, Similarity={details['similarity']:.3f}")
                    logger.info("---")

                logger.info("===============================================")

                # Log similarity distribution stats
                if all_similarities:
                    logger.info("===== SIMILARITY SCORE DISTRIBUTION =====")
                    logger.info(f"Total songs compared: {len(all_similarities)}")
                    logger.info(f"Min similarity: {min(all_similarities):.3f}")
                    logger.info(f"Max similarity: {max(all_similarities):.3f}")
                    logger.info(f"Mean similarity: {np.mean(all_similarities):.3f}")
                    logger.info(f"Median similarity: {np.median(all_similarities):.3f}")
                    logger.info(f"Songs with similarity >= 0.1: {above_01} ({above_01 / len(all_similarities) * 100:.1f}%)")
                    logger.info(f"Songs with similarity >= 0.2: {above_02} ({above_02 / len(all_similarities) * 100:.1f}%)")
                    logger.info(f"Songs with similarity >= 0.3: {above_03} ({above_03 / len(all_similarities) * 100:.1f}%)")
                    logger.info(f"Songs with similarity >= 0.4: {above_04} ({above_04 / len(all_similarities) * 100:.1f}%)")
                    logger.info(f"Songs with similarity >= 0.5: {above_05} ({above_05 / len(all_similarities) * 100:.1f}%)")
                    logger.info("=========================================")

                    # Histogram of similarity scores
                    hist, bin_edges = np.histogram(all_similarities, bins=10, range=(0, 1))
                    logger.info("Similarity score histogram:")
                    for i in range(len(hist)):
                        logger.info(f"  {bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f}: {hist[i]} songs ({hist[i] / len(all_similarities) * 100:.1f}%)")

                # Progressive refinement step
                if len(streaming_matches) > k * 2:
                    logger.info(f"Found {len(streaming_matches)} candidates, performing progressive refinement")
                    
                    # Extract scores
                    match_scores = [match[3] for match in streaming_matches]
                    
                    if match_scores:
                        # Calculate median score
                        median_score = np.median(match_scores)
                        # Set threshold based on median (85% of median)
                        refined_threshold = max(min_audio_sim, median_score * 0.85)
                        
                        logger.info(f"Progressive refinement: Using threshold {refined_threshold:.3f} (from median {median_score:.3f})")
                        
                        # Filter with stricter threshold
                        refined_matches = [match for match in streaming_matches if match[3] >= refined_threshold]
                        
                        # Only use refined matches if we still have enough
                        if len(refined_matches) >= k:
                            logger.info(f"After refinement: {len(refined_matches)} candidates (from {len(streaming_matches)})")
                            streaming_matches = refined_matches
                        else:
                            logger.info(f"Not enough candidates after refinement ({len(refined_matches)}), using original list")
                
                # Sort all matches and get top k
                streaming_matches.sort(key=lambda x: x[3], reverse=True)
                results = streaming_matches[:k]

                if not results:
                    logger.warning(f"❌ No songs with similarity score >= {current_min_similarity}")
                    return []

                # Show top matches for debugging
                logger.info("Top audio recommendation matches:")
                for match in results:
                    logger.info(f"  {match[1]} by {match[2]}: Audio score={match[3]:.3f}")

                logger.info(f"✅ Found {len(results)} audio-only recommendations for external features")
                
                # Format results to be consistent with other recommendation functions
                # Add dummy values for lyrics_score and sentiment_score
                formatted_results = [
                    (match[0], match[1], match[2], match[3], match[3], 0.0, 0.0)
                    for match in results
                ]
                
                return formatted_results

        except Exception as e:
            retry_count += 1
            logger.error(f"❌ Database connection error (attempt {retry_count}/{max_retries}): {e}")

            if retry_count >= max_retries:
                logger.error(f"❌ Failed after {max_retries} attempts.")
                return []

            logger.info(f"⏳ Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff

    return []


@lru_cache(maxsize=1024)
def calculate_similarity_cached(feat1_str, feat2_str, youtube_source=False):
    """Cached version of similarity calculation using string representation of features"""
    # Convert strings back to dictionaries
    features1 = json.loads(feat1_str)
    features2 = json.loads(feat2_str)
    return calculate_similarity(
        features1, features2, youtube_source=youtube_source, detailed=False
    )


def calculate_similarity(features1, features2, youtube_source=False, detailed=False):
    """
    Calculate similarity between two sets of audio features
    """
    # Check if we can use the cache (only when detailed=False)
    if not detailed:
        # Try to use cached version if possible
        try:
            # Convert dicts to strings for cache keys
            feat1_str = json.dumps(features1, sort_keys=True)
            feat2_str = json.dumps(features2, sort_keys=True)
            return calculate_similarity_cached(feat1_str, feat2_str, youtube_source), {}
        except (TypeError, ValueError):
            # If serialization fails, continue with regular calculation
            pass

    # Features to compare and their weights for standard similarity
    standard_weights = {
        "danceability": 0.15,
        "energy": 0.15,
        "tempo": 0.15,
        "valence": 0.10,
        "acousticness": 0.10,
        "instrumentalness": 0.10,
        "loudness": 0.10,
        "speechiness": 0.10,
        "liveness": 0.05,
    }

    # YouTube-optimized weights - focusing on more reliable features
    youtube_weights = {
        "tempo": 0.25,  # More weight on tempo which is more reliably extracted
        "energy": 0.20,  # Energy is also relatively consistent
        "loudness": 0.15,  # Adjusted for YouTube extraction
        "valence": 0.15,  # Emotional tone is important
        "acousticness": 0.10,  # Moderate weight on acoustic quality
        "danceability": 0.05,  # Lower weight as YT extraction can be inconsistent
        "speechiness": 0.05,  # Lower weight due to extraction differences
        "instrumentalness": 0.03,
        "liveness": 0.02,
    }

    # Feature weights to use based on source
    weights = youtube_weights if youtube_source else standard_weights

    # If either feature set is empty or not a dict, return 0 similarity
    if not isinstance(features1, dict) or not isinstance(features2, dict):
        if detailed:
            return 0.0, {"error": "Invalid feature format"}
        return 0.0

    # Only use features that are present in both and are in our weights
    # dictionary
    common_features = (
        set(features1.keys()) & set(features2.keys()) & set(weights.keys())
    )

    # Skip if too few features are present - use a lower threshold for YouTube
    min_required_features = 2 if youtube_source else 3
    if len(common_features) < min_required_features:
        if detailed:
            return 0.0, {
                "error": f"Not enough common features, found {len(common_features)}, need {min_required_features}"
            }
        return 0.0

    feature_scores = {}
    total_score = 0
    total_weight = 0

    for feature in common_features:
        weight = weights[feature]
        total_weight += weight

        # Get values, ensuring they're numeric
        try:
            # Use safe_float to handle any type
            val1 = safe_float(features1[feature])
            val2 = safe_float(features2[feature])
        except (ValueError, TypeError, IndexError):
            # Skip if conversion fails
            continue

        # Skip NaN values
        if math.isnan(val1) or math.isnan(val2):
            continue

        # Special handling for tempo - check for tempo relationships
        # (double/half)
        if feature == "tempo":
            tempo_similarity = calculate_tempo_similarity(val1, val2)
            feature_scores[feature] = tempo_similarity
            total_score += safe_float(tempo_similarity) * weight

        # Special handling for loudness which is on a different scale
        elif feature == "loudness":
            # Normalize loudness to 0-1 scale, considering typical range (-60 to 0 dB)
            # Use a logarithmic scale to handle the dB nature of loudness
            # Allow up to 20dB difference
            normalized_diff = min(1.0, abs(val1 - val2) / 20.0)
            similarity = 1.0 - normalized_diff
            feature_scores[feature] = similarity
            total_score += safe_float(similarity) * weight

        # Standard calculation for other features
        else:
            # Calculate normalized difference
            max_range = 1.0
            if feature == "duration_ms":
                max_range = 100000  # ~1.5 minute difference = 50% similarity

            normalized_diff = min(1.0, abs(val1 - val2) / max_range)
            similarity = 1.0 - normalized_diff
            feature_scores[feature] = similarity
            total_score += safe_float(similarity) * weight

    # Normalize by total weight used
    if total_weight > 0:
        final_score = total_score / total_weight
    else:
        final_score = 0.0

    # Return detailed breakdown if requested
    if detailed:
        return final_score, feature_scores
    return final_score


def calculate_tempo_similarity(tempo1, tempo2):
    """Calculate similarity between two tempos, accounting for double/half time relationships"""
    # Direct comparison
    direct_diff = abs(tempo1 - tempo2)

    # Check for double-time relationship
    double_diff = abs(tempo1 - 2 * tempo2)
    half_diff = abs(tempo1 - 0.5 * tempo2)

    # Use the minimum difference
    min_diff = min(direct_diff, double_diff, half_diff)

    # Normalize to a similarity score (0 to 1)
    # Allow up to 20 BPM difference for full similarity range
    if min_diff <= 5:
        similarity = 1.0
    elif min_diff <= 20:
        similarity = 1.0 - (min_diff - 5) / 15.0
    else:
        # Linear falloff beyond 20 BPM
        similarity = max(0.0, 1.0 - min_diff / 40.0)

    return similarity


def flatten_features(d):
    """Helper function to flatten feature dictionaries into vectors"""
    vec = []
    # Skip non-numeric fields like 'source'
    skip_fields = ["source", "id", "track_name", "track_artist", "lyrics"]
    
    if not isinstance(d, dict):
        logger.warning(f"flatten_features received non-dict input: {type(d)}")
        return []

    for k, v in d.items():
        if k in skip_fields:
            continue
            
        try:
            if isinstance(v, list):
                # Ensure all elements in list are numeric
                numeric_elements = []
                for item in v:
                    try:
                        numeric_elements.append(safe_float(item))
                    except (ValueError, TypeError):
                        pass  # Skip non-numeric elements
                vec.extend(numeric_elements)
            else:
                # Use safe_float to handle any type including tuples
                vec.append(safe_float(v))
        except (ValueError, TypeError, IndexError) as e:
            # Skip non-numeric values with better logging
            logger.debug(f"Skipping non-numeric feature {k}: {v} ({type(v)}) - {e}")
            continue
    return vec


def compute_lyrics_similarity(lyrics1, lyrics2):
    """Compute similarity between two lyrics texts"""
    if not lyrics1 or not lyrics2:
        return 0.0
    
    # Clean and preprocess lyrics
    def preprocess_lyrics(text):
        text = text.lower()
        text = re.sub(r"\[.*?\]", "", text)  # Remove [Verse], [Chorus], etc.
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
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


def compute_audio_similarity_yt(ref_features, db_features):
    """
    Compatibility function to maintain backward compatibility with existing code
    Use the new calculate_similarity function with youtube_source=True
    """
    # Clean the features to ensure we don't have tuples instead of single values
    cleaned_ref = {}
    cleaned_db = {}
    
    # Process reference features
    for key, value in ref_features.items():
        if isinstance(value, tuple):
            # Take the first element if it's a tuple
            if len(value) > 0:
                cleaned_ref[key] = float(value[0])
        else:
            # Keep the value as is
            cleaned_ref[key] = value
    
    # Process database features
    for key, value in db_features.items():
        if isinstance(value, tuple):
            # Take the first element if it's a tuple
            if len(value) > 0:
                cleaned_db[key] = float(value[0])
        else:
            # Keep the value as is
            cleaned_db[key] = value
    
    # Use the new calculate_similarity function with cleaned features
    # Always use detailed=True to maintain compatibility with callers
    return calculate_similarity(
        cleaned_ref, cleaned_db, youtube_source=True, detailed=True
    )


def extract_numeric_values(data):
    """
    Extract numeric values from a list or single element safely.
    Useful for converting data that might be a list/tuple of values to a list of floats.
    """
    if isinstance(data, (list, tuple)):
        # Process list/tuple elements
        result = []
        for item in data:
            try:
                result.append(float(item))
            except (ValueError, TypeError):
                pass  # Skip non-numeric elements
        return result if result else [0.0]  # Return at least one element
    else:
        # Process single element
        try:
            return [float(data)]
        except (ValueError, TypeError):
            return [0.0]
