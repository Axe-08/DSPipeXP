import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Dict, Any, List, Tuple
import json
from .database import db_manager, Song
from .config import settings
from sqlalchemy.sql import text
import os
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import select, func

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize data loader with optional custom data directory"""
        self.data_dir = data_dir if data_dir is not None else settings.DATA_DIR
        # Look for the file in multiple possible locations
        possible_paths = [
            Path("/app/data/spotify_songs.csv"),
            Path("/app/app/data/spotify_songs.csv"),
            Path(os.path.join(os.getcwd(), "app/data/spotify_songs.csv")),
            Path(os.path.join(os.getcwd(), "data/spotify_songs.csv"))
        ]
        
        for path in possible_paths:
            if path.exists():
                self.dataset_path = path
                logger.info(f"Found dataset at {path}")
                break
        else:
            raise FileNotFoundError(f"Dataset not found in any of: {possible_paths}")
        
    def process_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the dataset and clean data"""
        # Drop any completely empty rows
        df = df.dropna(how='all')
        
        # Fill NaN values with appropriate defaults for string columns
        string_cols = ['track_name', 'track_artist', 'track_album_name', 'playlist_genre', 'lyrics']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown').astype(str)
        
        # Convert numeric columns to appropriate types with NaN handling
        numeric_cols = {
            'danceability': 0.0,
            'energy': 0.0,
            'key': 0,
            'loudness': 0.0,
            'mode': 0,
            'speechiness': 0.0,
            'acousticness': 0.0,
            'instrumentalness': 0.0,
            'liveness': 0.0,
            'valence': 0.0,
            'tempo': 0.0,
            'duration_ms': 0
        }
        
        # Convert each numeric column with proper type handling
        for col, default in numeric_cols.items():
            if col in df.columns:
                # Convert to numeric, replace NaN with default
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default)
                
                # Convert to proper type (int or float)
                if isinstance(default, int):
                    df[col] = df[col].astype(int)
                else:
                    df[col] = df[col].astype(float)
        
        # Convert audio features to JSON string with explicit type conversion
        df['audio_features'] = df.apply(lambda row: json.dumps({
            'danceability': float(row['danceability']),
            'energy': float(row['energy']),
            'key': int(row['key']),
            'loudness': float(row['loudness']),
            'mode': int(row['mode']),
            'speechiness': float(row['speechiness']),
            'acousticness': float(row['acousticness']),
            'instrumentalness': float(row['instrumentalness']),
            'liveness': float(row['liveness']),
            'valence': float(row['valence']),
            'tempo': float(row['tempo']),
            'duration_ms': int(row['duration_ms'])
        }), axis=1)
        
        # Remove duplicates based on track_name and track_artist
        df = df.drop_duplicates(subset=['track_name', 'track_artist'], keep='first')
        
        # Ensure all required columns are present and properly formatted
        required_cols = ['track_name', 'track_artist', 'track_album_name', 'playlist_genre', 'lyrics', 'audio_features']
        return df[required_cols]

    async def get_song_count(self, session: AsyncSession) -> int:
        """Get the current count of songs in the database"""
        result = await session.execute(select(func.count()).select_from(Song))
        return result.scalar() or 0

    async def load_songs(self, session: AsyncSession, chunk_size: int = 500) -> Tuple[int, int]:
        """Load songs into database with improved error handling and progress tracking"""
        try:
            # Read and process the dataset
            logger.info(f"Reading dataset from {self.dataset_path}")
            df = pd.read_csv(self.dataset_path)
            initial_count = len(df)
            logger.info(f"Initial dataset size: {initial_count} rows")
            
            df = self.process_dataset(df)
            processed_count = len(df)
            logger.info(f"After processing and deduplication: {processed_count} rows")
            
            # Get initial song count
            initial_db_count = await self.get_song_count(session)
            logger.info(f"Current songs in database: {initial_db_count}")
            
            processed_rows = 0
            successful_inserts = 0
            error_count = 0
            
            # Process in chunks
            for i in range(0, processed_count, chunk_size):
                chunk = df.iloc[i:i+chunk_size]
                chunk_data = chunk.to_dict('records')
                
                try:
                    # Prepare the data for insertion with explicit type conversion
                    values = [
                        {
                            'track_name': str(row['track_name']).strip(),
                            'track_artist': str(row['track_artist']).strip(),
                            'track_album_name': str(row['track_album_name']).strip(),
                            'playlist_genre': str(row['playlist_genre']).strip(),
                            'lyrics': str(row['lyrics']).strip(),
                            'audio_features': row['audio_features'],
                            'is_original': False
                        }
                        for row in chunk_data
                    ]
                    
                    # Use PostgreSQL's INSERT ... ON CONFLICT DO NOTHING
                    stmt = insert(Song).values(values).on_conflict_do_nothing(
                        index_elements=['track_name', 'track_artist']
                    )
                    
                    result = await session.execute(stmt)
                    await session.commit()
                    
                    # Update progress
                    processed_rows += len(chunk)
                    successful_inserts += result.rowcount
                    
                    if i % (chunk_size * 5) == 0:  # Log every 5 chunks
                        logger.info(f"Progress: {processed_rows}/{processed_count} rows. Inserted: {successful_inserts}")
                    
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error processing chunk {i//chunk_size}: {str(e)}")
                    await session.rollback()
                    if error_count > 10:  # Stop if too many errors
                        logger.error("Too many errors, stopping data load")
                        break
                    continue
            
            final_db_count = await self.get_song_count(session)
            logger.info(f"""Data loading completed:
                Initial dataset size: {initial_count}
                Processed rows: {processed_count}
                Successfully inserted: {successful_inserts}
                Total songs in DB: {final_db_count}
                Errors encountered: {error_count}""")
            
            return successful_inserts, final_db_count
            
        except Exception as e:
            logger.error(f"Fatal error in load_songs: {str(e)}")
            raise

    async def load_initial_data(self):
        """Load initial data into the database if it's empty"""
        async with db_manager.SessionLocal() as session:
            # Check if database is empty
            count = await self.get_song_count(session)
            if count == 0:
                logger.info("Database is empty, loading initial data...")
                loaded, total = await self.load_songs(session)
                logger.info(f"Loaded {loaded} songs out of {total} total")
            else:
                logger.info(f"Database already contains {count} songs, skipping initial load")

# Create global instance
data_loader = DataLoader() 