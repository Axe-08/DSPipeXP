import os
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import json
import argparse
import sys

def setup_database(db_name, db_user, db_password, db_host="localhost", db_port=5432, force_recreate=False):
    """
    Set up a PostgreSQL database for DSPipeXP with tables and initial data
    from the spotify_songs.csv dataset.
    """
    print("Starting database initialization...")
    
    # Connect to postgres database to create our application database
    try:
        # First connect to 'postgres' database to create our app database
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            database="postgres",
            user=db_user,
            password=db_password
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Check if our database exists
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
        exists = cursor.fetchone()
        
        if exists and force_recreate:
            print(f"Database '{db_name}' exists. Dropping and recreating...")
            cursor.execute(f"DROP DATABASE IF EXISTS {db_name}")
            exists = None
        
        if not exists:
            print(f"Creating database '{db_name}'...")
            cursor.execute(f"CREATE DATABASE {db_name}")
            print(f"Database '{db_name}' created successfully.")
        else:
            print(f"Database '{db_name}' already exists.")
        
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error creating database: {e}")
        return False
    
    # Connect to our application database
    try:
        # Create SQLAlchemy engine
        db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        engine = create_engine(db_url)
        print(f"Connected to database '{db_name}'")
        
        # Create songs table
        print("Creating songs table...")
        create_songs_table_sql = """
        CREATE TABLE IF NOT EXISTS songs (
            id SERIAL PRIMARY KEY,
            track_id VARCHAR(255) UNIQUE,
            track_name VARCHAR(255),
            track_artist VARCHAR(255),
            track_album_name VARCHAR(255),
            playlist_name VARCHAR(255),
            playlist_genre VARCHAR(255),
            lyrics TEXT,
            audio_features JSONB,
            sentiment_features JSONB,
            word2vec_features JSONB,
            topic_features JSONB,
            spotify_url VARCHAR(255),
            youtube_url VARCHAR(255),
            duration_ms INTEGER,
            is_original BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        with engine.connect() as conn:
            conn.execute(create_songs_table_sql)
        
        # Find the dataset file
        dataset_path = find_dataset_file()
        if not dataset_path:
            print("Error: Could not find spotify_songs.csv dataset file.")
            return False
        
        print(f"Loading data from {dataset_path}...")
        df = pd.read_csv(dataset_path)
        print(f"Loaded {len(df)} records from dataset.")
        
        # Process and import the data
        print("Processing and importing data. This may take a few minutes...")
        
        # Convert audio features to JSON format
        audio_feature_columns = [
            'danceability', 'energy', 'key', 'loudness', 
            'mode', 'speechiness', 'acousticness', 'instrumentalness',
            'liveness', 'valence', 'tempo', 'duration_ms'
        ]
        
        # Create a smaller batch size to prevent memory issues
        batch_size = 1000
        total_records = len(df)
        imported_count = 0
        
        for start_idx in range(0, total_records, batch_size):
            end_idx = min(start_idx + batch_size, total_records)
            batch_df = df.iloc[start_idx:end_idx].copy()
            
            # Process this batch
            batch_df['audio_features'] = batch_df[audio_feature_columns].apply(
                lambda row: json.dumps({col: row[col] for col in audio_feature_columns}), 
                axis=1
            )
            
            # Select and rename columns for the songs table
            song_data = batch_df[[
                'track_id', 'track_name', 'track_artist', 'track_album_name',
                'playlist_name', 'playlist_genre', 'audio_features', 'duration_ms'
            ]].copy()
            
            # Add empty columns for features we'll calculate later
            song_data['lyrics'] = None
            song_data['sentiment_features'] = json.dumps({})
            song_data['word2vec_features'] = json.dumps({})
            song_data['topic_features'] = json.dumps({})
            song_data['youtube_url'] = None
            song_data['spotify_url'] = None
            song_data['is_original'] = False
            
            # Import into database
            song_data.to_sql('songs', engine, if_exists='append', index=False)
            
            imported_count += len(batch_df)
            print(f"Imported {imported_count} of {total_records} records...")
        
        # Create indexes for faster queries
        print("Creating indexes...")
        create_indexes_sql = """
        CREATE INDEX IF NOT EXISTS idx_track_name ON songs (track_name);
        CREATE INDEX IF NOT EXISTS idx_track_artist ON songs (track_artist);
        CREATE INDEX IF NOT EXISTS idx_playlist_genre ON songs (playlist_genre);
        """
        with engine.connect() as conn:
            conn.execute(create_indexes_sql)
        
        print("\nDatabase setup complete! 🎉")
        print(f"\nYour database connection string is: postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")
        print("\nTo use this database with DSPipeXP:")
        print("1. Create a file at '.streamlit/secrets.toml' with the following content:")
        print(f"""
# Database Configuration
db_url = "postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

# Other configuration
[api_keys]
genius_access_token = "your_genius_token_here"  # Optional, for lyrics fetching
        """)
        print("\n2. Now you can run the Streamlit app with: streamlit run dspipexp_streamlit/app.py")
        
        return True
    
    except Exception as e:
        print(f"Error setting up database: {e}")
        return False

def find_dataset_file():
    """Find the spotify_songs.csv file in standard locations"""
    possible_paths = [
        "lyric_dataset/spotify_songs.csv",
        "../lyric_dataset/spotify_songs.csv",
        "../../lyric_dataset/spotify_songs.csv",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lyric_dataset/spotify_songs.csv")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Initialize a local PostgreSQL database for DSPipeXP")
    parser.add_argument("--db-name", default="dspipexp", help="Database name (default: dspipexp)")
    parser.add_argument("--db-user", required=True, help="PostgreSQL username")
    parser.add_argument("--db-password", required=True, help="PostgreSQL password")
    parser.add_argument("--db-host", default="localhost", help="Database host (default: localhost)")
    parser.add_argument("--db-port", default=5432, type=int, help="Database port (default: 5432)")
    parser.add_argument("--force-recreate", action="store_true", help="Force recreation of database if it exists")
    
    args = parser.parse_args()
    
    success = setup_database(
        args.db_name, 
        args.db_user, 
        args.db_password, 
        args.db_host, 
        args.db_port,
        args.force_recreate
    )
    
    if not success:
        print("Database initialization failed.")
        sys.exit(1)

if __name__ == "__main__":
    main() 