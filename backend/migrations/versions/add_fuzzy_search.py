"""add fuzzy search capability

Revision ID: 20250429_2330_003
Create Date: 2024-03-31 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text
import logging
import time

# Set up logging
logger = logging.getLogger(__name__)

# revision identifiers, used by Alembic.
revision = '20250429_2330_003'
down_revision = None  # Set to None to make it a new root
branch_labels = None
depends_on = None

def retry_operation(func, max_attempts=3, delay=2):
    """Retry an operation with exponential backoff"""
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
            time.sleep(delay * (2 ** attempt))

def verify_extension() -> bool:
    """Verify pg_trgm extension is properly installed"""
    conn = op.get_bind()
    result = conn.execute(text("SELECT * FROM pg_extension WHERE extname = 'pg_trgm'"))
    return bool(result.first())

def verify_indexes() -> bool:
    """Verify the trigram indexes are properly created"""
    conn = op.get_bind()
    result = conn.execute(text("""
        SELECT indexname, indexdef 
        FROM pg_indexes 
        WHERE indexname IN ('idx_song_track_name_trgm', 'idx_song_artist_trgm')
    """))
    indexes = result.fetchall()
    return len(indexes) == 2

def verify_data_integrity() -> bool:
    """Verify data integrity after migration"""
    conn = op.get_bind()
    try:
        # Check if we can perform similarity searches
        result = conn.execute(text("""
            SELECT similarity('test', track_name), similarity('test', track_artist)
            FROM songs LIMIT 1
        """))
        result.first()
        return True
    except Exception as e:
        logger.error(f"Data integrity check failed: {e}")
        return False

def upgrade() -> None:
    try:
        logger.info("Starting fuzzy search migration upgrade...")
        
        # Drop any existing indexes first to avoid conflicts
        def drop_indexes():
            op.execute('DROP INDEX IF EXISTS idx_song_track_name_trgm;')
            op.execute('DROP INDEX IF EXISTS idx_song_artist_trgm;')
        retry_operation(drop_indexes)
        
        # Enable pg_trgm extension for fuzzy text search
        logger.info("Creating pg_trgm extension...")
        def create_extension():
            op.execute('CREATE EXTENSION IF NOT EXISTS pg_trgm;')
        retry_operation(create_extension)
        
        if not verify_extension():
            raise Exception("Failed to create pg_trgm extension")
        logger.info("Successfully created pg_trgm extension")
        
        # Create indexes for faster fuzzy search, with CONCURRENTLY to avoid locking
        logger.info("Creating trigram indexes...")
        def create_track_name_index():
            op.execute(
                'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_song_track_name_trgm ON songs USING gin (track_name gin_trgm_ops);'
            )
        retry_operation(create_track_name_index)
        
        def create_artist_index():
            op.execute(
                'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_song_artist_trgm ON songs USING gin (track_artist gin_trgm_ops);'
            )
        retry_operation(create_artist_index)
        
        if not verify_indexes():
            raise Exception("Failed to create trigram indexes")
        logger.info("Successfully created trigram indexes")
        
        # Verify data integrity
        if not verify_data_integrity():
            raise Exception("Data integrity check failed after migration")
        logger.info("Data integrity verified successfully")
        
        logger.info("Fuzzy search migration completed successfully")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise

def downgrade() -> None:
    try:
        logger.info("Starting fuzzy search migration downgrade...")
        
        # Remove the indexes
        logger.info("Removing trigram indexes...")
        def drop_indexes():
            op.execute('DROP INDEX IF EXISTS idx_song_track_name_trgm;')
            op.execute('DROP INDEX IF EXISTS idx_song_artist_trgm;')
        retry_operation(drop_indexes)
        
        # Verify indexes are removed
        if verify_indexes():
            raise Exception("Failed to remove trigram indexes")
        logger.info("Successfully removed trigram indexes")
        
        # Disable the extension
        logger.info("Removing pg_trgm extension...")
        def drop_extension():
            op.execute('DROP EXTENSION IF EXISTS pg_trgm;')
        retry_operation(drop_extension)
        
        # Verify extension is removed
        if verify_extension():
            raise Exception("Failed to remove pg_trgm extension")
        logger.info("Successfully removed pg_trgm extension")
        
        logger.info("Fuzzy search migration downgrade completed successfully")
        
    except Exception as e:
        logger.error(f"Downgrade failed: {e}")
        raise 