"""add fuzzy search capability

Revision ID: add_fuzzy_search
Create Date: 2024-03-31 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text
import logging

# Set up logging
logger = logging.getLogger(__name__)

# revision identifiers, used by Alembic.
revision = 'add_fuzzy_search'
down_revision = None
branch_labels = None
depends_on = None

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
        
        # Enable pg_trgm extension for fuzzy text search
        logger.info("Creating pg_trgm extension...")
        op.execute('CREATE EXTENSION IF NOT EXISTS pg_trgm;')
        
        if not verify_extension():
            raise Exception("Failed to create pg_trgm extension")
        logger.info("Successfully created pg_trgm extension")
        
        # Create indexes for faster fuzzy search
        logger.info("Creating trigram indexes...")
        op.execute(
            'CREATE INDEX idx_song_track_name_trgm ON songs USING gin (track_name gin_trgm_ops);'
        )
        op.execute(
            'CREATE INDEX idx_song_artist_trgm ON songs USING gin (track_artist gin_trgm_ops);'
        )
        
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
        op.execute('DROP INDEX IF EXISTS idx_song_track_name_trgm;')
        op.execute('DROP INDEX IF EXISTS idx_song_artist_trgm;')
        
        # Verify indexes are removed
        if verify_indexes():
            raise Exception("Failed to remove trigram indexes")
        logger.info("Successfully removed trigram indexes")
        
        # Disable the extension
        logger.info("Removing pg_trgm extension...")
        op.execute('DROP EXTENSION IF EXISTS pg_trgm;')
        
        # Verify extension is removed
        if verify_extension():
            raise Exception("Failed to remove pg_trgm extension")
        logger.info("Successfully removed pg_trgm extension")
        
        logger.info("Fuzzy search migration downgrade completed successfully")
        
    except Exception as e:
        logger.error(f"Downgrade failed: {e}")
        raise 