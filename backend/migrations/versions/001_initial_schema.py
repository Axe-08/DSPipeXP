"""Initial schema

Revision ID: 20250429_2235_001
Revises: 
Create Date: 2024-04-29 22:35:37.904853

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = '20250429_2235_001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Drop existing tables if they exist
    op.execute('DROP TABLE IF EXISTS vector_store CASCADE')
    op.execute('DROP TABLE IF EXISTS songs CASCADE')
    
    # Create songs table
    op.create_table(
        'songs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('track_name', sa.String(), nullable=False),
        sa.Column('track_artist', sa.String(), nullable=False),
        sa.Column('track_album_name', sa.String(), nullable=True),
        sa.Column('playlist_genre', sa.String(), nullable=True),
        sa.Column('lyrics', sa.Text(), nullable=True),
        sa.Column('clean_lyrics', sa.Text(), nullable=True),
        sa.Column('word2vec_features', JSONB, nullable=True),
        sa.Column('audio_features', JSONB, nullable=True),
        sa.Column('sentiment_features', JSONB, nullable=True),
        sa.Column('topic_features', JSONB, nullable=True),
        sa.Column('youtube_url', sa.String(), nullable=True),
        sa.Column('audio_path', sa.String(), nullable=True),
        sa.Column('is_original', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('added_date', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create vector store table
    op.create_table(
        'vector_store',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('song_id', sa.Integer(), nullable=False),
        sa.Column('feature_vector', sa.ARRAY(sa.Float), nullable=False),
        sa.Column('feature_type', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['song_id'], ['songs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes
    op.create_index('ix_songs_track_name', 'songs', ['track_name'])
    op.create_index('ix_songs_track_artist', 'songs', ['track_artist'])
    op.create_index('ix_songs_playlist_genre', 'songs', ['playlist_genre'])
    op.create_index('ix_vector_store_song_id', 'vector_store', ['song_id'])
    op.create_index('ix_vector_store_feature_type', 'vector_store', ['feature_type'])

def downgrade():
    op.drop_index('ix_vector_store_feature_type')
    op.drop_index('ix_vector_store_song_id')
    op.drop_index('ix_songs_playlist_genre')
    op.drop_index('ix_songs_track_artist')
    op.drop_index('ix_songs_track_name')
    op.drop_table('vector_store')
    op.drop_table('songs') 