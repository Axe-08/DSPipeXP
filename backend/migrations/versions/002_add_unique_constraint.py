"""Add unique constraint on track_name and track_artist

Revision ID: 20250429_2300_002
Revises: 20250429_2235_001
Create Date: 2024-04-29 23:00:37.904853

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '20250429_2300_002'
down_revision = '20250429_2235_001'
branch_labels = None
depends_on = None

def upgrade():
    # Add unique constraint
    op.create_unique_constraint(
        'uq_songs_track_name_artist',
        'songs',
        ['track_name', 'track_artist']
    )

def downgrade():
    # Remove unique constraint
    op.drop_constraint(
        'uq_songs_track_name_artist',
        'songs',
        type_='unique'
    ) 