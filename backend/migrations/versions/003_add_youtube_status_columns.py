"""Add YouTube status columns

Revision ID: 20250429_2330_003
Revises: 20250429_2300_002
Create Date: 2024-04-29 23:30:37.904853

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '20250429_2330_003'
down_revision = '20250429_2300_002'
branch_labels = None
depends_on = None

def upgrade():
    # Add YouTube status columns
    op.add_column('songs', sa.Column('youtube_url_updated_at', sa.DateTime(), nullable=True))
    op.add_column('songs', sa.Column('youtube_url_status', sa.String(), nullable=True))
    op.add_column('songs', sa.Column('youtube_url_error', sa.String(), nullable=True))

def downgrade():
    # Remove YouTube status columns
    op.drop_column('songs', 'youtube_url_error')
    op.drop_column('songs', 'youtube_url_status')
    op.drop_column('songs', 'youtube_url_updated_at') 