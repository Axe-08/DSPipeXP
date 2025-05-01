"""add youtube url columns

Revision ID: add_youtube_url_columns
Revises: 
Create Date: 2024-03-31

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'add_youtube_url_columns'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add new columns for YouTube URL management
    op.add_column('songs', sa.Column('youtube_url_updated_at', sa.DateTime(), nullable=True))
    op.add_column('songs', sa.Column('youtube_url_status', sa.String(), nullable=True))
    op.add_column('songs', sa.Column('youtube_url_error', sa.String(), nullable=True))


def downgrade() -> None:
    # Remove the columns if needed
    op.drop_column('songs', 'youtube_url_error')
    op.drop_column('songs', 'youtube_url_status')
    op.drop_column('songs', 'youtube_url_updated_at') 