"""merge multiple heads

Revision ID: merge_heads
Revises: add_fuzzy_search, 20250429_2330_003
Create Date: 2024-03-31 14:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
import logging

# revision identifiers, used by Alembic.
revision = 'merge_heads'
down_revision = ('add_fuzzy_search', '20250429_2330_003')  # Both head revisions
branch_labels = None
depends_on = None

logger = logging.getLogger(__name__)

def upgrade() -> None:
    """Merge point for multiple heads"""
    logger.info("Merging multiple migration heads")
    pass

def downgrade() -> None:
    """Downgrade point for merged heads"""
    logger.info("Downgrading merged heads")
    pass 