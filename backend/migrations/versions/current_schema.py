"""Current schema state

Revision ID: current_schema_001
Revises: 
Create Date: 2024-04-30 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = 'current_schema_001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # This is just a stamp migration - the schema is already in place
    pass

def downgrade():
    # This is just a stamp migration - no downgrade needed
    pass 