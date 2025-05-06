from alembic import command
from alembic.config import Config
import os

def stamp_database():
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate to the migrations directory
    migrations_dir = os.path.join(script_dir, '..', 'migrations')
    
    # Create Alembic configuration
    alembic_cfg = Config()
    alembic_cfg.set_main_option('script_location', migrations_dir)
    
    # Use the DATABASE_URL from environment
    database_url = os.getenv('DATABASE_URL')
    if database_url:
        alembic_cfg.set_main_option('sqlalchemy.url', database_url)
    
    # Stamp the database with current_schema_001
    command.stamp(alembic_cfg, "current_schema_001")
    print("Successfully stamped database with current_schema_001")

if __name__ == '__main__':
    stamp_database() 