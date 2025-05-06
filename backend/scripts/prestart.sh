#!/bin/bash
set -e

# Debug: Print current directory and Python path
echo "Current directory: $(pwd)"
echo "Python executable: $(which python)"
echo "PYTHONPATH: $PYTHONPATH"

# Start message
echo "Starting prestart.sh script..."

# Check if DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
    echo "ERROR: DATABASE_URL environment variable is not set"
    exit 1
fi

# Parse DATABASE_URL
echo "Parsing DATABASE_URL..."
echo "DATABASE_URL format (sensitive info redacted): ${DATABASE_URL//:*@/:***@}"

# Extract database connection details using regex
if [[ $DATABASE_URL =~ ^(postgresql(\+asyncpg)?|postgres)://([^:]+):([^@]+)@([^:]+):([0-9]+)/(.+)$ ]]; then
    export DB_USER="${BASH_REMATCH[3]}"
    export DB_PASSWORD="${BASH_REMATCH[4]}"
    export DB_HOST="${BASH_REMATCH[5]}"
    export DB_PORT="${BASH_REMATCH[6]}"
    export DB_NAME="${BASH_REMATCH[7]}"
elif [[ $DATABASE_URL =~ ^(postgresql(\+asyncpg)?|postgres)://([^:]+):([^@]+)@([^/]+)/(.+)$ ]]; then
    export DB_USER="${BASH_REMATCH[3]}"
    export DB_PASSWORD="${BASH_REMATCH[4]}"
    export DB_HOST="${BASH_REMATCH[5]}"
    export DB_PORT="5432"
    export DB_NAME="${BASH_REMATCH[6]}"
else
    echo "ERROR: Invalid DATABASE_URL format"
    exit 1
fi

echo "Successfully parsed DATABASE_URL"
echo "Host: $DB_HOST"
echo "Port: $DB_PORT"
echo "Database: $DB_NAME"
echo "User: $DB_USER"

# Function to test database connection
test_db_connection() {
    local url=$1
    local max_attempts=3
    local attempt=1
    local connected=false

    while [ $attempt -le $max_attempts ]; do
        if [ "$connected" = false ]; then
            echo "Attempt $attempt to connect to database..."
            export PGPASSWORD=$DB_PASSWORD
            if psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c '\dx'; then
                echo "Successfully connected to database on attempt $attempt"
                connected=true
            else
                echo "Failed to connect to database on attempt $attempt"
                if [ $attempt -eq $max_attempts ]; then
                    echo "ERROR: Failed to connect to database after $max_attempts attempts"
                    exit 1
                fi
                sleep 5
            fi
        fi
        attempt=$((attempt + 1))
    done
}

# Test database connection
test_db_connection "$DATABASE_URL"

# Run database schema updates
echo "Running schema updates..."
export PGPASSWORD=$DB_PASSWORD
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f scripts/add_youtube_columns.sql

# Run database migrations (for other tables/changes)
echo "Running database migrations..."
alembic upgrade head

# Run database health check
echo "Verifying database health..."

# Create a temporary health check script
cat > health_check_temp.py << 'EOL'
import asyncio
import logging
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

logger = logging.getLogger(__name__)

async def run_health_check():
    try:
        logger.info("Importing verify_database_health...")
        from app.core.testing import verify_database_health
        
        logger.info("Starting database health check...")
        healthy = await verify_database_health()
        if not healthy:
            logger.error("Database health check failed")
            sys.exit(1)
        logger.info("Database health check passed")
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error(f"Current directory: {os.getcwd()}")
        logger.error(f"Python path: {sys.path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unhandled exception in health check: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(run_health_check())
EOL

# Run the health check with proper Python path
PYTHONPATH=$PYTHONPATH:$(pwd) python health_check_temp.py 