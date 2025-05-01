#!/bin/bash
set -e

# Enable verbose output
set -x

echo "Starting prestart.sh script..."

# Function to convert URL for psql
convert_url_for_psql() {
    local url=$1
    # Convert postgresql+asyncpg:// to postgresql:// for psql
    url=${url/postgresql+asyncpg:\/\//postgresql:\/\/}
    # Convert postgres:// to postgresql:// for consistency
    url=${url/postgres:\/\//postgresql:\/\/}
    echo "$url"
}

# Function to test database connection
test_db_connection() {
    local url=$1
    local max_attempts=3
    local attempt=1
    local connected=false

    while [ $attempt -le $max_attempts ]; do
        if [ "$connected" = false ]; then
        echo "Attempt $attempt to connect to database..."
            if PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c '\dx'; then
            echo "Successfully connected to database on attempt $attempt"
            connected=true
        else
                if [ $attempt -eq $max_attempts ]; then
                    echo "Error: Failed to connect to database after $max_attempts attempts"
                    return 1
                fi
                echo "Connection attempt $attempt failed, retrying in 5 seconds..."
                sleep 5
            fi
        fi
        attempt=$((attempt + 1))
    done

    if [ "$connected" = false ]; then
        return 1
    fi
}

# Check if DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
    echo "Error: DATABASE_URL environment variable is not set"
    exit 1
fi

echo "Parsing DATABASE_URL..."
echo "DATABASE_URL format (sensitive info redacted): ${DATABASE_URL//:*@/:***@}"

# Parse DATABASE_URL components
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
    echo "Error: Invalid DATABASE_URL format"
        exit 1
fi

echo "Successfully parsed DATABASE_URL"
echo "Host: $DB_HOST"
echo "Port: $DB_PORT"
echo "Database: $DB_NAME"
echo "User: $DB_USER"

# Test database connection
test_db_connection "$DATABASE_URL"

# Run database migrations
echo "Running database migrations..."

# First try to merge heads if multiple exist
if ! alembic upgrade head; then
    echo "Multiple heads detected, attempting to merge..."
    
    # Get current heads
    HEADS=$(alembic heads)
    if [ $? -eq 0 ]; then
        echo "Current heads: $HEADS"
        
        # Try upgrading to all heads first
        if alembic upgrade heads; then
            echo "Successfully upgraded to all heads"
            
            # Then try the merge migration
            if alembic upgrade merge_heads; then
                echo "Successfully merged migration heads"
            else
                echo "Error: Failed to merge migration heads"
                exit 1
            fi
        else
            echo "Error: Failed to upgrade to all heads"
            exit 1
        fi
    else
        echo "Error: Failed to get current migration heads"
        exit 1
    fi
fi

# Verify database health
echo "Verifying database health..."
python -c "
import asyncio
from app.core.testing import verify_database_health

async def run_health_check():
    healthy = await verify_database_health()
    if not healthy:
        raise Exception('Database health check failed')
    print('Database health check passed')

asyncio.run(run_health_check())
"

if [ $? -ne 0 ]; then
    echo "Error: Database health check failed"
    exit 1
fi

echo "Prestart script completed successfully" 