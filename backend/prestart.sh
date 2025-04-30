#!/bin/bash

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

    while [ $attempt -le $max_attempts ] && [ "$connected" = false ]; do
        echo "Attempt $attempt to connect to database..."
        
        if PGPASSWORD=$DB_PASSWORD psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c '\dx' > /dev/null 2>&1; then
            echo "Successfully connected to database on attempt $attempt"
            connected=true
        else
            echo "Connection attempt $attempt failed"
            if [ $attempt -lt $max_attempts ]; then
                echo "Waiting 5 seconds before next attempt..."
                sleep 5
            fi
        fi
        attempt=$((attempt + 1))
    done

    if [ "$connected" = false ]; then
        echo "Failed to connect to database after $max_attempts attempts"
        return 1
    fi
    return 0
}

# Check if DATABASE_URL is set
if [ -z "${DATABASE_URL}" ]; then
    echo "Error: DATABASE_URL is not set"
    exit 1
fi

echo "Parsing DATABASE_URL..."
echo "DATABASE_URL format (sensitive info redacted): ${DATABASE_URL//:[^@]*@/:***@}"

# First try parsing URL with port number
if [[ $DATABASE_URL =~ ^(postgresql(\+asyncpg)?|postgres)://([^:]+):([^@]+)@([^:]+):([0-9]+)/(.+)$ ]]; then
    export DB_USER="${BASH_REMATCH[3]}"
    export DB_PASSWORD="${BASH_REMATCH[4]}"
    export DB_HOST="${BASH_REMATCH[5]}"
    export DB_PORT="${BASH_REMATCH[6]}"
    export DB_NAME="${BASH_REMATCH[7]}"
else
    # Try parsing URL without port number
    if [[ $DATABASE_URL =~ ^(postgresql(\+asyncpg)?|postgres)://([^:]+):([^@]+)@([^/]+)/(.+)$ ]]; then
        export DB_USER="${BASH_REMATCH[3]}"
        export DB_PASSWORD="${BASH_REMATCH[4]}"
        export DB_HOST="${BASH_REMATCH[5]}"
        export DB_PORT="5432"  # Default PostgreSQL port
        export DB_NAME="${BASH_REMATCH[6]}"
    else
        echo "Error: Could not parse DATABASE_URL"
        echo "Expected format: postgresql[+asyncpg]://user:password@host[:port]/dbname"
        exit 1
    fi
fi

echo "Successfully parsed DATABASE_URL"
echo "Host: $DB_HOST"
echo "Port: $DB_PORT"
echo "Database: $DB_NAME"
echo "User: $DB_USER"

# Test database connection
if ! test_db_connection "$DATABASE_URL"; then
    echo "Error: Could not connect to database"
    exit 1
fi

echo "Running database migrations..."
# Run migrations
if ! alembic upgrade head; then
    echo "Error: Database migration failed"
    exit 1
fi

echo "Verifying migrations..."
# Verify migrations
echo "Current migration version:"
alembic current

echo "Migration history:"
alembic history --verbose

echo "Prestart script completed successfully" 