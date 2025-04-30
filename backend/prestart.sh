#!/bin/bash

# Debug: Print environment variables (redacting sensitive info)
echo "Checking environment..."
echo "Raw DATABASE_URL length: ${#DATABASE_URL}"
echo "DATABASE_URL first 10 chars: ${DATABASE_URL:0:10}..."
echo "DATABASE_URL format: ${DATABASE_URL//:*/:*****@*****}"
echo "PORT: $PORT"
echo "HOST: $HOST"

# Check if DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
    echo "ERROR: DATABASE_URL is not set"
    exit 1
fi

echo "Setting up database connection..."

# Extract host and port from DATABASE_URL for nc check
DB_HOST="dpg-d08v2s49c44c73a9qeqg-a"
DB_PORT="5432"

echo "Using database connection info:"
echo "DB_HOST: $DB_HOST"
echo "DB_PORT: $DB_PORT"

# Wait for PostgreSQL with timeout
echo "Waiting for PostgreSQL at $DB_HOST:$DB_PORT..."
timeout=120
counter=0
until PGPASSWORD=${DATABASE_URL#*:*:} psql "${DATABASE_URL}" -c '\q' >/dev/null 2>&1; do
    counter=$((counter + 1))
    if [ $counter -gt $timeout ]; then
        echo "ERROR: Timeout waiting for PostgreSQL after ${timeout} seconds"
        exit 1
    fi
    echo "Waiting for PostgreSQL... ($counter/$timeout)"
    sleep 1
done
echo "PostgreSQL started successfully"

# Run migrations
echo "Running database migrations..."
alembic upgrade head

# Create required directories
echo "Creating storage directories..."
mkdir -p /tmp/audio /tmp/cache
chmod 777 /tmp/audio /tmp/cache

echo "Prestart tasks completed successfully" 