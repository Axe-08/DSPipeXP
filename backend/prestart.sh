#!/bin/bash

# Debug: Print environment variables (redacting sensitive info)
echo "Checking environment..."
echo "DATABASE_URL format: ${DATABASE_URL//:*/:*****@*****}"
echo "PORT: $PORT"
echo "HOST: $HOST"

# Extract host and port from DATABASE_URL
if [ -z "$DATABASE_URL" ]; then
    echo "ERROR: DATABASE_URL is not set"
    exit 1
fi

# Parse DATABASE_URL using more robust pattern matching
if [[ $DATABASE_URL =~ ^postgres(ql)?://[^:]+:[^@]+@([^:/]+):([0-9]+)/.*$ ]]; then
    DB_HOST="${BASH_REMATCH[2]}"
    DB_PORT="${BASH_REMATCH[3]}"
else
    echo "ERROR: DATABASE_URL format not recognized"
    echo "Expected format: postgresql://user:pass@host:port/db"
    exit 1
fi

# Debug: Print parsed values (without credentials)
echo "Parsed DB_HOST: $DB_HOST"
echo "Parsed DB_PORT: $DB_PORT"

if [ -z "$DB_HOST" ] || [ -z "$DB_PORT" ]; then
    echo "ERROR: Could not parse DATABASE_URL"
    echo "DB_HOST or DB_PORT is empty"
    exit 1
fi

# Wait for PostgreSQL with timeout
echo "Waiting for PostgreSQL at $DB_HOST:$DB_PORT..."
timeout=120
counter=0
until nc -z $DB_HOST $DB_PORT; do
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