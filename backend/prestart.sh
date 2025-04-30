#!/bin/bash

# Debug: Print environment variables (redacting sensitive info)
echo "Checking environment..."
echo "Raw DATABASE_URL length: ${#DATABASE_URL}"
echo "DATABASE_URL first 10 chars: ${DATABASE_URL:0:10}..."
echo "DATABASE_URL format: ${DATABASE_URL//:*/:*****@*****}"
echo "PORT: $PORT"
echo "HOST: $HOST"

# Extract host and port from DATABASE_URL
if [ -z "$DATABASE_URL" ]; then
    echo "ERROR: DATABASE_URL is not set"
    exit 1
fi

echo "Attempting to parse DATABASE_URL..."

# Try different parsing methods
# Method 1: Using grep
if echo "$DATABASE_URL" | grep -q "^postgresql://"; then
    echo "Found postgresql:// prefix"
    DB_HOST=$(echo "$DATABASE_URL" | grep -oP '(?<=@)[^:]+(?=:)')
    DB_PORT=$(echo "$DATABASE_URL" | grep -oP '(?<=:)[0-9]+(?=/)')
    echo "Method 1 parsing results - DB_HOST: $DB_HOST, DB_PORT: $DB_PORT"
fi

# Method 2: Using sed
if [ -z "$DB_HOST" ] || [ -z "$DB_PORT" ]; then
    echo "Trying method 2..."
    DB_HOST=$(echo "$DATABASE_URL" | sed -n 's/.*@\([^:]*\).*/\1/p')
    DB_PORT=$(echo "$DATABASE_URL" | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
    echo "Method 2 parsing results - DB_HOST: $DB_HOST, DB_PORT: $DB_PORT"
fi

# Method 3: Using bash regex
if [ -z "$DB_HOST" ] || [ -z "$DB_PORT" ]; then
    echo "Trying method 3..."
    if [[ $DATABASE_URL =~ @([^:]+):([0-9]+)/ ]]; then
        DB_HOST="${BASH_REMATCH[1]}"
        DB_PORT="${BASH_REMATCH[2]}"
        echo "Method 3 parsing results - DB_HOST: $DB_HOST, DB_PORT: $DB_PORT"
    fi
fi

# Final check
if [ -z "$DB_HOST" ] || [ -z "$DB_PORT" ]; then
    echo "ERROR: Could not parse DATABASE_URL"
    echo "DB_HOST or DB_PORT is empty"
    echo "Please ensure DATABASE_URL is in format: postgresql://user:pass@host:port/db"
    exit 1
fi

echo "Successfully parsed database connection info:"
echo "DB_HOST: $DB_HOST"
echo "DB_PORT: $DB_PORT"

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