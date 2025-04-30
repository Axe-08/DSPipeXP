#!/bin/bash

# Extract host and port from DATABASE_URL
if [ -z "$DATABASE_URL" ]; then
    echo "DATABASE_URL is not set"
    exit 1
fi

# Parse DATABASE_URL to get host and port
# Example URL: postgresql://user:pass@host:5432/db
DB_HOST=$(echo $DATABASE_URL | sed -n 's/.*@\([^:]*\).*/\1/p')
DB_PORT=$(echo $DATABASE_URL | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')

if [ -z "$DB_HOST" ] || [ -z "$DB_PORT" ]; then
    echo "Could not parse DATABASE_URL"
    exit 1
fi

# Wait for PostgreSQL with timeout
echo "Waiting for PostgreSQL at $DB_HOST:$DB_PORT..."
timeout=120
counter=0
until nc -z $DB_HOST $DB_PORT; do
    counter=$((counter + 1))
    if [ $counter -gt $timeout ]; then
        echo "Timeout waiting for PostgreSQL"
        exit 1
    fi
    echo "Waiting for PostgreSQL... ($counter/$timeout)"
    sleep 1
done
echo "PostgreSQL started"

# Run migrations
echo "Running database migrations..."
alembic upgrade head

# Create required directories
mkdir -p /tmp/audio /tmp/cache
chmod 777 /tmp/audio /tmp/cache

echo "Prestart tasks completed" 