#!/bin/bash

# Wait for PostgreSQL with timeout
echo "Waiting for PostgreSQL..."
timeout=60
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