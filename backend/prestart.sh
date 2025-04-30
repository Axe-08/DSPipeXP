#!/bin/bash

# Enable verbose logging
set -x

# Debug: Print environment variables (redacting sensitive info)
echo "Checking environment..."
echo "Raw DATABASE_URL length: ${#DATABASE_URL}"
echo "DATABASE_URL format: ${DATABASE_URL//:*/:*****@*****}"
echo "PORT: $PORT"
echo "HOST: $HOST"

# Check if DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
    echo "ERROR: DATABASE_URL is not set"
    exit 1
fi

echo "Setting up database connection..."

# Extract host from DATABASE_URL for logging
DB_HOST="dpg-d08v2s49c44c73a9qeqg-a"
DB_PORT="5432"
DB_EXTERNAL_HOST="dpg-d08v2s49c44c73a9qeqg-a.oregon-postgres.render.com"

# Convert postgresql:// to postgres:// for psql
PSQL_URL=$(echo "$DATABASE_URL" | sed 's/^postgresql:/postgres:/')

echo "Using database connection info:"
echo "DB_HOST (internal): $DB_HOST"
echo "DB_HOST (external): $DB_EXTERNAL_HOST"
echo "DB_PORT: $DB_PORT"

# Wait for PostgreSQL with timeout and verbose output
echo "Waiting for PostgreSQL..."
timeout=120
counter=0

# Try different connection methods with verbose output
until (
    echo "Attempting direct connection with DATABASE_URL..." &&
    PGPASSWORD=GJE1w9Br8L4auWLfSC4jes8fwZQDtbpv psql "${PSQL_URL}" -c '\dx' 2>&1 ||
    echo "Attempting connection with internal host..." &&
    PGPASSWORD=GJE1w9Br8L4auWLfSC4jes8fwZQDtbpv psql "postgres://dspipexp_user@$DB_HOST:$DB_PORT/dspipexp" -c '\dx' 2>&1 ||
    echo "Attempting connection with external host..." &&
    PGPASSWORD=GJE1w9Br8L4auWLfSC4jes8fwZQDtbpv psql "postgres://dspipexp_user@$DB_EXTERNAL_HOST:$DB_PORT/dspipexp" -c '\dx' 2>&1 ||
    echo "Attempting TCP connection to internal host..." &&
    nc -z -w 5 $DB_HOST $DB_PORT 2>&1 ||
    echo "Attempting TCP connection to external host..." &&
    nc -z -w 5 $DB_EXTERNAL_HOST $DB_PORT 2>&1
); do
    counter=$((counter + 1))
    if [ $counter -gt $timeout ]; then
        echo "ERROR: Timeout waiting for PostgreSQL after ${timeout} seconds"
        exit 1
    fi
    echo "Waiting for PostgreSQL... ($counter/$timeout)"
    sleep 1
done
echo "PostgreSQL started successfully"

# Run migrations with verbose output
echo "Running database migrations..."
alembic upgrade head --sql || {
    echo "ERROR: Migration generation failed"
    exit 1
}
echo "Checking current migration version..."
alembic current || {
    echo "ERROR: Could not get current migration version"
    exit 1
}
echo "Migration history:"
alembic history || {
    echo "ERROR: Could not get migration history"
    exit 1
}

# Create required directories
echo "Creating storage directories..."
mkdir -p /tmp/audio /tmp/cache
chmod 777 /tmp/audio /tmp/cache

echo "Prestart tasks completed successfully" 