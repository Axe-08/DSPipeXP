FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    ffmpeg \
    curl \
    netcat-openbsd \
    python3-dev \
    build-essential \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Create conda environment and activate it
RUN conda create -n myenv python=3.10 -y
RUN echo "source activate myenv" > ~/.bashrc
ENV PATH /opt/conda/envs/myenv/bin:$PATH

# Install Python packages
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Make prestart script executable
RUN chmod +x prestart.sh

# Create temporary directories
RUN mkdir -p /tmp/audio /tmp/cache && \
    chmod 777 /tmp/audio /tmp/cache

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8000
ENV PATH="/usr/local/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV AUDIO_STORAGE_PATH=/tmp/audio
ENV CACHE_STORAGE_PATH=/tmp/cache
ENV HOST=0.0.0.0

# Create non-root user and set permissions
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app/migrations

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/v1/health || exit 1

# Expose port
EXPOSE 8000

# Start command with prestart script
CMD ["bash", "-c", "./prestart.sh && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"] 