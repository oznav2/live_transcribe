# Multi-stage build for optimized image size
FROM python:3.11-slim AS base

# Install system dependencies including FFmpeg and build tools
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY static/ ./static/

# Create directory for Whisper models and copy the converted Ivrit model
RUN mkdir -p /root/.cache/whisper && \
    mkdir -p /app/models
COPY models/ivrit-whisper-large-v3-turbo.bin /app/models/

# Expose port
EXPOSE 8009

# Environment variables
ENV WHISPER_MODEL=ivrit-large-v3-turbo
ENV IVRIT_MODEL_PATH=/app/models/ivrit-whisper-large-v3-turbo.bin
ENV PORT=8009
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8009/health')" || exit 1

# Run the application
CMD ["python", "app.py"]
