#!/bin/bash
# Quick start script for Live Transcription App

set -e

echo "=========================================="
echo "Live Audio Stream Transcription"
echo "=========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed"
    echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✓ Docker is installed"
echo "✓ Docker Compose is installed"
echo ""

# Check if .env file exists, if not create from example
if [ ! -f .env ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo "✓ .env file created"
    echo ""
fi

# Build and start the application
echo "Building Docker image..."
docker-compose build

echo ""
echo "Starting application..."
docker-compose up -d

echo ""
echo "=========================================="
echo "✅ Application started successfully!"
echo "=========================================="
echo ""
echo "Access the application at:"
echo "  🌐 http://localhost:8000"
echo ""
echo "Useful commands:"
echo "  View logs:    docker-compose logs -f"
echo "  Stop app:     docker-compose down"
echo "  Restart:      docker-compose restart"
echo "  Check status: docker-compose ps"
echo ""
echo "Waiting for application to be ready..."
sleep 5

# Check if application is responding
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Application is responding and healthy!"
else
    echo "⚠️  Application may still be starting up..."
    echo "   Run 'docker-compose logs -f' to view startup progress"
fi

echo ""
echo "Happy transcribing! 🎙️"
