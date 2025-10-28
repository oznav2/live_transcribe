#!/bin/bash
# Quick start script for Live Transcription App (CUDA prebuilt runtime)

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

# Choose compose command (v2 preferred)
if docker compose version &> /dev/null; then
    COMPOSE="docker compose"
else
    COMPOSE="docker-compose"
fi
echo ""

# Check if .env file exists, if not create from example
if [ ! -f .env ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo "✓ .env file created"
    echo ""
fi

###############################
# Ensure prebuilt CUDA artifacts exist
###############################
CLI_PATH="whisper.cpp/build/bin/whisper-cli"
GGML_LIB_DIR="whisper.cpp/build/ggml/src"
WHISPER_LIB_DIR="whisper.cpp/build/src"

if [ ! -f "$CLI_PATH" ]; then
    echo "whisper-cli not found at $CLI_PATH"
    echo "Running prebuild.py to compile whisper.cpp with CUDA (host)..."
    python3 prebuild.py --cuda || {
        echo "❌ Failed to prebuild whisper.cpp. Ensure CMake >= 3.18 and CUDA toolkit installed."
        exit 1
    }
fi

if [ ! -d "$GGML_LIB_DIR" ] || ! ls -1 "$GGML_LIB_DIR"/libggml*.so* >/dev/null 2>&1; then
    echo "⚠️  ggml shared libraries not found under $GGML_LIB_DIR"
    echo "    Continuing, but CUDA backend may not load if libs are missing."
fi

if [ ! -d "$WHISPER_LIB_DIR" ] || ! ls -1 "$WHISPER_LIB_DIR"/libwhisper.so* >/dev/null 2>&1; then
    echo "⚠️  libwhisper.so not found under $WHISPER_LIB_DIR (may be fine for some builds)."
fi

# Build and start the application
echo "Building Docker image (runtime-prebuilt-with-cuda target)..."
$COMPOSE build

echo ""
echo "Starting application..."
$COMPOSE up -d

echo ""
echo "=========================================="
echo "✅ Application started successfully!"
echo "=========================================="
echo ""
echo "Access the application at:"
echo "  🌐 http://localhost:8009"
echo ""
echo "Useful commands:"
echo "  View logs:    $COMPOSE logs -f"
echo "  Stop app:     $COMPOSE down"
echo "  Restart:      $COMPOSE restart"
echo "  Check status: $COMPOSE ps"
echo ""
echo "Waiting for application to be ready..."
sleep 5

# Check if application is responding
if curl -s http://localhost:8009/health > /dev/null 2>&1; then
    echo "✅ Application is responding and healthy!"
else
    echo "⚠️  Application may still be starting up..."
    echo "   Run 'docker-compose logs -f' to view startup progress"
fi

echo ""
echo "Happy transcribing! 🎙️"
