#!/bin/bash

# Build script for Ivrit-enhanced transcription service
# This script builds the Docker image with Ivrit AI models support

set -e

echo "================================================"
echo "Building Ivrit-Enhanced Transcription Service"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for NVIDIA Docker runtime
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
else
    echo -e "${YELLOW}⚠ No NVIDIA GPU detected - models will run on CPU (slower)${NC}"
fi

# Check if .env file exists
if [ -f .env ]; then
    echo -e "${GREEN}✓ .env file found${NC}"
else
    echo -e "${YELLOW}⚠ No .env file found - using default settings${NC}"
    echo "  Consider creating .env with your API keys"
fi

# Parse command line arguments
BUILD_TYPE=${1:-"ivrit"}
CACHE_OPTION=${2:-""}

case $BUILD_TYPE in
    "ivrit")
        echo -e "\n${GREEN}Building with Ivrit CT2 models (recommended)${NC}"
        DOCKERFILE="Dockerfile.ivrit"
        COMPOSE_FILE="docker-compose.ivrit.yml"
        ;;
    "legacy")
        echo -e "\n${YELLOW}Building with legacy GGML model${NC}"
        DOCKERFILE="Dockerfile"
        COMPOSE_FILE="docker-compose.yml"
        ;;
    *)
        echo -e "${RED}Unknown build type: $BUILD_TYPE${NC}"
        echo "Usage: $0 [ivrit|legacy] [--no-cache]"
        exit 1
        ;;
esac

# Handle cache option
BUILD_ARGS=""
if [ "$CACHE_OPTION" == "--no-cache" ]; then
    echo -e "${YELLOW}Building without cache${NC}"
    BUILD_ARGS="--no-cache"
fi

# Create necessary directories
echo -e "\nCreating required directories..."
mkdir -p cache/audio cache/downloads cache/captures logs models

# Build the Docker image
echo -e "\n${GREEN}Starting Docker build...${NC}"
echo "Using Dockerfile: $DOCKERFILE"
echo "Using Compose file: $COMPOSE_FILE"

# Stop existing container if running
echo -e "\n${YELLOW}Stopping existing containers...${NC}"
docker-compose -f $COMPOSE_FILE down 2>/dev/null || true

# Build with docker-compose
echo -e "\n${GREEN}Building Docker image...${NC}"
if docker-compose -f $COMPOSE_FILE build $BUILD_ARGS; then
    echo -e "${GREEN}✓ Build successful!${NC}"
else
    echo -e "${RED}✗ Build failed!${NC}"
    exit 1
fi

# Option to start the service
echo -e "\n${GREEN}Build complete!${NC}"
echo ""
read -p "Do you want to start the service now? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "\n${GREEN}Starting service...${NC}"
    docker-compose -f $COMPOSE_FILE up -d
    
    # Wait for service to be healthy
    echo -e "\n${YELLOW}Waiting for service to be ready...${NC}"
    for i in {1..30}; do
        if curl -f http://localhost:8009/health &>/dev/null; then
            echo -e "${GREEN}✓ Service is ready!${NC}"
            echo ""
            echo "Access the application at: http://localhost:8009"
            echo ""
            echo "To view logs: docker-compose -f $COMPOSE_FILE logs -f"
            echo "To stop: docker-compose -f $COMPOSE_FILE down"
            break
        fi
        echo -n "."
        sleep 2
    done
else
    echo ""
    echo "To start the service later, run:"
    echo "  docker-compose -f $COMPOSE_FILE up -d"
fi

echo ""
echo "================================================"
echo "Build process complete!"
echo "================================================"