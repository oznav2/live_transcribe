#!/bin/bash

echo "Testing Docker build for Ivrit integration..."
echo "============================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test Docker build
echo -e "${YELLOW}Building Docker image...${NC}"
if docker build -f Dockerfile.ivrit -t live-transcription-ivrit:test .; then
    echo -e "${GREEN}✓ Docker build successful${NC}"
else
    echo -e "${RED}✗ Docker build failed${NC}"
    exit 1
fi

# Test running the container briefly to check imports
echo -e "${YELLOW}Testing Python imports in container...${NC}"
docker run --rm live-transcription-ivrit:test python3 -c "
import sys
errors = []

# Test core imports
try:
    import torch
    print('✓ PyTorch imported successfully')
    if torch.cuda.is_available():
        print('  CUDA is available')
    else:
        print('  CUDA is not available (CPU mode)')
except Exception as e:
    errors.append(f'PyTorch import failed: {e}')

try:
    import faster_whisper
    print('✓ faster_whisper imported successfully')
except Exception as e:
    errors.append(f'faster_whisper import failed: {e}')

try:
    import ivrit
    print('✓ ivrit package imported successfully')
except Exception as e:
    errors.append(f'ivrit import failed: {e}')

try:
    import whisper
    print('✓ openai-whisper imported successfully')
except Exception as e:
    print('⚠ openai-whisper not available (optional, will use faster_whisper)')

try:
    from deepgram import DeepgramClient
    print('✓ Deepgram SDK imported successfully')
except Exception as e:
    errors.append(f'Deepgram import failed: {e}')

# Test app.py imports
try:
    import app
    print('✓ app.py imported successfully')
    print(f'  Default model: {app.MODEL_SIZE}')
except Exception as e:
    errors.append(f'app.py import failed: {e}')

if errors:
    print('\\nERRORS FOUND:')
    for error in errors:
        print(f'  ✗ {error}')
    sys.exit(1)
else:
    print('\\n✓ All required imports successful!')
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo -e "${GREEN}The Docker image is ready to use.${NC}"
    echo ""
    echo "To run the container with GPU support:"
    echo "  docker-compose -f docker-compose.ivrit.yml up"
    echo ""
    echo "To run without GPU (CPU mode):"
    echo "  docker run -p 8009:8009 --env-file .env live-transcription-ivrit:test"
else
    echo -e "${RED}✗ Import tests failed${NC}"
    exit 1
fi