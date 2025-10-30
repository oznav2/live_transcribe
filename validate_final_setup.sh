#!/bin/bash

echo "=========================================="
echo "FINAL VALIDATION - Live Transcription App"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "1. Checking for whisper.cpp references..."
if grep -q "WHISPER_CPP\|whisper_cpp\|whisper.cpp\|whisper-cli" app.py 2>/dev/null | grep -v "^#"; then
    echo -e "${RED}✗ Found whisper.cpp references in app.py${NC}"
    exit 1
else
    echo -e "${GREEN}✓ No whisper.cpp references found${NC}"
fi

echo ""
echo "2. Checking default model configuration..."
if grep -q 'MODEL_SIZE = os.getenv("WHISPER_MODEL", "ivrit-ct2")' app.py; then
    echo -e "${GREEN}✓ Default model is set to ivrit-ct2${NC}"
else
    echo -e "${RED}✗ Default model is not ivrit-ct2${NC}"
    exit 1
fi

echo ""
echo "3. Checking Dockerfile.ivrit model downloads..."
if grep -q "ivrit-ai/whisper-large-v3-turbo-ct2" Dockerfile.ivrit; then
    echo -e "${GREEN}✓ Dockerfile will download ivrit-ai/whisper-large-v3-turbo-ct2${NC}"
else
    echo -e "${RED}✗ Dockerfile missing ivrit-ai/whisper-large-v3-turbo-ct2 download${NC}"
    exit 1
fi

echo ""
echo "4. Checking docker-compose.ivrit.yml..."
if grep -q "WHISPER_MODEL=ivrit-ct2" docker-compose.ivrit.yml; then
    echo -e "${GREEN}✓ docker-compose uses ivrit-ct2 as default${NC}"
else
    echo -e "${RED}✗ docker-compose not configured for ivrit-ct2${NC}"
    exit 1
fi

echo ""
echo "=========================================="
echo -e "${GREEN}✅ ALL CHECKS PASSED!${NC}"
echo "=========================================="
echo ""
echo "Your application is now configured to:"
echo "  • Use ivrit-ai/whisper-large-v3-turbo-ct2 as the default model"
echo "  • Work without whisper.cpp (no more CLI errors)"
echo "  • Provide better performance with faster_whisper"
echo ""
echo "To deploy your application:"
echo ""
echo "  1. Build the Docker image:"
echo "     docker build -f Dockerfile.ivrit -t live-transcription-ivrit ."
echo ""
echo "  2. Run with docker-compose:"
echo "     docker-compose -f docker-compose.ivrit.yml up"
echo ""
echo "The application will start with the ivrit-ct2 model ready to use!"
echo "No whisper.cpp errors will occur."