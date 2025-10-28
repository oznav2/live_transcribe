## CUDA builder stage for whisper.cpp (GGML_CUDA)
# Align builder CUDA version with PyTorch cu118 runtime for compatibility
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS whispercpp-builder

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Copy local whisper.cpp source to builder
COPY whisper.cpp/ /src/whisper.cpp/

# Build whisper.cpp with CUDA support (no cuBLAS to minimize runtime deps)
RUN cmake -S /src/whisper.cpp -B /build \
      -DGGML_CUDA=1 \
      -DGGML_CUBLAS=0 \
      -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /build -j --config Release

# Runtime stage
FROM python:3.11-slim AS base

# Install system dependencies including FFmpeg, build tools, and Python pip
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    build-essential \
    cmake \
    python3-pip \
    && pip3 install yt-dlp \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (PyTorch CUDA 11.8 wheels)
RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y torch torchvision torchaudio || true && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 \
        torch==2.1.2+cu118 torchaudio==2.1.2+cu118 && \
    python - <<'PY'
import torch
print('CUDA Available:', torch.cuda.is_available())
print('CUDA Version:', torch.version.cuda)
print('GPU Count:', torch.cuda.device_count())
PY

# Copy application code
COPY app.py .
COPY static/ ./static/

# Copy CUDA-built whisper.cpp binary and shared libraries from builder
RUN mkdir -p /app/whisper.cpp/build/bin && \
    mkdir -p /app/whisper.cpp/build/src && \
    mkdir -p /app/whisper.cpp/build/ggml/src && \
    mkdir -p /usr/local/cuda/lib64
COPY --from=whispercpp-builder /build/bin/whisper-cli /app/whisper.cpp/build/bin/whisper-cli
COPY --from=whispercpp-builder /build/src/libwhisper.so* /app/whisper.cpp/build/src/
COPY --from=whispercpp-builder /build/ggml/src/libggml*.so* /app/whisper.cpp/build/ggml/src/

# Bundle essential CUDA runtime libs required by ggml CUDA backend
COPY --from=whispercpp-builder /usr/local/cuda/lib64/libcudart.so* /usr/local/cuda/lib64/
COPY --from=whispercpp-builder /usr/local/cuda/lib64/libnvrtc.so* /usr/local/cuda/lib64/

# Set library path for whisper.cpp and CUDA runtime
ENV LD_LIBRARY_PATH=/app/whisper.cpp/build/src:/app/whisper.cpp/build/ggml/src:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Create directory for Whisper models and copy the converted Ivrit model
RUN mkdir -p /root/.cache/whisper && \
    mkdir -p /app/models
COPY models/ivrit-whisper-large-v3-turbo.bin /app/models/

# Expose port
EXPOSE 8009

# Environment variables
ENV WHISPER_MODEL=ivrit-large-v3-turbo
ENV IVRIT_MODEL_PATH=/app/models/ivrit-whisper-large-v3-turbo.bin
ENV WHISPER_CPP_PATH=/app/whisper.cpp/build/bin/whisper-cli
ENV PORT=8009
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8009/health')" || exit 1

# Run the application
CMD ["python", "app.py"]
