# Deployment Guide

This guide covers various deployment options for the Live Audio Stream Transcription application.

## Table of Contents

- [Docker Deployment (Recommended)](#docker-deployment)
- [Docker Compose](#docker-compose)
- [Cloud Platforms](#cloud-platforms)
- [Production Configuration](#production-configuration)
- [Security Considerations](#security-considerations)

---

## Docker Deployment

### Prerequisites

- Docker 20.10+
- At least 2GB RAM (4GB+ recommended)
- 10GB disk space for Docker images and Whisper models

### Quick Start

```bash
# Build the image
docker build -t live-transcription .

# Run the container
docker run -d \
  --name transcription-app \
  -p 8000:8000 \
  -e WHISPER_MODEL=base \
  -v whisper-models:/root/.cache/whisper \
  live-transcription

# Check logs
docker logs -f transcription-app

# Access the app
open http://localhost:8000
```

### Docker Image Tags

Build with specific tags for versioning:

```bash
docker build -t yourusername/live-transcription:1.0.0 .
docker build -t yourusername/live-transcription:latest .
```

---

## Docker Compose

### Development

```bash
# Start in foreground (see logs)
docker-compose up

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Production

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  transcription-app:
    build: .
    container_name: live-transcription
    restart: always
    ports:
      - "8000:8000"
    environment:
      - WHISPER_MODEL=small  # Better accuracy for production
      - PORT=8000
    volumes:
      - whisper-models:/root/.cache/whisper
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G

volumes:
  whisper-models:
    driver: local
```

Run with:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

---

## Cloud Platforms

### 1. DigitalOcean

#### App Platform (Dockerfile)

1. Create account at [DigitalOcean](https://digitalocean.com)
2. Go to App Platform → Create App
3. Connect your GitHub repository
4. Select Dockerfile deployment
5. Configure:
   - Resource Size: Basic (2 GB RAM minimum)
   - Environment Variables: `WHISPER_MODEL=base`
   - HTTP Port: 8000
6. Deploy

**Cost**: ~$12/month for Basic plan

#### Droplet (Manual)

```bash
# SSH into droplet
ssh root@your-droplet-ip

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Clone repository
git clone https://github.com/yourusername/webapp.git
cd webapp

# Start with Docker Compose
docker-compose up -d

# Setup Nginx reverse proxy (optional)
apt install nginx
# Configure Nginx...
```

---

### 2. AWS EC2

#### Launch Instance

1. Launch EC2 instance (t3.medium or larger)
2. Security Group: Allow port 8000 (or 80/443 with reverse proxy)
3. SSH into instance

```bash
# Install Docker
sudo yum update -y
sudo yum install docker -y
sudo service docker start
sudo usermod -a -G docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Deploy application
git clone https://github.com/yourusername/webapp.git
cd webapp
docker-compose up -d
```

#### Auto-start on reboot

```bash
# Create systemd service
sudo nano /etc/systemd/system/transcription-app.service
```

```ini
[Unit]
Description=Live Transcription App
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/ec2-user/webapp
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable transcription-app
sudo systemctl start transcription-app
```

---

### 3. Google Cloud Run

#### Deploy Container

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/live-transcription

# Deploy to Cloud Run
gcloud run deploy live-transcription \
  --image gcr.io/PROJECT_ID/live-transcription \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 3600 \
  --set-env-vars WHISPER_MODEL=base \
  --allow-unauthenticated
```

**Note**: Cloud Run has a 60-minute timeout limit.

---

### 4. Railway

1. Go to [Railway.app](https://railway.app)
2. New Project → Deploy from GitHub repo
3. Railway auto-detects Dockerfile
4. Set environment variables:
   - `WHISPER_MODEL=base`
5. Deploy

**Cost**: Pay-as-you-go, ~$5-20/month depending on usage

---

### 5. Render

1. Go to [Render.com](https://render.com)
2. New → Web Service
3. Connect repository
4. Configuration:
   - Environment: Docker
   - Build Command: (auto-detected)
   - Start Command: (auto-detected)
   - Plan: Standard ($25/month minimum for 2GB RAM)
5. Environment Variables:
   - `WHISPER_MODEL=base`
6. Create Web Service

---

### 6. Fly.io

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login
flyctl auth login

# Launch app
flyctl launch

# Deploy
flyctl deploy
```

Create `fly.toml`:
```toml
app = "live-transcription"

[build]
  dockerfile = "Dockerfile"

[env]
  WHISPER_MODEL = "base"
  PORT = "8000"

[[services]]
  internal_port = 8000
  protocol = "tcp"

  [[services.ports]]
    port = 80
    handlers = ["http"]
  
  [[services.ports]]
    port = 443
    handlers = ["tls", "http"]

[services.concurrency]
  type = "connections"
  hard_limit = 25
  soft_limit = 20

[[services.tcp_checks]]
  grace_period = "30s"
  interval = "15s"
  restart_limit = 0
  timeout = "2s"

[vm]
  memory = "2048mb"
  cpu_kind = "shared"
  cpus = 1
```

---

## Production Configuration

### Environment Variables

```bash
# Whisper model (larger = better accuracy, more resources)
WHISPER_MODEL=small

# Server port
PORT=8000

# Log level
LOG_LEVEL=info
```

### Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket timeout
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }
}
```

### SSL with Let's Encrypt

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo certbot renew --dry-run
```

---

## Security Considerations

### 1. Authentication

Add API key authentication:

```python
# In app.py
from fastapi import Header, HTTPException

API_KEY = os.getenv("API_KEY", "your-secret-key")

@app.websocket("/ws/transcribe")
async def websocket_transcribe(
    websocket: WebSocket,
    api_key: str = Header(None)
):
    if api_key != API_KEY:
        await websocket.close(code=1008)
        return
    # ... rest of code
```

### 2. Rate Limiting

Use a reverse proxy or implement rate limiting in the application.

### 3. CORS Configuration

For production, restrict CORS origins:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 4. Firewall

```bash
# UFW (Ubuntu)
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
```

---

## Monitoring

### Health Checks

```bash
# Simple health check
curl http://localhost:8000/health

# Continuous monitoring
watch -n 10 'curl -s http://localhost:8000/health | jq'
```

### Docker Logs

```bash
# View logs
docker logs -f transcription-app

# Last 100 lines
docker logs --tail 100 transcription-app
```

### Resource Monitoring

```bash
# Docker stats
docker stats transcription-app

# System resources
htop
```

---

## Backup and Recovery

### Backup Whisper Models

```bash
# Create backup
docker run --rm -v whisper-models:/data -v $(pwd):/backup \
  alpine tar czf /backup/whisper-models-backup.tar.gz -C /data .

# Restore backup
docker run --rm -v whisper-models:/data -v $(pwd):/backup \
  alpine tar xzf /backup/whisper-models-backup.tar.gz -C /data
```

---

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs transcription-app

# Check resource usage
docker stats

# Restart container
docker restart transcription-app
```

### Out of Memory

Increase Docker memory limits or use a smaller Whisper model:
```yaml
environment:
  - WHISPER_MODEL=tiny  # or base
```

### WebSocket Connection Issues

Check firewall and reverse proxy WebSocket configuration.

---

For more help, open an issue on GitHub.
