# Contributing to Live Audio Stream Transcription

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository**
2. **Clone your fork**:
   ```bash
   git clone https://github.com/yourusername/webapp.git
   cd webapp
   ```
3. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Using Docker (Recommended)
```bash
docker-compose up --build
```

### Local Development
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

## Making Changes

1. **Test your changes**: Ensure the application still works
2. **Follow code style**: Use PEP 8 for Python code
3. **Update documentation**: Update README.md if you add features
4. **Commit with clear messages**:
   ```bash
   git commit -m "Add: Feature description"
   ```

## Testing

Run the environment test:
```bash
python test_setup.py
```

## Submitting Changes

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
2. **Create a Pull Request** on GitHub
3. **Describe your changes** clearly in the PR description

## Feature Ideas

- [ ] Speaker diarization
- [ ] Multiple audio stream support
- [ ] Subtitle file export (SRT, VTT)
- [ ] Translation support
- [ ] Local file upload
- [ ] API authentication
- [ ] Batch processing queue

## Code of Conduct

Be respectful and constructive in all interactions.

## Questions?

Open an issue on GitHub for any questions or discussions.

Thank you for contributing! 🎉
