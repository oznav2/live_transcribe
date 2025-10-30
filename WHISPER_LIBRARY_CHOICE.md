# Whisper Library Choice: faster-whisper vs openai-whisper

## TL;DR
**Use ONLY `faster-whisper`. Do NOT install `openai-whisper`.**

## Why faster-whisper?

### Performance
- **2-4x faster** than openai-whisper
- **50% less memory** usage
- Optimized with CTranslate2 for production

### Compatibility
- Ivrit models (`ivrit-ai/whisper-large-v3-turbo-ct2`) are in CT2 format
- CT2 format is specifically for faster-whisper
- Better GPU utilization

### Known Conflicts with openai-whisper
Installing both libraries can cause:
1. **Dependency conflicts** - Different versions of tiktoken, transformers
2. **Memory issues** - Both libraries load models differently
3. **Model path conflicts** - Both try to manage model downloads
4. **Performance degradation** - Unnecessary overhead

## Current Configuration

### ✅ What We Use
```python
# requirements.ivrit.txt
faster-whisper>=1.1.1  # Primary transcription engine
```

### ❌ What We DON'T Use
```python
# NOT in requirements - causes conflicts
# openai-whisper==20231117  # DO NOT INSTALL
```

## Docker Configuration

The Dockerfile.ivrit is configured to:
1. Install ONLY faster-whisper
2. Pre-download Ivrit CT2 models
3. Skip openai-whisper to avoid conflicts

## App.py Handling

The application:
1. Prioritizes faster-whisper imports
2. Uses faster-whisper for all Ivrit models
3. Gracefully handles missing openai-whisper (it's not needed)

## If You See openai-whisper Anywhere

If you see `openai-whisper` in:
- requirements.txt → Remove it
- pip list → Uninstall it with `pip uninstall openai-whisper`
- Docker logs → The Docker build should skip it

## Performance Comparison

| Metric | faster-whisper | openai-whisper |
|--------|---------------|----------------|
| Speed | 2-4x faster | Baseline |
| Memory | 50% less | Baseline |
| GPU Utilization | Optimized | Standard |
| Ivrit Model Support | ✅ Native CT2 | ❌ Not supported |
| Production Ready | ✅ Yes | ⚠️ Development |

## Installation

### Correct Way
```bash
pip install faster-whisper>=1.1.1
```

### Incorrect Way (DO NOT DO THIS)
```bash
# DON'T DO THIS - causes conflicts
pip install openai-whisper faster-whisper  # ❌ BAD
```

## Troubleshooting

### If both are installed:
```bash
# Remove openai-whisper
pip uninstall openai-whisper -y

# Reinstall faster-whisper
pip install --upgrade faster-whisper>=1.1.1
```

### Check what's installed:
```bash
pip list | grep whisper
# Should show ONLY: faster-whisper
# Should NOT show: openai-whisper
```

## Conclusion

**faster-whisper is the ONLY whisper library you need for this application.**

Benefits:
- ✅ Better performance
- ✅ Lower memory usage
- ✅ Native Ivrit model support
- ✅ No conflicts
- ✅ Production optimized