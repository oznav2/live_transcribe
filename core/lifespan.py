"""Application lifespan management for startup and shutdown."""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from config.availability import (
    FASTER_WHISPER_AVAILABLE, OPENAI_WHISPER_AVAILABLE, 
    DEEPGRAM_AVAILABLE, MODEL_CONFIGS
)
from config.settings import DEEPGRAM_API_KEY
from config.constants import CACHE_ENABLED
from config.availability import MODEL_SIZE

from models.loader import load_model
from utils.cache import (
    init_cache_dir, init_capture_dir, init_download_cache_dir
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    
    # Critical dependency check
    if not FASTER_WHISPER_AVAILABLE and not OPENAI_WHISPER_AVAILABLE and not (DEEPGRAM_AVAILABLE and DEEPGRAM_API_KEY):
        logger.error("=" * 60)
        logger.error("CRITICAL: No transcription backend available!")
        logger.error("Please install one of the following:")
        logger.error("  • faster-whisper: pip install faster-whisper>=1.1.1")
        logger.error("  • openai-whisper: pip install openai-whisper")
        logger.error("  • Configure Deepgram API with DEEPGRAM_API_KEY")
        logger.error("=" * 60)
    
    try:
        if MODEL_CONFIGS:
            logger.info(f"Loading default model: {MODEL_SIZE}")
            logger.info("NOTE: First-time model download may take 3-5 minutes. Please be patient...")
            try:
                load_model(MODEL_SIZE)
                logger.info(f"Default model '{MODEL_SIZE}' loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load default model '{MODEL_SIZE}': {e}")
                # Try to load any available model
                for model_name in MODEL_CONFIGS.keys():
                    if model_name != MODEL_SIZE:
                        try:
                            logger.info(f"Trying alternative model: {model_name}")
                            load_model(model_name)
                            logger.info(f"Successfully loaded alternative model: {model_name}")
                            break
                        except Exception as e2:
                            logger.warning(f"Failed to load {model_name}: {e2}")
                            continue
        else:
            logger.warning("No models available at startup. Models can be loaded on-demand via API.")

        # Initialize audio cache
        init_cache_dir()
        logger.info(f"Audio cache initialized (enabled: {CACHE_ENABLED})")

        # Initialize capture directory
        init_capture_dir()
        logger.info("Capture directory initialized for first-60s mode")
        
        # Initialize download cache directory
        init_download_cache_dir()
        logger.info("Download cache initialized for URL-based caching")
        
        # Load and cache index.html to avoid blocking I/O on every request
        try:
            index_path = Path("static/index.html")
            if not index_path.exists():
                logger.error(f"index.html not found at {index_path.absolute()}")
                import core.state
                core.state.cached_index_html = None
            else:
                with open(index_path, "r", encoding="utf-8") as f:
                    import core.state
                    content = f.read()
                    core.state.cached_index_html = content
                    logger.info(f"✓ Cached index.html ({len(content)} bytes) for fast serving")
        except Exception as e:
            logger.error(f"Failed to cache index.html: {e}")
            import core.state
            core.state.cached_index_html = None
    except Exception as e:
        logger.error(f"Critical startup error: {e}")
        # Don't raise - allow the app to start even without models
        logger.warning("Application starting without pre-loaded models. Models will be loaded on-demand.")

    yield

    # Shutdown (if needed)
    logger.info("Application shutting down")