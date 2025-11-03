"""Model loading with thread-safe singleton pattern.

CRITICAL: This module handles global state and thread-safe model loading.
The double-check locking pattern MUST be preserved exactly.
"""
import os
import threading
import logging
from typing import Any, Optional

from config.settings import (
    IVRIT_MODEL_NAME,
    IVRIT_DEVICE,
    IVRIT_COMPUTE_TYPE,
)
from config.availability import (
    OPENAI_WHISPER_AVAILABLE,
    FASTER_WHISPER_AVAILABLE,
    PYANNOTE_AVAILABLE,
    MODEL_CONFIGS,
    CUDA_AVAILABLE,
)
from core.state import (
    whisper_models,
    diarization_pipeline,
    model_lock,
    diarization_pipeline_lock,
)

logger = logging.getLogger(__name__)

# Store current model info globally
current_model = None
current_model_name = None


def load_model(model_name: str):
    """Load a model based on its configuration"""
    global current_model, current_model_name

    # Fast path: check if model already loaded (no lock needed for read)
    if model_name == current_model_name and current_model is not None:
        return current_model
    
    # Slow path: need to load model, acquire lock to prevent race conditions
    with model_lock:
        # Double-check after acquiring lock (another thread might have loaded it)
        if model_name == current_model_name and current_model is not None:
            return current_model

        if model_name not in MODEL_CONFIGS:
            # Provide helpful error message about available models
            available = list(MODEL_CONFIGS.keys())
            if not available:
                raise ValueError(f"No models are available. Please install faster-whisper, openai-whisper, or configure Deepgram API.")
            raise ValueError(f"Unknown model: {model_name}. Available models: {', '.join(available)}")

        config = MODEL_CONFIGS[model_name]

        if config["type"] == "openai":
            if not OPENAI_WHISPER_AVAILABLE:
                # List alternative models
                alternatives = [m for m, c in MODEL_CONFIGS.items() if c["type"] != "openai"]
                alt_msg = f" Try one of these instead: {', '.join(alternatives)}" if alternatives else ""
                raise ValueError(f"openai-whisper is not installed. Cannot load OpenAI Whisper models.{alt_msg}")
            
            import whisper
            import torch
            
            logger.info(f"Loading OpenAI Whisper model: {config['name']}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device} (CUDA available: {torch.cuda.is_available()})")
            try:
                model = whisper.load_model(config["name"], device=device)
            except Exception as e:
                logger.warning(f"Failed to load model on {device}: {e}. Falling back to CPU.")
                model = whisper.load_model(config["name"], device="cpu")
        
        elif config["type"] == "faster_whisper":
            if not FASTER_WHISPER_AVAILABLE:
                raise ValueError(
                    "faster_whisper is not installed. Cannot load Ivrit CT2 models.\n"
                    "Install with: pip install faster-whisper>=1.1.1\n"
                    "Or use Docker: docker-compose -f docker-compose.ivrit.yml up"
                )
            
            import faster_whisper
            import torch
            
            model_name_or_path = config.get("name", "ivrit-ai/whisper-large-v3-turbo-ct2")
            device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            compute_type = config.get("compute_type", "float16" if device == "cuda" else "int8")
            
            logger.info(f"Loading faster_whisper model: {model_name_or_path}")
            logger.info(f"Device: {device}, Compute type: {compute_type}")
            
            try:
                # Load the faster_whisper model
                model = faster_whisper.WhisperModel(
                    model_name_or_path,
                    device=device,
                    compute_type=compute_type,
                    num_workers=1,
                    download_root="/root/.cache/whisper"
                )
                # Wrap in a dict to maintain consistency with other model types
                model = {
                    "type": "faster_whisper",
                    "model": model,
                    "config": config
                }
                logger.info(f"Successfully loaded faster_whisper model: {model_name_or_path}")
            except Exception as e:
                logger.error(f"Failed to load faster_whisper model: {e}")
                # Fallback to CPU with int8
                if device == "cuda":
                    logger.info("Attempting CPU fallback with int8 compute type...")
                    try:
                        model = faster_whisper.WhisperModel(
                            model_name_or_path,
                            device="cpu",
                            compute_type="int8",
                            num_workers=1
                        )
                        model = {
                            "type": "faster_whisper",
                            "model": model,
                            "config": config
                        }
                        logger.info("Successfully loaded model on CPU")
                    except Exception as e2:
                        raise ValueError(f"Failed to load faster_whisper model on both GPU and CPU: {e2}")
                else:
                    raise
        
        else:
            raise ValueError(f"Unknown model type: {config['type']}")

        current_model = model
        current_model_name = model_name
        return model


def get_diarization_pipeline():
    """Load and cache the pyannote diarization pipeline"""
    global diarization_pipeline
    
    if not PYANNOTE_AVAILABLE:
        logger.warning("Pyannote not available - diarization disabled")
        return None
    
    with diarization_pipeline_lock:
        if diarization_pipeline is None:
            try:
                from pyannote.audio import Pipeline
                import torch
                
                logger.info("Loading pyannote diarization pipeline...")
                # Try to load the Ivrit-optimized model first
                try:
                    diarization_pipeline = Pipeline.from_pretrained(
                        "ivrit-ai/pyannote-speaker-diarization-3.1",
                        use_auth_token=os.getenv("HUGGINGFACE_TOKEN")
                    )
                except Exception as e:
                    logger.warning(f"Failed to load Ivrit diarization model: {e}")
                    # Fall back to the standard model
                    diarization_pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=os.getenv("HUGGINGFACE_TOKEN")
                    )
                
                # Move to GPU if available
                if torch.cuda.is_available():
                    diarization_pipeline.to(torch.device("cuda"))
                
                logger.info("Diarization pipeline loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load diarization pipeline: {e}")
                return None
    
    return diarization_pipeline