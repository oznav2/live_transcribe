"""API routes for the transcription service."""
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from config.constants import (
    CACHE_DIR, DOWNLOAD_CACHE_DIR, CACHE_MAX_AGE_HOURS
)
from config.constants import CACHE_ENABLED
from config.availability import MODEL_SIZE
from config.settings import USE_OPENAI
from core.state import cached_index_html, URL_DOWNLOADS, current_model, current_model_name
from utils.validators import is_youtube_url
from utils.helpers import format_duration, format_view_count
from services.video_metadata import get_youtube_metadata
from services.translation import translate_text, get_translation_button_text
from utils.translation_cache import save_transcription_to_file, save_translation_to_file

logger = logging.getLogger(__name__)

# Create router instance
router = APIRouter()

# Request models
class TranscriptionRequest(BaseModel):
    url: str
    language: Optional[str] = None


class VideoInfoRequest(BaseModel):
    url: str


class TranslationRequest(BaseModel):
    text: str
    language: str
    url: Optional[str] = None
    video_title: Optional[str] = None


@router.get("/", response_class=HTMLResponse)
async def get_home():
    """Serve the main web interface (cached at startup)"""
    if cached_index_html is None:
        # Fallback - try to load it directly
        try:
            with open("static/index.html", "r", encoding="utf-8") as f:
                content = f.read()
                return HTMLResponse(content=content)
        except Exception as e:
            logger.error(f"Failed to load index.html: {e}")
            return HTMLResponse(content=f"<html><body><h1>Error loading UI: {e}</h1></body></html>")
    return HTMLResponse(content=cached_index_html)


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "whisper_model": current_model_name or MODEL_SIZE,
        "model_loaded": current_model is not None
    }


@router.post("/api/video-info")
async def get_video_info(request: VideoInfoRequest):
    """
    Fetch YouTube video metadata
    
    Returns JSON with video information or error
    """
    try:
        url = str(request.url)
        
        # Validate URL format
        if not url.startswith(('http://', 'https://')):
            return {
                "success": False,
                "error": "Invalid URL format"
            }
        
        # Check if it's a YouTube URL
        if not is_youtube_url(url):
            return {
                "success": False,
                "error": "Not a YouTube URL"
            }
        
        # Extract metadata
        metadata = await get_youtube_metadata(url)
        
        if metadata is None:
            return {
                "success": False,
                "error": "Failed to fetch video information"
            }
        
        # Format the response
        return {
            "success": True,
            "data": {
                "title": metadata['title'],
                "channel": metadata['channel'],
                "duration_seconds": metadata['duration'],
                "duration_formatted": format_duration(metadata['duration']),
                "view_count": metadata['view_count'],
                "view_count_formatted": format_view_count(metadata['view_count']),
                "thumbnail": metadata['thumbnail'],
                "is_youtube": metadata['is_youtube']
            }
        }
        
    except Exception as e:
        logger.error(f"Error in video info endpoint: {e}", exc_info=True)
        return {
            "success": False,
            "error": "Internal server error"
        }


@router.get("/gpu")
async def gpu_diagnostics():
    """GPU diagnostics endpoint"""
    # Import torch conditionally
    try:
        import torch
    except ImportError:
        return {
            "cuda_available": False,
            "error": "PyTorch not installed"
        }
    
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    cuda_version = torch.version.cuda if hasattr(torch.version, "cuda") else None
    devices = []
    for idx in range(device_count):
        try:
            props = torch.cuda.get_device_properties(idx)
            devices.append({
                "index": idx,
                "name": props.name,
                "total_memory_mb": round(props.total_memory / (1024 * 1024), 2),
                "multi_processor_count": props.multi_processor_count,
                "major": props.major,
                "minor": props.minor,
            })
        except Exception as e:
            devices.append({"index": idx, "error": str(e)})

    return {
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "device_count": device_count,
        "devices": devices,
        "env": {
            "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
            "NVIDIA_VISIBLE_DEVICES": os.getenv("NVIDIA_VISIBLE_DEVICES"),
        }
    }


@router.get("/api/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    if not CACHE_ENABLED:
        return {"enabled": False}

    try:
        cache_files = list(CACHE_DIR.glob("*.wav"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "enabled": True,
            "file_count": len(cache_files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "max_age_hours": CACHE_MAX_AGE_HOURS
        }
    except Exception as e:
        return {"enabled": True, "error": str(e)}


@router.post("/api/cache/clear")
async def clear_cache():
    """Clear all cached audio files"""
    if not CACHE_ENABLED:
        return {"enabled": False}

    try:
        deleted_count = 0
        for cache_file in CACHE_DIR.glob("*.wav"):
            cache_file.unlink()
            deleted_count += 1

        return {"success": True, "deleted": deleted_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/download-cache/stats")
async def download_cache_stats():
    """Get download cache statistics"""
    try:
        cache_files = list(DOWNLOAD_CACHE_DIR.glob("*.wav"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        # Get cache details
        cache_details = []
        for cache_file in cache_files:
            cache_details.append({
                "filename": cache_file.name,
                "size_mb": round(cache_file.stat().st_size / (1024 * 1024), 2),
                "modified": datetime.fromtimestamp(cache_file.stat().st_mtime).isoformat(),
                "url_hash": cache_file.name.split('_')[0] if '_' in cache_file.name else 'unknown'
            })
        
        return {
            "enabled": True,
            "file_count": len(cache_files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "max_age_hours": 24,
            "cache_directory": str(DOWNLOAD_CACHE_DIR),
            "files": cache_details
        }
    except Exception as e:
        return {"enabled": True, "error": str(e)}


@router.post("/api/download-cache/clear")
async def clear_download_cache():
    """Clear all cached download files"""
    try:
        deleted_count = 0
        deleted_size = 0

        for cache_file in DOWNLOAD_CACHE_DIR.glob("*.wav"):
            deleted_size += cache_file.stat().st_size
            cache_file.unlink()
            deleted_count += 1

        # Clear in-memory cache
        URL_DOWNLOADS.clear()

        return {
            "success": True,
            "deleted_files": deleted_count,
            "freed_space_mb": round(deleted_size / (1024 * 1024), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/translate")
async def translate_transcription(request: TranslationRequest):
    """
    Translate transcribed text using OpenAI GPT-4/5.

    Returns JSON with translation result or error.
    """
    try:
        # Check if OpenAI is configured
        if not USE_OPENAI:
            return {
                "success": False,
                "error": "Translation service not configured. Please set OPENAI_API_KEY in your .env file."
            }

        # Validate input
        if not request.text or not request.text.strip():
            return {
                "success": False,
                "error": "No text provided for translation"
            }

        if not request.language:
            return {
                "success": False,
                "error": "Source language not specified"
            }

        logger.info(f"Translation request: {request.language} → auto-detect target")

        # Save original transcription to cache if URL provided
        if request.url:
            save_transcription_to_file(
                url=request.url,
                transcription_text=request.text,
                language=request.language
            )

        # Perform translation
        translated_text, target_lang_code, target_lang_name = await translate_text(
            text=request.text,
            source_language=request.language,
            video_title=request.video_title,
            model="gpt-4"  # Use GPT-4 for high-quality translations
        )

        if not translated_text:
            return {
                "success": False,
                "error": "Translation failed. Please check logs for details."
            }

        # Save translation to cache if URL provided
        if request.url:
            save_translation_to_file(
                url=request.url,
                translation_text=translated_text,
                source_language=request.language,
                target_language=target_lang_code
            )

        # Return successful translation
        return {
            "success": True,
            "translation": translated_text,
            "source_language": request.language,
            "target_language": target_lang_code,
            "target_language_name": target_lang_name
        }

    except Exception as e:
        logger.error(f"Error in translation endpoint: {e}", exc_info=True)
        return {
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }