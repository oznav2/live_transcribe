"""Cache management utilities for audio, downloads, and transcriptions."""
import os
import hashlib
import json
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from config.constants import CACHE_DIR, DOWNLOAD_CACHE_DIR, CAPTURE_DIR, CACHE_MAX_AGE_HOURS, CACHE_ENABLED
from core.state import URL_DOWNLOADS, CAPTURES

logger = logging.getLogger(__name__)


def init_capture_dir():
    try:
        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning(f"Failed to create capture dir: {e}")


def init_download_cache_dir():
    """Initialize download cache directory for URL-based audio caching"""
    try:
        DOWNLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Clean old downloads (older than 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        for cache_file in DOWNLOAD_CACHE_DIR.glob("*.wav"):
            try:
                file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if file_time < cutoff_time:
                    cache_file.unlink()
                    # Remove from URL_DOWNLOADS if present
                    for url_hash, path in list(URL_DOWNLOADS.items()):
                        if path == str(cache_file):
                            del URL_DOWNLOADS[url_hash]
            except Exception as e:
                logger.warning(f"Failed to clean old download cache: {e}")
    except Exception as e:
        logger.warning(f"Failed to create download cache dir: {e}")


def get_url_hash(url: str) -> str:
    """Generate a unique hash for a URL to use as cache key"""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def get_cached_download(url: str) -> Optional[str]:
    """Check if we have a cached download for this URL"""
    url_hash = get_url_hash(url)
    
    # Check in-memory cache first
    if url_hash in URL_DOWNLOADS:
        cached_path = URL_DOWNLOADS[url_hash]
        if os.path.exists(cached_path):
            logger.info(f"Found cached download for URL (in-memory): {cached_path}")
            return cached_path
        else:
            # File was deleted, remove from cache
            del URL_DOWNLOADS[url_hash]
    
    # Check disk cache
    cache_pattern = f"{url_hash}_*.wav"
    for cached_file in DOWNLOAD_CACHE_DIR.glob(cache_pattern):
        if cached_file.exists():
            URL_DOWNLOADS[url_hash] = str(cached_file)
            logger.info(f"Found cached download for URL (disk): {cached_file}")
            return str(cached_file)
    
    return None


def save_download_to_cache(url: str, audio_file: str) -> str:
    """Save a downloaded audio file to URL cache and return the cached path"""
    try:
        url_hash = get_url_hash(url)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cached_filename = f"{url_hash}_{timestamp}.wav"
        cached_path = DOWNLOAD_CACHE_DIR / cached_filename
        
        # If the file is already in the cache directory, just update the mapping
        if DOWNLOAD_CACHE_DIR in Path(audio_file).parents:
            URL_DOWNLOADS[url_hash] = audio_file
            logger.info(f"Audio file already in cache directory: {audio_file}")
            return audio_file
        
        # Copy the file to cache directory
        shutil.copy2(audio_file, cached_path)
        URL_DOWNLOADS[url_hash] = str(cached_path)
        logger.info(f"Cached download for URL: {cached_path}")
        
        # Clean up the original temp file if it's outside cache dir
        try:
            if os.path.exists(audio_file) and DOWNLOAD_CACHE_DIR not in Path(audio_file).parents:
                os.unlink(audio_file)
                temp_dir = os.path.dirname(audio_file)
                if temp_dir and '/tmp' in temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            logger.debug(f"Failed to clean up temp file: {e}")
        
        return str(cached_path)
    except Exception as e:
        logger.error(f"Failed to cache download: {e}")
        return audio_file  # Return original file on cache failure


def init_cache_dir():
    """Initialize cache directory and clean old files"""
    if not CACHE_ENABLED:
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Clean old cache files
    cutoff_time = datetime.now() - timedelta(hours=CACHE_MAX_AGE_HOURS)
    cleaned_count = 0

    for cache_file in CACHE_DIR.glob("*.wav"):
        try:
            file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_time < cutoff_time:
                cache_file.unlink()
                cleaned_count += 1
        except Exception as e:
            logger.warning(f"Failed to clean cache file {cache_file}: {e}")

    if cleaned_count > 0:
        logger.info(f"Cleaned {cleaned_count} old cache files")


def generate_cache_key(audio_data: bytes, sample_rate: int, channels: int) -> str:
    """Generate cache key from audio data and parameters"""
    hasher = hashlib.sha256()
    hasher.update(audio_data)
    hasher.update(f"{sample_rate}:{channels}".encode())
    return hasher.hexdigest()


def get_cached_audio(cache_key: str) -> Optional[str]:
    """Get cached normalized audio file if it exists"""
    if not CACHE_ENABLED:
        return None

    cache_path = CACHE_DIR / f"{cache_key}.wav"
    if cache_path.exists():
        logger.debug(f"Cache hit for {cache_key}")
        return str(cache_path)
    return None


def save_to_cache(cache_key: str, audio_path: str) -> None:
    """Save normalized audio to cache"""
    if not CACHE_ENABLED:
        return

    try:
        cache_path = CACHE_DIR / f"{cache_key}.wav"
        shutil.copy2(audio_path, cache_path)
        logger.debug(f"Cached audio as {cache_key}")
    except Exception as e:
        logger.warning(f"Failed to cache audio: {e}")


# Cache statistics functions (will be called from api/routes.py)
def get_cache_stats() -> Dict[str, Any]:
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


def clear_cache() -> Dict[str, Any]:
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
        return {"success": False, "error": str(e)}


def get_download_cache_stats() -> Dict[str, Any]:
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


def clear_download_cache() -> Dict[str, Any]:
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
        return {"success": False, "error": str(e)}