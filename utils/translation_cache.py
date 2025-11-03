"""Cache utilities for storing transcriptions and translations as text files."""
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.constants import DOWNLOAD_CACHE_DIR
from utils.cache import get_url_hash

logger = logging.getLogger(__name__)


def save_transcription_to_file(url: str, transcription_text: str, language: str) -> Optional[str]:
    """
    Save transcription text to a file in the cache directory.

    Args:
        url: Original video URL
        transcription_text: Transcribed text to save
        language: Detected language code

    Returns:
        Path to saved file, or None on failure
    """
    try:
        # Generate URL hash for consistent naming
        url_hash = get_url_hash(url)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create filename: {url_hash}_transcription_{language}_{timestamp}.txt
        filename = f"{url_hash}_transcription_{language}_{timestamp}.txt"
        file_path = DOWNLOAD_CACHE_DIR / filename

        # Ensure cache directory exists
        DOWNLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Write transcription to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(transcription_text)

        logger.info(f"Saved transcription to: {file_path}")
        return str(file_path)

    except Exception as e:
        logger.error(f"Failed to save transcription to file: {e}")
        return None


def save_translation_to_file(
    url: str,
    translation_text: str,
    source_language: str,
    target_language: str
) -> Optional[str]:
    """
    Save translation text to a file in the cache directory.

    Args:
        url: Original video URL
        translation_text: Translated text to save
        source_language: Source language code
        target_language: Target language code

    Returns:
        Path to saved file, or None on failure
    """
    try:
        # Generate URL hash for consistent naming
        url_hash = get_url_hash(url)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create filename: {url_hash}_translation_{source}_{target}_{timestamp}.txt
        filename = f"{url_hash}_translation_{source_language}_to_{target_language}_{timestamp}.txt"
        file_path = DOWNLOAD_CACHE_DIR / filename

        # Ensure cache directory exists
        DOWNLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Write translation to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(translation_text)

        logger.info(f"Saved translation to: {file_path}")
        return str(file_path)

    except Exception as e:
        logger.error(f"Failed to save translation to file: {e}")
        return None


def get_cached_transcription(url: str, language: str) -> Optional[str]:
    """
    Retrieve cached transcription text for a URL and language.

    Args:
        url: Original video URL
        language: Language code

    Returns:
        Cached transcription text, or None if not found
    """
    try:
        url_hash = get_url_hash(url)
        pattern = f"{url_hash}_transcription_{language}_*.txt"

        # Find most recent matching file
        matching_files = sorted(
            DOWNLOAD_CACHE_DIR.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        if matching_files:
            file_path = matching_files[0]
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            logger.info(f"Found cached transcription: {file_path}")
            return text

        return None

    except Exception as e:
        logger.error(f"Failed to retrieve cached transcription: {e}")
        return None


def get_cached_translation(
    url: str,
    source_language: str,
    target_language: str
) -> Optional[str]:
    """
    Retrieve cached translation text for a URL and language pair.

    Args:
        url: Original video URL
        source_language: Source language code
        target_language: Target language code

    Returns:
        Cached translation text, or None if not found
    """
    try:
        url_hash = get_url_hash(url)
        pattern = f"{url_hash}_translation_{source_language}_to_{target_language}_*.txt"

        # Find most recent matching file
        matching_files = sorted(
            DOWNLOAD_CACHE_DIR.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        if matching_files:
            file_path = matching_files[0]
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            logger.info(f"Found cached translation: {file_path}")
            return text

        return None

    except Exception as e:
        logger.error(f"Failed to retrieve cached translation: {e}")
        return None
