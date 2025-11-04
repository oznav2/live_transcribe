"""Cache utilities for storing summaries as text files."""
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.constants import DOWNLOAD_CACHE_DIR
from utils.cache import get_url_hash

logger = logging.getLogger(__name__)


def save_summary_to_file(
    url: str,
    summary_text: str,
    source_language: str
) -> Optional[str]:
    """
    Save summary text to a file in the cache directory.

    Args:
        url: Original video URL
        summary_text: Hebrew summary text to save
        source_language: Source language code of the original transcription

    Returns:
        Path to saved file, or None on failure
    """
    try:
        # Generate URL hash for consistent naming
        url_hash = get_url_hash(url)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create filename: {url_hash}_summary_{source_language}_{timestamp}.txt
        filename = f"{url_hash}_summary_{source_language}_{timestamp}.txt"
        file_path = DOWNLOAD_CACHE_DIR / filename

        # Ensure cache directory exists
        DOWNLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Write summary to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)

        logger.info(f"Saved summary to: {file_path}")
        return str(file_path)

    except Exception as e:
        logger.error(f"Failed to save summary to file: {e}")
        return None


def get_cached_summary(url: str, source_language: str) -> Optional[str]:
    """
    Retrieve cached summary text for a URL and source language.

    Args:
        url: Original video URL
        source_language: Source language code

    Returns:
        Cached summary text, or None if not found
    """
    try:
        url_hash = get_url_hash(url)
        pattern = f"{url_hash}_summary_{source_language}_*.txt"

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
            logger.info(f"Found cached summary: {file_path}")
            return text

        return None

    except Exception as e:
        logger.error(f"Failed to retrieve cached summary: {e}")
        return None
