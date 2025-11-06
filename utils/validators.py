"""URL and input validation utilities."""
import re
from typing import Optional


def sanitize_url(url: str) -> str:
    """Remove invisible/unsafe characters and trim whitespace/backticks.

    This helps with pasted links that include zero-width characters, NBSP,
    or formatting artifacts that can break network requests.
    """
    if not isinstance(url, str):
        return url

    # Remove common zero-width/invisible characters and NBSP
    # \u200B (ZWSP), \u200C (ZWNJ), \u200D (ZWJ), \u2060 (WJ), \uFEFF (BOM), \u00A0 (NBSP)
    invisible_pattern = r"[\u200B\u200C\u200D\u2060\uFEFF\u00A0]"
    cleaned = re.sub(invisible_pattern, "", url)

    # Strip backticks and surrounding whitespace/newlines
    cleaned = cleaned.strip().strip('`').strip()

    # Collapse internal whitespace that can appear between path segments
    cleaned = re.sub(r"\s+", "", cleaned)

    return cleaned


def sanitize_token (token: Optional[str]) -> str:
    """Normalize API tokens/keys by stripping whitespace and invisible chars.

    Prevents subtle pasting artifacts (ZWSP, NBSP, BOM) that cause auth errors.
    Returns empty string if input is falsy.
    """
    if not token:
        return ''

    # Remove common zero-width/invisible characters and NBSP
    invisible_pattern = r"[\u200B\u200C\u200D\u2060\uFEFF\u00A0]"
    cleaned = re.sub(invisible_pattern, "", str(token))

    # Strip backticks and surrounding whitespace/newlines
    cleaned = cleaned.strip().strip('`').strip()

    # Collapse internal whitespace
    cleaned = re.sub(r"\s+", "", cleaned)

    return cleaned


def is_youtube_url(url: str) -> bool:
    """Check if URL is a YouTube URL"""
    youtube_patterns = [
        r'youtube\.com/watch\?v=',
        r'youtu\.be/',
        r'youtube\.com/embed/',
        r'youtube\.com/v/',
        r'm\.youtube\.com',
        r'youtube\.com/shorts/',
    ]
    
    url_lower = url.lower()
    return any(re.search(pattern, url_lower) for pattern in youtube_patterns)


def should_use_ytdlp(url: str) -> bool:
    """Determine if URL should use yt-dlp instead of direct FFmpeg streaming"""
    # Use yt-dlp for known video platforms and complex URLs
    ytdlp_patterns = [
        'youtube.com', 'youtu.be', 'vimeo.com', 'dailymotion.com',
        'facebook.com', 'twitter.com', 'twitch.tv', 'tiktok.com',
        'instagram.com', 'reddit.com'
    ]

    # Special handling for YouTube URLs with various formats
    youtube_patterns = [
        r'youtube\.com/watch\?v=',
        r'youtu\.be/',
        r'youtube\.com/embed/',
        r'youtube\.com/v/',
        r'm\.youtube\.com'
    ]

    url_lower = url.lower()

    # Check standard patterns
    if any(pattern in url_lower for pattern in ytdlp_patterns):
        return True

    # Check YouTube regex patterns
    for pattern in youtube_patterns:
        if re.search(pattern, url_lower):
            return True

    return False


def requires_audio_extraction(url: str) -> bool:
    """
    Check if URL requires special audio extraction (103fm, osimhistoria).

    These sites don't work with yt-dlp/ffmpeg directly and need HTML parsing
    to extract the actual MP3 URL.

    Args:
        url: URL to check

    Returns:
        True if URL needs audio extraction, False otherwise
    """
    url_lower = url.lower()
    return '103fm.maariv.co.il' in url_lower or 'osimhistoria.com' in url_lower