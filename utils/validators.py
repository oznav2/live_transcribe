"""URL and input validation utilities."""
import re
from typing import Optional


def is_youtube_url(url: str) -> bool:
    """Check if URL is a YouTube URL"""
    youtube_patterns = [
        r'youtube\.com/watch\?v=',
        r'youtu\.be/',
        r'youtube\.com/embed/',
        r'youtube\.com/v/',
        r'm\.youtube\.com'
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