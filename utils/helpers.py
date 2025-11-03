"""General helper utilities."""
from datetime import timedelta
import math
from typing import Any, Optional


def format_duration(seconds: int) -> str:
    """Format duration in seconds to MM:SS or HH:MM:SS"""
    if seconds is None:
        return "N/A"
    
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"


def format_view_count(count: int) -> str:
    """Format view count with K, M, B suffixes"""
    if count is None:
        return "N/A"
    
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.1f}B"
    elif count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.1f}K"
    else:
        return str(count)