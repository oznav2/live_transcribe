"""Video metadata extraction service."""
import asyncio
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


async def get_youtube_metadata(url: str) -> Optional[dict]:
    """
    Extract metadata from YouTube video using yt-dlp
    
    Args:
        url: YouTube video URL
    
    Returns:
        dict with keys: title, channel, duration, view_count, thumbnail
        None if extraction fails
    """
    try:
        logger.info(f"Extracting metadata for: {url}")
        
        # yt-dlp command to extract metadata without downloading
        cmd = [
            'yt-dlp',
            '--dump-json',
            '--no-playlist',
            '--skip-download',
            url
        ]
        
        # Run command asynchronously
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Wait for completion with timeout
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=10.0  # 10 second timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            logger.error(f"Timeout extracting metadata for {url}")
            return None
        
        if process.returncode != 0:
            error_msg = stderr.decode('utf-8', errors='ignore')
            logger.error(f"yt-dlp metadata extraction failed: {error_msg}")
            return None
        
        # Parse JSON output
        try:
            metadata = json.loads(stdout.decode('utf-8'))
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse yt-dlp JSON output: {e}")
            return None
        
        # Extract relevant fields
        result = {
            'title': metadata.get('title', 'Unknown Title'),
            'channel': metadata.get('uploader', metadata.get('channel', 'Unknown Channel')),
            'duration': metadata.get('duration', 0),
            'view_count': metadata.get('view_count', 0),
            'thumbnail': metadata.get('thumbnail', ''),
            'is_youtube': True
        }
        
        logger.info(f"Successfully extracted metadata: {result['title']}")
        return result
        
    except Exception as e:
        logger.error(f"Error extracting YouTube metadata: {e}", exc_info=True)
        return None