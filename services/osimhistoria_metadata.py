"""
Osim Historia metadata extractor for episode pages.

Extracts comprehensive metadata from osimhistoria.com episode pages:
- episode_number
- title
- summary
- episode_date
- episode_length
- transcript (pre-existing from page)

All metadata is cached to cache/metadata/{episode_id}/ for reuse.
"""

import logging
import re
import hashlib
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Configuration
METADATA_CACHE_DIR = Path("cache/metadata")
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "he-IL,he;q=0.9,en-US;q=0.8,en;q=0.7",
}
REQUEST_TIMEOUT = 15


def ensure_cache_dir():
    """Ensure metadata cache directory exists."""
    METADATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_episode_id(url: str) -> str:
    """
    Extract episode ID from osimhistoria URL.

    Examples:
        https://www.osimhistoria.com/osimhistoria/ep_454 -> ep_454
        https://osimhistoria.com/osimhistoria/episode-123 -> episode-123

    Fallback to URL hash if pattern doesn't match.
    """
    path = urlparse(url).path
    # Try to extract last path segment
    segments = [s for s in path.split('/') if s]
    if segments:
        return segments[-1]
    # Fallback: use URL hash (SHA256 for security)
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def fetch_html(url: str) -> str:
    """Fetch HTML content from URL."""
    logger.info(f"Fetching osimhistoria page: {url}")
    resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    resp.encoding = resp.encoding or "utf-8"
    return resp.text


def extract_episode_title_and_number(soup: BeautifulSoup) -> tuple[str, str]:
    """
    Extract title and episode number from page.

    Primary: <meta property="og:title" content="452: משחק החיים של קונוויי"/>
    Fallback: #comp-kf5277bw > h1 > span > span > span > span
    Format: "452: משחק החיים של קונוויי"

    Returns:
        (title, episode_number) - title without episode number prefix
    """
    full_text = ""

    # Try og:title meta tag first (most reliable)
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.has_attr("content"):
        full_text = og_title["content"].strip()
        logger.info(f"Found title in og:title: {full_text}")

    # Fallback to h1 selector
    if not full_text:
        node = soup.select_one("#comp-kf5277bw > h1 > span > span > span > span")
        if node:
            full_text = node.get_text(separator=" ", strip=True)
            logger.info(f"Found title in h1: {full_text}")

    if not full_text:
        logger.warning("Episode title not found in og:title or h1")
        return "", ""

    # Split on first colon to separate episode number from title
    # Format: "452: משחק החיים של קונוויי"
    if ":" in full_text:
        ep_num, _, ep_title = full_text.partition(":")
        return ep_title.strip(), ep_num.strip()

    return full_text.strip(), ""


def extract_summary(soup: BeautifulSoup) -> str:
    """
    Extract episode summary from page.

    Primary selector: #comp-kf53u3eb > p > span > span > span
    Fallback: Find rich text spans (excluding transcript div)
    """
    # Try primary selector
    node = soup.select_one("#comp-kf53u3eb > p > span > span > span")
    if node:
        txt = node.get_text(separator=" ", strip=True)
        if txt:
            return txt

    # Fallback heuristic
    spans = soup.find_all("span", class_="wixui-rich-text__text")
    for sp in spans:
        # Skip transcript div
        parent_div = sp.find_parent(id="comp-kf53xx1z")
        if parent_div:
            continue

        text = sp.get_text(separator=" ", strip=True)
        # Look for reasonable summary length
        if 20 < len(text) < 2000:
            return text

    logger.warning("Episode summary not found")
    return ""


def extract_episode_date(soup: BeautifulSoup) -> str:
    """
    Extract episode date from page.

    Selector: #comp-lda48wxn > p > span > span
    Format: "19.10.25"
    """
    node = soup.select_one("#comp-lda48wxn > p > span > span")
    if not node:
        logger.warning("Episode date not found (#comp-lda48wxn > p > span > span)")
        return ""

    text = node.get_text(separator=" ", strip=True)
    return text


def parse_duration_string(raw: str) -> Optional[int]:
    """
    Parse duration string to total seconds.

    Args:
        raw: Duration string like "00:00 / 01:04" or "01:04"

    Returns:
        Total seconds, or None if parsing fails
    """
    if not raw:
        return None

    # Handle "current / total" format
    if "/" in raw:
        right = raw.split("/", 1)[1].strip()
    else:
        right = raw.strip()

    # Try hh:mm:ss format
    m = re.match(r"^(\d{1,2}):(\d{2}):(\d{2})$", right)
    if m:
        h, mi, s = map(int, m.groups())
        return h * 3600 + mi * 60 + s

    # Try mm:ss format
    m = re.match(r"^(\d{1,2}):(\d{2})$", right)
    if m:
        mi, s = map(int, m.groups())
        return mi * 60 + s

    return None


def humanize_minutes(total_seconds: int) -> str:
    """
    Convert seconds to human-readable format.

    Examples:
        - 180 -> "3 minutes"
        - 3840 -> "1 hour and 4 minutes"
        - 7200 -> "2 hours"
    """
    total_minutes = total_seconds // 60

    if total_minutes < 60:
        return f"{total_minutes} minutes"

    hours = total_minutes // 60
    minutes = total_minutes % 60
    hour_word = "hour" if hours == 1 else "hours"

    if minutes == 0:
        return f"{hours} {hour_word}"

    return f"{hours} {hour_word} and {minutes} minutes"


def extract_episode_length(soup: BeautifulSoup) -> str:
    """
    Extract episode length from Wix player.

    Selector: [data-hook="timeStamp"]
    Format: "00:00 / 01:04"

    Returns:
        Human-readable duration like "64 minutes"
    """
    node = soup.find(attrs={"data-hook": "timeStamp"})
    if not node:
        logger.warning('Episode length not found ([data-hook="timeStamp"])')
        return ""

    raw = node.get_text(" ", strip=True)
    total_seconds = parse_duration_string(raw)

    if total_seconds is None:
        logger.warning(f"Could not parse duration: {raw!r}")
        return ""

    return humanize_minutes(total_seconds)


def clean_paragraph_text(s: str) -> str:
    """Clean paragraph text for transcript."""
    if s is None:
        return ""

    # Replace non-breaking spaces
    t = s.replace("\xa0", " ")
    # Normalize line endings
    t = re.sub(r"\r\n|\r", "\n", t)
    # Collapse multiple newlines
    t = re.sub(r"\n{2,}", "\n\n", t)

    return t.strip()


def extract_transcript_paragraphs(soup: BeautifulSoup) -> list[str]:
    """
    Extract transcript paragraphs from episode page.

    Selector: div#comp-kf53xx1z
    Extracts all block-level text elements in DOM order.
    """
    root = soup.find(id="comp-kf53xx1z")
    if not root:
        logger.warning("Transcript div not found (#comp-kf53xx1z)")
        return []

    paragraphs = []
    block_tags = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "div"}

    for node in root.descendants:
        if hasattr(node, 'name') and node.name is not None:
            name = node.name.lower()
            if name in block_tags:
                txt = node.get_text(separator=" ", strip=True)
                if txt:
                    cleaned = clean_paragraph_text(txt)
                    # Avoid duplicates
                    if cleaned and (not paragraphs or paragraphs[-1] != cleaned):
                        paragraphs.append(cleaned)

    # Remove empty paragraphs
    paragraphs = [p for p in paragraphs if p.strip()]
    logger.info(f"Extracted {len(paragraphs)} transcript paragraphs")

    return paragraphs


def save_metadata_to_cache(episode_id: str, metadata: Dict[str, str]):
    """
    Save metadata to cache directory.

    Structure:
        cache/metadata/{episode_id}/
            ├── title.txt
            ├── episode_number.txt
            ├── summary.txt
            ├── episode_date.txt
            ├── episode_length.txt
            └── transcript.txt
    """
    cache_dir = METADATA_CACHE_DIR / episode_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    for key, value in metadata.items():
        if value:  # Only save non-empty values
            file_path = cache_dir / f"{key}.txt"
            file_path.write_text(value, encoding="utf-8")

    logger.info(f"Saved metadata to cache: {cache_dir}")


def load_metadata_from_cache(episode_id: str) -> Optional[Dict[str, str]]:
    """
    Load metadata from cache if available.

    Returns:
        Dict with metadata fields, or None if cache doesn't exist
    """
    cache_dir = METADATA_CACHE_DIR / episode_id
    if not cache_dir.exists():
        return None

    metadata = {}
    fields = ["title", "episode_number", "summary", "episode_date", "episode_length", "transcript"]

    for field in fields:
        file_path = cache_dir / f"{field}.txt"
        if file_path.exists():
            try:
                metadata[field] = file_path.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"Failed to read cached {field}: {e}")
                metadata[field] = ""
        else:
            metadata[field] = ""

    logger.info(f"Loaded metadata from cache: {cache_dir}")
    return metadata


def extract_metadata(url: str, use_cache: bool = True) -> Dict[str, str]:
    """
    Extract all metadata from osimhistoria episode page.

    Args:
        url: Episode page URL
        use_cache: Whether to use cached metadata if available

    Returns:
        Dict containing:
            - title: Episode title
            - episode_number: Episode number
            - summary: Episode summary
            - episode_date: Publication date
            - episode_length: Duration in human-readable format
            - transcript: Full transcript text
            - thumbnail: Path to thumbnail image

    Raises:
        Exception: If extraction fails
    """
    ensure_cache_dir()
    episode_id = get_episode_id(url)

    # Try cache first
    if use_cache:
        cached = load_metadata_from_cache(episode_id)
        if cached:
            # Add thumbnail (not cached, always same)
            cached["thumbnail"] = "static/osim.png"
            logger.info(f"Using cached metadata for episode: {episode_id}")
            return cached

    # Fetch and parse HTML
    try:
        html = fetch_html(url)
        soup = BeautifulSoup(html, "html.parser")
    except Exception as e:
        logger.error(f"Failed to fetch osimhistoria page: {e}")
        raise

    # Extract all metadata fields
    title, episode_number = extract_episode_title_and_number(soup)
    summary = extract_summary(soup)
    episode_date = extract_episode_date(soup)
    episode_length = extract_episode_length(soup)
    transcript_paragraphs = extract_transcript_paragraphs(soup)
    transcript = "\n\n".join(transcript_paragraphs)

    metadata = {
        "title": title,
        "episode_number": episode_number,
        "summary": summary,
        "episode_date": episode_date,
        "episode_length": episode_length,
        "transcript": transcript,
        "thumbnail": "static/osim.png"
    }

    # Save to cache
    try:
        save_metadata_to_cache(episode_id, metadata)
    except Exception as e:
        logger.warning(f"Failed to cache metadata: {e}")

    logger.info(f"Extracted metadata for episode {episode_number}: {title}")
    return metadata
