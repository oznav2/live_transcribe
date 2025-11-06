"""
Multi-site audio URL extractor for sites that don't work with yt-dlp/ffmpeg.

Supported sites:
- 103fm.maariv.co.il  (radio episodes with mouthjs player)
- www.osimhistoria.com/osimhistoria/...  (Wix page that embeds Spreaker)

Strategy:
- Detect site by URL domain
- Run site-specific HTML extraction
- If site-specific fails, try generic .mp3 finder
- Return direct MP3 URL for ffmpeg download

This module is designed to be easily extensible: add another `elif` in `extract_audio_url()`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# GLOBAL CONFIG
# ─────────────────────────────────────────────────────────

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "he-IL,he;q=0.9,en-US;q=0.8,en;q=0.7",
}

# 103fm-specific defaults
HARDCODED_103FM_BASE = "https://awaod01.streamgates.net/103fm_aw/"
FALLBACK_103FM_SETTINGS_JS = "https://103fm.maariv.co.il/playerLive/IMA/aod_sttings_24.10.22.js"


# ─────────────────────────────────────────────────────────
# RESULT OBJECT
# ─────────────────────────────────────────────────────────

@dataclass
class ExtractionResult:
    """Result of audio URL extraction attempt."""
    ok: bool
    page_url: str
    audio_url: Optional[str]
    site: str
    extra: Dict[str, Any]
    errors: Dict[str, str]
    title: Optional[str] = None
    duration: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "page_url": self.page_url,
            "audio_url": self.audio_url,
            "site": self.site,
            "extra": self.extra,
            "errors": self.errors,
            "title": self.title,
            "duration": self.duration,
        }


# ─────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────

def fetch_text(url: str, referer: Optional[str] = None, timeout: int = 15) -> str:
    """Fetch HTML/text content from URL with proper headers."""
    headers = dict(DEFAULT_HEADERS)
    if referer:
        headers["Referer"] = referer
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def domain_of(url: str) -> str:
    """Extract domain from URL."""
    return urlparse(url).hostname or ""


def is_special_site(url: str) -> bool:
    """Quick check if URL requires special audio extraction."""
    host = domain_of(url).lower()
    return "103fm.maariv.co.il" in host or "osimhistoria.com" in host


# ─────────────────────────────────────────────────────────
# 103FM EXTRACTOR
# ─────────────────────────────────────────────────────────

def _103fm_extract(page_url: str, html: str) -> ExtractionResult:
    """
    Extract audio URL from 103fm.maariv.co.il page.

    Strategy:
    1. Find slug from .mouthjs-autoplay[data-file] or embed iframe
    2. Find base URL from player JS (or use hardcoded fallback)
    3. Construct: {base_url}{slug}.mp3
    """
    errors: Dict[str, str] = {}
    soup = BeautifulSoup(html, "html.parser")

    # 1) Extract slug
    slug = None
    el = soup.select_one(".mouthjs-autoplay")
    if el and el.has_attr("data-file"):
        slug = el["data-file"].strip()
        logger.info(f"[103fm] Found slug in main page: {slug}")
    else:
        # Try embed iframe
        iframe = soup.find("iframe", src=lambda x: x and "103embed.maariv.co.il" in x)
        if iframe and iframe.has_attr("src"):
            embed_url = urljoin(page_url, iframe["src"])
            try:
                embed_html = fetch_text(embed_url, referer=page_url)
                embed_soup = BeautifulSoup(embed_html, "html.parser")
                el2 = embed_soup.select_one(".mouthjs-autoplay")
                if el2 and el2.has_attr("data-file"):
                    slug = el2["data-file"].strip()
                    logger.info(f"[103fm] Found slug in embed: {slug}")
                else:
                    errors["embed_slug"] = "Embed loaded but no data-file found"
            except Exception as e:
                errors["embed_fetch"] = str(e)
                logger.warning(f"[103fm] Failed to fetch embed: {e}")
        else:
            errors["slug"] = "No .mouthjs-autoplay with data-file found"

    if not slug:
        return ExtractionResult(False, page_url, None, "103fm", {}, errors)

    # 2) Find player JS in page to extract base URL
    js_url = None
    for s in soup.find_all("script", src=True):
        src = s["src"]
        if "playerLive/IMA/" in src and "aod_" in src:
            js_url = urljoin("https://103fm.maariv.co.il/", src.lstrip("/"))
            break

    if not js_url:
        js_url = FALLBACK_103FM_SETTINGS_JS
        logger.info("[103fm] Using fallback JS URL")

    # 3) Read JS and extract base URL
    base_url = None
    try:
        js_text = fetch_text(js_url, referer=page_url)
        # Look for streamgates / 103fm_aw in JS
        for pat in (
            r'https?://[^\'"]*streamgates[^\'"]+?/',
            r'https?://[^\'"]+103fm_aw/',
        ):
            m = re.search(pat, js_text, flags=re.IGNORECASE)
            if m:
                base_url = m.group(0)
                logger.info(f"[103fm] Found base URL in JS: {base_url}")
                break
        if not base_url:
            errors["js_parse"] = "Player JS loaded but no base URL found"
    except Exception as e:
        errors["js_fetch"] = str(e)
        logger.warning(f"[103fm] Failed to fetch player JS: {e}")

    if not base_url:
        base_url = HARDCODED_103FM_BASE
        logger.info(f"[103fm] Using hardcoded base URL: {base_url}")

    if not base_url.endswith("/"):
        base_url += "/"

    audio_url = f"{base_url}{slug}.mp3"
    logger.info(f"[103fm] Extracted audio URL: {audio_url}")

    # Try to extract title from page
    title = None
    try:
        # Look for common title patterns in 103fm pages
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)
            # Clean up common suffixes
            if " - 103FM" in title:
                title = title.split(" - 103FM")[0].strip()
    except Exception as e:
        logger.debug(f"[103fm] Could not extract title: {e}")

    return ExtractionResult(True, page_url, audio_url, "103fm",
                            {"slug": slug, "base_url": base_url}, errors,
                            title=title)


# ─────────────────────────────────────────────────────────
# OSIM HISTORIA / WIX + SPREAKER EXTRACTOR
# ─────────────────────────────────────────────────────────

def _osimhistoria_extract(page_url: str, html: str) -> ExtractionResult:
    """
    Extract audio URL and metadata from osimhistoria.com page.

    Strategy:
    1. Look for Spreaker download URL pattern in HTML
    2. Fallback to generic Spreaker MP3 pattern
    3. Last resort: any .mp3 URL in page
    4. Extract full metadata (title, duration, transcript, etc.)
    """
    errors: Dict[str, str] = {}
    audio_url: Optional[str] = None

    # 1) Exact Spreaker download pattern
    m = re.search(r'https://api\.spreaker\.com/v2/episodes/\d+/download\.mp3', html)
    if m:
        audio_url = m.group(0)
        logger.info(f"[osimhistoria] Found Spreaker download URL: {audio_url}")

    # 2) Generic Spreaker MP3 pattern
    elif (m2 := re.search(r'https://api\.spreaker\.com[^\s"\'<>]+\.mp3', html)):
        audio_url = m2.group(0)
        logger.info(f"[osimhistoria] Found generic Spreaker MP3: {audio_url}")

    # 3) Generic MP3 fallback
    elif (generic := _find_first_mp3(html)):
        audio_url = generic
        logger.info(f"[osimhistoria] Found generic MP3: {audio_url}")

    if not audio_url:
        errors["spreaker"] = "No Spreaker download link found in page"
        return ExtractionResult(False, page_url, None, "osimhistoria", {}, errors)

    # 4) Extract full metadata using osimhistoria metadata service
    metadata = {}
    title = None
    duration = None

    try:
        from services.osimhistoria_metadata import extract_metadata
        metadata = extract_metadata(page_url, use_cache=True)
        title = metadata.get("title")
        duration = metadata.get("episode_length")
        logger.info(f"[osimhistoria] Extracted metadata: title={title}, duration={duration}")
    except Exception as e:
        logger.warning(f"[osimhistoria] Failed to extract metadata: {e}")
        errors["metadata"] = str(e)

    return ExtractionResult(
        True, page_url, audio_url, "osimhistoria-spreaker",
        metadata, errors, title=title, duration=duration
    )


# ─────────────────────────────────────────────────────────
# GENERIC FALLBACK
# ─────────────────────────────────────────────────────────

def _find_first_mp3(html: str) -> Optional[str]:
    """Find first .mp3 URL in HTML."""
    m = re.search(r'https?://[^\s"\'<>]+\.mp3', html)
    return m.group(0) if m else None


def _generic_extract(page_url: str, html: str) -> ExtractionResult:
    """Generic extractor: try to find any .mp3 link in HTML."""
    errors: Dict[str, str] = {}
    mp3 = _find_first_mp3(html)
    if mp3:
        logger.info(f"[generic] Found MP3: {mp3}")
        return ExtractionResult(True, page_url, mp3, "generic-mp3", {}, errors)
    errors["generic"] = "No .mp3 link found in HTML"
    return ExtractionResult(False, page_url, None, "generic", {}, errors)


# ─────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────

def extract_audio_url(page_url: str) -> ExtractionResult:
    """
    Main entry point for audio URL extraction.

    Detects site by domain and runs appropriate extractor.
    Returns ExtractionResult with direct MP3 URL if successful.

    Args:
        page_url: URL of page to extract audio from

    Returns:
        ExtractionResult with ok=True and audio_url if successful
    """
    try:
        html = fetch_text(page_url, referer=page_url)
    except Exception as e:
        logger.error(f"Failed to fetch page {page_url}: {e}")
        return ExtractionResult(False, page_url, None, "fetch", {}, {"page_fetch": str(e)})

    host = domain_of(page_url)

    if "103fm.maariv.co.il" in host:
        logger.info(f"Detected 103fm site, using specialized extractor")
        return _103fm_extract(page_url, html)

    if "osimhistoria.com" in host:
        logger.info(f"Detected osimhistoria site, using specialized extractor")
        return _osimhistoria_extract(page_url, html)

    # Unknown site - try generic MP3 finder
    logger.info(f"Unknown site, trying generic MP3 extraction")
    return _generic_extract(page_url, html)
