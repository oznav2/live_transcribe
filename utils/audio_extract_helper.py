#!/usr/bin/env python3
"""
Multi-site audio extractor.

Supported now:
- 103fm.maariv.co.il  (radio episodes with mouthjs)
- www.osimhistoria.com/osimhistoria/...  (Wix page that embeds Spreaker)

Strategy:
- Detect site by URL.
- Run the site-specific extractor.
- If site-specific fails, run a generic "find any .mp3" from the HTML.

This is written to be easy to extend: add another `elif` in `extract_audio_url(...)`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Dict, Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


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

DEBUG = False


def log(msg: str) -> None:
    if DEBUG:
        print(msg)


# ─────────────────────────────────────────────────────────
# RESULT OBJECT
# ─────────────────────────────────────────────────────────

@dataclass
class ExtractionResult:
    ok: bool
    page_url: str
    audio_url: Optional[str]
    site: str
    extra: Dict[str, Any]
    errors: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "page_url": self.page_url,
            "audio_url": self.audio_url,
            "site": self.site,
            "extra": self.extra,
            "errors": self.errors,
        }


# ─────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────

def fetch_text(url: str, referer: Optional[str] = None, timeout: int = 15) -> str:
    headers = dict(DEFAULT_HEADERS)
    if referer:
        headers["Referer"] = referer
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def domain_of(url: str) -> str:
    return urlparse(url).hostname or ""


# ─────────────────────────────────────────────────────────
# 103FM EXTRACTOR
# ─────────────────────────────────────────────────────────

def _103fm_extract(page_url: str, html: str) -> ExtractionResult:
    errors: Dict[str, str] = {}
    soup = BeautifulSoup(html, "html.parser")

    # 1) slug
    slug = None
    el = soup.select_one(".mouthjs-autoplay")
    if el and el.has_attr("data-file"):
        slug = el["data-file"].strip()
        log(f"[103fm] found slug in main page: {slug}")
    else:
        # try embed iframe
        iframe = soup.find("iframe", src=lambda x: x and "103embed.maariv.co.il" in x)
        if iframe and iframe.has_attr("src"):
            embed_url = urljoin(page_url, iframe["src"])
            try:
                embed_html = fetch_text(embed_url, referer=page_url)
                embed_soup = BeautifulSoup(embed_html, "html.parser")
                el2 = embed_soup.select_one(".mouthjs-autoplay")
                if el2 and el2.has_attr("data-file"):
                    slug = el2["data-file"].strip()
                    log(f"[103fm] found slug in embed: {slug}")
                else:
                    errors["embed_slug"] = "Embed loaded but no data-file."
            except Exception as e:
                errors["embed_fetch"] = str(e)
        else:
            errors["slug"] = "No .mouthjs-autoplay with data-file"

    if not slug:
        return ExtractionResult(False, page_url, None, "103fm", {}, errors)

    # 2) try to find player JS in page
    js_url = None
    for s in soup.find_all("script", src=True):
        src = s["src"]
        if "playerLive/IMA/" in src and "aod_" in src:
            js_url = urljoin("https://103fm.maariv.co.il/", src.lstrip("/"))
            break

    if not js_url:
        js_url = FALLBACK_103FM_SETTINGS_JS
        log("[103fm] using fallback JS")

    # 3) read JS and extract base
    base_url = None
    try:
        js_text = fetch_text(js_url, referer=page_url)
        # look for streamgates / 103fm_aw in JS
        for pat in (
            r'https?://[^\'"]*streamgates[^\'"]+?/',
            r'https?://[^\'"]+103fm_aw/',
        ):
            m = re.search(pat, js_text, flags=re.IGNORECASE)
            if m:
                base_url = m.group(0)
                break
        if not base_url:
            errors["js_parse"] = "Player JS loaded but no base found"
    except Exception as e:
        errors["js_fetch"] = str(e)

    if not base_url:
        base_url = HARDCODED_103FM_BASE

    if not base_url.endswith("/"):
        base_url += "/"

    audio_url = f"{base_url}{slug}.mp3"

    return ExtractionResult(True, page_url, audio_url, "103fm",
                            {"slug": slug, "base_url": base_url}, errors)


# ─────────────────────────────────────────────────────────
# OSIM HISTORIA / WIX + SPREAKER EXTRACTOR
# ─────────────────────────────────────────────────────────
#
# We saw in the HTML you attached that the page literally contains:
#   "url":"https://api.spreaker.com/v2/episodes/68200381/download.mp3"
# and also a button:
#   <a href="https://api.spreaker.com/v2/episodes/68200381/download.mp3" ...>
# so a regex is enough. :contentReference[oaicite:1]{index=1}

def _osimhistoria_extract(page_url: str, html: str) -> ExtractionResult:
    errors: Dict[str, str] = {}
    audio_url: Optional[str] = None

    # 1) first, exact Spreaker pattern
    m = re.search(r'https://api\.spreaker\.com/v2/episodes/\d+/download\.mp3', html)
    if m:
        audio_url = m.group(0)
        return ExtractionResult(True, page_url, audio_url, "osimhistoria-spreaker", {}, errors)

    # 2) fallback: any https://api.spreaker.com ... mp3
    m2 = re.search(r'https://api\.spreaker\.com[^\s"\'<>]+\.mp3', html)
    if m2:
        audio_url = m2.group(0)
        return ExtractionResult(True, page_url, audio_url, "osimhistoria-spreaker-generic", {}, errors)

    # 3) generic mp3 in page
    generic = _find_first_mp3(html)
    if generic:
        return ExtractionResult(True, page_url, generic, "osimhistoria-generic-mp3", {}, errors)

    errors["spreaker"] = "No spreaker download link found in page"
    return ExtractionResult(False, page_url, None, "osimhistoria", {}, errors)


# ─────────────────────────────────────────────────────────
# GENERIC FALLBACK
# ─────────────────────────────────────────────────────────

def _find_first_mp3(html: str) -> Optional[str]:
    m = re.search(r'https?://[^\s"\'<>]+\.mp3', html)
    return m.group(0) if m else None


def _generic_extract(page_url: str, html: str) -> ExtractionResult:
    errors: Dict[str, str] = {}
    mp3 = _find_first_mp3(html)
    if mp3:
        return ExtractionResult(True, page_url, mp3, "generic-mp3", {}, errors)
    errors["generic"] = "No .mp3 link found in HTML"
    return ExtractionResult(False, page_url, None, "generic", {}, errors)


# ─────────────────────────────────────────────────────────
# PUBLIC ENTRY
# ─────────────────────────────────────────────────────────

def extract_audio_url(page_url: str) -> ExtractionResult:
    """
    Main entry point.
    Decide which extractor to run based on domain.
    """
    try:
        html = fetch_text(page_url, referer=page_url)
    except Exception as e:
        return ExtractionResult(False, page_url, None, "fetch", {}, {"page_fetch": str(e)})

    host = domain_of(page_url)

    if "103fm.maariv.co.il" in host:
        return _103fm_extract(page_url, html)

    if "osimhistoria.com" in host:
        return _osimhistoria_extract(page_url, html)

    # if some other site — just try generic
    return _generic_extract(page_url, html)


# ─────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    DEBUG = True
    tests = [
        # 103fm
        "https://103fm.maariv.co.il/programs/media.aspx?ZrqvnVq=KEELLL&c41t4nzVQ=FJF",
        "https://103fm.maariv.co.il/programs/media.aspx?ZrqvnVq=KEEMLI&c41t4nzVQ=FJF",
        # osimhistoria
        "https://www.osimhistoria.com/osimhistoria/ep_454",
    ]
    for url in tests:
        res = extract_audio_url(url)
        print(res.to_dict())
