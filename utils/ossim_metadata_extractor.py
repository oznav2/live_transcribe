#!/usr/bin/env python3
"""
Improved extractor for osimhistoria episode pages.

Extracts:
- episode_number.txt
- title.txt
- summary.txt
- episode_date.txt
- episode_length.txt
- transcript.txt

All saved into: cache/metadata/

Selectors (from the real page):
- Title + episode number:
    #comp-lei4z7va > div > div
    e.g. "454: התשתית של הקפיטליזם - על הנהלת חשבונות כפולה"

- Summary:
    #comp-kf53u3eb > p > span > span > span

- Date:
    #comp-lda48wxn > p > span > span
    e.g. "19.10.25"

- Audio length (Wix player):
    [data-hook="timeStamp"]
    e.g. "00:00 / 01:04"

- Transcript:
    #comp-kf53xx1z
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag

# ---- config ----
DEFAULT_LOCAL_FILE = "/mnt/data/ossim.html"
OUTPUT_DIR = Path("cache/metadata")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ossim-extractor/1.2)"
}
REQUEST_TIMEOUT = 12
# -----------------


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_html(url: str) -> str:
    logging.info(f"Fetching URL: {url}")
    resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    resp.encoding = resp.encoding or "utf-8"
    return resp.text


def read_local_html(path: str) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8", errors="replace")


def soup_from_html(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


# ------------- specific extractors -------------

def extract_episode_title_and_number(soup: BeautifulSoup):
    """
    From: #comp-lei4z7va > div > div
    e.g. "454: התשתית של הקפיטליזם - על הנהלת חשבונות כפולה"
    """
    node = soup.select_one("#comp-lei4z7va > div > div")
    if not node:
        logging.warning("Episode title container (#comp-lei4z7va > div > div) not found.")
        return "", ""
    full_text = node.get_text(separator=" ", strip=True)
    if not full_text:
        return "", ""
    if ":" in full_text:
        ep_num, _, ep_title = full_text.partition(":")
        return ep_title.strip(), ep_num.strip()
    return full_text.strip(), ""


def extract_summary(soup: BeautifulSoup):
    """
    From: #comp-kf53u3eb > p > span > span > span
    """
    node = soup.select_one("#comp-kf53u3eb > p > span > span > span")
    if node:
        txt = node.get_text(separator=" ", strip=True)
        if txt:
            return txt

    # fallback to previous heuristic if needed
    spans = soup.find_all("span", class_="wixui-rich-text__text")
    for sp in spans:
        parent_div = sp.find_parent(id="comp-kf53xx1z")
        if parent_div:
            continue
        text = sp.get_text(separator=" ", strip=True)
        if 20 < len(text) < 2000:
            return text

    return ""


def extract_episode_date(soup: BeautifulSoup):
    """
    From: #comp-lda48wxn > p > span > span
    Example: "19.10.25"
    """
    node = soup.select_one("#comp-lda48wxn > p > span > span")
    if not node:
        logging.warning("Episode date node (#comp-lda48wxn > p > span > span) not found.")
        return ""
    text = node.get_text(separator=" ", strip=True)
    return text


def parse_duration_string(raw: str):
    """
    raw is something like "00:00 / 01:04"
    We want the right-hand side, e.g. "01:04".
    Then parse as hh:mm:ss or mm:ss.
    Return total_seconds (int) or None.
    """
    if not raw:
        return None
    # common format: "00:00 / 01:04"
    if "/" in raw:
        right = raw.split("/", 1)[1].strip()
    else:
        right = raw.strip()

    # try hh:mm:ss
    m = re.match(r"^(\d{1,2}):(\d{2}):(\d{2})$", right)
    if m:
        h, mi, s = map(int, m.groups())
        return h * 3600 + mi * 60 + s

    # try mm:ss
    m = re.match(r"^(\d{1,2}):(\d{2})$", right)
    if m:
        mi, s = map(int, m.groups())
        return mi * 60 + s

    return None


def humanize_minutes(total_seconds: int) -> str:
    """
    Convert seconds to the string format requested:
    - if < 60 minutes -> "X minutes"
    - if >= 60 minutes -> "H hour[s] and M minutes" (omit 'and M minutes' if M == 0)
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


def extract_episode_length(soup: BeautifulSoup):
    """
    From the Wix player span:
        [data-hook="timeStamp"]
    e.g. "00:00 / 01:04"
    """
    node = soup.find(attrs={"data-hook": "timeStamp"})
    if not node:
        logging.warning('Episode length node ([data-hook="timeStamp"]) not found.')
        return ""
    raw = node.get_text(" ", strip=True)
    total_seconds = parse_duration_string(raw)
    if total_seconds is None:
        logging.warning(f"Could not parse duration from: {raw!r}")
        return ""
    return humanize_minutes(total_seconds)


def clean_paragraph_text(s: str) -> str:
    if s is None:
        return ""
    t = s.replace("\xa0", " ")
    t = re.sub(r"\r\n|\r", "\n", t)
    t = re.sub(r"\n{2,}", "\n\n", t)
    return t.strip()


def extract_transcript_paragraphs(soup: BeautifulSoup):
    """
    Read from div#comp-kf53xx1z in DOM order.
    """
    root = soup.find(id="comp-kf53xx1z")
    if not root:
        logging.warning("Transcript div (#comp-kf53xx1z) not found.")
        return []

    paragraphs = []
    block_tags = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "div"}

    for node in root.descendants:
        if isinstance(node, Tag):
            name = node.name.lower()
            if name in block_tags:
                txt = node.get_text(separator=" ", strip=True)
                if txt:
                    cleaned = clean_paragraph_text(txt)
                    if cleaned and (not paragraphs or paragraphs[-1] != cleaned):
                        paragraphs.append(cleaned)

    paragraphs = [p for p in paragraphs if p.strip()]
    logging.info(f"Extracted {len(paragraphs)} transcript paragraphs")
    return paragraphs


# ------------- writing -------------

def write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def process_html(html: str):
    soup = soup_from_html(html)

    # title + ep number
    title, ep_num = extract_episode_title_and_number(soup)
    # summary
    summary = extract_summary(soup)
    # date
    episode_date = extract_episode_date(soup)
    # length
    episode_length = extract_episode_length(soup)
    # transcript
    transcript_paragraphs = extract_transcript_paragraphs(soup)
    transcript_text = "\n\n".join(transcript_paragraphs)

    ensure_output_dir()
    write_text(OUTPUT_DIR / "title.txt", title)
    write_text(OUTPUT_DIR / "episode_number.txt", ep_num)
    write_text(OUTPUT_DIR / "summary.txt", summary)
    write_text(OUTPUT_DIR / "episode_date.txt", episode_date)
    write_text(OUTPUT_DIR / "episode_length.txt", episode_length)
    write_text(OUTPUT_DIR / "transcript.txt", transcript_text)

    return {
        "title": title,
        "episode_number": ep_num,
        "summary_len": len(summary),
        "episode_date": episode_date,
        "episode_length": episode_length,
        "transcript_paragraphs": len(transcript_paragraphs),
    }


def main():
    ap = argparse.ArgumentParser(description="Extract Osim Historia episode metadata")
    ap.add_argument("--url", "-u", help="Episode URL to fetch")
    ap.add_argument("--file", "-f", help="Local HTML file to read")
    args = ap.parse_args()

    html = None
    last_err = None

    if args.file:
        try:
            html = read_local_html(args.file)
        except Exception as e:
            logging.warning(f"Failed to read local file {args.file}: {e!r}")
            last_err = e

    if html is None and args.url:
        try:
            html = fetch_html(args.url)
        except Exception as e:
            logging.warning(f"Failed to fetch {args.url}: {e!r}")
            last_err = e

    if html is None:
        # final fallback to uploaded file
        try:
            html = read_local_html(DEFAULT_LOCAL_FILE)
            logging.info(f"Using local fallback {DEFAULT_LOCAL_FILE}")
        except Exception as e:
            raise RuntimeError("Could not load any HTML source") from (last_err or e)

    result = process_html(html)
    logging.info("Done:")
    for k, v in result.items():
        logging.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
