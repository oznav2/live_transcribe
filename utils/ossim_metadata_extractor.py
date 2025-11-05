#!/usr/bin/env python3
"""
Improved extractor for osimhistoria page.

- Episode number + title from:
    #comp-lei4z7va > div > div
  Example text: "454: התשתית של הקפיטליזם - על הנהלת חשבונות כפולה"
  -> episode_number = "454"
  -> title = "התשתית של הקפיטליזם - על הנהלת חשבונות כפולה"

- Summary from:
    #comp-kf53u3eb > p > span > span > span

- Transcript from:
    #comp-kf53xx1z
  paragraphs in page order, separated by a blank line

Outputs to cache/metadata/*.txt
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag, NavigableString

# ---- config ----
DEFAULT_LOCAL_FILE = "/mnt/data/ossim.html"
OUTPUT_DIR = Path("cache/metadata")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ossim-extractor/1.1)"
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
    Target the exact container:
        #comp-lei4z7va > div > div
    Text format:
        "<number>: <title>"
    We return (title, number) where number may be "" if not found.
    """
    title_sel = soup.select_one("#comp-lei4z7va > div > div")
    if not title_sel:
        logging.warning("Episode title container (#comp-lei4z7va > div > div) not found.")
        return "", ""

    full_text = title_sel.get_text(separator=" ", strip=True)
    if not full_text:
        return "", ""

    # Split only on FIRST colon
    if ":" in full_text:
        ep_num, _, ep_title = full_text.partition(":")
        ep_num = ep_num.strip()
        ep_title = ep_title.strip()
        return ep_title, ep_num
    else:
        # no colon: treat whole thing as title
        return full_text.strip(), ""

def extract_summary(soup: BeautifulSoup):
    """
    Target:
        #comp-kf53u3eb > p > span > span > span
    If missing/empty, fallback to older heuristic.
    """
    node = soup.select_one("#comp-kf53u3eb > p > span > span > span")
    if node:
        text = node.get_text(separator=" ", strip=True)
        if text:
            return text

    # fallback: previous heuristic
    spans = soup.find_all("span", class_="wixui-rich-text__text")
    for sp in spans:
        text = sp.get_text(separator=" ", strip=True)
        # pick a mid-length span not in transcript
        parent_div = sp.find_parent(id="comp-kf53xx1z")
        if parent_div:
            continue
        if 20 < len(text) < 2000:
            return text

    return ""


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
    Keep h1-h6, p, li, blockquote, div textual blocks.
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

    # remove empties
    paragraphs = [p for p in paragraphs if p.strip()]
    logging.info(f"Extracted {len(paragraphs)} transcript paragraphs")
    return paragraphs


# ------------- writing -------------

def write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def process_html(html: str):
    soup = soup_from_html(html)

    # 1) episode title + number
    title, ep_num = extract_episode_title_and_number(soup)
    logging.info(f"Episode title: {title!r}")
    logging.info(f"Episode number: {ep_num!r}")

    # 2) summary
    summary = extract_summary(soup)
    logging.info(f"Summary length: {len(summary)}")

    # 3) transcript
    transcript_paragraphs = extract_transcript_paragraphs(soup)
    transcript_text = "\n\n".join(transcript_paragraphs)

    # 4) write
    ensure_output_dir()
    write_text(OUTPUT_DIR / "title.txt", title)
    write_text(OUTPUT_DIR / "episode_number.txt", ep_num)
    write_text(OUTPUT_DIR / "summary.txt", summary)
    write_text(OUTPUT_DIR / "transcript.txt", transcript_text)

    return {
        "title": title,
        "episode_number": ep_num,
        "summary_length": len(summary),
        "transcript_paragraphs": len(transcript_paragraphs),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", "-u", help="URL to fetch")
    parser.add_argument("--file", "-f", help="Local HTML file to read")
    args = parser.parse_args()

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
        # fallbacks: try sample, then local default
        try:
            html = read_local_html(DEFAULT_LOCAL_FILE)
            logging.info(f"Using local fallback: {DEFAULT_LOCAL_FILE}")
        except Exception as e:
            logging.error("Could not obtain HTML at all.")
            raise RuntimeError("No HTML source available") from (last_err or e)

    result = process_html(html)
    logging.info("Done:")
    for k, v in result.items():
        logging.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
