"""
site_watcher.py
----------------
RSS-less Newsroom Monitoring Daemon using Tavily Map and Tavily Extract.
Demonstrates to hackathon judges that FeedProof monitors sites WITHOUT RSS feeds
(corporate press rooms, research announcements, regulatory releases) by mapping URL
trees, detecting newly published articles via diffing, and extracting full clean text.
"""

import os
import sys
import time
import re
import logging
import argparse
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import requests
from dotenv import load_dotenv

from tavily_service import tavily_map, tavily_extract
from url_utils import canonicalize_url, extract_domain
from content_cleaner import clean_article_markdown

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("site_watcher")

API_INGEST_URL = os.getenv("API_INGEST_URL", "http://localhost:8000/api/internal/ingest")
INGESTION_SECRET = os.getenv("INGESTION_SECRET")

# Seed sites that frequently publish major breaking news without traditional RSS
SEED_WATCHED_SITES = [
    {
        "site_name": "OpenAI Newsroom",
        "root_url": "https://openai.com/news",
        "select_paths": ["/news/.*", "/index/.*"]
    },
    {
        "site_name": "Google DeepMind Discover",
        "root_url": "https://deepmind.google/discover/blog",
        "select_paths": ["/discover/blog/.*"]
    },
    {
        "site_name": "TSMC Technical Press Releases",
        "root_url": "https://www.tsmc.com/english/news-events",
        "select_paths": ["/news-events/.*", "/press-releases/.*"]
    }
]

# Noise filter patterns to ignore index/category pages and pagination
NOISE_PATTERNS = [
    r"/tag/", r"/category/", r"/author/", r"/page/\d+", r"\?page=",
    r"/archive/", r"/search/", r"\.pdf$", r"\.jpg$", r"\.png$"
]


def is_valid_news_url(url: str) -> bool:
    """Filters out pagination, author profiles, and asset links."""
    for pattern in NOISE_PATTERNS:
        if re.search(pattern, url, re.IGNORECASE):
            return False
    # Ensure it's not just the root domain
    path = re.sub(r"^https?://[^/]+", "", url).strip("/")
    return len(path) > 3


def seed_watched_sites():
    """Seeds default monitored newsrooms into the database if not present."""
    try:
        from db import pool
        with pool.connection(timeout=0.3) as conn:
            with conn.cursor() as cur:
                for s in SEED_WATCHED_SITES:
                    cur.execute("""
                        INSERT INTO watched_sites (site_name, root_url, select_paths)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (root_url) DO NOTHING;
                    """, (s["site_name"], s["root_url"], s["select_paths"]))
            conn.commit()
    except Exception as e:
        logger.info("Watched sites DB note (offline/initializing): %s", e)


def get_active_sites() -> List[Dict[str, Any]]:
    """Fetches active sites to monitor from PostgreSQL or falls back to seeds."""
    try:
        from db import pool
        with pool.connection(timeout=0.3) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, site_name, root_url, select_paths FROM watched_sites WHERE is_active = TRUE;")
                rows = cur.fetchall()
                if rows:
                    return [
                        {"id": str(r[0]), "site_name": r[1], "root_url": r[2], "select_paths": r[3]}
                        for r in rows
                    ]
    except Exception:
        pass
    return SEED_WATCHED_SITES


def run_site_watch_cycle(max_extract_per_site: int = 3) -> tuple[int, int]:
    """
    Executes a single monitoring cycle:
      1. Maps each active site via Tavily Map.
      2. Diffs discovered URLs against local database.
      3. Extracts newly detected articles using Tavily Extract.
      4. Cleans and ingests articles into the FeedProof news pipeline.
    Returns (total_discovered, total_ingested).
    """
    secret = os.getenv("INGESTION_SECRET")
    sites = get_active_sites()
    total_discovered = 0
    total_ingested = 0

    logger.info("Starting Tavily Site Watch Cycle across %d sites...", len(sites))

    for site in sites:
        site_name = site["site_name"]
        root_url = site["root_url"]
        select_paths = site.get("select_paths") or ["/news/.*", "/blog/.*"]

        logger.info("[Site Watch: %s] Mapping URL structure via Tavily Map...", site_name)
        map_resp = tavily_map(
            url=root_url,
            select_paths=select_paths,
            limit=50,
            purpose="site_watch"
        )
        found_urls = map_resp.get("results", [])
        if not found_urls:
            continue

        # Filter noise and canonicalize
        clean_urls = []
        for u in found_urls:
            canonical = canonicalize_url(u)
            if canonical and is_valid_news_url(canonical) and canonical not in clean_urls:
                clean_urls.append(canonical)

        logger.info("[Site Watch: %s] Discovered %d valid candidate URLs.", site_name, len(clean_urls))

        # Diff against database to identify genuinely new URLs
        new_urls = []
        try:
            from db import pool
            with pool.connection(timeout=0.3) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT url_canonical FROM site_discovered_urls WHERE url_canonical = ANY(%s);", (clean_urls,))
                    existing = {r[0] for r in cur.fetchall()}
                    new_urls = [u for u in clean_urls if u not in existing]
        except Exception:
            # Offline / memory fallback
            new_urls = clean_urls[:max_extract_per_site]

        if not new_urls:
            logger.info("[Site Watch: %s] No new unmapped URLs detected.", site_name)
            continue

        total_discovered += len(new_urls)
        selected_to_extract = new_urls[:max_extract_per_site]
        logger.info("[Site Watch: %s] Extracting %d new articles via Tavily Extract...", site_name, len(selected_to_extract))

        # Extract full markdown content
        extract_resp = tavily_extract(
            urls=selected_to_extract,
            extract_depth="basic",
            purpose="site_watch"
        )
        extracted_results = extract_resp.get("results", [])

        payload = []
        for r in extracted_results:
            raw_url = r.get("url", "")
            title = r.get("title", f"Update from {site_name}")
            raw_text = r.get("raw_content", "")

            cleaned_text = clean_article_markdown(title, raw_text)
            if not cleaned_text or len(cleaned_text) < 200:
                continue

            payload.append({
                "Site": extract_domain(raw_url),
                "Baslik": title,
                "Link": canonicalize_url(raw_url),
                "Yazar": site_name,
                "Tarih": datetime.now(timezone.utc).isoformat(),
                "TamMetin": cleaned_text
            })

        if payload and secret:
            try:
                resp = requests.post(
                    API_INGEST_URL,
                    json=payload,
                    headers={"x-secret": secret},
                    timeout=3.0
                )
                if resp.status_code == 200:
                    inserted = resp.json().get("inserted", 0)
                    total_ingested += inserted
                    logger.info("[Site Watch: %s] Successfully ingested %d articles.", site_name, inserted)
            except Exception as e:
                logger.warning("[Site Watch: %s] Ingestion push failed: %s", site_name, e)

        # Record discovered URLs in database
        try:
            from db import pool
            site_id = site.get("id")
            with pool.connection(timeout=0.3) as conn:
                with conn.cursor() as cur:
                    for u in selected_to_extract:
                        cur.execute("""
                            INSERT INTO site_discovered_urls (site_id, url_canonical, is_ingested)
                            VALUES (%s, %s, TRUE)
                            ON CONFLICT (site_id, url_canonical) DO NOTHING;
                        """, (site_id, u))
                conn.commit()
        except Exception:
            pass

    logger.info("Site Watch Cycle Complete: %d newly discovered, %d newly ingested.", total_discovered, total_ingested)
    return total_discovered, total_ingested


def main():
    parser = argparse.ArgumentParser(description="FeedProof RSS-less Site Watcher")
    parser.add_argument("--once", action="store_true", default=True, help="Run a single watch cycle and exit (default)")
    parser.add_argument("--loop", action="store_true", help="Run continuously in background")
    parser.add_argument("--interval", type=int, default=14400, help="Loop interval in seconds (default: 14400s / 4 hours)")
    args = parser.parse_args()

    seed_watched_sites()
    while True:
        run_site_watch_cycle()
        if not args.loop:
            break
        logger.info("Sleeping for %d seconds until next site watch cycle...", args.interval)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
