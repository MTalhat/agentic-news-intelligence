"""
site_watcher.py
----------------
RSS-less Newsroom Monitoring Daemon using Tavily Map, Tavily Extract, and Tavily Crawl.
Demonstrates to hackathon judges that FeedProof monitors sites WITHOUT RSS feeds
(corporate press rooms, research announcements, regulatory releases) by mapping URL
trees, recording full site baselines, diffing newly published articles, inferring
publication dates, and harvesting full clean Markdown.
"""

import os
import sys
import time
import re
import logging
import argparse
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timezone
from dateutil import parser as date_parser
import requests
from dotenv import load_dotenv

from tavily_service import tavily_map, tavily_extract, tavily_crawl
from url_utils import canonicalize_url, extract_domain
from content_cleaner import clean_article_markdown

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("site_watcher")

API_INGEST_URL = os.getenv("API_INGEST_URL", "http://localhost:8000/api/internal/ingest")
INGESTION_SECRET = os.getenv("INGESTION_SECRET")

# In-memory URL baseline tracker (used alongside PostgreSQL site_discovered_urls)
_memory_known_urls: Dict[str, Set[str]] = {}

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
    path = re.sub(r"^https?://[^/]+", "", url).strip("/")
    return len(path) > 3


def _infer_article_date(url: str, raw_text: str) -> str:
    """
    Infers publication date from URL path (/YYYY/MM/DD/ or YYYY-MM-DD) or lead article text
    before falling back to current UTC timestamp (fixes D1/N6).
    """
    now_utc = datetime.now(timezone.utc)
    # 1. Check URL path for YYYY/MM/DD or YYYY-MM-DD
    url_match = re.search(r"/(202\d)[/-](0[1-9]|1[0-2])[/-](0[1-9]|[12]\d|3[01])\b", url)
    if url_match:
        try:
            y, m, d = int(url_match.group(1)), int(url_match.group(2)), int(url_match.group(3))
            dt = datetime(y, m, d, tzinfo=timezone.utc)
            if dt <= now_utc:
                return dt.isoformat()
        except Exception:
            pass

    # 2. Check lead 400 characters of Markdown text for explicit date strings
    if raw_text:
        head = raw_text[:400]
        iso_match = re.search(
            r"\b(202\d-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])(?:T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})?)?)\b",
            head
        )
        if iso_match:
            try:
                dt = date_parser.parse(iso_match.group(1))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt <= now_utc:
                    return dt.isoformat()
            except Exception:
                pass

    return now_utc.isoformat()


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
    """Seeds defaults and fetches active sites to monitor from PostgreSQL, with seed fallback."""
    seed_watched_sites()
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
      1. Maps each active site via Tavily Map (with instructions & exclude_paths).
      2. Diffs discovered URLs against known baseline (PostgreSQL + in-memory).
      3. Persists ALL discovered URLs into site_discovered_urls so old pages are never re-flagged as new.
      4. Uses Tavily Crawl on multi-article bursts (>=5 new URLs) or Tavily Extract for targeted updates.
      5. Infers publication dates, cleans Markdown, and ingests into the pipeline.
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
        site_id = site.get("id")
        site_key = str(site_id or root_url)
        select_paths = site.get("select_paths") or ["/news/.*", "/blog/.*"]

        logger.info("[Site Watch: %s] Mapping URL structure via Tavily Map...", site_name)
        map_resp = tavily_map(
            url=root_url,
            select_paths=select_paths,
            exclude_paths=NOISE_PATTERNS,
            instructions="news articles, blog posts and press releases only",
            limit=50,
            purpose="site_watch"
        )
        found_urls = map_resp.get("results", [])
        if not found_urls:
            continue

        # Filter noise and canonicalize
        clean_urls: List[str] = []
        for u in found_urls:
            canonical = canonicalize_url(u)
            if canonical and is_valid_news_url(canonical) and canonical not in clean_urls:
                clean_urls.append(canonical)

        logger.info("[Site Watch: %s] Discovered %d valid candidate URLs.", site_name, len(clean_urls))

        # Diff against database + memory baseline to identify genuinely new URLs
        existing: Set[str] = set(_memory_known_urls.get(site_key, set()))
        try:
            from db import pool
            with pool.connection(timeout=0.3) as conn:
                with conn.cursor() as cur:
                    if site_id:
                        cur.execute(
                            "SELECT url_canonical FROM site_discovered_urls WHERE site_id = %s OR url_canonical = ANY(%s);",
                            (site_id, clean_urls)
                        )
                    else:
                        cur.execute(
                            "SELECT url_canonical FROM site_discovered_urls WHERE url_canonical = ANY(%s);",
                            (clean_urls,)
                        )
                    existing.update(r[0] for r in cur.fetchall() if r[0])
        except Exception:
            pass

        new_urls = [u for u in clean_urls if u not in existing]
        if not new_urls:
            logger.info("[Site Watch: %s] No new unmapped URLs detected.", site_name)
            continue

        total_discovered += len(new_urls)
        selected_to_extract = new_urls[:max_extract_per_site]
        selected_set = set(selected_to_extract)

        # Persist ALL discovered URLs into baseline immediately (N6 fix: prevents remaining 47 URLs
        # from being mistaken as "new" on subsequent cycles)
        _memory_known_urls.setdefault(site_key, set()).update(clean_urls)
        try:
            from db import pool
            with pool.connection(timeout=0.3) as conn:
                with conn.cursor() as cur:
                    if site_id:
                        cur.execute("UPDATE watched_sites SET last_mapped_at = NOW() WHERE id = %s;", (site_id,))
                        for u in new_urls:
                            cur.execute("""
                                INSERT INTO site_discovered_urls (site_id, url_canonical, is_ingested)
                                VALUES (%s, %s, %s)
                                ON CONFLICT (site_id, url_canonical) DO UPDATE
                                SET is_ingested = site_discovered_urls.is_ingested OR EXCLUDED.is_ingested;
                            """, (site_id, u, u in selected_set))
                conn.commit()
        except Exception:
            pass

        # Use Tavily Crawl on burst updates (>=5 new URLs requested) or Tavily Extract for <=4 URLs (N9)
        if len(new_urls) >= 5 and max_extract_per_site >= 5:
            logger.info("[Site Watch: %s] Burst of %d new URLs detected -> harvesting via Tavily Crawl...", site_name, len(new_urls))
            crawl_resp = tavily_crawl(
                url=root_url,
                select_paths=select_paths,
                exclude_paths=NOISE_PATTERNS,
                instructions="recent news announcements, blog posts and press releases",
                max_depth=1,
                limit=max_extract_per_site,
                purpose="site_watch_burst"
            )
            extracted_results = crawl_resp.get("results", [])
        else:
            logger.info("[Site Watch: %s] Extracting %d new articles via Tavily Extract...", site_name, len(selected_to_extract))
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

            inferred_date = _infer_article_date(raw_url, raw_text)
            payload.append({
                "Site": extract_domain(raw_url),
                "Baslik": title,
                "Link": canonicalize_url(raw_url),
                "Yazar": site_name,
                "Tarih": inferred_date,
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
