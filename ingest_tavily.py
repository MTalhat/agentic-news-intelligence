import os
import sys
import time
import argparse
import logging
from datetime import datetime, timezone
import requests
from dotenv import load_dotenv

from tavily_service import search_news
from url_utils import canonicalize_url, extract_domain

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tavily_ingestion")

API_INGEST_URL = os.getenv("API_INGEST_URL", "http://localhost:8000/api/internal/ingest")
INGESTION_SECRET = os.getenv("INGESTION_SECRET")

# Fail-fast check on startup
if not INGESTION_SECRET:
    logger.critical("FATAL: INGESTION_SECRET environment variable is missing. Halting ingestion.")
    sys.exit(1)

# Complete taxonomy coverage without external dependency
TAXONOMY_QUERIES = {
    "AI": ("autonomous AI agents LLM inference reasoning models", "news"),
    "Hardware": ("semiconductor chip fabrication TSMC 2nm packaging GPUs", "news"),
    "Tech": ("cloud infrastructure cybersecurity software developer platforms", "news"),
    "Automotive": ("electric vehicle solid state battery autonomous driving robotaxi", "news"),
    "Gaming": ("video games console Steam Deck PC gaming engine releases", "news"),
    "Finance": ("macro economy central banks fintech interest rates cryptocurrency", "finance"),
    "Science": ("space exploration NASA fusion energy quantum computing materials", "news"),
}

def harvest_and_ingest(max_per_topic: int = 4) -> tuple[int, int]:
    total_fetched = 0
    total_inserted = 0

    for category, (query, search_topic) in TAXONOMY_QUERIES.items():
        logger.info("[Ingest] Harvesting '%s' (Query: '%s')...", category, query[:40])
        results = search_news(
            topic=query, 
            max_results=max_per_topic, 
            time_range="week", 
            search_topic=search_topic,
            include_raw_content="markdown",
            timeout=18.0
        )
        total_fetched += len(results)

        payload = []
        for r in results:
            raw_url = r.get("url", "").strip()
            clean_url = canonicalize_url(raw_url)
            title = r.get("title", "").strip()
            text = (r.get("raw_content") or r.get("content") or "").strip()

            if len(text) < 250 or not clean_url or not title:
                continue

            if len(text) > 25000:
                text = text[:25000]

            payload.append({
                "Site": extract_domain(clean_url),
                "Baslik": title,
                "Link": clean_url,
                "Yazar": r.get("author") or "Intelligence Desk",
                "Tarih": r.get("published_date") or datetime.now(timezone.utc).isoformat(),
                "TamMetin": text
            })

        if not payload:
            continue

        try:
            resp = requests.post(
                API_INGEST_URL,
                json=payload,
                headers={"x-secret": INGESTION_SECRET},
                timeout=25.0
            )
            if resp.status_code == 200:
                data = resp.json()
                ins = data.get("inserted", 0)
                total_inserted += ins
                logger.info("  -> [%s] %d articles fetched | %d NEW stored.", category, len(payload), ins)
            else:
                logger.error("  -> Ingest API error (%d): %s", resp.status_code, resp.text)
        except Exception as e:
            logger.error("  -> Ingest connection failed: %s", e)

    return total_fetched, total_inserted

def main():
    parser = argparse.ArgumentParser(description="Tavily News Ingestion Engine")
    parser.add_argument("--once", action="store_true", default=True, help="Run a single harvest cycle and exit (default)")
    parser.add_argument("--loop", action="store_true", help="Run continuously in background (overrides --once)")
    parser.add_argument("--interval", type=int, default=43200, help="Loop interval in seconds (default: 43200s / 12 hours)")
    parser.add_argument("--max", type=int, default=4, help="Max results per topic (default: 4)")
    args = parser.parse_args()

    is_loop = args.loop
    logger.info("Starting Tavily Ingestion (Mode: %s)...", "CONTINUOUS DAEMON" if is_loop else "ONE-SHOT")
    
    while True:
        fetched, inserted = harvest_and_ingest(max_per_topic=args.max)
        logger.info("Cycle complete: %d fetched, %d newly stored.", fetched, inserted)
        if not is_loop:
            break
        logger.info("Sleeping for %d seconds (%.1f hours) until next cycle...", args.interval, args.interval / 3600)
        time.sleep(args.interval)

if __name__ == "__main__":
    main()
