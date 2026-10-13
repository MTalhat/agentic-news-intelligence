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
from content_cleaner import clean_article_markdown

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tavily_ingestion")

API_INGEST_URL = os.getenv("API_INGEST_URL", "http://localhost:8000/api/internal/ingest")
INGESTION_SECRET = os.getenv("INGESTION_SECRET")

# Fail-fast check on startup
if not INGESTION_SECRET:
    logger.critical("FATAL: INGESTION_SECRET environment variable is missing. Halting ingestion.")
    sys.exit(1)

# Balanced 7-category taxonomy with 3 diverse sub-queries per category
TAXONOMY_QUERIES = {
    "AI": [
        ("autonomous AI agents LLM inference reasoning models", "news"),
        ("frontier foundation models multimodal AI neural networks", "news"),
        ("open source artificial intelligence local LLM fine tuning", "news"),
    ],
    "Hardware": [
        ("semiconductor chip fabrication TSMC 2nm packaging GPUs", "news"),
        ("NVIDIA AMD processor microarchitecture wafer manufacturing", "news"),
        ("silicon photonics high bandwidth memory HBM quantum chips", "news"),
    ],
    "Tech": [
        ("cloud infrastructure cybersecurity software developer platforms", "news"),
        ("open source enterprise software Linux Kubernetes architecture", "news"),
        ("data engineering distributed systems database protocols", "news"),
    ],
    "Automotive": [
        ("electric vehicle solid state battery charging infrastructure EV", "news"),
        ("autonomous driving robotaxi sensor fusion lidar radar self driving", "news"),
        ("automotive software defined vehicle EV manufacturing supply chain", "news"),
    ],
    "Gaming": [
        ("video games console Steam Deck PC gaming engine releases", "news"),
        ("PlayStation Xbox Nintendo handheld hardware game development", "news"),
        ("Unreal Engine graphics ray tracing PC gaming releases", "news"),
    ],
    "Finance": [
        ("macro economy central banks inflation interest rates Federal Reserve", "finance"),
        ("fintech digital banking venture capital corporate earnings markets", "finance"),
        ("cryptocurrency bitcoin ethereum blockchain sovereign wealth funds", "finance"),
    ],
    "Science": [
        ("space exploration NASA James Webb telescope astronomy missions", "news"),
        ("nuclear fusion clean energy tokamak physics renewable breakthroughs", "news"),
        ("biotechnology genomics CRISPR neuroscience biomedical discoveries", "news"),
    ],
}

def harvest_and_ingest(max_per_topic: int = 4, target_total: int = 0) -> tuple[int, int]:
    total_fetched = 0
    total_inserted = 0

    num_categories = len(TAXONOMY_QUERIES)
    target_per_category = max(1, target_total // num_categories) if target_total > 0 else max_per_topic

    logger.info(
        "Harvest strategy: %d categories | Target per category: ~%d articles (Total target: %s)",
        num_categories, target_per_category, f"{target_total}" if target_total > 0 else f"{target_per_category * num_categories}"
    )

    category_stats = {}

    for category, query_tuples in TAXONOMY_QUERIES.items():
        category_fetched = 0
        category_inserted = 0
        
        # Calculate how many queries we need to cover the target
        for query, search_topic in query_tuples:
            needed = target_per_category - category_inserted
            if needed <= 0:
                break
            
            fetch_count = min(10, needed) if target_total > 0 else max_per_topic
            logger.info("[Ingest: %s] Query: '%s' (requesting %d)...", category, query[:40], fetch_count)
            
            results = search_news(
                topic=query, 
                max_results=fetch_count, 
                time_range="week", 
                search_topic=search_topic,
                include_raw_content="markdown",
                timeout=18.0
            )
            category_fetched += len(results)
            total_fetched += len(results)

            payload = []
            for r in results:
                raw_url = r.get("url", "").strip()
                clean_url = canonicalize_url(raw_url)
                title = r.get("title", "").strip()
                text = (r.get("raw_content") or r.get("content") or "").strip()
                text = clean_article_markdown(title, text)

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
                    category_inserted += ins
                    total_inserted += ins
                else:
                    logger.error("  -> Ingest API error (%d): %s", resp.status_code, resp.text)
            except Exception as e:
                logger.error("  -> Ingest connection failed: %s", e)

        category_stats[category] = {"fetched": category_fetched, "stored": category_inserted}
        logger.info("  -> [%s Summary] %d fetched | %d stored.", category, category_fetched, category_inserted)

    logger.info("================ HARVEST BALANCE REPORT ================")
    for cat, stats in category_stats.items():
        logger.info("  * %-12s: %3d stored (fetched %3d)", cat, stats["stored"], stats["fetched"])
    logger.info("========================================================")

    return total_fetched, total_inserted

def main():
    parser = argparse.ArgumentParser(description="Tavily News Ingestion Engine")
    parser.add_argument("--once", action="store_true", default=True, help="Run a single harvest cycle and exit (default)")
    parser.add_argument("--loop", action="store_true", help="Run continuously in background (overrides --once)")
    parser.add_argument("--interval", type=int, default=14400, help="Loop interval in seconds (default: 14400s / 4 hours)")
    parser.add_argument("--max", type=int, default=4, help="Max results per topic query (default: 4)")
    parser.add_argument("--target", type=int, default=0, help="Target total articles to harvest equally distributed across all 7 categories (e.g. --target 200)")
    args = parser.parse_args()

    is_loop = args.loop
    mode_desc = f"TARGET EQUAL HARVEST ({args.target} articles)" if args.target > 0 else ("CONTINUOUS DAEMON" if is_loop else "ONE-SHOT")
    logger.info("Starting Tavily Ingestion (Mode: %s)...", mode_desc)
    
    while True:
        fetched, inserted = harvest_and_ingest(max_per_topic=args.max, target_total=args.target)
        logger.info("Cycle complete: %d fetched, %d newly stored.", fetched, inserted)
        if not is_loop:
            break
        logger.info("Sleeping for %d seconds (%.1f hours) until next cycle...", args.interval, args.interval / 3600)
        time.sleep(args.interval)

if __name__ == "__main__":
    main()
