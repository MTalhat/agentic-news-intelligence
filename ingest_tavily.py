import os
import sys
import time
import json
import re
import email.utils
import hashlib
import argparse
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Optional
import requests
from dotenv import load_dotenv

from tavily_service import search_news, tavily_extract
from url_utils import canonicalize_url, extract_domain
from content_cleaner import clean_article_markdown, extract_normalized_shingles

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tavily_ingestion")

API_INGEST_URL = os.getenv("API_INGEST_URL", "http://localhost:8000/api/internal/ingest")
INGESTION_SECRET = os.getenv("INGESTION_SECRET")

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ingest_state.json")

# Balanced 7-category taxonomy with 3 diverse sub-queries per category
TAXONOMY_QUERIES = {
    "AI": [
        ("autonomous AI agents LLM inference reasoning models", "news"),
        ("open source AI models weights benchmarks Ollama vLLM", "news"),
        ("AI chip clusters data centers compute scaling", "news"),
    ],
    "Hardware": [
        ("semiconductor chip fabrication TSMC 2nm packaging GPUs", "news"),
        ("consumer CPUs GPUs gaming laptops handheld consoles", "news"),
        ("solid state storage DDR5 memory architectures", "news"),
    ],
    "Tech": [
        ("cloud infrastructure cybersecurity software developer platforms", "news"),
        ("open source software frameworks programming languages", "news"),
        ("enterprise SaaS cloud native databases devops", "news"),
    ],
    "Automotive": [
        ("electric vehicle solid state battery autonomous driving robotaxi", "news"),
        ("EV charging infrastructure battery manufacturing range", "news"),
        ("software defined vehicles automotive silicon compute", "news"),
    ],
    "Gaming": [
        ("video games console Steam Deck PC gaming engine releases", "news"),
        ("PlayStation Xbox Nintendo handheld hardware game development", "news"),
        ("Unreal Engine 5 graphics ray tracing PC gaming hardware", "news"),
    ],
    "Finance": [
        ("macro economy central banks fintech interest rates cryptocurrency", "finance"),
        ("venture capital tech startup investments IPO market", "finance"),
        ("digital banking payment infrastructure crypto regulation", "finance"),
    ],
    "Science": [
        ("space exploration NASA fusion energy quantum computing materials", "news"),
        ("biotechnology genomics CRISPR medical hardware discoveries", "news"),
        ("renewable energy grid storage nuclear fission reactors", "news"),
    ],
}


def _load_ingest_state() -> dict:
    """Loads rotation state from .ingest_state.json with safe defaults."""
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return {
                        "cycle_index": int(data.get("cycle_index", 0)),
                        "last_successful_run_utc": data.get("last_successful_run_utc")
                    }
    except Exception as e:
        logger.warning("[State] Could not read %s, using defaults: %s", STATE_FILE, e)
    return {"cycle_index": 0, "last_successful_run_utc": None}


def _save_ingest_state(state: dict):
    """Atomically writes rotation state to .ingest_state.json."""
    try:
        temp_file = f"{STATE_FILE}.tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(temp_file, STATE_FILE)
    except Exception as e:
        logger.error("[State] Failed to persist state to %s: %s", STATE_FILE, e)


def _parse_and_validate_date(raw_date: Any, max_age_days: int = 14) -> Optional[str]:
    """
    Parses ISO-8601 or RFC-2822 timestamps, enforces UTC normalization,
    and guards against future or expired timestamps.
    Falls back gracefully to current UTC time for missing publisher metadata.
    """
    now_utc = datetime.now(timezone.utc)
    if raw_date:
        dt = None
        if isinstance(raw_date, datetime):
            dt = raw_date
        elif isinstance(raw_date, str) and raw_date.strip():
            clean_str = raw_date.strip().replace("Z", "+00:00")
            try:
                dt = datetime.fromisoformat(clean_str)
            except Exception:
                try:
                    dt = email.utils.parsedate_to_datetime(raw_date)
                except Exception:
                    pass

        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)

            # Reject future timestamps (> now + 2 hours tolerance for clock drift)
            if dt > now_utc + timedelta(hours=2):
                logger.warning("[Date Gate] Dropped future date: %s", raw_date)
                return None

            # Reject stale dates older than max_age_days
            if dt < now_utc - timedelta(days=max_age_days):
                logger.warning("[Date Gate] Dropped stale date (>%d days old): %s", max_age_days, raw_date)
                return None

            return dt.isoformat()

    # Safe Hackathon Fallback: default to current time so valid articles aren't lost
    return now_utc.isoformat()


def _truncate_markdown_safely(text: str, max_chars: int = 25000) -> str:
    """
    Slices text at max_chars, breaking on a paragraph or sentence boundary,
    and balances open code fences (```) to prevent formatting corruption.
    """
    if len(text) <= max_chars:
        return text

    sliced = text[:max_chars]
    # Break on paragraph boundary if near the cutoff
    last_para = sliced.rfind('\n\n')
    if last_para >= max_chars - 1200:
        sliced = sliced[:last_para].strip()
    else:
        last_sent = sliced.rfind('. ')
        if last_sent >= max_chars - 600:
            sliced = sliced[:last_sent + 1].strip()

    # Code block balancing: if odd number of ``` fences, close the block
    code_fences = len(re.findall(r'^```', sliced, flags=re.MULTILINE))
    if code_fences % 2 == 1:
        sliced += "\n```\n"

    return sliced


def harvest_and_ingest(
    max_per_topic: int = 4, 
    target_total: int = 0, 
    all_queries: bool = False, 
    time_range: str = "day"
) -> tuple[int, int]:
    """
    Autonomous multi-category news harvesting pipeline.
    Preserves exact signature: max_per_topic is 1st argument, target_total is 2nd argument.
    Returns (total_fetched, total_inserted).
    """
    secret = os.getenv("INGESTION_SECRET")
    if not secret:
        logger.critical("FATAL: INGESTION_SECRET environment variable is unset. Aborting harvest.")
        return 0, 0

    state = _load_ingest_state()
    cycle_index = state.get("cycle_index", 0)

    total_fetched = 0
    total_inserted = 0

    # Multi-layer in-memory deduplication tracking for this cycle
    seen_urls: set[str] = set()
    seen_content_hashes: set[str] = set()
    seen_shingle_sets: list[set[tuple[str, ...]]] = []

    category_stats = {}
    abort_all = False

    logger.info(
        "Harvest cycle starting | Mode: %s | cycle_index: %d | time_range: %s | target_total: %s",
        "ALL-QUERIES" if all_queries else "ROTATING-SINGLE",
        cycle_index,
        time_range,
        target_total or "N/A"
    )

    for category, query_tuples in TAXONOMY_QUERIES.items():
        if abort_all:
            break

        category_fetched = 0
        category_inserted = 0

        # Determine which queries to run for this category
        if all_queries:
            selected_queries = query_tuples
        else:
            selected_query = query_tuples[cycle_index % len(query_tuples)]
            selected_queries = [selected_query]

        for query, search_topic in selected_queries:
            logger.info("[Ingest: %s] Query: '%s' (topic=%s, requesting %d)...", category, query[:45], search_topic, max_per_topic)

            try:
                results = search_news(
                    topic=query,
                    max_results=max_per_topic,
                    time_range=time_range,
                    search_topic=search_topic,
                    include_raw_content="markdown",
                    timeout=18.0
                )
            except Exception as e:
                logger.error("[Search Error] Tavily search failed for '%s': %s", query[:30], e)
                continue

            category_fetched += len(results)
            total_fetched += len(results)

            payload = []
            for r in results:
                raw_url = r.get("url", "").strip()
                clean_url = canonicalize_url(raw_url)
                title = r.get("title", "").strip()
                raw_text = (r.get("raw_content") or r.get("content") or "").strip()

                if not clean_url or not title:
                    continue

                # Layer 1: In-Memory URL Deduplication
                if clean_url in seen_urls:
                    continue
                seen_urls.add(clean_url)

                # Clean content through hardened Markdown pipeline
                text = clean_article_markdown(title, raw_text)
                if not text or len(text) < 250:
                    # Tavily Extract Fallback for paywalls or JS-heavy sites
                    extract_res = tavily_extract(clean_url, purpose="ingest_extract_fallback")
                    for ep in extract_res.get("results", []):
                        if ep.get("raw_content"):
                            text = clean_article_markdown(title, ep["raw_content"])
                            break
                    if not text or len(text) < 250:
                        continue

                # Layer 2: Content Hash Deduplication (SHA-256)
                content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if content_hash in seen_content_hashes:
                    continue
                seen_content_hashes.add(content_hash)

                # Layer 3: Normalized Shingle Jaccard Similarity (Syndicated wire copies)
                shingles = extract_normalized_shingles(text, k=5)
                if len(shingles) >= 5:
                    is_wire_dup = False
                    for prev_shingles in seen_shingle_sets:
                        intersection = len(shingles & prev_shingles)
                        union = len(shingles | prev_shingles)
                        if union > 0 and (intersection / union) >= 0.65:
                            logger.info("[Dedup Gate] Dropped syndicated wire copy (Jaccard: %.2f) for %s", intersection / union, clean_url)
                            is_wire_dup = True
                            break
                    if is_wire_dup:
                        continue
                    seen_shingle_sets.append(shingles)

                # Boundary-safe markdown truncation and code block balancing
                text = _truncate_markdown_safely(text, max_chars=25000)

                # Date parsing and validation
                published_iso = _parse_and_validate_date(r.get("published_date"))
                if not published_iso:
                    continue

                # Strict Ingest API contract payload
                payload.append({
                    "Site": extract_domain(clean_url),
                    "Baslik": title,
                    "Link": clean_url,
                    "Yazar": r.get("author") or extract_domain(clean_url),
                    "Tarih": published_iso,
                    "TamMetin": text
                })

            if not payload:
                continue

            # Robust POST to API Ingest with retries and fail-fast abort on 401/403
            batch_inserted = 0
            for attempt in range(3):
                try:
                    resp = requests.post(
                        API_INGEST_URL,
                        json=payload,
                        headers={"x-secret": secret},
                        timeout=25.0
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        batch_inserted = data.get("inserted", 0)
                        category_inserted += batch_inserted
                        total_inserted += batch_inserted
                        break
                    elif resp.status_code in (401, 403):
                        logger.critical("FATAL: Ingest authentication failed (HTTP %d). Halting ingestion cycle immediately.", resp.status_code)
                        abort_all = True
                        break
                    else:
                        logger.warning("Ingest API error HTTP %d (attempt %d/3): %s", resp.status_code, attempt + 1, resp.text)
                        if attempt < 2:
                            time.sleep(1.0 if attempt == 0 else 3.0)
                except Exception as e:
                    logger.warning("Ingest connection failed (attempt %d/3): %s", attempt + 1, e)
                    if attempt < 2:
                        time.sleep(1.0 if attempt == 0 else 3.0)

            if abort_all:
                break

        category_stats[category] = {"fetched": category_fetched, "stored": category_inserted}
        logger.info("  -> [%s Summary] %d fetched | %d stored.", category, category_fetched, category_inserted)

    # State update and rotation
    max_queries = max(len(q) for q in TAXONOMY_QUERIES.values())
    new_state = {
        "cycle_index": (cycle_index + 1) % max_queries,
        "last_successful_run_utc": datetime.now(timezone.utc).isoformat()
    }
    _save_ingest_state(new_state)

    logger.info("================ HARVEST BALANCE REPORT ================")
    for cat, stats in category_stats.items():
        logger.info("  * %-12s: %3d stored (fetched %3d)", cat, stats["stored"], stats["fetched"])
    logger.info("========================================================")

    return total_fetched, total_inserted


def main():
    parser = argparse.ArgumentParser(description="Tavily News Ingestion Engine")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--once", action="store_true", default=True, help="Run a single harvest cycle and exit (default)")
    group.add_argument("--loop", action="store_true", help="Run continuously in background (overrides --once)")
    parser.add_argument("--all-queries", action="store_true", help="Execute all 3 sub-queries per category sequentially")
    parser.add_argument("--interval", type=int, default=43200, help="Loop interval in seconds (default: 43200s / 12 hours)")
    parser.add_argument("--max", type=int, default=4, help="Max results per topic query (default: 4)")
    parser.add_argument("--target", type=int, default=0, help="Target total articles to harvest (legacy compatibility)")
    args = parser.parse_args()

    # Fail-fast check on startup CLI invocation
    if not os.getenv("INGESTION_SECRET"):
        logger.critical("FATAL: INGESTION_SECRET environment variable is missing. Halting ingestion.")
        sys.exit(1)

    is_loop = args.loop
    mode_desc = f"ALL-QUERIES ({'DAEMON' if is_loop else 'ONE-SHOT'})" if args.all_queries else ("CONTINUOUS DAEMON" if is_loop else "ONE-SHOT ROTATION")
    logger.info("Starting Tavily Ingestion (Mode: %s)...", mode_desc)

    while True:
        fetched, inserted = harvest_and_ingest(
            max_per_topic=args.max,
            target_total=args.target,
            all_queries=args.all_queries,
            time_range="day"
        )
        logger.info("Cycle complete: %d fetched, %d newly stored.", fetched, inserted)
        if not is_loop:
            break
        logger.info("Sleeping for %d seconds (%.1f hours) until next cycle...", args.interval, args.interval / 3600)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
