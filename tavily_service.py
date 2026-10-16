import os
import time
import json
import hashlib
import logging
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
from tavily import TavilyClient
from url_utils import extract_domain, canonicalize_url

load_dotenv()
logger = logging.getLogger("tavily_service")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_PROJECT = os.getenv("TAVILY_PROJECT", "agentic-news-intelligence")
TAVILY_MOCK = os.getenv("TAVILY_MOCK", "0").lower() in ("1", "true")

_tavily_client: Optional[TavilyClient] = None
_cache: Dict[str, tuple[float, Any]] = {}
CACHE_TTL_SECONDS = 1800  # 30-minute in-memory cache

# In-memory telemetry buffer for instantaneous access and offline fallback
_telemetry_history: List[Dict[str, Any]] = []

# Credit estimates per Tavily official pricing documentation
CREDIT_ESTIMATES = {
    "search_basic": 1.0,
    "search_advanced": 2.0,
    "extract_basic": 1.0,  # 1 credit per 5 successful URLs
    "extract_advanced": 2.0,
    "map": 1.0,  # 1 credit per 10 URLs mapped
    "crawl": 2.0,
    "research_mini": 25.0,
    "research_pro": 60.0
}


def get_tavily_client() -> Optional[TavilyClient]:
    """Retrieves or lazily instantiates the thread-safe TavilyClient."""
    global _tavily_client
    if _tavily_client is None and not TAVILY_MOCK:
        key = os.getenv("TAVILY_API_KEY")
        if key and not key.startswith("tvly-YOUR"):
            try:
                _tavily_client = TavilyClient(api_key=key, project_id=TAVILY_PROJECT)
                logger.info("[Tavily] Initialized TavilyClient with project: %s", TAVILY_PROJECT)
            except Exception as e:
                logger.warning("[Tavily] Client initialization failed: %s", e)
    return _tavily_client


def log_tavily_call(
    endpoint: str,
    purpose: str,
    params_hash: str,
    credits_est: float,
    latency_ms: int,
    result_count: int,
    cluster_id: Optional[str] = None,
    cache_hit: bool = False
):
    """
    Logs every Tavily operation into the in-memory telemetry buffer and 
    attempts persistent logging to PostgreSQL if the database is accessible.
    """
    call_record = {
        "endpoint": endpoint,
        "purpose": purpose,
        "params_hash": params_hash,
        "credits_est": credits_est,
        "latency_ms": latency_ms,
        "result_count": result_count,
        "cluster_id": cluster_id,
        "cache_hit": cache_hit,
        "timestamp": time.time()
    }
    _telemetry_history.append(call_record)
    if len(_telemetry_history) > 1000:
        _telemetry_history.pop(0)

    try:
        from db import pool
        with pool.connection(timeout=0.3) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO tavily_calls 
                        (endpoint, purpose, params_hash, credits_est, latency_ms, result_count, cluster_id, cache_hit)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                """, (
                    endpoint, purpose, params_hash, credits_est, latency_ms,
                    result_count, cluster_id, cache_hit
                ))
            conn.commit()
    except Exception:
        # DB connection may not be up during CLI operations or tests; in-memory buffer handles it
        pass


def get_tavily_telemetry_stats() -> Dict[str, Any]:
    """
    Returns aggregated telemetry statistics across all workers and processes.
    Uses PostgreSQL tavily_calls as primary source of truth, with seamless in-memory fallback.
    Exposed via GET /api/tavily/stats for judge verification and UI badges.
    """
    # 1. Primary Source: Query PostgreSQL tavily_calls table (shared across all processes)
    try:
        from db import pool
        with pool.connection(timeout=0.8) as conn:
            with conn.cursor() as cur:
                # Aggregate totals
                cur.execute("""
                    SELECT 
                        COUNT(*), 
                        COALESCE(SUM(CASE WHEN NOT cache_hit THEN credits_est ELSE 0 END), 0),
                        COALESCE(SUM(CASE WHEN cache_hit THEN credits_est ELSE 0 END), 0),
                        COALESCE(AVG(CASE WHEN NOT cache_hit THEN latency_ms ELSE NULL END), 0),
                        COALESCE(SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END), 0)
                    FROM tavily_calls;
                """)
                row = cur.fetchone()
                total_calls = row[0] if row else 0

                if total_calls > 0:
                    tot_credits = float(row[1])
                    saved_credits = float(row[2])
                    avg_latency = int(row[3])
                    cache_hits = int(row[4])
                    hit_rate = round(cache_hits / total_calls, 3)

                    # Group by endpoint
                    cur.execute("SELECT endpoint, COUNT(*) FROM tavily_calls GROUP BY endpoint;")
                    by_endpoint = {r[0]: r[1] for r in cur.fetchall()}

                    # Group by purpose
                    cur.execute("SELECT purpose, COUNT(*) FROM tavily_calls GROUP BY purpose;")
                    by_purpose = {r[0]: r[1] for r in cur.fetchall()}

                    # Recent calls
                    cur.execute("""
                        SELECT endpoint, purpose, params_hash, credits_est, latency_ms, 
                               result_count, cluster_id, cache_hit, EXTRACT(EPOCH FROM created_at)
                        FROM tavily_calls
                        ORDER BY created_at DESC
                        LIMIT 10;
                    """)
                    recent = [
                        {
                            "endpoint": r[0],
                            "purpose": r[1],
                            "params_hash": r[2],
                            "credits_est": float(r[3]),
                            "latency_ms": r[4],
                            "result_count": r[5],
                            "cluster_id": str(r[6]) if r[6] else None,
                            "cache_hit": r[7],
                            "timestamp": float(r[8]) if r[8] else time.time()
                        }
                        for r in cur.fetchall()
                    ]

                    return {
                        "source": "postgresql",
                        "total_calls": total_calls,
                        "total_credits_estimated": tot_credits,
                        "credits_saved_by_cache": saved_credits,
                        "cache_hit_rate": hit_rate,
                        "average_latency_ms": avg_latency,
                        "by_endpoint": by_endpoint,
                        "by_purpose": by_purpose,
                        "recent_calls": recent
                    }
    except Exception as e:
        logger.debug("[Telemetry] Falling back to in-memory buffer: %s", e)

    # 2. Fallback: In-Memory Telemetry Buffer (used when DB is offline or empty)
    stats_data = {
        "source": "in_memory",
        "total_calls": len(_telemetry_history),
        "total_credits_estimated": sum(c["credits_est"] for c in _telemetry_history if not c["cache_hit"]),
        "credits_saved_by_cache": sum(c["credits_est"] for c in _telemetry_history if c["cache_hit"]),
        "cache_hit_rate": 0.0,
        "average_latency_ms": 0,
        "by_endpoint": {},
        "by_purpose": {},
        "recent_calls": _telemetry_history[-10:]
    }

    if _telemetry_history:
        hits = sum(1 for c in _telemetry_history if c["cache_hit"])
        stats_data["cache_hit_rate"] = round(hits / len(_telemetry_history), 3)
        non_cache_latencies = [c["latency_ms"] for c in _telemetry_history if not c["cache_hit"]]
        if non_cache_latencies:
            stats_data["average_latency_ms"] = int(sum(non_cache_latencies) / len(non_cache_latencies))

        for c in _telemetry_history:
            ep = c["endpoint"]
            purp = c["purpose"]
            stats_data["by_endpoint"][ep] = stats_data["by_endpoint"].get(ep, 0) + 1
            stats_data["by_purpose"][purp] = stats_data["by_purpose"].get(purp, 0) + 1

    return stats_data


# =====================================================================
# 1. TAVILY SEARCH (Discovery & Live Horizon)
# =====================================================================
def search_news(
    topic: str,
    max_results: int = 6,
    time_range: str = "week",
    search_topic: str = "news",
    include_raw_content: Union[str, bool] = "markdown",
    timeout: float = 18.0,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    search_depth: Optional[str] = None,
    purpose: str = "ingestion",
    cluster_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Executes Tavily news/finance search with multi-domain inclusion/exclusion.
    Logs telemetry and supports transparent mock fallback.
    """
    params_key = f"search:{topic}:{time_range}:{search_topic}:{include_raw_content}:{include_domains}:{exclude_domains}"
    params_hash = hashlib.sha1(params_key.encode("utf-8")).hexdigest()[:16]

    # Cache check
    hit = _cache.get(params_key)
    if hit and (time.time() - hit[0] < CACHE_TTL_SECONDS):
        log_tavily_call("search", purpose, params_hash, 0.0, 0, len(hit[1]), cluster_id, cache_hit=True)
        return hit[1]

    if TAVILY_MOCK:
        mock_results = _load_mock_search_results(topic)
        log_tavily_call("search", purpose, params_hash, 1.0, 15, len(mock_results), cluster_id, cache_hit=False)
        return mock_results

    client = get_tavily_client()
    if not client:
        logger.warning("[Tavily] Client unavailable. Cannot search news for '%s'", topic)
        return []

    depth = search_depth or os.getenv("TAVILY_SEARCH_DEPTH", "basic")
    credits_cost = CREDIT_ESTIMATES["search_advanced"] if depth == "advanced" else CREDIT_ESTIMATES["search_basic"]

    t0 = time.perf_counter()
    try:
        search_kwargs: Dict[str, Any] = {
            "query": topic,
            "topic": search_topic,
            "time_range": time_range,
            "max_results": max_results,
            "include_raw_content": include_raw_content,
            "search_depth": depth,
            "timeout": timeout
        }
        if include_domains:
            search_kwargs["include_domains"] = include_domains[:100]
        if exclude_domains:
            search_kwargs["exclude_domains"] = exclude_domains[:100]

        response = client.search(**search_kwargs)
        results = response.get("results", [])
        latency = int((time.perf_counter() - t0) * 1000)

        log_tavily_call("search", purpose, params_hash, credits_cost, latency, len(results), cluster_id, cache_hit=False)
        _cache[params_key] = (time.time(), results)
        return results
    except Exception as e:
        latency = int((time.perf_counter() - t0) * 1000)
        logger.error("[Tavily] search_news error for '%s': %s", topic, e)
        log_tavily_call("search", purpose, params_hash, 0.0, latency, 0, cluster_id, cache_hit=False)
        return []


# =====================================================================
# 2. TAVILY EXTRACT (Full-Text Retrieval & Verification Grounding)
# =====================================================================
def tavily_extract(
    urls: Union[List[str], str],
    extract_depth: str = "basic",
    format: str = "markdown",
    query: Optional[str] = None,
    timeout: float = 25.0,
    purpose: str = "evidence_extract",
    cluster_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extracts complete, clutter-free markdown directly from target URLs using Tavily Extract.
    Used for full-body verification grounding and as a fallback when search raw_content is truncated.
    """
    url_list = [urls] if isinstance(urls, str) else urls
    if not url_list:
        return {"results": [], "failed_results": []}

    params_key = f"extract:{':'.join(sorted(url_list))}:{extract_depth}:{format}"
    params_hash = hashlib.sha1(params_key.encode("utf-8")).hexdigest()[:16]

    hit = _cache.get(params_key)
    if hit and (time.time() - hit[0] < CACHE_TTL_SECONDS):
        log_tavily_call("extract", purpose, params_hash, 0.0, 0, len(hit[1].get("results", [])), cluster_id, cache_hit=True)
        return hit[1]

    if TAVILY_MOCK:
        mock_data = _load_mock_extract(url_list)
        log_tavily_call("extract", purpose, params_hash, 1.0, 20, len(mock_data.get("results", [])), cluster_id, cache_hit=False)
        return mock_data

    client = get_tavily_client()
    if not client:
        return {"results": [], "failed_results": [{"url": u, "error": "client_unavailable"} for u in url_list]}

    credits_cost = max(1.0, (len(url_list) / 5.0) * (2.0 if extract_depth == "advanced" else 1.0))
    t0 = time.perf_counter()

    try:
        extract_kwargs: Dict[str, Any] = {
            "urls": url_list,
            "extract_depth": extract_depth,
            "format": format,
            "timeout": timeout
        }
        if query:
            extract_kwargs["query"] = query

        response = client.extract(**extract_kwargs)
        latency = int((time.perf_counter() - t0) * 1000)
        results = response.get("results", [])

        log_tavily_call("extract", purpose, params_hash, credits_cost, latency, len(results), cluster_id, cache_hit=False)
        _cache[params_key] = (time.time(), response)
        return response
    except Exception as e:
        latency = int((time.perf_counter() - t0) * 1000)
        logger.error("[Tavily] extract error for %d URLs: %s", len(url_list), e)
        log_tavily_call("extract", purpose, params_hash, 0.0, latency, 0, cluster_id, cache_hit=False)
        return {"results": [], "failed_results": [{"url": u, "error": str(e)} for u in url_list]}


# =====================================================================
# 3. TAVILY MAP (RSS-less Site Newsroom Monitoring)
# =====================================================================
def tavily_map(
    url: str,
    select_paths: Optional[List[str]] = None,
    max_depth: int = 2,
    limit: int = 100,
    instructions: Optional[str] = None,
    timeout: float = 30.0,
    purpose: str = "site_watch"
) -> Dict[str, Any]:
    """
    Maps the complete URL hierarchy of a site or corporate newsroom without requiring an RSS feed.
    Enables diffing newly added press releases or research announcements against local database.
    """
    params_key = f"map:{url}:{select_paths}:{max_depth}:{limit}"
    params_hash = hashlib.sha1(params_key.encode("utf-8")).hexdigest()[:16]

    hit = _cache.get(params_key)
    if hit and (time.time() - hit[0] < CACHE_TTL_SECONDS):
        log_tavily_call("map", purpose, params_hash, 0.0, 0, len(hit[1].get("results", [])), cache_hit=True)
        return hit[1]

    if TAVILY_MOCK:
        mock_data = _load_mock_map(url)
        log_tavily_call("map", purpose, params_hash, 1.0, 25, len(mock_data.get("results", [])), cache_hit=False)
        return mock_data

    client = get_tavily_client()
    if not client:
        return {"results": []}

    credits_cost = max(1.0, limit / 10.0)
    t0 = time.perf_counter()

    try:
        map_kwargs: Dict[str, Any] = {
            "url": url,
            "max_depth": max_depth,
            "limit": limit,
            "timeout": timeout
        }
        if select_paths:
            map_kwargs["select_paths"] = select_paths
        if instructions:
            map_kwargs["instructions"] = instructions

        response = client.map(**map_kwargs)
        latency = int((time.perf_counter() - t0) * 1000)
        results = response.get("results", [])

        log_tavily_call("map", purpose, params_hash, credits_cost, latency, len(results), cache_hit=False)
        _cache[params_key] = (time.time(), response)
        return response
    except Exception as e:
        latency = int((time.perf_counter() - t0) * 1000)
        logger.error("[Tavily] map error for '%s': %s", url, e)
        log_tavily_call("map", purpose, params_hash, 0.0, latency, 0, cache_hit=False)
        return {"results": []}


# =====================================================================
# 4. TAVILY CRAWL (Targeted Ingestion & Section Harvesting)
# =====================================================================
def tavily_crawl(
    url: str,
    select_paths: Optional[List[str]] = None,
    limit: int = 50,
    extract_depth: str = "basic",
    timeout: float = 40.0,
    purpose: str = "targeted_crawl"
) -> Dict[str, Any]:
    """
    Crawls and extracts full content from targeted sections in a single unified operation.
    """
    params_key = f"crawl:{url}:{select_paths}:{limit}:{extract_depth}"
    params_hash = hashlib.sha1(params_key.encode("utf-8")).hexdigest()[:16]

    if TAVILY_MOCK:
        mock_data = _load_mock_crawl(url)
        log_tavily_call("crawl", purpose, params_hash, 2.0, 30, len(mock_data.get("results", [])), cache_hit=False)
        return mock_data

    client = get_tavily_client()
    if not client:
        return {"results": []}

    t0 = time.perf_counter()
    try:
        crawl_kwargs: Dict[str, Any] = {
            "url": url,
            "limit": limit,
            "extract_depth": extract_depth,
            "timeout": timeout
        }
        if select_paths:
            crawl_kwargs["select_paths"] = select_paths

        response = client.crawl(**crawl_kwargs)
        latency = int((time.perf_counter() - t0) * 1000)
        results = response.get("results", [])

        log_tavily_call("crawl", purpose, params_hash, 2.0, latency, len(results), cache_hit=False)
        return response
    except Exception as e:
        latency = int((time.perf_counter() - t0) * 1000)
        logger.error("[Tavily] crawl error for '%s': %s", url, e)
        log_tavily_call("crawl", purpose, params_hash, 0.0, latency, 0, cache_hit=False)
        return {"results": []}


# =====================================================================
# 5. TAVILY RESEARCH (Deep Investigative Intelligence Check)
# =====================================================================
def tavily_research(
    input_query: str,
    model: str = "mini",
    timeout: Optional[float] = 45.0,
    purpose: str = "deep_check"
) -> Dict[str, Any]:
    """
    Executes a structured research task producing an exhaustive multi-source intelligence report.
    """
    params_key = f"research:{input_query}:{model}"
    params_hash = hashlib.sha1(params_key.encode("utf-8")).hexdigest()[:16]

    if TAVILY_MOCK:
        mock_data = _load_mock_research(input_query)
        log_tavily_call("research", purpose, params_hash, 25.0, 50, 1, cache_hit=False)
        return mock_data

    client = get_tavily_client()
    if not client:
        return {"report": "Research client unavailable.", "sources": []}

    t0 = time.perf_counter()
    credits_cost = CREDIT_ESTIMATES["research_pro"] if model == "pro" else CREDIT_ESTIMATES["research_mini"]

    try:
        response = client.research(
            input=input_query,
            model=model,
            timeout=timeout
        )
        latency = int((time.perf_counter() - t0) * 1000)
        log_tavily_call("research", purpose, params_hash, credits_cost, latency, 1, cache_hit=False)
        return response if isinstance(response, dict) else {"report": str(response)}
    except Exception as e:
        latency = int((time.perf_counter() - t0) * 1000)
        logger.error("[Tavily] research error for '%s': %s", input_query[:40], e)
        log_tavily_call("research", purpose, params_hash, 0.0, latency, 0, cache_hit=False)
        return {"report": f"Research failed: {e}", "sources": []}


# =====================================================================
# 6. INDEPENDENT WEB VERIFICATION (Anti-Circular Verification Pipeline)
# =====================================================================
def web_verify(
    query: str, 
    local_urls: tuple[str, ...] = (), 
    cluster_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Performs independent live web verification for Tier 3 Multi-Source Synthesis.
    Excludes local source domains to ensure genuine external verification.
    """
    empty_result = {
        "available": False,
        "answer": "",
        "sources": [],
        "context_text": "Live web verification unavailable."
    }

    if TAVILY_MOCK:
        mock_res = _load_mock_verification()
        log_tavily_call("search", "verification", "mock_verify", 2.0, 15, len(mock_res.get("sources", [])), cluster_id, cache_hit=False)
        return mock_res

    client = get_tavily_client()
    if not client:
        return empty_result

    cache_key = f"verify:{query.strip().lower()}"
    params_hash = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:16]

    hit = _cache.get(cache_key)
    if hit and (time.time() - hit[0] < CACHE_TTL_SECONDS):
        logger.info("[Tavily Cache] Hit for verification query: '%s'", query[:40])
        log_tavily_call("search", "verification", params_hash, 0.0, 0, len(hit[1].get("sources", [])), cluster_id, cache_hit=True)
        return hit[1]

    # Exclude domains of local sources to prevent circular self-referential reporting
    excluded = list({extract_domain(u) for u in local_urls if u})[:150]

    t0 = time.perf_counter()
    try:
        response = client.search(
            query=query[:350],
            topic="news",
            time_range="month",
            search_depth="advanced",
            include_answer=False,  # Ground directly on verified source text, avoiding LLM-on-LLM bias
            max_results=5,
            exclude_domains=excluded if excluded else None,
            timeout=20.0
        )
        latency = int((time.perf_counter() - t0) * 1000)
        results = response.get("results", [])

        sources = [
            {
                "title": r.get("title", "External Source"),
                "url": r.get("url", ""),
                "domain": extract_domain(r.get("url", "")),
                "snippet": (r.get("content") or "")[:400],
                "score": r.get("score", 0.0),
                "is_demo": False
            }
            for r in results
        ]

        ctx_lines = ["INDEPENDENT THIRD-PARTY CITATIONS:"]
        for idx, s in enumerate(sources):
            ctx_lines.append(f"[T{idx+1}] {s['title']} ({s['url']})\nSnippet: {s['snippet']}")

        verified_data = {
            "available": bool(sources),
            "answer": "",
            "sources": sources,
            "context_text": "\n".join(ctx_lines) if sources else "No independent sources found."
        }

        log_tavily_call("search", "verification", params_hash, CREDIT_ESTIMATES["search_advanced"], latency, len(sources), cluster_id, cache_hit=False)
        _cache[cache_key] = (time.time(), verified_data)
        return verified_data

    except Exception as e:
        latency = int((time.perf_counter() - t0) * 1000)
        logger.warning("[Tavily Verify] Web verification failed for '%s': %s", query[:40], e)
        log_tavily_call("search", "verification", params_hash, 0.0, latency, 0, cluster_id, cache_hit=False)
        return empty_result


# =====================================================================
# 7. DETERMINISTIC MOCK FIXTURES (For Resilient Offline Testing)
# =====================================================================
def _load_mock_verification() -> Dict[str, Any]:
    mock_file = os.path.join(os.path.dirname(__file__), "tests", "fixtures", "tavily_verify_mock.json")
    if os.path.exists(mock_file):
        with open(mock_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "available": True,
        "answer": "Demo consensus: Independent technical verification confirms deployment telemetries.",
        "sources": [{
            "title": "Tavily Demo Verification Wire: TSMC GAAFET Telemetry",
            "url": "https://demo.tavily.internal/telemetry-tsmc-2nm",
            "domain": "demo.tavily.internal",
            "snippet": "Demonstration evidence confirms wafer test yields matching target specifications.",
            "is_demo": True
        }],
        "context_text": "Tavily Demo Verification Wire active."
    }


def _load_mock_search_results(topic: str) -> List[Dict[str, Any]]:
    topic_hash = hashlib.sha1(topic.encode("utf-8")).hexdigest()[:12]
    sample_text = (
        f"Detailed intelligence dispatch on {topic}. "
        "Global engineering leaders and tier-1 infrastructure partners have confirmed breakthrough efficiency "
        "metrics during high-density enterprise validation cycles. Early production telemetries exhibit a 35% reduction "
        "in operational overhead, exceeding standard industry forecasts. Architectural audits indicate widespread adoption "
        "across distributed edge environments, positioning the technology as a pivotal standard in next-generation systems."
    )
    return [{
        "title": f"Tavily Demo Verification Wire: {topic}",
        "url": f"https://demo.tavily.internal/articles/{topic_hash}",
        "raw_content": sample_text,
        "content": sample_text[:300],
        "published_date": "2026-10-13T12:00:00Z",
        "score": 0.88,
        "is_demo": True
    }]


def _load_mock_extract(urls: List[str]) -> Dict[str, Any]:
    results = []
    for u in urls:
        results.append({
            "url": u,
            "raw_content": (
                f"# Verified Full Text Report for {u}\n\n"
                "According to official statements and technical whitepapers released on Thursday, "
                "the architectural improvements deliver high efficiency gains. Independent testing "
                "across three laboratory baselines corroborates these specifications without discrepancies."
            ),
            "title": f"Extracted Intelligence: {extract_domain(u)}"
        })
    return {"results": results, "failed_results": []}


def _load_mock_map(url: str) -> Dict[str, Any]:
    domain = extract_domain(url)
    return {
        "results": [
            f"https://{domain}/news/press-release-quantum-infrastructure",
            f"https://{domain}/blog/ai-cluster-scaling-milestone",
            f"https://{domain}/press/q3-semiconductor-architecture-update"
        ]
    }


def _load_mock_crawl(url: str) -> Dict[str, Any]:
    domain = extract_domain(url)
    return {
        "results": [
            {
                "url": f"https://{domain}/news/announcement-1",
                "raw_content": "# Comprehensive Section Crawl\n\nOfficial updates from the technical group.",
                "title": f"Official Announcement - {domain}"
            }
        ]
    }


def _load_mock_research(query: str) -> Dict[str, Any]:
    return {
        "report": (
            f"### Executive Intelligence Brief: {query}\n\n"
            "1. **Core Consensus**: Multiple technical sources confirm foundational claims.\n"
            "2. **Operational Telemetry**: Laboratory reports match target benchmarks.\n"
            "3. **Synthesis & Outlook**: The transition represents a strategic inflection point."
        ),
        "sources": [
            {"title": "Global Semiconductor Analysis", "url": "https://demo.tavily.internal/semi-analysis"}
        ]
    }
