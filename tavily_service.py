import os
import time
import json
import hashlib
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from tavily import TavilyClient
from url_utils import extract_domain

load_dotenv()
logger = logging.getLogger("tavily_service")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_PROJECT = os.getenv("TAVILY_PROJECT", "agentic-news-intelligence")
TAVILY_MOCK = os.getenv("TAVILY_MOCK", "0").lower() in ("1", "true")

_tavily_client: Optional[TavilyClient] = None
_cache: Dict[str, tuple[float, Dict[str, Any]]] = {}
CACHE_TTL_SECONDS = 1800  # 30-minute in-memory cache

def get_tavily_client() -> Optional[TavilyClient]:
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

def search_news(
    topic: str, 
    max_results: int = 6, 
    time_range: str = "week", 
    search_topic: str = "news",
    include_raw_content: str | bool = "markdown",
    timeout: float = 18.0
) -> List[Dict[str, Any]]:
    """Fetches news articles. Configurable include_raw_content and timeout for fast search snippets."""
    if TAVILY_MOCK:
        return _load_mock_search_results(topic)
    
    client = get_tavily_client()
    if not client:
        logger.warning("[Tavily] Client unavailable. Cannot search news for '%s'", topic)
        return []

    try:
        response = client.search(
            query=topic,
            topic=search_topic,
            time_range=time_range,
            max_results=max_results,
            include_raw_content=include_raw_content,
            search_depth=os.getenv("TAVILY_SEARCH_DEPTH", "basic"),
            timeout=timeout
        )
        return response.get("results", [])
    except Exception as e:
        logger.error("[Tavily] search_news error for '%s': %s", topic, e)
        return []

def web_verify(query: str, local_urls: tuple[str, ...] = ()) -> Dict[str, Any]:
    """
    Performs independent live web verification for Tier 3 Multi-Source Synthesis.
    Excludes local source domains to ensure independent external verification.
    """
    empty_result = {
        "available": False,
        "answer": "",
        "sources": [],
        "context_text": "Live web verification unavailable."
    }

    if TAVILY_MOCK:
        return _load_mock_verification()

    client = get_tavily_client()
    if not client:
        return empty_result

    cache_key = query.strip().lower()
    hit = _cache.get(cache_key)
    if hit and (time.time() - hit[0] < CACHE_TTL_SECONDS):
        logger.info("[Tavily Cache] Hit for query: '%s'", query[:40])
        return hit[1]

    # Exclude domains of local sources to prevent self-referential reporting
    excluded = list({extract_domain(u) for u in local_urls if u})[:150]

    try:
        response = client.search(
            query=query[:350],
            topic="news",
            time_range="month",
            search_depth="advanced",
            include_answer="advanced",
            max_results=5,
            exclude_domains=excluded if excluded else None,
            timeout=20.0
        )
        
        answer = response.get("answer", "")
        results = response.get("results", [])

        sources = [
            {
                "title": r.get("title", "External Source"),
                "url": r.get("url", ""),
                "domain": extract_domain(r.get("url", "")),
                "snippet": (r.get("content") or "")[:400],
                "is_demo": False
            }
            for r in results
        ]

        ctx_lines = []
        if answer:
            ctx_lines.append(f"EXTERNAL AI CONSENSUS OVERVIEW:\n{answer}\n")
        ctx_lines.append("INDEPENDENT THIRD-PARTY CITATIONS:")
        for idx, s in enumerate(sources):
            ctx_lines.append(f"[T{idx+1}] {s['title']} ({s['url']})\nSnippet: {s['snippet']}")

        verified_data = {
            "available": bool(answer or sources),
            "answer": answer,
            "sources": sources,
            "context_text": "\n".join(ctx_lines) if (answer or sources) else "No independent sources found."
        }
        _cache[cache_key] = (time.time(), verified_data)
        return verified_data

    except Exception as e:
        logger.warning("[Tavily Verify] Web verification failed for '%s': %s", query[:40], e)
        return empty_result

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
        "is_demo": True
    }]
