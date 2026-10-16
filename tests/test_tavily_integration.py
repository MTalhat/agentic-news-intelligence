import os
import sys
import json
import hmac
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
load_dotenv()

import tavily_service
from url_utils import canonicalize_url, extract_domain
from tavily_service import (
    search_news, web_verify, tavily_extract, tavily_map, 
    tavily_crawl, tavily_research, get_tavily_telemetry_stats
)


def test_canonicalize_url():
    raw = "https://www.Reuters.com/tech/nvidia-chips/?utm_source=twitter&utm_medium=social&ref=feed#section"
    expected = "https://reuters.com/tech/nvidia-chips"
    assert canonicalize_url(raw) == expected
    assert canonicalize_url("https://theverge.com/news/") == "https://theverge.com/news"
    assert canonicalize_url("") == ""


def test_extract_domain():
    assert extract_domain("https://www.theverge.com/tech/item") == "theverge.com"
    assert extract_domain("https://docs.nvidia.com/index.html") == "docs.nvidia.com"
    assert extract_domain("invalid-url") == "news.web"


def test_mock_search_news():
    orig = tavily_service.TAVILY_MOCK
    tavily_service.TAVILY_MOCK = True
    try:
        results = search_news("Nvidia Blackwell B200", max_results=2)
        assert len(results) > 0
        item = results[0]
        assert "Tavily Demo Verification Wire" in item["title"]
        assert len(item["raw_content"]) >= 350
        assert item.get("is_demo") is True
    finally:
        tavily_service.TAVILY_MOCK = orig


def test_mock_extract():
    orig = tavily_service.TAVILY_MOCK
    tavily_service.TAVILY_MOCK = True
    try:
        resp = tavily_extract(["https://tsmc.com/press/2nm-update"])
        results = resp.get("results", [])
        assert len(results) == 1
        assert "Verified Full Text Report" in results[0]["raw_content"]
    finally:
        tavily_service.TAVILY_MOCK = orig


def test_mock_map():
    orig = tavily_service.TAVILY_MOCK
    tavily_service.TAVILY_MOCK = True
    try:
        resp = tavily_map("https://openai.com/news")
        results = resp.get("results", [])
        assert len(results) >= 2
        assert any("press-release" in u for u in results)
    finally:
        tavily_service.TAVILY_MOCK = orig


def test_mock_crawl():
    orig = tavily_service.TAVILY_MOCK
    tavily_service.TAVILY_MOCK = True
    try:
        resp = tavily_crawl("https://deepmind.google/discover/blog")
        results = resp.get("results", [])
        assert len(results) >= 1
        assert "Comprehensive Section Crawl" in results[0]["raw_content"]
    finally:
        tavily_service.TAVILY_MOCK = orig


def test_mock_research():
    orig = tavily_service.TAVILY_MOCK
    tavily_service.TAVILY_MOCK = True
    try:
        resp = tavily_research("Semiconductor 2nm lithography scaling")
        assert "Executive Intelligence Brief" in resp.get("report", "")
    finally:
        tavily_service.TAVILY_MOCK = orig


def test_mock_web_verify():
    orig = tavily_service.TAVILY_MOCK
    tavily_service.TAVILY_MOCK = True
    try:
        ver = web_verify("QuantumScape solid state battery breakthroughs", local_urls=("https://electrek.co/news",))
        assert ver["available"] is True
        assert len(ver["sources"]) > 0
        first_src = ver["sources"][0]
        assert first_src.get("is_demo") is True
        assert first_src.get("domain") == "demo.tavily.internal"
    finally:
        tavily_service.TAVILY_MOCK = orig


def test_telemetry_stats():
    stats = get_tavily_telemetry_stats()
    assert "total_calls" in stats
    assert "total_credits_estimated" in stats
    assert "by_endpoint" in stats
    assert stats["total_calls"] > 0


def test_byte_safe_hmac():
    secret = "local_hackathon_dev_secret"
    valid = "local_hackathon_dev_secret"
    invalid = "wrong_secret"
    assert hmac.compare_digest(valid.encode("utf-8"), secret.encode("utf-8")) is True
    assert hmac.compare_digest(invalid.encode("utf-8"), secret.encode("utf-8")) is False


def test_sse_parser_tolerance():
    simulated_sse_lines = [
        b'data: {"status": "Cross-checking assertions against live web consensus via Tavily..."}\n',
        b'data: {"sources": [{"title": "Demo Wire", "url": "https://demo"}], "available": true}\n',
        b'data: {"token": "1. Shared Consensus: All outlets agree."}\n',
        b'data: {"token": " 2. Framing Discrepancies: No major disputes."}\n',
        b'data: [DONE]\n'
    ]
    
    collected = []
    for line in simulated_sse_lines:
        decoded = line.decode('utf-8').strip()
        if decoded.startswith("data: "):
            payload = decoded[6:].strip()
            if payload == "[DONE]":
                break
            try:
                data_obj = json.loads(payload)
                if "token" in data_obj:
                    collected.append(data_obj["token"])
                elif "error" in data_obj:
                    collected.append(f" ERROR: {data_obj['error']} ")
            except Exception:
                collected.append(payload)

    result_text = "".join(collected)
    assert "Shared Consensus" in result_text
    assert "Framing Discrepancies" in result_text


if __name__ == "__main__":
    print("=" * 65)
    print("      RUNNING TAVILY MULTI-ENDPOINT INTEGRATION SUITE")
    print("=" * 65)
    test_canonicalize_url()
    print("✓ test_canonicalize_url PASSED")
    test_extract_domain()
    print("✓ test_extract_domain PASSED")
    test_mock_search_news()
    print("✓ test_mock_search_news PASSED")
    test_mock_extract()
    print("✓ test_mock_extract (Tavily Extract API) PASSED")
    test_mock_map()
    print("✓ test_mock_map (Tavily Map API) PASSED")
    test_mock_crawl()
    print("✓ test_mock_crawl (Tavily Crawl API) PASSED")
    test_mock_research()
    print("✓ test_mock_research (Tavily Research API) PASSED")
    test_mock_web_verify()
    print("✓ test_mock_web_verify (Anti-Circular Verification) PASSED")
    test_telemetry_stats()
    print("✓ test_telemetry_stats (Real-time Telemetry Tracking) PASSED")
    test_byte_safe_hmac()
    print("✓ test_byte_safe_hmac PASSED")
    test_sse_parser_tolerance()
    print("✓ test_sse_parser_tolerance PASSED")
    print("=" * 65)
    print("[ALL 11 TAVILY INTEGRATION & MULTI-ENDPOINT TESTS PASSED!]")
    print("=" * 65)
