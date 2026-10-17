import os
import sys
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
load_dotenv()

from datetime import datetime, timezone
from evidence_verifier import (
    span_is_grounded, assign_origins, verify_cluster_evidence,
    _classify_sentence_stance, _is_fresh_cluster
)
import tavily_service


def test_span_grounding_valid():
    source = (
        "Taiwan Semiconductor Manufacturing Company (TSMC) today reaffirmed during its quarterly "
        "earnings conference call that high-volume manufacturing of its 2-nanometer process node "
        "remains on track for late 2025 across facilities in Hsinchu and Kaohsiung."
    )
    quote = "high-volume manufacturing of its 2-nanometer process node remains on track"
    assert span_is_grounded(quote, source) is True


def test_span_grounding_hallucination_rejection():
    source = (
        "TSMC reported strong Q3 financial results with revenue growing 39% year over year. "
        "Smartphone chip demand remained steady while AI accelerator allocations increased."
    )
    # Generative AI hallucination not in source
    fabricated_quote = "TSMC has postponed all 2nm production indefinitely due to extreme wafer yield defects"
    assert span_is_grounded(fabricated_quote, source) is False
    assert span_is_grounded("short", source) is False


def test_wire_origin_syndication_detection():
    # 3 articles covering the same news, but two are syndicated reprints of Reuters
    reuters_wire = {
        "title": "ECB cuts interest rates by 25 basis points as inflation cools",
        "url": "https://reuters.com/markets/europe/ecb-cuts-rates-2026-10-15/",
        "text": "(Reuters) - The European Central Bank lowered its key deposit rate by 25 basis points on Thursday, marking its third cut this year as euro zone inflation decelerated faster than previously forecast.",
        "domain": "reuters.com"
    }
    yahoo_reprint = {
        "title": "ECB cuts interest rates by 25 basis points as inflation cools",
        "url": "https://finance.yahoo.com/news/ecb-rates-cut-october.html",
        "text": "(Reuters) - The European Central Bank lowered its key deposit rate by 25 basis points on Thursday, marking its third cut this year as euro zone inflation decelerated faster than previously forecast.",
        "domain": "finance.yahoo.com"
    }
    politico_original = {
        "title": "Lagarde steers ECB toward easier monetary stance amid growth concerns",
        "url": "https://politico.eu/article/lagarde-ecb-rate-cut-growth-concerns/",
        "text": "Frankfurt officials confirmed Thursday's policy pivot, but Governing Council members privately expressed diverging views regarding the pace of subsequent monetary easing into early 2027.",
        "domain": "politico.eu"
    }

    docs = [reuters_wire, yahoo_reprint, politico_original]
    result = assign_origins(docs)

    # 3 total articles, but only 2 independent origins (Reuters wire + Politico original)
    assert result["independent_origins_count"] == 2
    origins = [o["origin"] for o in result["origins"]]
    assert "Reuters" in origins
    assert "politico.eu" in origins
    print("  -> Origin detection: 3 articles correctly reduced to 2 independent origins (detected Reuters wire syndication).")


def test_evidence_verifier_mock_pipeline():
    orig = tavily_service.TAVILY_MOCK
    tavily_service.TAVILY_MOCK = True
    try:
        sample_articles = [
            {
                "id": "11111111-1111-1111-1111-111111111111",
                "title": "TSMC 2nm GAAFET Wafer Yield Telemetry Confirmed",
                "tldr": "TSMC engineers confirmed wafer yields matching target metrics for 2nm volume fabrication.",
                "text": "Official statements from TSMC leadership confirm 2nm production readiness.",
                "url": "https://tech-source.com/tsmc-2nm"
            }
        ]
        card = verify_cluster_evidence(sample_articles, cluster_id=None)
        assert "summary_verdict" in card
        assert "independent_origins_count" in card
        assert "claims" in card
        assert len(card["claims"]) > 0
    finally:
        tavily_service.TAVILY_MOCK = orig


def test_negation_aware_stance_classification():
    claim = "TSMC 2nm volume production begins on schedule"
    # Double negation / benign negation should be classified as 'support', not 'contradict'
    s_benign = "Independent testing across three laboratory baselines corroborates TSMC 2nm specifications without discrepancies"
    assert _classify_sentence_stance(s_benign, claim, "TSMC") == "support"

    s_rumor = "TSMC engineers confirmed that rumors of 2nm delay were false"
    assert _classify_sentence_stance(s_rumor, claim, "TSMC") == "support"

    # Direct refutation should be 'contradict'
    s_contra = "TSMC officially denied reports that 2nm volume production begins this quarter"
    assert _classify_sentence_stance(s_contra, claim, "TSMC") == "contradict"


def test_fresh_story_rule():
    fresh_articles = [{
        "id": "22222222-2222-2222-2222-222222222222",
        "title": "Breaking: Startup unveils experimental photonic chip prototype",
        "tldr": "A stealth startup unveiled an experimental photonic chip prototype minutes ago.",
        "text": "The company announced the prototype in a brief blog post today.",
        "url": "https://stealth-photonics.example.com/launch",
        "published_at": datetime.now(timezone.utc).isoformat()
    }]
    assert _is_fresh_cluster(fresh_articles, max_hours=6.0) is True


def test_wire_origin_incidental_mention_no_false_positive():
    from evidence_verifier import detect_wire_origin
    # Incidental body mention of Reuters or Bloomberg must NOT collapse independent origin into wire agency (N4)
    incidental_tc = (
        "Nvidia unveiled its latest rack-scale cooling architecture at GTC today. "
        "Analysts speaking to Bloomberg and Reuters last month had speculated about liquid cooling yields."
    )
    assert detect_wire_origin(incidental_tc, "https://techcrunch.com/2026/10/17/nvidia-cooling/") == "techcrunch.com"
    # Genuine dateline/byline MUST be detected as wire origin
    genuine_wire = "TAIPEI (Reuters) - Taiwan Semiconductor Manufacturing Co reported record quarterly profit."
    assert detect_wire_origin(genuine_wire, "https://yahoo.com/finance/news/tsmc-profit.html") == "Reuters"


def test_balanced_candidate_selection_and_anti_circular_guard():
    from evidence_verifier import _select_balanced_candidate_urls
    official = [{"url": "https://nvidianews.nvidia.com/news/b200"}, {"url": "https://nvidia.com/extra"}]
    support = [
        {"url": "https://reuters.com/tech/b200-1"},
        {"url": "https://Tomshardware.com/tech/b200-2"},
        {"url": "https://anandtech.com/tech/b200-3"}
    ]
    contra = [
        {"url": "https://ft.com/content/b200-delay-1"},
        {"url": "https://wsj.com/articles/b200-delay-2"},
        {"url": "https://bloomberg.com/news/b200-delay-3"}
    ]
    selected = _select_balanced_candidate_urls(
        official_results=official,
        support_results=support,
        contra_results=contra,
        local_urls=["https://techcrunch.com/local-story"]
    )
    # 1 official + 2 support + 2 contradiction = 5 balanced URLs
    assert len(selected) == 5
    assert "https://nvidianews.nvidia.com/news/b200" in selected
    assert "https://ft.com/content/b200-delay-1" in selected
    assert "https://wsj.com/articles/b200-delay-2" in selected

    # Verify social/video platforms (youtube.com, reddit.com) are rejected and regional mirrors (me.pcmag.com / au.pcmag.com) deduplicated
    noisy_support = [
        {"url": "https://www.youtube.com/watch?v=12345", "score": 0.99},
        {"url": "https://www.reddit.com/r/hardware/comments/abc", "score": 0.95},
        {"url": "https://me.pcmag.com/chips/999/tsmc-2nm", "score": 0.88},
        {"url": "https://au.pcmag.com/chips/999/tsmc-2nm-au", "score": 0.87},
        {"url": "https://www.reuters.com/technology/tsmc-2nm-yields", "score": 0.80}
    ]
    clean_selected = _select_balanced_candidate_urls(
        official_results=[],
        support_results=noisy_support,
        contra_results=[],
        local_urls=set()
    )
    assert all("youtube.com" not in u and "reddit.com" not in u for u in clean_selected)
    # Reuters (Tier-1 authority) ranked first, and only 1 PCMag regional mirror kept on initial pass
    assert clean_selected[0] == "https://www.reuters.com/technology/tsmc-2nm-yields"


if __name__ == "__main__":
    print("=" * 65)
    print("    RUNNING EVIDENCE VERIFIER & ORIGINS TEST SUITE")
    print("=" * 65)
    test_span_grounding_valid()
    print("✓ test_span_grounding_valid PASSED")
    test_span_grounding_hallucination_rejection()
    print("✓ test_span_grounding_hallucination_rejection PASSED")
    test_wire_origin_syndication_detection()
    print("✓ test_wire_origin_syndication_detection PASSED")
    test_evidence_verifier_mock_pipeline()
    print("✓ test_evidence_verifier_mock_pipeline PASSED")
    test_negation_aware_stance_classification()
    print("✓ test_negation_aware_stance_classification PASSED")
    test_fresh_story_rule()
    print("✓ test_fresh_story_rule PASSED")
    test_wire_origin_incidental_mention_no_false_positive()
    print("✓ test_wire_origin_incidental_mention_no_false_positive PASSED")
    test_balanced_candidate_selection_and_anti_circular_guard()
    print("✓ test_balanced_candidate_selection_and_anti_circular_guard PASSED")
    print("=" * 65)
    print("[ALL 8 EVIDENCE & ORIGIN TESTS PASSED!]")
    print("=" * 65)


