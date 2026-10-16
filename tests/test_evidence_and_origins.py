import os
import sys
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
load_dotenv()

from evidence_verifier import span_is_grounded, assign_origins, verify_cluster_evidence
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
    print("=" * 65)
    print("[ALL 4 EVIDENCE & ORIGIN TESTS PASSED!]")
    print("=" * 65)
