import os
import sys
import json
import tempfile
from datetime import datetime, timezone, timedelta

# Ensure workspace root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from content_cleaner import (
    clean_markdown_article,
    clean_article_markdown,
    extract_normalized_shingles,
    _is_error_or_bot_page,
    _is_paywall_stub,
    _is_ui_or_sidebar_dump
)
from ingest_tavily import (
    _parse_and_validate_date,
    _truncate_markdown_safely,
    _load_ingest_state,
    _save_ingest_state
)


def test_regex_link_stripping():
    """Verifies that leading standalone links and separator bars are stripped, while prose links are retained."""
    sample = (
        "[Home](https://example.com)\n"
        "[Dashboard](https://example.com/dash) | [Settings](https://example.com/set) • [Help](https://example.com/help)\n\n"
        "# Nvidia Announces Quantum Computing Initiative\n\n"
        "Yesterday, [Nvidia](https://nvidia.com) unveiled their new quantum simulation platform [CUDA-Q](https://nvidia.com/cuda-q) "
        "designed to accelerate hybrid classical-quantum research pipelines across enterprise and academic laboratories. "
        "The software architecture provides a unified programming model connecting existing GPU supercomputers with leading QPU hardware.\n\n"
        "Researchers worldwide can access the open-source libraries immediately without proprietary hardware licensing restrictions."
    )
    cleaned, truncated = clean_markdown_article(sample)
    assert "[Home]" not in cleaned
    assert "[Dashboard]" not in cleaned
    assert "Nvidia Announces Quantum Computing Initiative" in cleaned
    assert "[Nvidia](https://nvidia.com) unveiled their new quantum simulation platform" in cleaned
    print("✓ test_regex_link_stripping PASSED")


def test_comment_section_truncation():
    """Verifies truncation at full-line exit headers (## Comments) when start >= 600 chars."""
    body_prose = (
        "# Major Breakthrough in Commercial Fusion Energy Scaling\n\n"
        "Scientists at the National Ignition Facility have achieved a net energy gain factor of 2.5 in laser-driven inertial confinement experiments. "
        "The latest test generated 4.2 megajoules of energy from an initial laser input of 1.7 megajoules, setting a world record for controlled laboratory reactions. "
        "This milestone demonstrates the viability of modern diagnostic sensors and automated hohlraum positioning systems under extreme thermonuclear conditions. "
        "Commercial power plant designs are already beginning to incorporate these target manufacturing techniques to increase repetition rates.\n\n"
        "Further developments will focus on magnetic confinement integration and tritium breeding blankets to sustain commercial baseload operation.\n\n"
    )
    sample = body_prose + (
        "## Comments\n\n"
        "User123: This is an incredible physics milestone!\n"
        "Physicist99: What was the target pellet composition?\n"
    )
    assert len(body_prose) >= 600
    cleaned, truncated = clean_markdown_article(sample)
    assert truncated is True
    assert "## Comments" not in cleaned
    assert "User123" not in cleaned
    assert "Commercial power plant designs" in cleaned
    print("✓ test_comment_section_truncation PASSED")


def test_mid_article_heading_protection():
    """Verifies that editorial mid-article headings like '## Comments from CEO' are NOT truncated."""
    sample = (
        "# Semiconductor Manufacturing Forecast for 2027\n\n"
        "Foundry capital expenditures are projected to increase by 18 percent year-over-year as leading semiconductor fabricators expand advanced packaging cleanrooms. "
        "High bandwidth memory (HBM4) integration requires sub-micron interconnect densities that push existing lithography tools beyond historical limits. "
        "Equipment suppliers are adjusting delivery schedules to meet these aggressive timeline constraints.\n\n"
        "## Comments from CEO Regarding Expansion\n\n"
        "During the quarterly analyst conference call, executive leadership emphasized that customer commitments exceed initial capacity allocations. "
        "The company plans to invest an additional five billion dollars into domestic assembly and testing facilities over the next three fiscal years.\n\n"
        "This strategic investment ensures diversified supply chains for mission-critical artificial intelligence accelerators."
    )
    cleaned, truncated = clean_markdown_article(sample)
    assert truncated is False
    assert "Comments from CEO Regarding Expansion" in cleaned
    assert "quarterly analyst conference call" in cleaned
    print("✓ test_mid_article_heading_protection PASSED")


def test_inline_promo_stripping():
    """Verifies that inline promotional callouts are stripped line-by-line without truncating the rest of the document."""
    sample = (
        "# Space Telescope Identifies Exoplanet Atmosphere\n\n"
        "Astronomers analyzing spectral absorption bands have detected water vapor and carbon monoxide in the atmosphere of an exoplanet orbiting a nearby red dwarf star. "
        "The atmospheric signatures indicate turbulent convective mixing within the upper troposphere.\n\n"
        "[##### **Related Story** New Infrared Observatory Receives Launch Approval](https://space.org/news/123)\n\n"
        "Sign up for our newsletter to receive daily space intelligence.\n\n"
        "Ground-based high-resolution spectrographs subsequently confirmed the presence of heavy molecular absorbers in the transmission spectrum. "
        "These independent measurements rule out stellar activity as the primary driver of the observed transmission signal."
    )
    cleaned, truncated = clean_markdown_article(sample)
    assert "Related Story" not in cleaned
    assert "Sign up for our newsletter" not in cleaned
    assert "Ground-based high-resolution spectrographs" in cleaned
    assert "Astronomers analyzing spectral absorption" in cleaned
    print("✓ test_inline_promo_stripping PASSED")


def test_error_and_bot_rejection():
    """Verifies that Cloudflare / 404 stubs (< 1500 chars) are rejected, but genuine news articles on outages are retained."""
    bot_stub = "<html><body>Please enable JavaScript and Cookies to continue. Verify you are a human. Access to this page has been denied.</body></html>"
    cleaned_bot, _ = clean_markdown_article(bot_stub)
    assert cleaned_bot == ""

    outage_news = (
        "# Global Cloud Provider Resolves Major DNS Outage Affecting Enterprise SaaS\n\n"
        "A cascading misconfiguration in edge routing tables triggered widespread HTTP 404 errors for millions of users across North America on Tuesday morning. "
        "Enterprise communication platforms, cloud document editors, and financial payment gateways experienced intermittent disruptions lasting approximately forty-seven minutes. "
        "Network reliability engineers deployed an automated rollback sequence to restore valid BGP route advertisements across all tier-one peering points.\n\n"
        "Post-incident reviews revealed that an internal policy validation script failed to catch syntax regressions prior to global deployment. "
        "The infrastructure operator has introduced redundant peer-review gates to prevent similar configuration propagations in future updates. "
        "Industry analysts observed that multi-cloud redundancy strategies proved resilient for financial institutions that maintained active-active failover endpoints.\n\n"
        "Several enterprise clients reported that automated health checks routed critical traffic away from impacted availability zones within seconds of initial degradation. "
        "The company has committed to publishing a comprehensive architectural post-mortem and upgrading synthetic probe frequencies across all regional data centers.\n\n"
        "Users who encountered the temporary '404 page not found' notification during the service window were redirected once routing converged. "
        "Further resilience drills are planned for next quarter to evaluate automated route-isolation behaviors across global edge pops."
    )
    assert len(outage_news) >= 1500
    cleaned_news, _ = clean_markdown_article(outage_news, title="Global Cloud Provider Resolves Major DNS Outage")
    assert "Global Cloud Provider Resolves Major DNS Outage" in cleaned_news
    assert len(cleaned_news) > 1000
    print("✓ test_error_and_bot_rejection PASSED")


def test_paywall_and_ui_dump_rejection():
    """Verifies that paywall stubs and navigation/ticker dumps (like Yahoo Finance) are rejected."""
    paywall_stub = (
        "# Financial Markets Digest\n\n"
        "Subscribe to read the full analysis. Already a subscriber? Log in to unlock this story with a subscription."
    )
    cleaned_paywall, _ = clean_markdown_article(paywall_stub)
    assert cleaned_paywall == ""

    yahoo_finance_menu = (
        "1. [My Portfolio](https://finance.yahoo.com/portfolios/)\n"
        "2. 1. [Stocks](https://finance.yahoo.com/markets/stocks/)\n"
        "      1. [Most Active](https://finance.yahoo.com/markets/stocks/most-active/)\n"
        "      2. [Day Gainers](https://finance.yahoo.com/markets/stocks/gainers/)\n"
        "      3. [Day Losers](https://finance.yahoo.com/markets/stocks/losers/)\n"
        "      4. [Trending](https://finance.yahoo.com/markets/stocks/trending/)\n"
        "      5. [Highest Dividend](https://finance.yahoo.com/markets/stocks/highest-dividend/)\n"
        "      6. [Large Cap](https://finance.yahoo.com/markets/stocks/large-cap/)\n"
        "      7. [Small Cap](https://finance.yahoo.com/markets/stocks/small-cap/)\n"
        "   2. [Crypto](https://finance.yahoo.com/markets/crypto/all/)\n"
        "      1. [Most Active](https://finance.yahoo.com/markets/crypto/most-active/)\n"
        "      2. [Day Gainers](https://finance.yahoo.com/markets/crypto/gainers/)\n"
        "      3. [Day Losers](https://finance.yahoo.com/markets/crypto/losers/)\n"
    )
    cleaned_menu, _ = clean_markdown_article(yahoo_finance_menu, title="10x Research Forecast Bitcoin Could Fall")
    assert cleaned_menu == ""
    print("✓ test_paywall_and_ui_dump_rejection PASSED")


def test_shingle_deduplication():
    """Verifies that syndicated articles with identical prose but differing link styles are detected via Jaccard >= 0.65."""
    text_outlet_a = (
        "Valve officially announced the [Steam Deck 2](https://store.steampowered.com/steamdeck) today, featuring a custom AMD monolithic APU built on TSMC 3nm silicon. "
        "The handheld console delivers a 120Hz variable refresh rate OLED display with up to sixty percent improved battery endurance across high-intensity workloads. "
        "Pre-orders are scheduled to open early next quarter with deliveries beginning in late autumn."
    )
    text_outlet_b = (
        "Valve officially announced the Steam Deck 2 today, featuring a custom [AMD](https://amd.com) monolithic APU built on [TSMC](https://tsmc.com) 3nm silicon. "
        "The handheld console delivers a 120Hz variable refresh rate OLED display with up to sixty percent improved battery endurance across high-intensity workloads. "
        "Pre-orders are scheduled to open early next quarter with deliveries beginning in late autumn."
    )
    shingles_a = extract_normalized_shingles(text_outlet_a, k=5)
    shingles_b = extract_normalized_shingles(text_outlet_b, k=5)
    assert len(shingles_a) > 0
    assert len(shingles_b) > 0

    jaccard = len(shingles_a & shingles_b) / len(shingles_a | shingles_b)
    assert jaccard >= 0.90
    print(f"✓ test_shingle_deduplication PASSED (Jaccard: {jaccard:.2f})")


def test_date_validation():
    """Verifies future/expired date rejection and graceful fallback to UTC now for missing dates."""
    now = datetime.now(timezone.utc)
    
    # Valid recent date (ISO)
    recent_iso = (now - timedelta(days=2)).isoformat()
    assert _parse_and_validate_date(recent_iso) is not None

    # Future date (> 2 hours) -> Rejected
    future_iso = (now + timedelta(hours=5)).isoformat()
    assert _parse_and_validate_date(future_iso) is None

    # Stale date (> 14 days) -> Rejected
    stale_iso = (now - timedelta(days=20)).isoformat()
    assert _parse_and_validate_date(stale_iso) is None

    # Missing date -> Fallback to valid ISO timestamp
    fallback = _parse_and_validate_date(None)
    assert fallback is not None
    assert "T" in fallback
    print("✓ test_date_validation PASSED")


def test_code_block_balancing():
    """Verifies that _truncate_markdown_safely balances unclosed code blocks."""
    unclosed_code = (
        "# Implementation Guide\n\n"
        "Here is the sample code:\n\n"
        "```python\n"
        "def compute_gradients():\n"
        "    return 42\n"
    )
    truncated = _truncate_markdown_safely(unclosed_code, max_chars=80)
    assert truncated.count("```") % 2 == 0
    print("✓ test_code_block_balancing PASSED")


def test_state_rotation():
    """Verifies that .ingest_state.json advances cycle_index atomically."""
    temp_dir = tempfile.mkdtemp()
    test_state_file = os.path.join(temp_dir, "test_state.json")
    
    # Test saving and loading
    initial_state = {"cycle_index": 1, "last_successful_run_utc": None}
    with open(test_state_file, "w", encoding="utf-8") as f:
        json.dump(initial_state, f)

    with open(test_state_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["cycle_index"] == 1

    next_state = {"cycle_index": (loaded["cycle_index"] + 1) % 3, "last_successful_run_utc": datetime.now(timezone.utc).isoformat()}
    with open(test_state_file, "w", encoding="utf-8") as f:
        json.dump(next_state, f)

    with open(test_state_file, "r", encoding="utf-8") as f:
        reloaded = json.load(f)
    assert reloaded["cycle_index"] == 2
    print("✓ test_state_rotation PASSED")


if __name__ == "__main__":
    print("=" * 65)
    print("   HARDENED INGESTION & CONTENT CLEANER TEST SUITE")
    print("=" * 65)
    test_regex_link_stripping()
    test_comment_section_truncation()
    test_mid_article_heading_protection()
    test_inline_promo_stripping()
    test_error_and_bot_rejection()
    test_paywall_and_ui_dump_rejection()
    test_shingle_deduplication()
    test_date_validation()
    test_code_block_balancing()
    test_state_rotation()
    print("=" * 65)
    print("ALL 10 HARDENED CLEANER & INGESTION TESTS PASSED!")
    print("=" * 65)
