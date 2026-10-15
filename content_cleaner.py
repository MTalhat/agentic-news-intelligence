import re
import logging
from typing import Tuple

logger = logging.getLogger("content_cleaner")

# Universal stop words for title relevance checking
STOP_WORDS = {
    "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with",
    "by", "is", "it", "its", "from", "as", "that", "this", "are", "was", "were",
    "be", "has", "have", "had", "will", "would", "can", "could", "should", "not"
}

# 1. Strong Tail Headings: Full-line editorial exit headers anchored to the entire line ($)
STRONG_TAIL_PATTERN = (
    r'^\s*#{1,4}\s*(?:comments?|discussion|join the conversation|'
    r'related (?:stories|articles|posts|content)|further reading|'
    r'recommended (?:stories|articles)|popular (?:stories|articles)|trending (?:stories|now))\s*$'
)
STRONG_TAIL_REGEX = re.compile(STRONG_TAIL_PATTERN, flags=re.IGNORECASE | re.MULTILINE)

# 2. Weak Tail Markers: Looser closing markers requiring position depth (>= 55% of doc)
WEAK_TAIL_PATTERN = (
    r'^\s*(?:\*{0,2}|_{0,2})\b(?:about the (?:\[?(?:writer|author|reporter)\]?(?:\([^\)]+\))?|team)|'
    r'copyright reserved|all rights reserved|©\s*\d{4}|privacy policy|cookie policy)\b'
)
WEAK_TAIL_REGEX = re.compile(WEAK_TAIL_PATTERN, flags=re.IGNORECASE | re.MULTILINE)

# 3. Inline Promotional & Boilerplate Lines: Removed line-by-line without truncating the document
INLINE_BOILERPLATE_PATTERN = (
    r'^\s*(?:'
    r'(?:sign up for|subscribe to) (?:our |the )?newsletter.*|'
    r'follow\s+.*?on google.*|'
    r'share (?:this|on).*|'
    r'\[read all\s+comments\b.*|'
    r'\[skip to\b.*|'
    r'\[?\s*#{0,5}\s*\*{0,2}\s*(?:read more|see also|also read|related (?:story|article|reading|coverage)):?\s*\*{0,2}\s*.*?\]\(https?://[^\)]+\)'
    r')\s*$'
)
INLINE_BOILERPLATE_REGEX = re.compile(INLINE_BOILERPLATE_PATTERN, flags=re.IGNORECASE | re.MULTILINE)


def _is_error_or_bot_page(text: str) -> bool:
    """Detects bot challenges or 404 error pages while protecting news articles reporting on outages."""
    challenge_phrases = [
        "please enable javascript",
        "verify you are a human",
        "just a moment...",
        "404 page not found",
        "access to this page has been denied"
    ]
    text_lower = text.lower()
    lead_350 = text_lower[:350]
    for phrase in challenge_phrases:
        if phrase in lead_350 or (len(text) < 1500 and phrase in text_lower):
            return True
    return False


def _is_paywall_stub(text: str) -> bool:
    """Detects shallow paywall gates while allowing long articles that mention subscriptions."""
    if len(text) >= 2500:
        return False
    paywall_phrases = [
        "subscribe to read",
        "subscribers only",
        "already a subscriber? log in",
        "unlock this story with a subscription"
    ]
    text_lower = text.lower()
    return any(p in text_lower for p in paywall_phrases)


def _is_ui_or_sidebar_dump(text: str, title: str = "") -> bool:
    """
    Detects navigation menus, stock ticker lists, or sidebar aggregator dumps
    that lack genuine editorial prose.
    """
    # Strip code blocks prior to checking metrics
    non_code = re.sub(r'```[\s\S]*?```', '', text)
    lines = [l.strip() for l in non_code.split('\n') if l.strip()]
    if len(lines) >= 6:
        avg_line_len = sum(len(l) for l in lines) / len(lines)
        punct_count = sum(1 for l in lines if re.search(r'[.,!?:;]', l))
        punct_ratio = punct_count / len(lines)
        if avg_line_len < 35 and punct_ratio < 0.35:
            return True

    # Title word relevance: If title has >= 3 significant words and NONE appear in text
    if title:
        sig_words = [
            w for w in re.findall(r'\b[a-z0-9]+\b', title.lower())
            if len(w) >= 3 and w not in STOP_WORDS
        ]
        if len(sig_words) >= 3:
            text_lower = text.lower()
            if not any(w in text_lower for w in sig_words):
                return True

    return False


def _strip_leading_boilerplate(text: str) -> str:
    """Strips author bylines, standalone navigation links, and sharing bars before prose."""
    lines = text.split('\n')
    cleaned_lines = []
    in_lead = True
    for line in lines:
        stripped = line.strip()
        if in_lead:
            if not stripped:
                continue
            if stripped.startswith('# '):
                cleaned_lines.append(line)
                continue
            if re.match(r'^\[Skip to', stripped, re.I):
                continue
            # Standalone markdown link
            if re.match(r'^\[[^\][]+\]\([^\(\)]+\)$', stripped):
                continue
            # Multi-link separator line: [Home](...) | [News](...)
            if re.match(r'^(?:\[[^\][]+\]\([^\(\)]+\)\s*[•·|\-*]\s*)+(?:\[[^\][]+\]\([^\(\)]+\))?$', stripped):
                continue
            # Standalone byline link
            if re.match(r'(?i)^(?:by|author|posted by|published by):?\s*\[.*?\]\(.*?\)', stripped):
                continue
            if stripped.lower() in {'log in', 'sign in', 'sign up', 'oops, something went wrong', 'join', 'home', 'menu'}:
                continue
            in_lead = False
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines).strip()


def _is_high_link_density_list(lines: list[str]) -> bool:
    """Evaluates whether a group of bullet items is an aggregator link widget (> 60% link chars)."""
    if len(lines) < 2:
        return False
    total_chars = sum(len(l.strip()) for l in lines)
    if total_chars == 0:
        return False
    link_chars = sum(len(m.group(0)) for l in lines for m in re.finditer(r'\[.*?\]\(.*?\)', l))
    return (link_chars / total_chars) > 0.60


def _prune_link_density_lists(text: str) -> str:
    """Detects and prunes list blocks that are overwhelmingly links."""
    paragraphs = text.split('\n\n')
    cleaned_paragraphs = []
    for p in paragraphs:
        lines = [l.strip() for l in p.split('\n') if l.strip()]
        is_bullet_block = len(lines) >= 2 and all(
            l.startswith(('*', '-', '•')) or re.match(r'^\d+\.', l) for l in lines
        )
        if is_bullet_block and _is_high_link_density_list(lines):
            continue
        cleaned_paragraphs.append(p)
    return '\n\n'.join(cleaned_paragraphs).strip()


def _deduplicate_paragraphs(text: str) -> str:
    """
    Code-safe paragraph deduplication: drops duplicate text blocks while
    preserving all code fences, markdown headings, and short lines.
    """
    paragraphs = text.split('\n\n')
    retained = []
    in_code_block = False

    for p in paragraphs:
        p_stripped = p.strip()
        lines = p_stripped.split('\n')
        # Check code fence state toggles within lines
        code_fence_count = sum(1 for l in lines if l.strip().startswith('```'))
        if in_code_block:
            retained.append(p)
            if code_fence_count % 2 == 1:
                in_code_block = False
            continue

        if code_fence_count % 2 == 1:
            in_code_block = True
            retained.append(p)
            continue

        # Bypass deduplication for headings and short items
        if p_stripped.startswith('#') or len(p_stripped) < 40:
            retained.append(p)
            continue

        p_lower = p_stripped.lower()
        # Drop exact duplicates or substrings of recent paragraphs
        is_dup = False
        for prev in retained[-5:]:
            prev_lower = prev.strip().lower()
            if p_lower == prev_lower or (len(p_stripped) > 80 and p_lower in prev_lower):
                is_dup = True
                break

        if not is_dup:
            retained.append(p)

    return '\n\n'.join(retained).strip()


def extract_normalized_shingles(text: str, k: int = 5) -> set[tuple[str, ...]]:
    """
    Tokenizes text into k-word shingles after stripping links and markdown syntax.
    Used for syndicated wire deduplication via Jaccard similarity.
    """
    stripped = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    stripped = re.sub(r'[`#*_]', '', stripped)
    words = re.findall(r'\b[a-z0-9]+\b', stripped.lower())[:300]
    if len(words) < k:
        return set()
    return {tuple(words[i:i + k]) for i in range(len(words) - k + 1)}


def clean_markdown_article(raw_markdown: str, title: str = "") -> Tuple[str, bool]:
    """
    Cleans raw Markdown content by removing CMS widgets, promo callouts, and link lists.
    Signature: raw_markdown is 1st argument, title is 2nd argument.
    Returns (cleaned_text, was_truncated).
    """
    if not raw_markdown or len(raw_markdown.strip()) == 0:
        return "", False

    original_length = len(raw_markdown)
    processed = raw_markdown.strip()

    # Pre-cleaning Context-Safe Rejection Gates
    if _is_error_or_bot_page(processed):
        logger.info("[Sanitizer] Rejected bot challenge or error page: %s", title[:40])
        return "", False

    if _is_paywall_stub(processed):
        logger.info("[Sanitizer] Rejected paywall stub: %s", title[:40])
        return "", False

    # Pass 1: Strip leading bylines, standalone links, and nav remnants
    processed = _strip_leading_boilerplate(processed)

    # Pass 2: Two-Tier Tail Truncation
    was_truncated = False
    earliest_tail_start = None

    # Tier 2A: Strong exit headings (>= 600 chars depth, no percentage guard)
    for m in STRONG_TAIL_REGEX.finditer(processed):
        if m.start() >= 600:
            if earliest_tail_start is None or m.start() < earliest_tail_start:
                earliest_tail_start = m.start()
            break

    # Tier 2B: Weak closing markers (>= 600 chars AND >= 55% document depth)
    for m in WEAK_TAIL_REGEX.finditer(processed):
        if m.start() >= 600 and m.start() >= (len(processed) * 0.55):
            if earliest_tail_start is None or m.start() < earliest_tail_start:
                earliest_tail_start = m.start()
            break

    if earliest_tail_start is not None:
        processed = processed[:earliest_tail_start].strip()
        was_truncated = True

    # Pass 3: Strip inline promotional banners line-by-line
    processed = INLINE_BOILERPLATE_REGEX.sub('', processed)

    # Pass 4: Prune high-density link lists
    processed = _prune_link_density_lists(processed)

    # Pass 5: Code-safe paragraph deduplication
    processed = _deduplicate_paragraphs(processed)

    # Normalize excessive newlines
    processed = re.sub(r'\n{3,}', '\n\n', processed).strip()

    # Post-cleaning UI / Sidebar Dump Rejection Gate
    if _is_ui_or_sidebar_dump(processed, title=title):
        logger.info("[Sanitizer] Rejected UI/sidebar dump lacking prose: %s", title[:40])
        return "", False

    # Safety Guardrail: Prevent catastrophic over-truncation
    if len(processed) < 400 and original_length >= 600:
        logger.warning(
            "[Sanitizer] Cleaning resulted in excessively short article (%d chars from %d). Reverting safely.",
            len(processed), original_length
        )
        if earliest_tail_start is not None and earliest_tail_start >= 400:
            return raw_markdown[:earliest_tail_start].strip(), True
        return raw_markdown.strip(), False

    return processed, was_truncated


def clean_article_markdown(title: str, text: str) -> str:
    """
    Backward-compatible helper returning cleaned text string.
    Signature: title is 1st argument, text is 2nd argument.
    """
    cleaned, _ = clean_markdown_article(text, title=title)
    return cleaned
