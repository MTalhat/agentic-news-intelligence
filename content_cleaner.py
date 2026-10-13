import re
import logging
from typing import Tuple

logger = logging.getLogger("content_cleaner")

# Universal Markdown headings and labels where editorial body ends and peripheral CMS widgets begin
TAIL_HEADER_PATTERNS = [
    r'^\s*#{1,4}\s*(?:further reading|related (?:stories|coverage|articles|posts|content)|trending (?:stories|now|articles)|popular (?:discussions|stories|posts)|recommended (?:stories|articles)|more (?:from|coverage)|read (?:more|next)|see also)\b',
    r'^\s*(?:\*{1,2}|_{1,2})?\s*about the (?:\[?author\]?|writer|reporter)\b',
    r'^\s*\[read all\s+comments\b',
    r'^\s*#{1,4}\s*(?:comments|discussion|join the conversation)\b',
    r'^\s*(?:sign up for|subscribe to) (?:our |the )?newsletter\b',
    r'^\s*follow\s+.*?on google\b',
    r'^\s*\*{0,2}(?:copyright reserved|all rights reserved|©\s*\d{4})',
    r'^\s*#{1,4}\s*(?:knowledge centers|share this|share on)',
    r'^\s*(?:\*{0,2}|_{0,2})\b(?:privacy notice|privacy dashboard|cookie policy)\b'
]

TAIL_BOUNDARY_REGEX = re.compile('|'.join(TAIL_HEADER_PATTERNS), flags=re.IGNORECASE | re.MULTILINE)

# Inline promotional callout blocks embedded between paragraphs
INLINE_PROMO_REGEX = re.compile(
    r'^\s*\[?\s*#{1,5}\s*\*{0,2}(?:related (?:story|article|reading)|read more|see also|also read):?\s*\*{0,2}\s*.*?\]\(https?://[^\)]+\)\s*$',
    flags=re.IGNORECASE | re.MULTILINE
)

def _strip_leading_boilerplate(text: str) -> str:
    """Strips author byline link debris, sharing links, and nav remnants before the first body paragraph."""
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
            if (stripped.startswith('[') and stripped.endswith(')') and len(stripped) < 140) or \
               re.match(r'^\[.*?\]\(.*?\)\s*[•·|\-]', stripped) or \
               re.match(r'^\[.*?\]\(.*?\)\s*\[.*?\]\(.*?\)', stripped) or \
               re.match(r'(?i)^(?:by|author|posted by|published by):?\s*\[.*?\]\(.*?\)', stripped) or \
               re.match(r'^\[(?:comments|add to preferr)', stripped, re.I) or \
               stripped.lower() in {'log in', 'sign in', 'sign up', 'oops, something went wrong', 'join', 'home', 'menu'}:
                continue
            in_lead = False
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines).strip()

def _is_high_link_density_list(lines: list[str]) -> bool:
    """Evaluates whether a group of bullet items is an aggregator link widget."""
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
        is_bullet_block = len(lines) >= 2 and all(l.startswith(('*', '-', '•')) or re.match(r'^\d+\.', l) for l in lines)
        if is_bullet_block and _is_high_link_density_list(lines):
            continue
        cleaned_paragraphs.append(p)
    return '\n\n'.join(cleaned_paragraphs).strip()

def clean_markdown_article(raw_markdown: str, title: str = "") -> Tuple[str, bool]:
    """
    Cleans raw Markdown content by removing CMS widgets, promo callouts, and link lists.
    Returns (cleaned_text, was_truncated).
    """
    if not raw_markdown or len(raw_markdown.strip()) == 0:
        return "", False

    original_length = len(raw_markdown)
    processed = raw_markdown.strip()

    # Pass 1: Strip leading byline and comment links
    processed = _strip_leading_boilerplate(processed)

    # Pass 2: Chop off everything after the first tail marker
    was_truncated = False
    match = TAIL_BOUNDARY_REGEX.search(processed)
    if match:
        processed = processed[:match.start()].strip()
        was_truncated = True

    # Pass 3: Remove mid-article promotional callouts
    processed = INLINE_PROMO_REGEX.sub('', processed)

    # Pass 4: Prune high-density list widgets that might remain before the tail boundary
    processed = _prune_link_density_lists(processed)

    # Normalize excessive newlines
    processed = re.sub(r'\n{3,}', '\n\n', processed).strip()

    # Safety Guardrail: Prevent catastrophic over-truncation
    if len(processed) < 400 and original_length >= 600:
        logger.warning(
            "[Sanitizer] Cleaning resulted in excessively short article (%d chars from %d). Reverting safely.",
            len(processed), original_length
        )
        if match and match.start() >= 400:
            return raw_markdown[:match.start()].strip(), True
        return raw_markdown.strip(), False

    return processed, was_truncated

def clean_article_markdown(title: str, text: str) -> str:
    """Backward-compatible helper returning cleaned text string."""
    cleaned, _ = clean_markdown_article(text, title=title)
    return cleaned
