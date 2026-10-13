import urllib.parse
from typing import Set

TRACKING_PARAMS: Set[str] = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "ref", "source", "mc_cid", "mc_eid"
}

def canonicalize_url(raw_url: str) -> str:
    """Canonicalizes a URL: strips tracking parameters, fragments, trailing slashes, and lowercases host."""
    if not raw_url:
        return ""
    try:
        parsed = urllib.parse.urlsplit(raw_url.strip())
        query_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=False)
        cleaned_pairs = [
            (k, v) for k, v in query_pairs 
            if k.lower() not in TRACKING_PARAMS and not k.lower().startswith("utm_")
        ]
        cleaned_query = urllib.parse.urlencode(cleaned_pairs)
        path = parsed.path.rstrip("/")
        
        return urllib.parse.urlunsplit((
            parsed.scheme.lower() or "https",
            parsed.netloc.lower().removeprefix("www."),
            path,
            cleaned_query,
            ""  # Strip fragment (#)
        ))
    except Exception:
        return raw_url.strip()

def extract_domain(raw_url: str) -> str:
    """Safely extracts clean domain name from URL without throwing IndexError."""
    try:
        netloc = urllib.parse.urlsplit(raw_url).netloc.lower()
        return netloc.removeprefix("www.") or "news.web"
    except Exception:
        return "news.web"
