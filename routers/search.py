from fastapi import APIRouter, Query

from db import get_db
from embedder import get_embedder
from tavily_service import search_news
from url_utils import canonicalize_url, extract_domain

router = APIRouter(tags=["Hybrid RRF Search"])


def execute_hybrid_search(q: str, grouped: bool = False, limit: int = 12):
    """Core modular 3-tier hybrid search engine fusing dense vectors, BM25 FTS, and tag likeness."""
    STOP_WORDS = {
        "i", "want", "need", "give", "me", "the", "a", "an", "and", "or", 
        "in", "on", "at", "to", "for", "of", "with", "by", "show", "find", 
        "news", "articles", "article", "latest", "about", "some", "any",
        "kind", "that", "this", "talk", "talking", "tell", "its", "it", 
        "what", "which", "type", "sort", "small", "big", "good", "like"
    }

    # Step 1: Dense Vector Embedding
    embedder = get_embedder()
    q_vec = embedder.embed_text(q)

    # Step 2: Content Token Normalization for Lexical/Tag Substring Overlap
    clean_words = [w.strip(".,?!\"'").lower() for w in q.split()]
    meaningful_tokens = [w for w in clean_words if len(w) > 2 and w not in STOP_WORDS]
    if not meaningful_tokens:
        meaningful_tokens = [w for w in clean_words if len(w) > 1]
    kw_like_patterns = [f"%{w}%" for w in meaningful_tokens] if meaningful_tokens else ["%__empty__%"]

    # Step 3: Reciprocal Rank Fusion (RRF) SQL Engine with Calibrated Noise Cutoff (< 0.78 cosine distance)
    SQL_HYBRID_RRF = """
    WITH vector_matches AS (
        SELECT id, 
               (embedding <=> %(q_vec)s::vector) as v_dist,
               1.0 / (60.0 + ROW_NUMBER() OVER (ORDER BY embedding <=> %(q_vec)s::vector ASC)) AS score
        FROM articles
        WHERE is_processed = TRUE 
          AND (macro_tag IS NULL OR macro_tag != 'Off-Topic')
          AND embedding IS NOT NULL
          AND (embedding <=> %(q_vec)s::vector) < 0.78
        ORDER BY (embedding <=> %(q_vec)s::vector) ASC
        LIMIT 25
    ),
    fts_matches AS (
        SELECT id, 
               ts_rank_cd(to_tsvector('english', title || ' ' || clean_text), websearch_to_tsquery('english', %(q)s)) as fts_rank,
               1.0 / (60.0 + ROW_NUMBER() OVER (
                   ORDER BY ts_rank_cd(to_tsvector('english', title || ' ' || clean_text), websearch_to_tsquery('english', %(q)s)) DESC
               )) AS score
        FROM articles
        WHERE is_processed = TRUE 
          AND (macro_tag IS NULL OR macro_tag != 'Off-Topic')
          AND to_tsvector('english', title || ' ' || clean_text) @@ websearch_to_tsquery('english', %(q)s)
        ORDER BY ts_rank_cd(to_tsvector('english', title || ' ' || clean_text), websearch_to_tsquery('english', %(q)s)) DESC
        LIMIT 25
    ),
    tag_matches AS (
        SELECT id, 
               (
                   (CASE WHEN title ILIKE ANY(%(kw_like_patterns)s::text[]) THEN 4 ELSE 0 END) +
                   (SELECT count(*) * 2 FROM unnest(articles.micro_tags) t WHERE t ILIKE ANY(%(kw_like_patterns)s::text[]))
               ) as tag_pts,
               1.0 / (60.0 + ROW_NUMBER() OVER (
                   ORDER BY (
                       (CASE WHEN title ILIKE ANY(%(kw_like_patterns)s::text[]) THEN 4 ELSE 0 END) +
                       (SELECT count(*) * 2 FROM unnest(articles.micro_tags) t WHERE t ILIKE ANY(%(kw_like_patterns)s::text[]))
                   ) DESC, published_at DESC
               )) AS score
        FROM articles
        WHERE is_processed = TRUE 
          AND (macro_tag IS NULL OR macro_tag != 'Off-Topic')
          AND cardinality(%(kw_like_patterns)s::text[]) > 0
          AND (
              title ILIKE ANY(%(kw_like_patterns)s::text[])
              OR EXISTS (SELECT 1 FROM unnest(articles.micro_tags) t WHERE t ILIKE ANY(%(kw_like_patterns)s::text[]))
          )
        LIMIT 25
    ),
    fused_scores AS (
        SELECT id, SUM(score) AS rrf_score
        FROM (
            SELECT id, score FROM vector_matches
            UNION ALL
            SELECT id, score FROM fts_matches
            UNION ALL
            SELECT id, score FROM tag_matches
        ) combined
        GROUP BY id
    )
    SELECT a.id, a.source_feed, a.title, a.tldr, a.micro_tags, a.original_url, a.macro_tag, a.macro_tags,
           fs.rrf_score,
           vm.v_dist,
           fm.fts_rank,
           tm.tag_pts
    FROM fused_scores fs
    JOIN articles a ON fs.id = a.id
    LEFT JOIN vector_matches vm ON a.id = vm.id
    LEFT JOIN fts_matches fm ON a.id = fm.id
    LEFT JOIN tag_matches tm ON a.id = tm.id
    WHERE vm.v_dist IS NOT NULL OR fm.fts_rank IS NOT NULL OR (tm.tag_pts IS NOT NULL AND tm.tag_pts >= 4)
    ORDER BY fs.rrf_score DESC, a.published_at DESC
    LIMIT %(limit)s;
    """

    results = []
    with get_db() as cur:
        cur.execute(SQL_HYBRID_RRF, {
            "q_vec": q_vec,
            "q": q,
            "kw_like_patterns": kw_like_patterns,
            "limit": limit
        })
        results = cur.fetchall()

    direct_matches = []
    explore_matches = []

    for r in results:
        art_id, source, title, tldr, micro_tags, url, macro_tag, macro_tags, rrf_score, v_dist, fts_rank, tag_pts = r
        m_tags = macro_tags if macro_tags else ([macro_tag] if macro_tag else [])
        tags = micro_tags if micro_tags else []

        is_direct = False
        if fts_rank is not None:
            is_direct = True
        elif tag_pts is not None and tag_pts >= 4:
            is_direct = True
        elif v_dist is not None and v_dist <= 0.68:
            is_direct = True

        if v_dist is not None:
            if is_direct:
                relevance_score = int(min(99, max(75, 75 + (0.68 - v_dist) / 0.28 * 24)))
                possibility_desc = "Direct Focus"
            else:
                relevance_score = int(min(68, max(45, 45 + (0.78 - v_dist) / 0.10 * 23)))
                possibility_desc = "Adjacent Horizon"
        else:
            relevance_score = 85 if is_direct else 55
            possibility_desc = "Direct Focus" if is_direct else "Adjacent Horizon"

        if fts_rank is not None:
            relevance_score = min(99, relevance_score + 8)
        elif tag_pts is not None and tag_pts >= 4:
            relevance_score = min(98, relevance_score + 5)

        matching_tags = [t for t in tags if any(tok in t for tok in meaningful_tokens)]
        if matching_tags:
            explore_angle = f"Related via: {', '.join(matching_tags[:3])}"
        elif tags:
            explore_angle = f"Domain context: {macro_tag or 'Tech'} ({', '.join(tags[:2])})"
        else:
            explore_angle = f"Cross-domain discovery: {macro_tag or 'General Tech'}"

        item = {
            "id": str(art_id),
            "source": source,
            "title": title,
            "tldr": tldr,
            "macro_tag": macro_tag,
            "macro_tags": m_tags,
            "micro_tags": tags,
            "url": url,
            "match_tier": "direct" if is_direct else "explore",
            "relevance_score": relevance_score,
            "possibility": possibility_desc,
            "explore_angle": explore_angle
        }

        if is_direct:
            direct_matches.append(item)
        else:
            explore_matches.append(item)

    if grouped is True or (isinstance(grouped, str) and grouped.lower() in ("true", "1")):
        return {
            "query": q,
            "total_direct": len(direct_matches),
            "total_explore": len(explore_matches),
            "direct": direct_matches,
            "explore": explore_matches
        }

    return direct_matches + explore_matches


@router.get("/api/search")
def natural_search(
    q: str = Query(..., min_length=2), 
    grouped: bool = Query(False, description="Separate results into direct matches and 'want to explore' discoveries"),
    live_web: bool = Query(False, description="Include live external web discovery horizon")
):
    is_grouped = grouped is True or (isinstance(grouped, str) and grouped.lower() in ("true", "1"))
    is_live_web = live_web is True or (isinstance(live_web, str) and live_web.lower() in ("true", "1"))
    res = execute_hybrid_search(q, grouped=is_grouped, limit=12)
    if not is_live_web:
        return res

    # Live web discovery with 5.0s timeout and canonical URL deduplication
    local_urls = set()
    if is_grouped and isinstance(res, dict):
        for item in res.get("direct", []) + res.get("explore", []):
            local_urls.add(canonicalize_url(item.get("url", "")))
    elif isinstance(res, list):
        for item in res:
            local_urls.add(canonicalize_url(item.get("url", "")))

    raw_web = search_news(q, max_results=4, search_topic="news", include_raw_content=False, timeout=5.0, purpose="live_horizon")
    live_formatted = [
        {
            "title": w.get("title", "Live Web Result"),
            "url": canonicalize_url(w.get("url", "")),
            "source": extract_domain(w.get("url", "")),
            "snippet": (w.get("content") or "")[:250],
            "tier": "live_web"
        }
        for w in raw_web
        if canonicalize_url(w.get("url", "")) not in local_urls
    ]

    if is_grouped and isinstance(res, dict):
        res["live_web"] = live_formatted
        return res
    return {"local_results": res, "live_web": live_formatted}
