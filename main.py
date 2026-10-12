import os
import hashlib
import json
from datetime import datetime
from dateutil import parser as date_parser
from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from db import get_db
from ai import ai_client, FAST_MODEL, REASONING_MODEL
from embedder import get_embedder
from schemas import (
    ScrapedArticleIn, ArticleChatRequest, CommentCreate, 
    SynthesizeRequest, QueryIntentOut, UserLoginRequest
)

app = FastAPI(title="Intelligence RSS Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

INGESTION_SECRET = os.getenv("INGESTION_SECRET")

# 1. Ingestion Endpoint
@app.post("/api/internal/ingest")
def ingest_articles(items: list[ScrapedArticleIn], x_secret: str = Header(...)):
    if x_secret != INGESTION_SECRET:
        raise HTTPException(status_code=401, detail="Invalid ingestion secret")

    inserted = 0
    with get_db() as cur:
        for item in items:
            url_hash = hashlib.sha256(item.Link.encode("utf-8")).hexdigest()
            try:
                pub_date = date_parser.parse(item.Tarih) if item.Tarih else datetime.utcnow()
            except Exception:
                pub_date = datetime.utcnow()

            cur.execute("""
                INSERT INTO articles (url_hash, source_feed, title, original_url, author, published_at, clean_text)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (url_hash) DO NOTHING
                RETURNING id;
            """, (url_hash, item.Site, item.Baslik, item.Link, item.Yazar, pub_date, item.TamMetin))
            if cur.fetchone():
                inserted += 1

    return {"received": len(items), "inserted": inserted}

# 2. Feed Endpoint
@app.get("/api/articles")
def get_articles(page: int = 1, page_size: int = 20, macro: str | None = None):
    offset = (page - 1) * page_size
    with get_db() as cur:
        if macro:
            macro_clean = macro.strip()
            cur.execute("""
                SELECT id, source_feed, title, original_url, author, published_at, macro_tag, macro_tags, micro_tags, tldr
                FROM articles WHERE is_processed = TRUE 
                  AND (macro_tag ILIKE %s OR EXISTS (
                      SELECT 1 FROM unnest(macro_tags) m WHERE m ILIKE %s
                  ))
                ORDER BY published_at DESC LIMIT %s OFFSET %s;
            """, (macro_clean, macro_clean, page_size, offset))
        else:
            cur.execute("""
                SELECT id, source_feed, title, original_url, author, published_at, macro_tag, macro_tags, micro_tags, tldr
                FROM articles WHERE is_processed = TRUE AND (macro_tag IS NULL OR macro_tag != 'Off-Topic')
                ORDER BY published_at DESC LIMIT %s OFFSET %s;
            """, (page_size, offset))
        rows = cur.fetchall()

    return [
        {
            "id": str(r[0]), "source": r[1], "title": r[2], "url": r[3],
            "author": r[4], "published_at": r[5], "macro_tag": r[6],
            "macro_tags": r[7] if r[7] else ([r[6]] if r[6] else []),
            "micro_tags": r[8], "tldr": r[9]
        }
        for r in rows
    ]

# 3. Single Article Full Detail
@app.get("/api/articles/{article_id}")
def get_article(article_id: str):
    with get_db() as cur:
        cur.execute("""
            SELECT id, source_feed, title, original_url, author, published_at, clean_text, macro_tag, macro_tags, micro_tags, tldr
            FROM articles WHERE id = %s;
        """, (article_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Article not found")
        return {
            "id": str(row[0]), "source": row[1], "title": row[2], "url": row[3],
            "author": row[4], "published_at": row[5], "clean_text": row[6],
            "macro_tag": row[7],
            "macro_tags": row[8] if row[8] else ([row[7]] if row[7] else []),
            "micro_tags": row[9], "tldr": row[10]
        }

# 4. Hackathon-Grade Autonomous 3-Tier Hybrid Search (pgvector + BM25 FTS + RRF)
@app.get("/api/search")
def natural_search(q: str = Query(..., min_length=2), grouped: bool = Query(False, description="Separate results into direct matches and 'want to explore' discoveries")):
    # Conversational stop words to prevent spurious substring ILIKE false-positives
    STOP_WORDS = {
        "i", "want", "need", "give", "me", "the", "a", "an", "and", "or", 
        "in", "on", "at", "to", "for", "of", "with", "by", "show", "find", 
        "news", "articles", "article", "latest", "about", "some", "any",
        "kind", "that", "this", "talk", "talking", "tell", "its", "it", 
        "what", "which", "type", "sort", "small", "big", "good", "like"
    }

    # Step 1: Sub-10ms Dense Vector Embedding
    embedder = get_embedder()
    q_vec = embedder.embed_text(q)

    # Step 2: Content Token Normalization for Lexical/Tag Overlap
    clean_words = [w.strip(".,?!\"'").lower() for w in q.split()]
    meaningful_tokens = [w for w in clean_words if len(w) > 2 and w not in STOP_WORDS]
    if not meaningful_tokens:
        meaningful_tokens = [w for w in clean_words if len(w) > 1]
    kw_like_patterns = [f"%{w}%" for w in meaningful_tokens] if meaningful_tokens else ["%__empty__%"]

    # Step 3: Reciprocal Rank Fusion (RRF) SQL Engine with Calibrated Noise Cutoff (< 0.88 cosine distance)
    SQL_HYBRID_RRF = """
    WITH vector_matches AS (
        SELECT id, 
               (embedding <=> %(q_vec)s::vector) as v_dist,
               1.0 / (60.0 + ROW_NUMBER() OVER (ORDER BY embedding <=> %(q_vec)s::vector ASC)) AS score
        FROM articles
        WHERE is_processed = TRUE 
          AND (macro_tag IS NULL OR macro_tag != 'Off-Topic')
          AND embedding IS NOT NULL
          AND (embedding <=> %(q_vec)s::vector) < 0.88
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
    LIMIT 12;
    """

    results = []
    with get_db() as cur:
        cur.execute(SQL_HYBRID_RRF, {
            "q_vec": q_vec,
            "q": q,
            "kw_like_patterns": kw_like_patterns
        })
        results = cur.fetchall()

    direct_matches = []
    explore_matches = []

    for r in results:
        art_id, source, title, tldr, micro_tags, url, macro_tag, macro_tags, rrf_score, v_dist, fts_rank, tag_pts = r
        m_tags = macro_tags if macro_tags else ([macro_tag] if macro_tag else [])
        tags = micro_tags if micro_tags else []

        # Mathematical tier calibration:
        # Direct: High semantic similarity (v_dist <= 0.80), or explicit FTS text match, or strong title keyword match
        # Explore: Adjacent peripheral similarity (0.80 < v_dist < 0.88)
        is_direct = False
        if v_dist is not None and v_dist <= 0.80:
            is_direct = True
        elif fts_rank is not None:
            is_direct = True
        elif tag_pts is not None and tag_pts >= 4:
            is_direct = True

        # Calibrated relevance score and possibility tier
        if v_dist is not None:
            if is_direct:
                relevance_score = int(min(99, max(82, 82 + (0.80 - v_dist) * 120)))
                possibility_desc = "Direct Focus"
            else:
                relevance_score = int(min(75, max(52, 52 + (0.88 - v_dist) / 0.08 * 22)))
                possibility_desc = "Adjacent Horizon"
        else:
            relevance_score = 85 if is_direct else 62
            possibility_desc = "Direct Focus" if is_direct else "Adjacent Horizon"

        if fts_rank is not None:
            relevance_score = min(99, relevance_score + 8)

        # Contextual exploration angle
        if any("battery" in t for t in tags):
            explore_angle = "Energy & Materials: Next-gen solid-state & semiconductor integration"
        elif any("nvidia" in t or "gpu" in t or "inference" in t for t in tags):
            explore_angle = "Compute & Acceleration: GPU architecture, parallel inference & silicon engines"
        elif any("agent" in t or "llm" in t or "runtime" in t for t in tags):
            explore_angle = "Autonomous Systems: AI runtime architecture & silicon synergy"
        elif any("space" in t or "physics" in t or "jwst" in t for t in tags):
            explore_angle = "Frontier Science: Astrophysics instrumentation & optical sensor arrays"
        elif any("crypto" in t or "security" in t or "quantum" in t for t in tags):
            explore_angle = "Cybersecurity & Cryptography: Post-quantum standards & hardware security"
        elif any("finance" in t or "euro" in t or "settlement" in t for t in tags):
            explore_angle = "Fintech & Infrastructure: Distributed ledger & interbank clearing systems"
        else:
            top_tags = ", ".join(tags[:3]) if tags else (macro_tag or "Cross-domain")
            explore_angle = f"Adjacent Domain: Cross-cutting developments in {top_tags}"

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

    # Default flat list preserves direct matches first, followed by exploratory matches
    return direct_matches + explore_matches



# 5. Tier 2 Grounded Copilot (Streaming)
@app.post("/api/articles/{article_id}/chat")
def chat_with_article(article_id: str, req: ArticleChatRequest):
    with get_db() as cur:
        cur.execute("SELECT title, clean_text FROM articles WHERE id = %s;", (article_id,))
        art = cur.fetchone()
        if not art:
            raise HTTPException(status_code=404, detail="Article not found")
    
    title, text = art[0], art[1]
    system_msg = {
        "role": "system",
        "content": (
            f"You are a strictly grounded copilot for the article: '{title}'.\n"
            "Answer using ONLY facts from the provided text. If not in the text, say 'The article does not state that.'\n\n"
            f"--- CONTENT ---\n{text[:10000]}"
        )
    }
    bounded_history = [t.model_dump() for t in req.history[-4:]]
    messages = [system_msg, *bounded_history, {"role": "user", "content": req.question}]

    def stream_tokens():
        stream = ai_client.chat.completions.create(
            model=REASONING_MODEL,
            messages=messages,
            stream=True,
            temperature=0.2
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    return StreamingResponse(stream_tokens(), media_type="text/plain")

# 6. Tier 3 Multi-Source Consensus Agent
@app.post("/api/intelligence/synthesize")
def synthesize_sources(req: SynthesizeRequest):
    if len(req.article_ids) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 article IDs.")

    with get_db() as cur:
        cur.execute("""
            SELECT source_feed, title, tldr, clean_text 
            FROM articles WHERE id = ANY(%s::uuid[]);
        """, (req.article_ids,))
        rows = cur.fetchall()

    if len(rows) < 2:
        raise HTTPException(status_code=404, detail="Insufficient articles found.")

    blocks = ""
    for idx, (source, title, tldr, text) in enumerate(rows):
        blocks += f"\n--- SOURCE {idx+1}: [{source}] '{title}' ---\nTLDR: {tldr}\nEXCERPT: {text[:1500]}\n"

    system_msg = {
        "role": "system",
        "content": (
            "You are an investigative news analyst. Compare the provided news sources.\n"
            "Provide:\n"
            "1. Shared Consensus (What all outlets agree on)\n"
            "2. Framing Discrepancies & Contradictions (Where outlets disagree, omit facts, or spin)\n"
            "3. Singular / Unverified Claims"
        )
    }

    def stream_synthesis():
        stream = ai_client.chat.completions.create(
            model=REASONING_MODEL,
            messages=[system_msg, {"role": "user", "content": f"Analyze these sources:\n{blocks}"}],
            stream=True,
            temperature=0.2
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    return StreamingResponse(stream_synthesis(), media_type="text/plain")

# 7. Social Annotations
@app.post("/api/articles/{article_id}/comments")
def post_comment(article_id: str, c: CommentCreate):
    with get_db() as cur:
        cur.execute("""
            INSERT INTO article_comments (article_id, user_id, selected_text, comment_text)
            VALUES (%s, %s, %s, %s)
            RETURNING id, created_at;
        """, (article_id, c.user_id, c.selected_text, c.comment_text))
        res = cur.fetchone()
        return {"id": str(res[0]), "created_at": res[1]}

@app.get("/api/articles/{article_id}/comments")
def get_comments(article_id: str):
    with get_db() as cur:
        cur.execute("""
            SELECT c.id, c.selected_text, c.comment_text, c.created_at, u.username, u.avatar_url
            FROM article_comments c
            JOIN users u ON c.user_id = u.id
            WHERE c.article_id = %s
            ORDER BY c.created_at ASC;
        """, (article_id,))
        rows = cur.fetchall()
        return [
            {"id": str(r[0]), "selected_text": r[1], "comment": r[2], "created_at": r[3], "username": r[4], "avatar_url": r[5]}
            for r in rows
        ]

# 8. Frictionless User Profiles
@app.post("/api/users/login")
def user_quick_login(req: UserLoginRequest):
    raw_name = req.username or req.name or ""
    username = raw_name.strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username cannot be empty")
    with get_db() as cur:
        cur.execute("SELECT id, username, avatar_url FROM users WHERE username = %s;", (username,))
        row = cur.fetchone()
        if row:
            return {"id": str(row[0]), "username": row[1], "avatar_url": row[2]}
        avatar = f"https://avatar.iran.liara.run/public/boy?username={username}"
        cur.execute("""
            INSERT INTO users (username, avatar_url) VALUES (%s, %s)
            RETURNING id, username, avatar_url;
        """, (username, avatar))
        new_row = cur.fetchone()
        return {"id": str(new_row[0]), "username": new_row[1], "avatar_url": new_row[2]}

@app.get("/api/users")
def get_all_users():
    with get_db() as cur:
        cur.execute("SELECT id, username, avatar_url, created_at FROM users ORDER BY created_at ASC;")
        rows = cur.fetchall()
        return [
            {"id": str(r[0]), "username": r[1], "avatar_url": r[2], "created_at": r[3]}
            for r in rows
        ]
