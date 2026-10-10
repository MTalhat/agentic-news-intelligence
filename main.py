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
            cur.execute("""
                SELECT id, source_feed, title, original_url, author, published_at, macro_tag, micro_tags, tldr
                FROM articles WHERE is_processed = TRUE AND macro_tag ILIKE %s
                ORDER BY published_at DESC LIMIT %s OFFSET %s;
            """, (macro, page_size, offset))
        else:
            cur.execute("""
                SELECT id, source_feed, title, original_url, author, published_at, macro_tag, micro_tags, tldr
                FROM articles WHERE is_processed = TRUE
                ORDER BY published_at DESC LIMIT %s OFFSET %s;
            """, (page_size, offset))
        rows = cur.fetchall()

    return [
        {
            "id": str(r[0]), "source": r[1], "title": r[2], "url": r[3],
            "author": r[4], "published_at": r[5], "macro_tag": r[6],
            "micro_tags": r[7], "tldr": r[8]
        }
        for r in rows
    ]

# 3. Single Article Full Detail
@app.get("/api/articles/{article_id}")
def get_article(article_id: str):
    with get_db() as cur:
        cur.execute("""
            SELECT id, source_feed, title, original_url, author, published_at, clean_text, macro_tag, micro_tags, tldr
            FROM articles WHERE id = %s;
        """, (article_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Article not found")
        return {
            "id": str(row[0]), "source": row[1], "title": row[2], "url": row[3],
            "author": row[4], "published_at": row[5], "clean_text": row[6],
            "macro_tag": row[7], "micro_tags": row[8], "tldr": row[9]
        }

# 4. Agentic Natural Language Search
@app.get("/api/search")
def natural_search(q: str = Query(..., min_length=2)):
    # Step A: Tier 1 extracts structured intent
    micro_tags = []
    macro_tag = None
    try:
        search_messages = [
            {
                "role": "system",
                "content": (
                    "Convert this news query into 2-5 lowercase search tags. "
                    "Return JSON: {\"macro_tag\": \"Tech|AI|Hardware|Automotive|Gaming|Finance|Science|None\", "
                    "\"micro_tags\": [\"tag1\", \"tag2\"]}"
                )
            },
            {"role": "user", "content": q}
        ]
        try:
            resp = ai_client.chat.completions.create(
                model=FAST_MODEL,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=search_messages,
                timeout=10.0
            )
        except Exception as fe:
            if "json_schema" in str(fe):
                resp = ai_client.chat.completions.create(
                    model=FAST_MODEL,
                    temperature=0.0,
                    response_format={"type": "json_schema", "json_schema": {"name": "query_intent", "schema": QueryIntentOut.model_json_schema()}},
                    messages=search_messages,
                    timeout=10.0
                )
            else:
                raise fe
        data = json.loads(resp.choices[0].message.content)
        parsed = QueryIntentOut(**data)
        micro_tags = [t.lower().strip() for t in parsed.micro_tags if t.strip()]
        if parsed.macro_tag and parsed.macro_tag.lower() not in ("none", "null"):
            macro_tag = parsed.macro_tag.strip()
    except Exception:
        micro_tags = [w.lower() for w in q.split() if len(w) > 2]
        macro_tag = None

    # Step B: Match against PostgreSQL with Tag Likeness & Macro Alignment
    results = []
    with get_db() as cur:
        tag_patterns = [f"%{t}%" for t in micro_tags] if micro_tags else []
        macro_pattern = f"%{macro_tag}%" if macro_tag else None
        q_pattern = f"%{q.strip()}%"

        cur.execute("""
            SELECT id, source_feed, title, tldr, micro_tags, original_url
            FROM articles 
            WHERE is_processed = TRUE 
              AND (
                  (%s::text IS NOT NULL AND macro_tag ILIKE %s)
                  OR (cardinality(%s::text[]) > 0 AND EXISTS (
                      SELECT 1 FROM unnest(articles.micro_tags) AS t 
                      WHERE t ILIKE ANY(%s::text[])
                  ))
                  OR array_to_string(articles.micro_tags, ' ') ILIKE %s
                  OR title ILIKE %s
              )
            ORDER BY 
                CASE 
                    WHEN %s::text IS NOT NULL AND macro_tag ILIKE %s THEN 1
                    WHEN title ILIKE %s THEN 2
                    ELSE 3
                END,
                published_at DESC 
            LIMIT 8;
        """, (
            macro_pattern, macro_pattern,
            tag_patterns, tag_patterns,
            q_pattern,
            q_pattern,
            macro_pattern, macro_pattern,
            q_pattern
        ))
        results = cur.fetchall()

        # Step C: Full-text fallback if likeness yields 0 results
        if not results:
            cur.execute("""
                SELECT id, source_feed, title, tldr, micro_tags, original_url
                FROM articles WHERE is_processed = TRUE 
                  AND to_tsvector('english', title || ' ' || clean_text) @@ plainto_tsquery('english', %s)
                ORDER BY published_at DESC LIMIT 6;
            """, (q,))
            results = cur.fetchall()

    return [
        {"id": str(r[0]), "source": r[1], "title": r[2], "tldr": r[3], "micro_tags": r[4], "url": r[5]}
        for r in results
    ]

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
