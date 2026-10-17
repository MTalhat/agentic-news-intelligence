import hmac
import hashlib
from datetime import datetime, timezone
from dateutil import parser as date_parser
from fastapi import APIRouter, HTTPException, Header

from db import get_db
from schemas import ScrapedArticleIn
from url_utils import canonicalize_url
from content_cleaner import clean_article_markdown
from routers.deps import INGESTION_SECRET, validate_uuid
from routers.users import get_personalized_feed_data

router = APIRouter(tags=["Articles & Ingestion"])


@router.post("/api/internal/ingest")
def ingest_articles(items: list[ScrapedArticleIn], x_secret: str = Header(...)):
    if not hmac.compare_digest(x_secret.encode("utf-8"), INGESTION_SECRET.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid ingestion secret")

    inserted = 0
    with get_db() as cur:
        for item in items:
            canonical_link = canonicalize_url(item.Link)
            url_hash = hashlib.sha256(canonical_link.encode("utf-8")).hexdigest()
            try:
                pub_date = date_parser.parse(item.Tarih) if item.Tarih else datetime.now(timezone.utc)
                if pub_date.tzinfo is None:
                    pub_date = pub_date.replace(tzinfo=timezone.utc)
            except Exception:
                pub_date = datetime.now(timezone.utc)

            cleaned_content = clean_article_markdown(item.Baslik, item.TamMetin)

            cur.execute("""
                INSERT INTO articles (url_hash, source_feed, title, original_url, author, published_at, clean_text)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (url_hash) DO NOTHING
                RETURNING id;
            """, (url_hash, item.Site, item.Baslik, canonical_link, item.Yazar, pub_date, cleaned_content))
            if cur.fetchone():
                inserted += 1

    return {"received": len(items), "inserted": inserted}


# Static route declared BEFORE dynamic /api/articles/{article_id}
@router.get("/api/articles")
def get_articles(
    page: int = 1, 
    page_size: int = 20, 
    macro: str | None = None,
    user_id: str | None = None
):
    if user_id:
        pdata = get_personalized_feed_data(user_id=user_id, page=page, page_size=page_size)
        return pdata["articles"]

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
                FROM articles 
                WHERE is_processed = TRUE 
                  AND macro_tag IS NOT NULL 
                  AND macro_tag != 'Off-Topic'
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


@router.get("/api/articles/{article_id}")
def get_article(article_id: str):
    valid_id = validate_uuid(article_id, "article_id")
    with get_db() as cur:
        cur.execute("""
            SELECT id, source_feed, title, original_url, author, published_at, clean_text, macro_tag, macro_tags, micro_tags, tldr
            FROM articles WHERE id = %s;
        """, (valid_id,))
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
