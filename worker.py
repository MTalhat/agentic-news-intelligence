import os
import time
import json
import logging
from db import get_db
from ai import ai_client, FAST_MODEL
from schemas import ArticleEnrichmentOut, ArticleTldrOut
from classifier import LayaClassifier
from embedder import get_embedder
from content_cleaner import clean_markdown_article

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("worker")

ENRICHMENT_ENGINE = os.getenv("ENRICHMENT_ENGINE", "hybrid").lower()

SYSTEM_PROMPT_FULL = """
You are an RSS news tagging and summarization engine. 
Analyze the input text and extract metadata.
You must respond with valid JSON matching this schema:
{
  "macro_tag": "One of: Tech, AI, Hardware, Automotive, Gaming, Finance, Science",
  "micro_tags": ["lowercase_tag1", "tag2", "tag3"],
  "tldr": "A 2-sentence empirical summary of the news."
}
"""

SYSTEM_PROMPT_TLDR = """
You are an RSS news summarization engine. 
The article topic has already been classified. Provide a strictly grounded 2-sentence empirical summary of the news.
You must respond with valid JSON matching this schema:
{
  "tldr": "A 2-sentence empirical summary of the news."
}
"""

# Lazy-loaded classifier instance
_classifier = None

def get_classifier():
    global _classifier
    if _classifier is None and ENRICHMENT_ENGINE == "hybrid":
        _classifier = LayaClassifier()
    return _classifier

def init_db_schema():
    """Ensure optional observability, error tracking, and vector columns exist in the articles table."""
    try:
        with get_db() as cur:
            cur.execute("""
                ALTER TABLE articles 
                ADD COLUMN IF NOT EXISTS confidence_score FLOAT,
                ADD COLUMN IF NOT EXISTS on_topic_score FLOAT,
                ADD COLUMN IF NOT EXISTS macro_tags TEXT[],
                ADD COLUMN IF NOT EXISTS embedding vector(384),
                ADD COLUMN IF NOT EXISTS error_message TEXT,
                ADD COLUMN IF NOT EXISTS retry_count INT DEFAULT 0;
                CREATE INDEX IF NOT EXISTS idx_articles_macro_tags ON articles USING GIN (macro_tags);
                CREATE INDEX IF NOT EXISTS idx_articles_embedding ON articles USING hnsw (embedding vector_cosine_ops);
            """)
        logger.info("Database schema columns verified (confidence_score, on_topic_score, macro_tags, embedding, error_message, retry_count).")
    except Exception as e:
        logger.warning("Could not verify DB columns: %s", e)

def clean_and_parse_json(content: str) -> dict:
    """Robust JSON extraction that strips markdown code fences and whitespace."""
    text = (content or "").strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    
    # Locate outer brackets
    s = text.find("{")
    e = text.rfind("}")
    if s != -1 and e != -1 and e > s:
        text = text[s:e+1]
    return json.loads(text)

def generate_tldr_only(title: str, text: str, macro_tag: str, micro_tags: list) -> str:
    """Fast targeted LLM call that only generates the 2-sentence TL;DR summary."""
    truncated = text[:3500]
    prompt = f"Category: {macro_tag} ({', '.join(micro_tags)})\nTitle: {title}\n\nContent:\n{truncated}"
    
    # 1. Fast direct prompt (5-8x faster on local LM Studio without GBNF grammar lag)
    try:
        resp = ai_client.chat.completions.create(
            model=FAST_MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_TLDR},
                {"role": "user", "content": prompt}
            ],
            timeout=50.0
        )
        data = clean_and_parse_json(resp.choices[0].message.content)
        tldr = data.get("tldr", "").strip()
        if tldr:
            return tldr
    except Exception as e:
        logger.warning("[TLDR Fast Call Retry] %s. Falling back to structured schema...", e)

    # 2. Structured json_schema fallback
    try:
        resp = ai_client.chat.completions.create(
            model=FAST_MODEL,
            temperature=0.0,
            response_format={
                "type": "json_schema", 
                "json_schema": {"name": "tldr_only", "schema": ArticleTldrOut.model_json_schema()}
            },
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_TLDR},
                {"role": "user", "content": prompt}
            ],
            timeout=50.0
        )
        data = clean_and_parse_json(resp.choices[0].message.content)
        return data.get("tldr", "").strip()
    except Exception as fe:
        logger.error("[TLDR Error] Failed to generate summary: %s", fe)
        # Safe fallback summary from content rather than dropping the article
        first_sentence = text.strip().split(".")[0] if text else title
        return f"{title}. {first_sentence}."

def generate_full_enrichment(title: str, text: str, hint: str = "") -> tuple[str, list, str]:
    """Fallback LLM call that extracts macro_tag, micro_tags, and tldr in a single prompt."""
    truncated = text[:3500]
    prompt = f"{hint}Title: {title}\n\nContent:\n{truncated}".strip()
    
    try:
        resp = ai_client.chat.completions.create(
            model=FAST_MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_FULL},
                {"role": "user", "content": prompt}
            ],
            timeout=50.0
        )
        raw = clean_and_parse_json(resp.choices[0].message.content)
    except Exception:
        resp = ai_client.chat.completions.create(
            model=FAST_MODEL,
            temperature=0.0,
            response_format={
                "type": "json_schema", 
                "json_schema": {"name": "enrichment", "schema": ArticleEnrichmentOut.model_json_schema()}
            },
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_FULL},
                {"role": "user", "content": prompt}
            ],
            timeout=50.0
        )
        raw = clean_and_parse_json(resp.choices[0].message.content)

    parsed = ArticleEnrichmentOut(**raw)
    from taxonomy import clean_micro_tags
    clean_micro = clean_micro_tags(parsed.micro_tags, max_tags=6)
    return parsed.macro_tag, clean_micro, parsed.tldr

def save_article_enrichment(art_id: str, macro: str, macro_tags: list, micro: list, tldr: str, title: str = "", conf: float = None, on_topic: float = None):
    """Saves enriched article to DB, generating and storing 384-d dense vector embeddings."""
    embedding = None
    if title or tldr:
        try:
            embedder = get_embedder()
            embedding = embedder.embed_text(f"{title}. {tldr}")
        except Exception as ee:
            logger.warning("Could not generate embedding for article %s: %s", art_id, ee)

    with get_db() as cur:
        try:
            cur.execute("""
                UPDATE articles 
                SET macro_tag = %s, macro_tags = %s, micro_tags = %s, tldr = %s, 
                    confidence_score = %s, on_topic_score = %s, embedding = %s, is_processed = TRUE 
                WHERE id = %s;
            """, (macro, macro_tags, micro, tldr, conf, on_topic, embedding, art_id))
        except Exception:
            cur.execute("""
                UPDATE articles 
                SET macro_tag = %s, micro_tags = %s, tldr = %s, embedding = %s, is_processed = TRUE 
                WHERE id = %s;
            """, (macro, micro, tldr, embedding, art_id))


def process_batch() -> bool:
    with get_db() as cur:
        cur.execute("""
            SELECT id, title, clean_text, COALESCE(retry_count, 0)
            FROM articles 
            WHERE is_processed = FALSE 
              AND (retry_count IS NULL OR retry_count < 3)
            ORDER BY created_at ASC 
            LIMIT 5
            FOR UPDATE SKIP LOCKED;
        """)
        batch = cur.fetchall()

    if not batch:
        return False

    classifier = get_classifier()
    has_classifier = classifier is not None and classifier.is_available()

    for art_id, title, text, r_count in batch:
        try:
            # Defensive cleaning to strip tail widgets before Laya triage & embedding
            cleaned_body, _ = clean_markdown_article(text or "")
            safe_text = cleaned_body[:20000] if len(cleaned_body) >= 200 else (text or "")[:20000]
            t0 = time.perf_counter()
            triage = None

            # --- STAGE 1: Fast Sub-100ms SLM Triage (if enabled) ---
            if has_classifier:
                triage = classifier.classify(title, safe_text)

            if triage is not None:
                # 1A. Noise / Spam Filter Gate
                if triage.is_off_topic:
                    save_article_enrichment(
                        art_id, 
                        macro="Off-Topic", 
                        macro_tags=["Off-Topic"],
                        micro=["off_topic"], 
                        tldr="Filtered out: Article does not match monitored news topics.",
                        title=title,
                        conf=triage.confidence,
                        on_topic=triage.on_topic_score
                    )
                    logger.info(
                        "[Worker Filtered] %s -> Off-Topic (Score: %.2f in %.0fms)", 
                        title[:35], triage.on_topic_score, triage.latency_ms
                    )
                    continue

                # 1B. SLM-Led Triage -> Generative TL;DR Only
                tldr = generate_tldr_only(title, safe_text, triage.macro_tag, triage.micro_tags)
                save_article_enrichment(
                    art_id,
                    macro=triage.macro_tag,
                    macro_tags=triage.macro_tags,
                    micro=triage.micro_tags,
                    tldr=tldr,
                    title=title,
                    conf=triage.confidence,
                    on_topic=triage.on_topic_score
                )
                total_ms = (time.perf_counter() - t0) * 1000
                logger.info(
                    "[Worker Hybrid] %s -> %s / %s (Laya: %.0fms | Conf: %.2f | Total: %.0fms)",
                    title[:30], triage.macro_tags, triage.micro_tags, triage.latency_ms, triage.confidence, total_ms
                )
                continue

            # --- STAGE 2: Fallback Standard LLM Enrichment (only if Laya completely disabled) ---
            macro, micro, tldr = generate_full_enrichment(title, safe_text)
            save_article_enrichment(art_id, macro=macro, macro_tags=[macro], micro=micro, tldr=tldr, title=title)
            logger.info("[Worker Legacy Enriched] %s -> %s / %s", title[:30], macro, micro)

        except Exception as e:
            err_msg = str(e)[:450]
            current_retries = (r_count or 0) + 1
            give_up = current_retries >= 3
            logger.error("[Worker Error] Article %s (attempt %d/3): %s", art_id, current_retries, err_msg)
            with get_db() as cur:
                cur.execute("""
                    UPDATE articles 
                    SET retry_count = %s,
                        error_message = %s,
                        is_processed = %s
                    WHERE id = %s;
                """, (current_retries, err_msg, give_up, art_id))
    
    return True

if __name__ == "__main__":
    init_db_schema()
    classifier = get_classifier()
    mode_str = "HYBRID (Laya SLM Triage + Generative TLDR)" if (classifier and classifier.is_available()) else "LEGACY LLM-ONLY"
    logger.info("Worker started in %s mode. Polling for unprocessed articles...", mode_str)
    
    while True:
        had_work = process_batch()
        if not had_work:
            time.sleep(3)
