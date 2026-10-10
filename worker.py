import time
import json
from db import get_db
from ai import ai_client, FAST_MODEL
from schemas import ArticleEnrichmentOut

SYSTEM_PROMPT = """
You are an RSS news tagging and summarization engine. 
Analyze the input text and extract metadata.
You must respond with valid JSON matching this schema:
{
  "macro_tag": "One of: Tech, AI, Hardware, Automotive, Gaming, Finance, Science",
  "micro_tags": ["lowercase_tag1", "tag2", "tag3"],
  "tldr": "A 2-sentence empirical summary of the news."
}
"""

def process_batch():
    with get_db() as cur:
        cur.execute("""
            SELECT id, title, clean_text 
            FROM articles 
            WHERE is_processed = FALSE 
            ORDER BY created_at ASC 
            LIMIT 5;
        """)
        batch = cur.fetchall()

    if not batch:
        return False

    for art_id, title, text in batch:
        truncated_text = text[:3500]
        prompt = f"Title: {title}\n\nContent:\n{truncated_text}"
        
        try:
            try:
                resp = ai_client.chat.completions.create(
                    model=FAST_MODEL,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    timeout=25.0
                )
            except Exception as fe:
                if "json_schema" in str(fe):
                    resp = ai_client.chat.completions.create(
                        model=FAST_MODEL,
                        temperature=0.0,
                        response_format={"type": "json_schema", "json_schema": {"name": "enrichment", "schema": ArticleEnrichmentOut.model_json_schema()}},
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}
                        ],
                        timeout=25.0
                    )
                else:
                    raise fe
            raw = json.loads(resp.choices[0].message.content)
            parsed = ArticleEnrichmentOut(**raw)
            clean_micro = [t.lower().strip() for t in parsed.micro_tags]

            with get_db() as cur:
                cur.execute("""
                    UPDATE articles 
                    SET macro_tag = %s, micro_tags = %s, tldr = %s, is_processed = TRUE 
                    WHERE id = %s;
                """, (parsed.macro_tag, clean_micro, parsed.tldr, art_id))
            print(f"[Worker Enriched] {title[:30]} -> {parsed.macro_tag} / {clean_micro}")
        except Exception as e:
            print(f"[Worker Error] {art_id}: {e}")
            with get_db() as cur:
                cur.execute("UPDATE articles SET is_processed = TRUE WHERE id = %s;", (art_id,))
    return True

if __name__ == "__main__":
    print("Worker running. Polling for unprocessed articles...")
    while True:
        had_work = process_batch()
        if not had_work:
            time.sleep(3)
