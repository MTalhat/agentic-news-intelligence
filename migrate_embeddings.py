"""One-time migration script to generate and persist 384-d dense vector embeddings
for all existing articles in the database.
"""

import time
import logging
from db import get_db
from embedder import get_embedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("migrate_embeddings")

def run_migration():
    embedder = get_embedder()
    logger.info("Embedder initialized on device: %s", embedder.device)

    with get_db() as cur:
        cur.execute("""
            SELECT id, title, tldr, clean_text 
            FROM articles 
            ORDER BY created_at ASC;
        """)
        rows = cur.fetchall()

    logger.info("Found %d articles to embed.", len(rows))
    if not rows:
        logger.info("No articles found to migrate.")
        return

    updated_count = 0
    t0 = time.perf_counter()

    for row in rows:
        art_id, title, tldr, clean_text = row[0], row[1], row[2], row[3]
        
        # Build semantic representation
        text_to_embed = f"{title}. {tldr if tldr else clean_text[:300]}"
        vector = embedder.embed_text(text_to_embed)

        with get_db() as cur:
            cur.execute("""
                UPDATE articles 
                SET embedding = %s 
                WHERE id = %s;
            """, (vector, art_id))
        
        updated_count += 1
        logger.info("[%d/%d] Embedded article: '%s'", updated_count, len(rows), title[:60])

    elapsed = time.perf_counter() - t0
    logger.info("Successfully updated %d articles with 384-d embeddings in %.2fs (avg %.1fms/art).",
                updated_count, elapsed, (elapsed / updated_count) * 1000)

    # Verification query
    with get_db() as cur:
        cur.execute("SELECT count(*) FROM articles WHERE embedding IS NOT NULL;")
        embedded_count = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM articles;")
        total_count = cur.fetchone()[0]

    logger.info("Database verification: %d/%d articles have valid embeddings.", embedded_count, total_count)
    assert embedded_count == total_count, f"Mismatch: {embedded_count} embedded vs {total_count} total articles"

if __name__ == "__main__":
    run_migration()
