"""Migration script to clean raw scraped markdown from all database articles.

Removes header navbars, breadcrumbs, footers, related stories, and cookie notices,
ensuring clean_text displays pristine article content on the frontend.
"""

import sys
import logging
from db import get_db
from content_cleaner import clean_article_markdown
from taxonomy import extract_domain_keywords, clean_micro_tags, refine_macro_classification

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("clean_articles_db")

def run_migration():
    logger.info("Starting database markdown cleaning migration...")
    
    with get_db() as cur:
        cur.execute("SELECT id, title, clean_text, micro_tags, macro_tag, macro_tags FROM articles ORDER BY created_at ASC;")
        rows = cur.fetchall()

    logger.info("Found %d articles to evaluate.", len(rows))

    cleaned_count = 0
    total_chars_saved = 0
    tags_corrected = 0

    with get_db() as cur:
        for aid, title, raw_text, old_micros, macro_tag, macro_tags in rows:
            if not raw_text:
                continue

            orig_len = len(raw_text)
            cleaned_text = clean_article_markdown(title, raw_text)
            
            # Recalculate micro_tags on the clean text to fix any false-positives (like 'electric_vehicles' from 'dev')
            computed_micros = clean_micro_tags(extract_domain_keywords(title, cleaned_text))
            
            # Merge with existing tags if valid, but drop false positive 'electric_vehicles' if no EV match in cleaned text
            updated_micros = list(old_micros) if old_micros else []
            if "electric_vehicles" in updated_micros and "electric_vehicles" not in computed_micros:
                updated_micros = [m for m in updated_micros if m != "electric_vehicles"]
                tags_corrected += 1

            updated_macro = macro_tag
            updated_macros = list(macro_tags) if macro_tags else ([macro_tag] if macro_tag else [])
            if updated_macro and updated_macro != "Off-Topic":
                updated_macro, updated_macros = refine_macro_classification(
                    title, cleaned_text, updated_macro, updated_macros
                )

            has_text_changed = (len(cleaned_text) < orig_len)
            has_tags_changed = (
                (updated_micros != (old_micros or []))
                or (updated_macro != macro_tag)
                or (updated_macros != (macro_tags or []))
            )

            if has_text_changed or has_tags_changed:
                cur.execute("""
                    UPDATE articles 
                    SET clean_text = %s, micro_tags = %s, macro_tag = %s, macro_tags = %s
                    WHERE id = %s;
                """, (cleaned_text, updated_micros, updated_macro, updated_macros, aid))
                
                if has_text_changed:
                    cleaned_count += 1
                    total_chars_saved += (orig_len - len(cleaned_text))

    logger.info("=" * 60)
    logger.info("MIGRATION COMPLETED SUCCESSFULLY")
    logger.info("Total articles evaluated: %d", len(rows))
    logger.info("Articles cleaned: %d", cleaned_count)
    logger.info("Boilerplate noise removed: %s characters", f"{total_chars_saved:,}")
    logger.info("False-positive micro-tags corrected: %d", tags_corrected)
    logger.info("=" * 60)

if __name__ == "__main__":
    run_migration()
