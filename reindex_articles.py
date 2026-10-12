"""Re-index all existing articles in the database using LayaClassifier.

Ensures that all articles have standardized taxonomy tags (e.g. TSMC gets 'semiconductors',
Steam Deck gets 'hardware_consoles', etc.) matching taxonomy.py.
"""

import sys
import time
from db import get_db
from classifier import LayaClassifier

sys.stdout.reconfigure(encoding="utf-8")

def reindex_all():
    print("=" * 70)
    print("       RE-INDEXING DATABASE ARTICLES WITH LAYA TAXONOMY")
    print("=" * 70)

    classifier = LayaClassifier()
    if not classifier.is_available():
        print("ERROR: LayaClassifier is not available.")
        return

    with get_db() as cur:
        cur.execute("SELECT id, title, clean_text, macro_tag, micro_tags FROM articles ORDER BY created_at ASC;")
        articles = cur.fetchall()

    print(f"\nFound {len(articles)} articles to re-index.\n")

    updated = 0
    for art_id, title, text, old_macro, old_micro in articles:
        t0 = time.perf_counter()
        res = classifier.classify(title, text)
        latency = (time.perf_counter() - t0) * 1000

        if res:
            new_macro = res.macro_tag
            new_macro_tags = res.macro_tags
            new_micro = res.micro_tags
            conf = res.confidence
            on_topic = res.on_topic_score

            with get_db() as cur:
                try:
                    cur.execute("""
                        UPDATE articles 
                        SET macro_tag = %s, macro_tags = %s, micro_tags = %s, 
                            confidence_score = %s, on_topic_score = %s 
                        WHERE id = %s;
                    """, (new_macro, new_macro_tags, new_micro, conf, on_topic, art_id))
                except Exception:
                    cur.execute("""
                        UPDATE articles 
                        SET macro_tag = %s, micro_tags = %s 
                        WHERE id = %s;
                    """, (new_macro, new_micro, art_id))

            updated += 1
            print(f"[{updated}/{len(articles)}] Re-Indexed in {latency:.1f}ms:")
            print(f"    Title: {title[:65]}...")
            print(f"    Old: [{old_macro}] {old_micro}")
            print(f"    New: Macro={new_macro_tags} | Micro={new_micro} (Conf: {conf:.2f} | OnTopic: {on_topic:.2f})\n")

    print("=" * 70)
    print(f"SUCCESS: Re-indexed {updated} articles with multi-macro and rich micro taxonomy!")
    print("=" * 70)


if __name__ == "__main__":
    reindex_all()
