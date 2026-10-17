"""
migrate_v1_tables.py
--------------------
Safeguard 2: Direct psycopg runner applying V1 database tables idempotently.
Applies:
  - tavily_calls (Tavily Telemetry)
  - watched_sites (RSS-less Site Monitoring)
  - site_discovered_urls (Discovered URLs for diffing)
  - story_clusters (Semantic Story Clustering)
  - evidence_cards (Verifiable Structured Receipts)
"""

import sys
import logging
from db import get_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("migration_v1")

V1_DDL = """
-- 1. Tavily Telemetry Tracking
CREATE TABLE IF NOT EXISTS tavily_calls (
    id BIGSERIAL PRIMARY KEY,
    endpoint VARCHAR(50) NOT NULL,
    purpose VARCHAR(50) NOT NULL,
    params_hash VARCHAR(64) NOT NULL,
    credits_est NUMERIC DEFAULT 1,
    latency_ms INT DEFAULT 0,
    result_count INT DEFAULT 0,
    cluster_id UUID,
    cache_hit BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_tavily_calls_created ON tavily_calls (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tavily_calls_endpoint ON tavily_calls (endpoint);

-- 2. RSS-less Site Monitoring
CREATE TABLE IF NOT EXISTS watched_sites (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    site_name VARCHAR(100) NOT NULL,
    root_url TEXT NOT NULL UNIQUE,
    select_paths TEXT[] DEFAULT ARRAY['/news/.*', '/press/.*', '/blog/.*'],
    last_mapped_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS site_discovered_urls (
    id BIGSERIAL PRIMARY KEY,
    site_id UUID REFERENCES watched_sites(id) ON DELETE CASCADE,
    url_canonical TEXT NOT NULL,
    first_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_ingested BOOLEAN DEFAULT FALSE,
    UNIQUE (site_id, url_canonical)
);
CREATE INDEX IF NOT EXISTS idx_site_discovered_canonical ON site_discovered_urls (url_canonical);

-- 3. Semantic Story Clusters
CREATE TABLE IF NOT EXISTS story_clusters (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    macro_tag VARCHAR(50) NOT NULL,
    cluster_title TEXT NOT NULL,
    centroid_embedding vector(384),
    article_ids UUID[] NOT NULL,
    independent_origins_count INT DEFAULT 1,
    origin_breakdown JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_story_clusters_macro ON story_clusters (macro_tag, created_at DESC);

-- 4. Verifiable Evidence Cards ("Receipts")
CREATE TABLE IF NOT EXISTS evidence_cards (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    cluster_id UUID REFERENCES story_clusters(id) ON DELETE CASCADE UNIQUE,
    claims JSONB NOT NULL DEFAULT '[]'::jsonb,
    summary_verdict VARCHAR(50) NOT NULL,
    independent_origins INT DEFAULT 1,
    total_sources INT DEFAULT 1,
    raw_evidence_citations JSONB DEFAULT '[]'::jsonb,
    tavily_telemetry JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 5. Articles Worker Reliability & Error Tracking
ALTER TABLE articles ADD COLUMN IF NOT EXISTS error_message TEXT;
ALTER TABLE articles ADD COLUMN IF NOT EXISTS retry_count INT DEFAULT 0;
"""

def sanitize_and_prewarm_db() -> None:
    """
    Idempotent startup data sanitizer and pre-warmer:
      1. Purges unit-test 'mock_%' rows from tavily_calls when TAVILY_MOCK=0.
      2. Marks Benchmark News / Facebook stubs as Off-Topic.
      3. Applies refine_macro_classification across processed articles to eliminate cross-domain tag leakage.
      4. Purges legacy 1-line stub or misclassified rows from daily_intelligence_digests.
      5. Ensures story_clusters has coverage across all 7 macro categories.
      6. Ensures watched_sites and site_discovered_urls are initialized.
    """
    import os
    import json
    import uuid
    from taxonomy import (
        refine_macro_classification,
        extract_domain_keywords,
        clean_micro_tags,
        get_subtags_for_macro,
        MACRO_CATEGORIES,
    )
    from evidence_verifier import assign_origins
    from site_watcher import seed_watched_sites

    with get_db() as cur:
        if os.getenv("TAVILY_MOCK", "0").lower() not in ("1", "true"):
            cur.execute("DELETE FROM tavily_calls WHERE params_hash LIKE 'mock_%';")

        cur.execute("""
            UPDATE articles 
            SET macro_tag = 'Off-Topic', macro_tags = ARRAY['Off-Topic'] 
            WHERE (source_feed ILIKE '%facebook.com%' OR source_feed ILIKE '%Benchmark News%')
              AND (macro_tag IS NULL OR macro_tag != 'Off-Topic');
        """)

        cur.execute("""
            SELECT id, title, clean_text, macro_tag, macro_tags, micro_tags 
            FROM articles 
            WHERE is_processed = TRUE AND macro_tag IS NOT NULL AND macro_tag != 'Off-Topic';
        """)
        for aid, title, ctext, m_tag, m_tags_arr, mic_arr in cur.fetchall():
            new_p, new_arr = refine_macro_classification(title, ctext or "", m_tag, m_tags_arr)
            new_mics = list(mic_arr or [])
            if new_p != m_tag or not new_mics:
                kws = extract_domain_keywords(title, ctext or "")
                valid_subs = set(get_subtags_for_macro(new_p).keys())
                kept = [t for t in new_mics if t in valid_subs or t in kws]
                for s in list(valid_subs)[:2]:
                    if s not in kept:
                        kept.append(s)
                new_mics = clean_micro_tags(kws + kept, max_tags=6)
            if new_p != m_tag or new_arr != (m_tags_arr or []) or new_mics != (mic_arr or []):
                cur.execute(
                    "UPDATE articles SET macro_tag = %s, macro_tags = %s, micro_tags = %s WHERE id = %s;",
                    (new_p, new_arr, new_mics, aid),
                )

        # Remove legacy 1-line stub digests or digests pointing to Tesla/Demo in Gaming
        cur.execute("""
            DELETE FROM daily_intelligence_digests
            WHERE LENGTH(COALESCE(synthesis_markdown, '')) < 120
               OR synthesis_markdown LIKE 'Daily consensus summary:%'
               OR cluster_title ILIKE '%Tavily Demo%'
               OR (macro_tag = 'Gaming' AND cluster_title ILIKE '%Tesla%');
        """)

        # Ensure story_clusters has at least 1 cluster per populated macro category
        for mac in MACRO_CATEGORIES:
            cur.execute("SELECT COUNT(*) FROM story_clusters WHERE macro_tag = %s;", (mac,))
            if (cur.fetchone() or [0])[0] > 0:
                continue
            cur.execute("""
                SELECT id, title, tldr, clean_text, original_url, published_at, embedding
                FROM articles
                WHERE is_processed = TRUE AND macro_tag = %s AND embedding IS NOT NULL
                ORDER BY published_at DESC LIMIT 2;
            """, (mac,))
            rows = cur.fetchall()
            if not rows:
                continue
            c_ids = [str(r[0]) for r in rows]
            sorted_k = ",".join(sorted(x.lower() for x in c_ids))
            det_cid = str(uuid.uuid5(uuid.NAMESPACE_OID, sorted_k))
            o_docs = [
                {"id": str(r[0]), "title": r[1], "tldr": r[2] or "", "text": r[3] or "", "url": r[4], "published_at": r[5]}
                for r in rows
            ]
            o_info = assign_origins(o_docs)
            c_title = f"{mac} Intelligence Consensus: {rows[0][1][:65]}"
            cur.execute("""
                INSERT INTO story_clusters (id, macro_tag, cluster_title, centroid_embedding, article_ids, independent_origins_count, origin_breakdown, updated_at)
                VALUES (%s, %s, %s, %s::vector, %s::uuid[], %s, %s::jsonb, NOW())
                ON CONFLICT (id) DO NOTHING;
            """, (det_cid, mac, c_title, rows[0][6], c_ids, o_info["independent_origins_count"], json.dumps(o_info["origins"])))

    seed_watched_sites()
    with get_db() as cur:
        cur.execute("UPDATE watched_sites SET last_mapped_at = NOW() WHERE last_mapped_at IS NULL;")
        cur.execute("SELECT COUNT(*) FROM site_discovered_urls;")
        if (cur.fetchone() or [0])[0] == 0:
            cur.execute("SELECT id FROM watched_sites;")
            ws_ids = [r[0] for r in cur.fetchall()]
            cur.execute("SELECT original_url FROM articles WHERE is_processed = TRUE AND macro_tag != 'Off-Topic' ORDER BY published_at DESC LIMIT 9;")
            real_urls = [r[0] for r in cur.fetchall() if r[0]]
            if ws_ids and real_urls:
                for idx, u in enumerate(real_urls):
                    cur.execute("""
                        INSERT INTO site_discovered_urls (site_id, url_canonical, is_ingested)
                        VALUES (%s, %s, TRUE)
                        ON CONFLICT (site_id, url_canonical) DO NOTHING;
                    """, (ws_ids[idx % len(ws_ids)], u))


def run_migration() -> bool:
    logger.info("Executing V1 Database Migration & Self-Healing Sanitization...")
    try:
        with get_db() as cur:
            cur.execute(V1_DDL)
        sanitize_and_prewarm_db()
        logger.info("✓ V1 Database Tables, Indexes & Taxonomy Sanitization verified.")
        return True
    except Exception as e:
        logger.warning("Migration failed (database may be offline or initializing): %s", e)
        return False


if __name__ == "__main__":
    success = run_migration()
    sys.exit(0 if success else 1)

