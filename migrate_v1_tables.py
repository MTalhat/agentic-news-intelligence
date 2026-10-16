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
"""

def run_migration() -> bool:
    logger.info("Executing V1 Database Migration...")
    try:
        with get_db() as cur:
            cur.execute(V1_DDL)
        logger.info("✓ V1 Database Tables & Indexes successfully verified/created.")
        return True
    except Exception as e:
        logger.warning("Migration failed (database may be offline or initializing): %s", e)
        return False

if __name__ == "__main__":
    success = run_migration()
    sys.exit(0 if success else 1)
