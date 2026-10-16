CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Users
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    avatar_url TEXT,
    preferred_macro_tags TEXT[] DEFAULT '{}',
    preferred_micro_tags TEXT[] DEFAULT '{}',
    onboarding_completed BOOLEAN DEFAULT FALSE,
    interest_embedding vector(384),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Backward-compatible migrations for existing user tables
ALTER TABLE users ADD COLUMN IF NOT EXISTS preferred_macro_tags TEXT[] DEFAULT '{}';
ALTER TABLE users ADD COLUMN IF NOT EXISTS preferred_micro_tags TEXT[] DEFAULT '{}';
ALTER TABLE users ADD COLUMN IF NOT EXISTS onboarding_completed BOOLEAN DEFAULT FALSE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS interest_embedding vector(384);

-- 2. Articles (Normalized from Scraper JSON)
CREATE TABLE IF NOT EXISTS articles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    url_hash VARCHAR(64) UNIQUE NOT NULL,
    source_feed VARCHAR(100) NOT NULL,
    title TEXT NOT NULL,
    original_url TEXT NOT NULL,
    author VARCHAR(255),
    published_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    clean_text TEXT NOT NULL,
    macro_tag VARCHAR(50),
    macro_tags TEXT[],
    micro_tags TEXT[],
    tldr TEXT,
    confidence_score FLOAT,
    on_topic_score FLOAT,
    embedding vector(384),
    is_processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Backward-compatible migrations for existing databases
ALTER TABLE articles ADD COLUMN IF NOT EXISTS confidence_score FLOAT;
ALTER TABLE articles ADD COLUMN IF NOT EXISTS on_topic_score FLOAT;
ALTER TABLE articles ADD COLUMN IF NOT EXISTS macro_tags TEXT[];
ALTER TABLE articles ADD COLUMN IF NOT EXISTS embedding vector(384);


-- 3. Social Paragraph Annotations
CREATE TABLE IF NOT EXISTS article_comments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    article_id UUID REFERENCES articles(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    selected_text TEXT,
    comment_text TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 4. Social Interactions & Likes (Toggleable)
CREATE TABLE IF NOT EXISTS user_article_interactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    article_id UUID REFERENCES articles(id) ON DELETE CASCADE,
    interaction_type VARCHAR(20) NOT NULL, -- 'like', 'bookmark', 'read'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE (user_id, article_id, interaction_type)
);

CREATE INDEX IF NOT EXISTS idx_interactions_user_type ON user_article_interactions (user_id, interaction_type);
CREATE INDEX IF NOT EXISTS idx_interactions_article ON user_article_interactions (article_id);

-- 5. Daily Intelligence Digests Cache (Rotates Every 24 Hours)
CREATE TABLE IF NOT EXISTS daily_intelligence_digests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    digest_date DATE NOT NULL DEFAULT CURRENT_DATE,
    macro_tag VARCHAR(50) NOT NULL,
    cluster_title TEXT NOT NULL,
    article_ids UUID[] NOT NULL,
    tavily_evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
    synthesis_markdown TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE (digest_date, macro_tag)
);

CREATE INDEX IF NOT EXISTS idx_digests_date_macro ON daily_intelligence_digests (digest_date, macro_tag);

-- 6. High-Performance Search & Filter Indexes
CREATE INDEX IF NOT EXISTS idx_articles_url_hash ON articles (url_hash);
CREATE INDEX IF NOT EXISTS idx_articles_is_processed ON articles (is_processed);
CREATE INDEX IF NOT EXISTS idx_articles_macro_tag ON articles (macro_tag);
CREATE INDEX IF NOT EXISTS idx_articles_macro_tags ON articles USING GIN (macro_tags);
CREATE INDEX IF NOT EXISTS idx_articles_micro_tags ON articles USING GIN (micro_tags);
CREATE INDEX IF NOT EXISTS idx_articles_fts ON articles USING GIN (to_tsvector('english', title || ' ' || clean_text));
CREATE INDEX IF NOT EXISTS idx_articles_embedding ON articles USING hnsw (embedding vector_cosine_ops);

-- Seed default test user
INSERT INTO users (id, username, avatar_url)
VALUES ('a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11', 'local_tester', 'https://avatar.iran.liara.run/public')
ON CONFLICT (username) DO NOTHING;

-- ====================================================================
-- 7. V1 TAVILY INTELLIGENCE EXPANSIONS (Nebius x NVIDIA Hackathon)
-- ====================================================================

-- 7.1 Tavily Telemetry Tracking (Endpoint, Latency, Credits, Cache Rate)
CREATE TABLE IF NOT EXISTS tavily_calls (
    id BIGSERIAL PRIMARY KEY,
    endpoint VARCHAR(50) NOT NULL,       -- search, extract, map, crawl, research
    purpose VARCHAR(50) NOT NULL,        -- discovery, support, contradiction, official, evidence_extract, site_watch, deep_check
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

-- 7.2 RSS-less Site Monitoring (Tavily Map & Extract)
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

-- 7.3 Semantic Story Clusters
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

-- 7.4 Verifiable Evidence Cards ("Receipts")
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

-- 7.5 Articles Worker Reliability & Error Tracking
ALTER TABLE articles ADD COLUMN IF NOT EXISTS error_message TEXT;
ALTER TABLE articles ADD COLUMN IF NOT EXISTS retry_count INT DEFAULT 0;

