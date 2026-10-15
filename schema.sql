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
