CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Users
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    avatar_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

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
    micro_tags TEXT[],
    tldr TEXT,
    is_processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 3. Social Paragraph Annotations
CREATE TABLE IF NOT EXISTS article_comments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    article_id UUID REFERENCES articles(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    selected_text TEXT,
    comment_text TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 4. High-Performance Search & Filter Indexes
CREATE INDEX IF NOT EXISTS idx_articles_url_hash ON articles (url_hash);
CREATE INDEX IF NOT EXISTS idx_articles_is_processed ON articles (is_processed);
CREATE INDEX IF NOT EXISTS idx_articles_macro_tag ON articles (macro_tag);
CREATE INDEX IF NOT EXISTS idx_articles_micro_tags ON articles USING GIN (micro_tags);
CREATE INDEX IF NOT EXISTS idx_articles_fts ON articles USING GIN (to_tsvector('english', title || ' ' || clean_text));

-- Seed default test user
INSERT INTO users (id, username, avatar_url)
VALUES ('a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11', 'local_tester', 'https://avatar.iran.liara.run/public')
ON CONFLICT (username) DO NOTHING;
