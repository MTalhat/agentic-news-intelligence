# AGENTS.md — System Architecture & Codebase Manifest

> **Repository**: `MTalhat/agentic-news-intelligence`  
> **Stack**: Python 3.11+, FastAPI, PostgreSQL 16 (`pgvector`), PyTorch, `sentence-transformers`, `laya` SLM, Nebius / LM Studio / OpenRouter runtimes.  
> **Status**: Verified by 9-Part End-to-End Regression Suite (`tests/comprehensive_test_evaluation.py`).

---

## 1. Hackathon Project Profile & Mission

### 1.1 Executive Summary
**Agentic News Intelligence** is an autonomous, production-grade intelligence newsroom designed to solve the noise, fragility, and confirmation bias of modern information aggregation. Instead of relying on brittle web scrapers or naive LLM wrappers, it combines an always-on ingestion pipeline, sub-100ms local decision SLM triage, sub-50ms hybrid vector search, and an investigative multi-source synthesis agent that cross-checks news against live web consensus.

* **Target Event**: Nebius - NVIDIA Hackathon
* **Target Tracks**:
  * **Best Apps & Agents** (Primary)
  * **Best Use of Tavily** ($3,000 Sponsor Prize Track)
* **Core Philosophy**: Cost-effective local SLM triage at the edge + high-throughput cloud reasoning on Nebius Token Factory + independent live web verification via Tavily.

---

### 1.2 The Problem We Solve
1. **The Ingestion & Scraper Reality Wall**: Traditional scrapers break constantly due to CDN anti-bot blocks, fragile CSS/DOM selectors, and low yields (~40%). Moreover, raw webpage dumps flood vector databases with CMS junk (author bios, social buttons, "Trending Stories" widgets) that pollute embeddings and trigger lexical false positives.
2. **The LLM Token & Cost Bottleneck**: Running 70B parameter models or cloud LLM APIs over every raw incoming news article is prohibitively slow and expensive.
3. **Circular Confirmation Bias in AI Fact-Checking**: Standard AI summarizers read one outlet's claim and search the web only to find the exact same article syndicated elsewhere, blindly declaring claims "verified."

---

### 1.3 How the System Works (The 4-Pillar Solution)
1. **Autonomous, Clutter-Free Ingestion**:
   * Uses Tavily's dedicated news and finance search engine with `include_raw_content="markdown"` across 7 taxonomy categories (`Tech`, `AI`, `Hardware`, `Automotive`, `Gaming`, `Finance`, `Science`).
   * Passes content through a custom, zero-dependency 4-pass Markdown cleaner (`content_cleaner.py`) that uses boundary sentinel detection to strip tail link dumps and promo callouts while preserving 100% of the core reporting.
   * Strips tracking parameters (`utm_*`, `fbclid`) and hashes canonical URLs with SHA-256 for zero-duplicate idempotency.

2. **Sub-100ms Edge SLM Triage (`Laya`)**:
   * Incoming articles hit a local SLM decision model (`convaiinnovations/laya`) that executes macro/micro taxonomy tagging and noise gating in **under 100ms**.
   * Low-confidence edge cases escalate to a generative reasoning model, while verified on-topic articles immediately receive a grounded 2-sentence empirical summary, slashing token usage by over 80%.

3. **Sub-50ms 3-Tier Hybrid RRF Retrieval**:
   * Embeds articles with 384-dimensional dense vectors (`all-MiniLM-L6-v2`) indexed via PostgreSQL 16 `pgvector` (HNSW).
   * Fuses Dense Cosine Distance, BM25 Full-Text Search (`websearch_to_tsquery`), and GIN tag likeness into a single Reciprocal Rank Fusion (RRF) score.
   * Segregates results into **Direct Matches** ($d \le 0.68$ or FTS/tag hits), dynamic **"Want to Explore?"** horizons ($0.68 < d < 0.78$), and an optional **Live Web Horizon** (`live_web=true`) without adding latency to the local vector engine.

4. **Investigative Multi-Source Synthesis & Grounded Fact-Checking**:
   * When comparing multiple articles, the Tier 3 synthesis agent (`POST /api/intelligence/synthesize`) extracts core assertions using a fast model.
   * Runs `web_verify()` with **independent domain exclusion (`exclude_domains`)** to force Tavily to locate third-party corroboration rather than circular repeats.
   * Streams a structured intelligence brief over Server-Sent Events (SSE) featuring claim-level verdicts:
     * `[SUPPORTED]` — corroborated by independent third-party outlets (`[T1]`, `[T2]`).
     * `[CONTRADICTED]` — disputed by live consensus data.
     * `[UNVERIFIED]` — claimed by only one outlet with zero outside confirmation.

---

### 1.4 Technology Stack & Track Alignment

| Component | Technology | Role in Hackathon Architecture |
|---|---|---|
| **Inference Runtime** | **Nebius Token Factory** | Hosts NVIDIA Nemotron 70B & Llama 3.1 8B for deep synthesis and fast claim extraction. |
| **Edge Decision SLM** | **Laya (`convaiinnovations/laya`)** | Executes sub-100ms macro/micro classification & noise gating. |
| **Web Intelligence** | **Tavily Search API** | Autonomous multi-category ingestion & independent domain-excluded fact-checking. |
| **Vector Database** | **PostgreSQL 16 + `pgvector`** | HNSW cosine indexing, BM25 full-text indexing, and atomic JSONB metadata storage. |
| **Embeddings** | **`sentence-transformers/all-MiniLM-L6-v2`** | Generates 384-dimensional dense vectors for sub-50ms hybrid semantic search. |

---

## 2. System Overview

**Agentic News Intelligence** is a real-time news aggregation, asynchronous AI enrichment, and multi-agent synthesis platform. It solves the performance and accuracy bottlenecks of traditional RSS readers through a **hybrid multi-tier pipeline**:
1. **Sub-100ms SLM Triage (`Laya`)**: Classifies macro and micro taxonomy tags, filters off-topic noise, and routes articles based on confidence thresholds without incurring expensive autoregressive LLM calls.
2. **Generative Grounded Summarization**: Produces empirical, strictly grounded 2-sentence TL;DR summaries via fast cloud/local reasoning models.
3. **Dense Vector & Reciprocal Rank Fusion Search**: Combines 384-dimensional dense vectors (HNSW index), PostgreSQL BM25 Full-Text Search (`websearch_to_tsquery`), and tag substring likeness into a unified RRF score with segregated **Direct Focus** and **Want to Explore?** tiers. Empirically measured search latency is **25–65 ms**.
4. **Grounded Copilot & Multi-Source Synthesis**: Real-time Server-Sent Events (SSE) streaming Q&A strictly bound to article ground truth with hallucination guardrails, and comparative multi-source synthesis highlighting consensus vs. framing discrepancies.
5. **Autonomous Web Sourcing & Independent Verification (`Tavily`)**: Continuously harvests full-markdown breaking news across 7 macro taxonomy categories directly into the Laya triage pipeline, while providing independent third-party web cross-verification (`exclude_domains`) for Tier 3 synthesis with structured claim verdicts (`[SUPPORTED]`, `[CONTRADICTED]`, `[UNVERIFIED]`).

---

## 3. File-by-File Manifest

```text
rss-backend/
├── main.py                  # Primary FastAPI API Gateway (~65 lines, CORS & router wiring)
├── routers/                 # Modular Domain APIRouter Package
│   ├── __init__.py          # Router exports (articles, search, intelligence, users, comments)
│   ├── deps.py              # Shared dependencies (INGESTION_SECRET, validate_uuid, parse_vector)
│   ├── articles.py          # Feed retrieval, article detail & HMAC-protected ingestion
│   ├── search.py            # 3-Tier Hybrid RRF search engine & Live Web Horizon
│   ├── intelligence.py      # Copilot SSE, Multi-Source Synthesis SSE, Daily Digest & Tavily Ops
│   ├── users.py             # User onboarding, preferences, EMA interactions & personalized feed
│   └── comments.py          # Contextual paragraph annotations & taxonomy dictionary
├── worker.py                # Asynchronous RSS Ingestion & AI Enrichment Worker (SKIP LOCKED)
├── classifier.py            # Local Decision SLM (convaiinnovations/laya)
├── embedder.py              # Dense Vector Embedding Engine (sentence-transformers/all-MiniLM-L6-v2)
├── taxonomy.py              # Domain Taxonomy, Tagging Criteria & Keyword Normalization
├── content_cleaner.py       # Generalized 4-Pass Markdown Article Sanitizer & Boilerplate Pruner
├── tavily_service.py        # 5-Endpoint Tavily Client (Search, Extract, Map, Crawl, Research) & Telemetry
├── evidence_verifier.py     # Verifiable Receipts Orchestrator (Span Grounding & Wire Origin Analysis)
├── site_watcher.py          # RSS-less Newsroom Monitoring Daemon (Tavily Map & Extract)
├── ingest_tavily.py         # Autonomous Tavily Ingestion Daemon Across 7 Macro Taxonomy Topics
├── url_utils.py             # Canonical URL Normalization & Safe Domain Extraction
├── db.py                    # PostgreSQL Connection Pooling (psycopg-pool)
├── ai.py                    # OpenAI-Compatible Client (Nebius / LM Studio / OpenRouter)
├── schemas.py               # Pydantic V2 Request/Response Data Models
├── schema.sql               # PostgreSQL DDL, Extensions & Indexes (pgvector, HNSW, GIN, V1 tables)
├── migrate_v1_tables.py     # Standalone Idempotent V1 Migration Runner (psycopg)
├── docker-compose.yml       # Local PostgreSQL 16 + pgvector Container Config
├── migrate_embeddings.py    # Database Migration Script: Embeds Existing Articles
├── clean_articles_db.py     # Database Maintenance Script: Batch Cleans Markdown Boilerplate
├── reindex_articles.py      # Maintenance Script: Batch Re-enriches & Re-classifies Articles
├── requirements.txt         # Core Python Runtime Dependencies (tavily-python>=0.8.5)
├── .env / .env.example      # Environment Configurations, Secrets & Routing Thresholds
├── legacy/                  # Deprecated V0 Prototypes
│   └── scraping_algorithm.py # Deprecated V0 1,480-line Scraper (Superseded by Tavily)
└── tests/                   # Consolidated Test & Benchmark Suite
    ├── test_tavily_integration.py      # Tavily 5-Endpoint Integration & Telemetry Test Suite
    ├── test_evidence_and_origins.py    # Span Grounding & Wire Origin Analysis Test Suite
    ├── test_content_cleaner.py         # Markdown Sanitizer 4-Pass Unit Test Suite
    ├── test_ingestion_and_cleaner.py   # Ingestion, Shingle Dedup & Date Validation Suite
    ├── test_user_personalization.py    # User Onboarding, Preferences, Feed & Daily Digest Verification
    ├── comprehensive_test_evaluation.py # Master 9-Part End-to-End Evaluation Suite
    ├── fixtures/tavily_verify_mock.json # Deterministic Demo Mode Verification Fixtures
    ├── fixtures/test_nvidia_raw.json   # Test Fixture Data
    ├── test_search_speed.py            # Hybrid Vector Search Latency Benchmarks
    ├── test_hybrid_enrichment.py       # SLM Triage vs LLM Latency & Accuracy Verification
    ├── test_copilot_demo.py            # Grounded Streaming Copilot & Guardrail Test
    ├── test_live_endpoints.py          # Live REST API Contract Verification
    ├── test_search_suite.py            # Multi-Domain Natural Language Search Verification
    └── test_seed.py                    # Bootstrap Seed Ingestion Script
```

---

### Detailed File Responsibilities

#### `main.py` & `routers/`
- **Role**: Primary application entry point (`main.py`) and modular domain `APIRouter` modules (`routers/articles.py`, `routers/search.py`, `routers/intelligence.py`, `routers/users.py`, `routers/comments.py`, `routers/deps.py`).
- **Key Responsibilities**:
  - `GET /api/articles`: Paginated feed retrieval with macro tag filtering (`macro=Gaming`, `macro=Tech`, etc.).
  - `GET /api/articles/{article_id}`: Full article detail with clean text, TL;DR, macro/micro tags, and metadata.
  - `POST /api/internal/ingest`: Protected ingestion endpoint validating `x-secret` with byte-safe `hmac.compare_digest` and fail-fast startup checks. Hashes canonicalized URLs (`canonicalize_url()`) with SHA-256 for zero-duplicate idempotency (`ON CONFLICT (url_hash) DO NOTHING`). Accepts array of `ScrapedArticleIn`.
  - `GET /api/search` & `execute_hybrid_search()`: 3-Tier Hybrid Search engine (Dense vector cosine distance + BM25 FTS + tag likeness fused via RRF `UNION ALL`). Includes calibrated noise cutoff ($d < 0.78$) and segregates results into **Direct Matches** ($d \le 0.68$ or FTS/tag hits) and **Want to Explore?** ($0.68 < d < 0.78$ adjacent horizons) with dynamic tag-driven discovery angles (`explore_angle`). Supports flat array, `?grouped=true` split output, and an optional `?live_web=true` segregated external web discovery tier (5.0s timeout, deduplicated).
  - `POST /api/articles/{article_id}/chat`: Tier 2 Article Copilot streaming token responses over Server-Sent Events (`text/event-stream` with `data: {"token": "..."}\n\n`). Bounded to the article's text with strict anti-hallucination instructions.
  - `POST /api/intelligence/synthesize`: Tier 3 Multi-Source Consensus Agent. Uses `FAST_MODEL` to formulate a neutral verification query, runs `web_verify()` with `exclude_domains` to pull third-party evidence via Tavily, and streams structured claim verdicts (`[SUPPORTED]`, `[CONTRADICTED]`, `[UNVERIFIED]`) backed by `[T1]`, `[T2]` citations over SSE.
  - `POST /api/users/login` & `GET /api/users`: Zero-friction user onboarding with deterministic avatars assigned via DiceBear / Liara avatar generator, returning onboarding status and preferences.
  - `GET /api/taxonomy`: Hierarchical macro-to-micro taxonomy dictionary for onboarding modals with zero cross-category tag leakage.
  - `PUT /api/users/{user_id}/preferences` & `GET`: Stores user macro/micro category preferences, dynamically generating a 384d semantic interest embedding via natural language synthesis.
  - `GET /api/feed/personalized`: Multi-factor personalized newsfeed combining micro-tag cardinality overlap, macro alignment, cosine vector proximity (`<=>`), and exponential recency decay with zero-leakage filtering.
  - `POST /api/articles/{article_id}/interactions`: Real-time like/unlike toggle and bookmark tracking with dynamic Exponential Moving Average (EMA) vector learning (`alpha=0.15`).
  - `GET /api/intelligence/daily-digest`: 24-hour rotating daily fact-checked intelligence digest clustering top trending articles per macro category, cross-checked with Tavily and cached in PostgreSQL.
  - `POST /api/articles/{id}/comments` & `GET`: Social annotations anchored to highlighted paragraph excerpts, stored in `article_comments`.

#### `tavily_service.py`
- **Role**: Centralized Tavily Web Intelligence & Fact-Checking client wrapper.
- **Key Responsibilities**:
  - Thread-safe singleton client managing `TAVILY_API_KEY` and project tracking (`TAVILY_PROJECT`).
  - `search_news()`: Pulls full-markdown articles across domains with `include_raw_content="markdown"` and customizable timeouts.
  - `web_verify()`: Performs independent external claim verification with `exclude_domains` to prevent self-referential bias.
  - In-memory 30-minute TTL cache (`CACHE_TTL_SECONDS = 1800`) to conserve API credit limits.
  - Deterministic offline demo mode (`TAVILY_MOCK=1`) using SHA-1 hashing and `tavily_verify_mock.json` fixtures for resilient conference presentations.

#### `content_cleaner.py`
- **Role**: Generalized 4-Pass Markdown Article Sanitizer, Boilerplate Pruner & Context-Safe Rejection Engine.
- **Key Responsibilities**:
  - **Pass 1 (Leading Junk Stripping)**: Removes standalone author byline links, comment anchors, view-count badges, and "Skip to content" remnants before the first body paragraph.
  - **Pass 2 (Two-Tier Tail Boundary Cutoff)**:
    - *Strong Tail Boundary*: Unconditionally terminates text at definite termination markers (`Related Stories/Articles`, `Trending Stories`, `Popular Discussions`, `Comments`, `Discussion`, `Leave a Reply`, `Recommended Stories`) once depth exceeds 600 characters.
    - *Weak Tail Boundary*: Position-guarded termination at boundary headers (`About the Author`, `Further Reading`, `Newsletter`, `Sources`) only when appearing at $\ge 55\%$ document depth and $\ge 600$ characters, preventing false cutoffs on early narrative mentions.
  - **Pass 3 (Inline Promo & Banner Stripping)**: Removes mid-article interruption banners line-by-line (e.g. `[##### **Related Story** ...]`, `Read More: [Link]`, `SEE ALSO: [Link]`, newsletter subscription callouts) without truncating subsequent paragraphs.
  - **Pass 4 (Link-Density List Pruning)**: Algorithmic detector pruning bullet lists where $>60\%$ of tokens are hyperlinks, dropping aggregator widgets while preserving genuine comparison lists.
  - **Context-Safe Rejection Gates**:
    - `_is_error_or_bot_page`: Flags 404, Cloudflare, CAPTCHA, or bot-blocked stub pages ($< 1500$ chars or in lead 350 chars), while preserving legitimate editorial coverage of cloud outages ($\ge 1500$ chars).
    - `_is_paywall_stub`: Drops shallow subscription prompts ($< 2500$ chars).
    - `_is_ui_or_sidebar_dump`: Drops navigation chrome, stock ticker price dumps, and sidebar link aggregators lacking body punctuation or title-word alignment.
  - **Paragraph Deduplication**: Strips repeated CMS paragraphs with strict code-block fence (` ``` `) awareness.
  - **Shingle Extraction**: `extract_normalized_shingles(text, k=5)` extracts normalized word shingles for downstream Jaccard-based syndicated wire copy deduplication.

#### `ingest_tavily.py`
- **Role**: Autonomous ingestion daemon harvesting live news across all 7 macro taxonomy categories.
- **Key Responsibilities**:
  - **Taxonomy Sub-Query Rotation**: Maps each of the 7 macro categories to 3 curated, high-yield sub-queries (21 total search vectors) rotating cyclically across harvest cycles via atomic `.ingest_state.json` persistence.
  - **Date Poisoning Protection**: `_parse_and_validate_date()` rejects future timestamps ($> \text{now} + 2\text{h}$) and stale articles ($> 14$ days old), with safe fallback to `datetime.now(timezone.utc)`.
  - **Syndicated Wire Copy Deduplication**: In addition to canonical URL tracking and SHA-256 full-content hashing, computes Jaccard similarity across 5-word shingles to drop syndicated duplicates ($\ge 0.65$ similarity).
  - **Safe Markdown Truncation**: `_truncate_markdown_safely()` respects sentence and paragraph boundaries when capping articles at 25,000 characters and balances unclosed markdown code fences (` ``` `).
  - **Network & Auth Resilience**: Implements exponential backoff retries (up to 2 retries) on 5xx or connection errors, while aborting immediately on 401/403 authorization failures.
  - **Execution Modes**: Defaults to credit-safe one-shot rotation (`--once`, 7 queries, `--max 4`), continuous daemon operation (`--loop`, interval 43,200s), or all-query batch ingestion (`--all-queries`).

#### `url_utils.py`
- **Role**: URL sanitization and domain parsing utility.
- **Key Responsibilities**:
  - `canonicalize_url()`: Strips marketing tracking parameters (`utm_*`, `fbclid`, `gclid`, `ref`), removes fragment identifiers, lowercases hostnames, and trims trailing slashes to guarantee clean SHA-256 deduplication.
  - `extract_domain()`: Safely extracts network domain names without throwing `IndexError` on malformed URLs.

#### `worker.py`
- **Role**: Standalone background daemon polling un-enriched articles from PostgreSQL.
- **Key Responsibilities**:
  - Continuously queries `SELECT * FROM articles WHERE is_processed = FALSE LIMIT 5 FOR UPDATE SKIP LOCKED`.
  - Runs defensive sanitization with `clean_markdown_article()` before token truncation to ensure SLM triage and embeddings are computed over editorial prose.
  - Runs **Stage 1 (Triage)**: Invokes `LayaClassifier` to obtain `macro_tag`, `macro_tags`, `micro_tags`, `confidence_score`, and `on_topic_score`.
  - Handles **Noise Gating**: If classified as `Off-Topic`, marks `is_processed = TRUE` and bypasses expensive LLM summarization.
  - Runs **Stage 2 (Summarization)**: If confidence is high ($\ge 0.65$), sends a targeted, low-token TL;DR prompt to the LLM. If low, escalates to a full classification prompt.
  - Runs **Stage 3 (Embedding)**: Computes a 384-dimensional vector embedding of `title + " " + tldr` using `embedder.py`.
  - Persists the enriched data, vector, tags, and timestamps atomically to PostgreSQL.

#### `classifier.py`
- **Role**: Deterministic decision SLM wrapper.
- **Key Responsibilities**:
  - Implements the singleton `LayaClassifier`.
  - Loads the local SLM decision model (`convaiinnovations/laya`) using the `laya` Python library on GPU/CPU.
  - Evaluates candidate macro categories against `TAXONOMY_CRITERIA` defined in `taxonomy.py`.
  - Applies calibrated thresholds:
    - `LAYA_CONFIDENCE_THRESHOLD` (default: `0.65`): Determines if LLM escalation is required.
    - `LAYA_ON_TOPIC_THRESHOLD` (default: `0.40`): Rejects spam, clickbait, and irrelevant articles.
  - Automatically extracts and cleans domain micro-tags.

#### `embedder.py`
- **Role**: Dense vector embedding generator (~10ms GPU latency).
- **Key Responsibilities**:
  - Implements the thread-safe singleton `TextEmbedder`.
  - Utilizes `sentence-transformers/all-MiniLM-L6-v2` (384 float dimensions).
  - Automatically selects `cuda` when available, falling back to `cpu`.
  - Implements a resilient cache-first offline pattern: checks local Hugging Face cache first with `HF_HUB_OFFLINE=1`; downloads once on cache miss and permanently locks to offline execution.
  - Normalizes embeddings to unit length for direct cosine distance calculation via PostgreSQL `<=>` operator.

#### `taxonomy.py`
- **Role**: Central source of truth for taxonomy, ontology rules, and tag normalization.
- **Key Responsibilities**:
  - Defines supported macro categories: `Tech`, `AI`, `Hardware`, `Automotive`, `Gaming`, `Finance`, `Science`, `Off-Topic`.
  - Defines domain keywords and subtag dictionaries for automatic micro-tagging (e.g. `2nm`, `gaafet`, `solid_state_battery`, `steam_deck`, `autonomous_agents`).
  - Enforces strict grouped regex word boundaries (`\b(?:electric vehicles?|evs?)\b`, `\b(?:james webb|jwst)\b`, `\b(?:european central bank|ecb)\b`) to prevent substring false-positive contamination (e.g. `dev` matching `ev`).
  - Provides helper functions: `normalize_macro_tag()`, `extract_domain_keywords()`, `clean_micro_tags()`.

#### `db.py`
- **Role**: Database connection lifecycle management.
- **Key Responsibilities**:
  - Initializes `psycopg_pool.ConnectionPool` using `DATABASE_URL`.
  - Provides the `get_db()` context manager for safe cursor allocation with automatic transaction commit on exit and rollback on exceptions.

#### `ai.py`
- **Role**: Central OpenAI-compatible client configuration.
- **Key Responsibilities**:
  - Instantiates `ai_client` via `openai.OpenAI`.
  - Configures `AI_BASE_URL` and `AI_API_KEY` (compatible with Nebius Token Factory, LM Studio, Ollama, OpenRouter).
  - Declares model constants: `FAST_MODEL` (e.g. `meta-llama-3.1-8b-instruct`) and `REASONING_MODEL` (e.g. `nvidia/llama-3.1-nemotron-70b-instruct`).

#### `schemas.py`
- **Role**: Pydantic V2 models for request validation and serialization.
- **Key Models**:
  - `ScrapedArticleIn`: Ingestion payload structure (`Site`, `Baslik`, `Link`, `Yazar`, `Tarih`, `TamMetin`).
  - `ArticleChatRequest`: Copilot request body with conversation `history` and user `question`.
  - `SynthesizeRequest`: Multi-source consensus request containing a list of `article_ids`.
  - `UserLoginRequest`: Username and display name payload.
  - `UserPreferencesUpdate`: Onboarding payload with `preferred_macro_tags` and `preferred_micro_tags`.
  - `UserInteractionRequest`: Article interaction payload (`like`, `bookmark`, `read`).
  - `UserProfileOut`: User profile with onboarding status, preferred categories, and activity counts.
  - `DailyDigestOut`: 24-hour rotating synthesized intelligence brief with consensus bullets and citations.
  - `CommentCreate`: Contextual comment payload with `user_id`, `selected_text`, and `comment_text`.

#### `schema.sql`
- **Role**: Database DDL schema file.
- **Key Elements**:
  - Extensions: `uuid-ossp`, `vector`.
  - Tables: `articles`, `users`, `article_comments`, `user_article_interactions`, `daily_intelligence_digests`.
  - Indexes:
    - `idx_articles_embedding`: HNSW index using `vector_cosine_ops` (`CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)`).
    - `idx_articles_macro_tags` & `idx_articles_micro_tags`: GIN inverted indexes for exact array searches.
    - `idx_articles_url_hash`: Unique B-Tree index enforcing URL hash deduplication.
    - `idx_user_interactions_lookup`: Unique B-Tree index on `(user_id, article_id, interaction_type)`.
    - `idx_daily_digests_date_macro`: Unique B-Tree index on `(digest_date, macro_tag)`.
    - Full-text search index on `to_tsvector('english', title || ' ' || clean_text)`.

#### `docker-compose.yml`
- **Role**: Local containerized development database.
- **Configuration**:
  - Spins up `pgvector/pgvector:pg16` on port `5432`.
  - Pre-configures user `dev_user`, password `dev_password123`, database `rss_intelligence`.

#### `migrate_embeddings.py`
- **Role**: One-off migration script.
- **Function**: Scans all processed articles where `embedding IS NULL`, generates dense embeddings using `TextEmbedder`, and updates the rows in PostgreSQL.

#### `clean_articles_db.py`
- **Role**: Database maintenance migration script.
- **Function**: Scans all articles in PostgreSQL, strips leading navbars, trailing CMS aggregators, author social cards, and inline promotional interruption banners via `content_cleaner.py`, and cleans false-positive micro-tags in-place.

#### `reindex_articles.py`
- **Role**: Maintenance utility.
- **Function**: Re-runs the `LayaClassifier` and `taxonomy` extraction over existing records to backfill new taxonomy tags and confidence scores.

#### `tests/` Directory
- `comprehensive_test_evaluation.py`: Master test suite executing all 9 evaluation tiers with dynamic SSE stream parsing and live article lookup.
- `test_ingestion_and_cleaner.py`: 10-Tier unit and regression test suite verifying two-tier tail boundaries, inline promo stripping, leading boilerplate cleanup, error/bot gate, UI ticker dump gate, code fence paragraph deduplication, safe markdown truncation, future/stale date validation, and syndicated wire copy shingle detection.
- `test_user_personalization.py`: 7-Tier verification suite covering taxonomy hierarchy, natural language interest embedding, zero-leakage personalized feed, like/unlike toggling with EMA vector shift, and daily rotating digest caching.
- `test_content_cleaner.py`: 4-pass Markdown sanitizer unit test suite verifying Wccftech CMS clutter stripping, technical code block preservation, link-density list pruning, and short article guardrails.
- `test_tavily_integration.py`: Tavily integration, URL canonicalization, Markdown cleaning, and SSE streaming fact-checking tests.
- `fixtures/tavily_verify_mock.json`: Deterministic Demo Mode verification fixtures.
- `test_search_speed.py`: Measures execution latency for search queries across semantic vector spaces (empirically 25–65 ms).
- `test_hybrid_enrichment.py`: Validates LayaClassifier cold-start, warmup time, and multi-domain classification accuracy.
- `test_copilot_demo.py`: Verifies SSE chunk delivery and checks that hallucination prompts trigger grounded refusal.
- `test_live_endpoints.py`: Hits a running FastAPI instance with HTTP requests to verify REST contracts.
- `test_search_suite.py`: Verifies tag likeness and recall across diverse query patterns.
- `test_seed.py`: Seeds the database with high-quality reference articles across Gaming, Hardware, AI, Automotive, and Finance.

---

## 4. Data Pipelines & Execution Flow

```
[Tavily News Ingestion / RSS Feeds]
       │
       ▼ (canonicalize_url() + Ingestion Secret Auth)
       ▼ (HTTP POST /api/internal/ingest)
[SHA-256 Hash Deduplication] ──── Duplicate ───► Discard (HTTP 200, inserted: 0)
       │
       ▼ (New Article)
[PostgreSQL: articles table] (is_processed = FALSE)
       │
       ▼ (Background Polling by worker.py)
[Stage 1: LayaClassifier SLM Triage] ──── Off-Topic ───► Mark Processed & Skip
       │
       ├── High Confidence (>= 0.65) ──► Targeted TL;DR Prompt
       └── Low Confidence (< 0.65)   ──► Escalated Full LLM Prompt
       │
       ▼
[Stage 2: LLM (Nemotron / Llama-3.1)] ──► 2-Sentence Empirical TL;DR
       │
       ▼
[Stage 3: TextEmbedder (GPU/CPU)] ──► 384d Dense Vector (all-MiniLM-L6-v2)
       │
       ▼
[PostgreSQL Update] ──► Sets is_processed = TRUE, tags, and embedding

                                 Tier 3 Multi-Source Verification
                                                 │
                                                 ▼
[Synthesize Request] ──► FAST_MODEL Claim Extraction ──► Tavily web_verify() [exclude_domains]
                                                                │
                                                                ▼
                                      SSE Stream with [SUPPORTED]/[CONTRADICTED]/[UNVERIFIED]
```

---

## 5. Search Architecture (`/api/search`)

```
User Query: "i want news about gaming and chips and small conductors"
       │
       ├── 1. TextEmbedder: Generates 384d Dense Vector (q_vec)
       ├── 2. Token Normalization: Filters stop words -> ['gaming', 'chips', 'conductors']
       ├── 3. SQL Reciprocal Rank Fusion (RRF):
       │     ├── Vector Matches: (embedding <=> q_vec) < 0.88 LIMIT 25
       │     ├── BM25 FTS Matches: to_tsvector @@ websearch_to_tsquery LIMIT 25
       │     └── Tag Matches: ILIKE substring patterns on title/micro_tags LIMIT 25
       │           │
       │           ▼ (UNION ALL + SUM(score) grouped by article ID)
       │     ┌────────────────────────────────────────────────────────┐
       │     │             Local Result Classification & Tiering      │
       │     └────────────────────────────────────────────────────────┘
       │           │
       │   ┌───────┴───────────┐
       │   ▼                   ▼
       │ [Direct Matches]   [Want to Explore?]        [Noise Cutoff]
       │ - dist <= 0.80     - 0.80 < dist < 0.88      - dist >= 0.88
       │ - FTS text match   - Peripheral tags         - COMPLETELY DROPPED
       │ - Tag matches      - Tag explore angle       - (e.g. ECB Digital Euro)
       │
       └── 4. Optional Live Web Discovery (?live_web=true):
             └── Tavily search_news() [5.0s timeout, deduplicated vs local URLs]
                   └── Returned in unified/isolated JSON payload
```

---

## 6. Quick Reference Commands

### Prerequisites & Environment Setup
```powershell
# 1. Activate Python virtual environment
.\venv\Scripts\Activate.ps1

# 2. Check / configure .env variables
# Make sure INGESTION_SECRET, TAVILY_API_KEY, and AI_API_KEY are configured:
# TAVILY_MOCK=1   # 1 = Zero-credit deterministic demo mode; 0 = Live Tavily Web API
```

---

### Step-by-Step Startup Sequence

#### Step 1: Start PostgreSQL with pgvector
```powershell
docker compose up -d
```

#### Step 2: Start Backend API Server
```powershell
.\venv\Scripts\uvicorn.exe main:app --host 0.0.0.0 --port 8000 --reload
```
> The API will be available at `http://localhost:8000` (Swagger docs at `http://localhost:8000/docs`).

#### Step 3: Start Asynchronous AI Enrichment Worker
In a new terminal window:
```powershell
.\venv\Scripts\python.exe worker.py
```
> Continuously watches for new articles, runs sub-100ms Laya SLM triage, generates 2-sentence grounded TL;DR summaries, and computes 384d vector embeddings.

#### Step 4: Run Tavily Autonomous News Ingestion
In a new terminal window:
```powershell
# Balanced bulk harvest (e.g. 200 articles equally split across all 7 categories):
.\venv\Scripts\python.exe ingest_tavily.py --target 200

# Quick one-shot harvest (4 per category):
.\venv\Scripts\python.exe ingest_tavily.py --once

# Or continuous daemon mode (runs every 4 hours, credit-safe):
.\venv\Scripts\python.exe ingest_tavily.py --loop --interval 14400 --max 4
```

---

### API Verification & Testing

#### 1. Test Hybrid Search API
```powershell
# Standard Flat Search
curl "http://localhost:8000/api/search?q=gaming+chips"

# Grouped Search (Direct Matches vs Want to Explore?)
curl "http://localhost:8000/api/search?q=gaming+chips&grouped=true"

# Hybrid Search + Live Tavily Web Discovery
curl "http://localhost:8000/api/search?q=intel+chips&live_web=true"
```

#### 2. Test Multi-Source Consensus & Fact-Checking (SSE Stream)
```powershell
# Synthesize two articles and run independent Tavily claim verification:
curl -X POST "http://localhost:8000/api/intelligence/synthesize" `
  -H "Content-Type: application/json" `
  -d '{"article_ids": [1, 2]}'
```

#### 3. Run Automated Test Suites
```powershell
# Content Sanitizer Unit Test Suite (4-Pass Markdown Cleaner):
.\venv\Scripts\python.exe tests/test_content_cleaner.py

# Tavily Unit & Integration Test Suite (7 test scenarios):
.\venv\Scripts\python.exe tests/test_tavily_integration.py

# User Personalization & Feed Engine Test Suite (7 Tiers):
.\venv\Scripts\python.exe tests/test_user_personalization.py

# Master 9-Part End-to-End Regression Suite (100% Score across all 9 Tiers):
.\venv\Scripts\python.exe tests/comprehensive_test_evaluation.py

# Database Maintenance: Batch Sanitize Markdown & Clean Micro-Tags:
.\venv\Scripts\python.exe clean_articles_db.py
```

---

### Public Deployment & Tunneling
```powershell
# Expose Local Backend to the Web via LocalTunnel:
npx localtunnel --port 8000 --subdomain agentic-news-backend
```
