# AGENTS.md — System Architecture & Codebase Manifest

> **Repository**: `MTalhat/agentic-news-intelligence`  
> **Stack**: Python 3.11+, FastAPI, PostgreSQL 16 (`pgvector`), PyTorch, `sentence-transformers`, `laya` SLM, Nebius / LM Studio / OpenRouter runtimes.  
> **Status**: Verified by 9-Part End-to-End Regression Suite (`tests/comprehensive_test_evaluation.py`).

---

## 1. System Overview

**Agentic News Intelligence** is a real-time news aggregation, asynchronous AI enrichment, and multi-agent synthesis platform. It solves the performance and accuracy bottlenecks of traditional RSS readers through a **3-tier hybrid pipeline**:
1. **Sub-100ms SLM Triage (`Laya`)**: Classifies macro and micro taxonomy tags, filters off-topic noise, and routes articles based on confidence thresholds without incurring expensive autoregressive LLM calls.
2. **Generative Grounded Summarization**: Produces empirical, strictly grounded 2-sentence TL;DR summaries via fast cloud/local reasoning models.
3. **Dense Vector & Reciprocal Rank Fusion Search**: Combines 384-dimensional dense vectors (HNSW index), PostgreSQL BM25 Full-Text Search (`websearch_to_tsquery`), and tag substring likeness into a unified RRF score with segregated **Direct Focus** and **Want to Explore?** tiers. Empirically measured search latency is **25–65 ms**.
4. **Grounded Copilot & Multi-Source Synthesis**: Real-time Server-Sent Events (SSE) streaming Q&A strictly bound to article ground truth with hallucination guardrails, and comparative multi-source synthesis highlighting consensus vs. framing discrepancies.

---

## 2. File-by-File Manifest

```text
rss-backend/
├── main.py                  # Primary FastAPI API Gateway, Search Engine & Streaming Handlers
├── worker.py                # Asynchronous RSS Ingestion & AI Enrichment Worker
├── classifier.py            # Local Decision SLM (convaiinnovations/laya)
├── embedder.py              # Dense Vector Embedding Engine (sentence-transformers/all-MiniLM-L6-v2)
├── taxonomy.py              # Domain Taxonomy, Tagging Criteria & Keyword Normalization
├── tavily_service.py        # Tavily Live Search, Grounded Fact-Checking & TTL Cache
├── ingest_tavily.py         # Autonomous Tavily Ingestion Daemon Across 7 Macro Taxonomy Topics
├── url_utils.py             # Canonical URL Normalization & Safe Domain Extraction
├── db.py                    # PostgreSQL Connection Pooling (psycopg-pool)
├── ai.py                    # OpenAI-Compatible Client (Nebius / LM Studio / OpenRouter)
├── schemas.py               # Pydantic V2 Request/Response Data Models
├── schema.sql               # PostgreSQL DDL, Extensions & Indexes (pgvector, HNSW, GIN)
├── docker-compose.yml       # Local PostgreSQL 16 + pgvector Container Config
├── migrate_embeddings.py    # Database Migration Script: Embeds Existing Articles
├── reindex_articles.py      # Maintenance Script: Batch Re-enriches & Re-classifies Articles
├── requirements.txt         # Core Python Runtime Dependencies
├── .env / .env.example      # Environment Configurations, Secrets & Routing Thresholds
└── tests/                   # Consolidated Test & Benchmark Suite
    ├── comprehensive_test_evaluation.py # Master 9-Part End-to-End Evaluation Suite
    ├── test_tavily_integration.py      # Tavily Integration, URL Canonicalization & SSE Tests
    ├── fixtures/tavily_verify_mock.json # Deterministic Demo Mode Verification Fixtures
    ├── test_search_speed.py            # Hybrid Vector Search Latency Benchmarks
    ├── test_hybrid_enrichment.py       # SLM Triage vs LLM Latency & Accuracy Verification
    ├── test_copilot_demo.py            # Grounded Streaming Copilot & Guardrail Test
    ├── test_live_endpoints.py          # Live REST API Contract Verification
    ├── test_search_suite.py            # Multi-Domain Natural Language Search Verification
    └── test_seed.py                    # Bootstrap Seed Ingestion Script
```

---

### Detailed File Responsibilities

#### `main.py`
- **Role**: Primary application entry point running Uvicorn / FastAPI.
- **Key Responsibilities**:
  - `GET /api/articles`: Paginated feed retrieval with macro tag filtering (`macro=Gaming`, `macro=Tech`, etc.).
  - `GET /api/articles/{article_id}`: Full article detail with clean text, TL;DR, macro/micro tags, and metadata.
  - `POST /api/internal/ingest`: Protected ingestion endpoint (validates `x-secret` header). Hashes article URLs with SHA-256 for instantaneous zero-duplicate idempotency (`ON CONFLICT (url_hash) DO NOTHING`). Accepts array of `ScrapedArticleIn`.
  - `GET /api/search` & `execute_hybrid_search()`: 3-Tier Hybrid Search engine (Dense vector cosine distance + BM25 FTS + tag likeness fused via RRF `UNION ALL`). Includes calibrated noise cutoff ($d < 0.88$) and segregates results into **Direct Matches** ($d \le 0.80$ or FTS/tag hits) and **Want to Explore?** ($0.80 < d < 0.88$ adjacent horizons) with dynamic tag-driven discovery angles (`explore_angle`). Supports both flat array and `?grouped=true` split output.
  - `POST /api/articles/{article_id}/chat`: Tier 2 Article Copilot streaming token responses over Server-Sent Events (`text/event-stream` with `data: {"token": "..."}\n\n`). Bounded to the article's text with strict anti-hallucination instructions.
  - `POST /api/intelligence/synthesize`: Tier 3 Multi-Source Consensus Agent. Cross-examines 2+ articles on a story over SSE to isolate shared facts from editorial divergence.
  - `POST /api/users/login` & `GET /api/users`: Zero-friction user onboarding with deterministic avatars assigned via DiceBear / Liara avatar generator.
  - `POST /api/articles/{id}/comments` & `GET`: Social annotations anchored to highlighted paragraph excerpts, stored in `article_comments`.

#### `worker.py`
- **Role**: Standalone background daemon polling un-enriched articles from PostgreSQL.
- **Key Responsibilities**:
  - Continuously queries `SELECT * FROM articles WHERE is_processed = FALSE LIMIT 5 FOR UPDATE SKIP LOCKED`.
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
  - `CommentCreate`: Contextual comment payload with `user_id`, `selected_text`, and `comment_text`.

#### `schema.sql`
- **Role**: Database DDL schema file.
- **Key Elements**:
  - Extensions: `uuid-ossp`, `vector`.
  - Tables: `articles`, `users`, `article_comments`.
  - Indexes:
    - `idx_articles_embedding`: HNSW index using `vector_cosine_ops` (`CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)`).
    - `idx_articles_macro_tags` & `idx_articles_micro_tags`: GIN inverted indexes for exact array searches.
    - `idx_articles_url_hash`: Unique B-Tree index enforcing URL hash deduplication.
    - Full-text search index on `to_tsvector('english', title || ' ' || clean_text)`.

#### `docker-compose.yml`
- **Role**: Local containerized development database.
- **Configuration**:
  - Spins up `pgvector/pgvector:pg16` on port `5432`.
  - Pre-configures user `dev_user`, password `dev_password123`, database `rss_intelligence`.

#### `migrate_embeddings.py`
- **Role**: One-off migration script.
- **Function**: Scans all processed articles where `embedding IS NULL`, generates dense embeddings using `TextEmbedder`, and updates the rows in PostgreSQL.

#### `reindex_articles.py`
- **Role**: Maintenance utility.
- **Function**: Re-runs the `LayaClassifier` and `taxonomy` extraction over existing records to backfill new taxonomy tags and confidence scores.

#### `tests/` Directory
- `comprehensive_test_evaluation.py`: Master test suite executing all 9 evaluation tiers with dynamic SSE stream parsing and live article lookup.
- `test_search_speed.py`: Measures execution latency for search queries across semantic vector spaces (empirically 25–65 ms).
- `test_hybrid_enrichment.py`: Validates LayaClassifier cold-start, warmup time, and multi-domain classification accuracy.
- `test_copilot_demo.py`: Verifies SSE chunk delivery and checks that hallucination prompts trigger grounded refusal.
- `test_live_endpoints.py`: Hits a running FastAPI instance with HTTP requests to verify REST contracts.
- `test_search_suite.py`: Verifies tag likeness and recall across diverse query patterns.
- `test_seed.py`: Seeds the database with high-quality reference articles across Gaming, Hardware, AI, Automotive, and Finance.

---

## 3. Data Pipelines & Execution Flow

```
[Raw RSS Feeds]
       │
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
```

---

## 4. Search Architecture (`/api/search`)

```
User Query: "i want news about gaming and chips and small conductors"
       │
       ├── 1. TextEmbedder: Generates 384d Dense Vector (q_vec)
       ├── 2. Token Normalization: Filters stop words -> ['gaming', 'chips', 'conductors']
       └── 3. SQL Reciprocal Rank Fusion (RRF):
             ├── Vector Matches: (embedding <=> q_vec) < 0.88 LIMIT 25
             ├── BM25 FTS Matches: to_tsvector @@ websearch_to_tsquery LIMIT 25
             └── Tag Matches: ILIKE substring patterns on title/micro_tags LIMIT 25
                   │
                   ▼ (UNION ALL + SUM(score) grouped by article ID)
       ┌────────────────────────────────────────────────────────┐
       │             Result Classification & Tiering            │
       └────────────────────────────────────────────────────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
  [Direct Matches]     [Want to Explore?]          [Noise Cutoff]
  - dist <= 0.80       - 0.80 < dist < 0.88        - dist >= 0.88
  - FTS text match     - Peripheral tags           - No text/tag overlap
  - Strong tag match   - Tag-driven explore angle  - COMPLETELY DROPPED
  - Score: 82-99%      - Score: 52-75%             - (e.g. ECB Digital Euro)
```

---

## 5. Quick Reference Commands

```powershell
# 1. Start Database
docker compose up -d

# 2. Start Backend API Server
.\venv\Scripts\uvicorn.exe main:app --host 0.0.0.0 --port 8000 --reload

# 3. Start Background Ingestion Worker
.\venv\Scripts\python.exe worker.py

# 4. Run Tavily Autonomous News Ingestion (One-shot or 12h Daemon)
.\venv\Scripts\python.exe ingest_tavily.py --once
# Or continuous daemon mode:
# .\venv\Scripts\python.exe ingest_tavily.py --loop

# 5. Run Tavily Unit & Integration Suite
.\venv\Scripts\python.exe tests/test_tavily_integration.py

# 6. Expose Public Tunnel via LocalTunnel
npx localtunnel --port 8000 --subdomain agentic-news-backend

# 7. Run Full Evaluation Suite (100% Score across all 9 Tiers)
.\venv\Scripts\python.exe tests/comprehensive_test_evaluation.py
```
