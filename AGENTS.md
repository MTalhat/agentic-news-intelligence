# AGENTS.md — System Architecture & Codebase Manifest

> **Repository**: `MTalhat/agentic-news-intelligence`  
> **Stack**: Python 3.11+, FastAPI, PostgreSQL 16 (`pgvector`), PyTorch, `sentence-transformers`, `laya` SLM, Nebius / LM Studio / OpenRouter runtimes.  
> **Status**: Verified by 9-Part End-to-End Regression Suite (`tests/comprehensive_test_evaluation.py`).

---

## 1. System Overview

**Agentic News Intelligence** is a real-time news aggregation, asynchronous AI enrichment, and multi-agent synthesis platform. It solves the performance and accuracy bottlenecks of traditional RSS readers through a **hybrid multi-tier pipeline**:
1. **Sub-100ms SLM Triage (`Laya`)**: Classifies macro and micro taxonomy tags, filters off-topic noise, and routes articles based on confidence thresholds without incurring expensive autoregressive LLM calls.
2. **Generative Grounded Summarization**: Produces empirical, strictly grounded 2-sentence TL;DR summaries via fast cloud/local reasoning models.
3. **Dense Vector & Reciprocal Rank Fusion Search**: Combines 384-dimensional dense vectors (HNSW index), PostgreSQL BM25 Full-Text Search (`websearch_to_tsquery`), and tag substring likeness into a unified RRF score with segregated **Direct Focus** and **Want to Explore?** tiers. Empirically measured search latency is **25–65 ms**.
4. **Grounded Copilot & Multi-Source Synthesis**: Real-time Server-Sent Events (SSE) streaming Q&A strictly bound to article ground truth with hallucination guardrails, and comparative multi-source synthesis highlighting consensus vs. framing discrepancies.
5. **Autonomous Web Sourcing & Independent Verification (`Tavily`)**: Continuously harvests full-markdown breaking news across 7 macro taxonomy categories directly into the Laya triage pipeline, while providing independent third-party web cross-verification (`exclude_domains`) for Tier 3 synthesis with structured claim verdicts (`[SUPPORTED]`, `[CONTRADICTED]`, `[UNVERIFIED]`).

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
  - `POST /api/internal/ingest`: Protected ingestion endpoint validating `x-secret` with byte-safe `hmac.compare_digest` and fail-fast startup checks. Hashes canonicalized URLs (`canonicalize_url()`) with SHA-256 for zero-duplicate idempotency (`ON CONFLICT (url_hash) DO NOTHING`). Accepts array of `ScrapedArticleIn`.
  - `GET /api/search` & `execute_hybrid_search()`: 3-Tier Hybrid Search engine (Dense vector cosine distance + BM25 FTS + tag likeness fused via RRF `UNION ALL`). Includes calibrated noise cutoff ($d < 0.88$) and segregates results into **Direct Matches** ($d \le 0.80$ or FTS/tag hits) and **Want to Explore?** ($0.80 < d < 0.88$ adjacent horizons) with dynamic tag-driven discovery angles (`explore_angle`). Supports flat array, `?grouped=true` split output, and an optional `?live_web=true` segregated external web discovery tier (5.0s timeout, deduplicated).
  - `POST /api/articles/{article_id}/chat`: Tier 2 Article Copilot streaming token responses over Server-Sent Events (`text/event-stream` with `data: {"token": "..."}\n\n`). Bounded to the article's text with strict anti-hallucination instructions.
  - `POST /api/intelligence/synthesize`: Tier 3 Multi-Source Consensus Agent. Uses `FAST_MODEL` to formulate a neutral verification query, runs `web_verify()` with `exclude_domains` to pull third-party evidence via Tavily, and streams structured claim verdicts (`[SUPPORTED]`, `[CONTRADICTED]`, `[UNVERIFIED]`) backed by `[T1]`, `[T2]` citations over SSE.
  - `POST /api/users/login` & `GET /api/users`: Zero-friction user onboarding with deterministic avatars assigned via DiceBear / Liara avatar generator.
  - `POST /api/articles/{id}/comments` & `GET`: Social annotations anchored to highlighted paragraph excerpts, stored in `article_comments`.

#### `tavily_service.py`
- **Role**: Centralized Tavily Web Intelligence & Fact-Checking client wrapper.
- **Key Responsibilities**:
  - Thread-safe singleton client managing `TAVILY_API_KEY` and project tracking (`TAVILY_PROJECT`).
  - `search_news()`: Pulls full-markdown articles across domains with `include_raw_content="markdown"` and customizable timeouts.
  - `web_verify()`: Performs independent external claim verification with `exclude_domains` to prevent self-referential bias.
  - In-memory 30-minute TTL cache (`CACHE_TTL_SECONDS = 1800`) to conserve API credit limits.
  - Deterministic offline demo mode (`TAVILY_MOCK=1`) using SHA-1 hashing and `tavily_verify_mock.json` fixtures for resilient conference presentations.

#### `ingest_tavily.py`
- **Role**: Autonomous ingestion daemon harvesting live news across all 7 macro taxonomy categories.
- **Key Responsibilities**:
  - Polls categories (`Tech`, `AI`, `Hardware`, `Automotive`, `Gaming`, `Finance`, `Science`).
  - Enforces URL canonicalization via `canonicalize_url()` prior to ingestion.
  - Rejects shallow stubs (`len(text) < 250`) and caps maximum article length to 25,000 characters.
  - Defaults to one-shot mode (`--once`), with a credit-safe 12-hour interval (43,200s) and `--max 4` articles per category for continuous daemon operation (`--loop`).
  - Has a fail-fast startup check that halts execution (`sys.exit(1)`) if `INGESTION_SECRET` is unset.

#### `url_utils.py`
- **Role**: URL sanitization and domain parsing utility.
- **Key Responsibilities**:
  - `canonicalize_url()`: Strips marketing tracking parameters (`utm_*`, `fbclid`, `gclid`, `ref`), removes fragment identifiers, lowercases hostnames, and trims trailing slashes to guarantee clean SHA-256 deduplication.
  - `extract_domain()`: Safely extracts network domain names without throwing `IndexError` on malformed URLs.

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

## 4. Search Architecture (`/api/search`)

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

## 5. Quick Reference Commands

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
# One-shot harvest across all 7 macro taxonomy categories:
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
# Tavily Unit & Integration Test Suite (6 test scenarios):
.\venv\Scripts\python.exe tests/test_tavily_integration.py

# Master 9-Part End-to-End Regression Suite (100% Score across all 9 Tiers):
.\venv\Scripts\python.exe tests/comprehensive_test_evaluation.py
```

---

### Public Deployment & Tunneling
```powershell
# Expose Local Backend to the Web via LocalTunnel:
npx localtunnel --port 8000 --subdomain agentic-news-backend
```
