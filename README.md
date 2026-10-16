# FeedProof — Agentic News Intelligence 🚀
> *"Your news feed, with receipts."*  
> **Target Event**: Nebius × NVIDIA Hackathon  
> **Target Tracks**: **Best Use of Tavily ($3,000 Sponsor Prize Track)** & **Best Apps & Agents**

FeedProof is an autonomous, production-grade intelligence newsroom designed to solve the noise, fragility, and circular confirmation bias of modern information aggregation. Instead of relying on brittle web scrapers or naive LLM wrappers, it combines local SLM edge triage (`Laya`), sub-50ms hybrid vector search (PostgreSQL `pgvector` HNSW), and an investigative multi-source synthesis agent that cross-examines news against live web consensus.

---

## 🌟 The 5-Endpoint Tavily Intelligence Stack

FeedProof deeply integrates the entire **Tavily Python SDK** across 5 native endpoints:

| Endpoint | Role in Architecture | File / Implementation |
|---|---|---|
| **Tavily Search** | Autonomous multi-category news harvesting + live discovery horizon (`?live_web=true`) + tri-angle verification queries. | `tavily_service.py`, `ingest_tavily.py` |
| **Tavily Extract** | Automatic full-text fallback for paywalls/JS stubs + full evidence page retrieval for verifiable claim grounding. | `tavily_service.py`, `ingest_tavily.py`, `evidence_verifier.py` |
| **Tavily Map** | **"RSS-less Newsroom Monitoring"**: Discovers newly published announcements from corporate press rooms and research labs without RSS feeds. | `site_watcher.py`, `tavily_service.py` |
| **Tavily Crawl** | Targeted single-operation mapping & full-text extraction across dedicated editorial sections. | `tavily_service.py` |
| **Tavily Research** | On-demand deep multi-source investigative briefings for complex or disputed assertions. | `tavily_service.py` |

---

## 🏆 Key Innovations

### 1. Verifiable "Receipts" (Evidence Cards & Span Grounding)
Standard AI fact-checkers suffer from circular confirmation bias (reading syndicated copy and declaring it "verified"). FeedProof implements:
- **Verbatim Span Grounding (`span_is_grounded`)**: Extracted quotes must exist verbatim in the source text pulled via Tavily Extract. Fabricated LLM claims are dropped immediately.
- **Wire Syndication & Origin Analysis (`assign_origins`)**: Detects wire bylines (Reuters, AP, Bloomberg, PR Newswire) and shingle overlap. Proves to readers that *"7 articles often equal only 1 Reuters wire syndication"*.
- **Fresh News Rule**: Breaking news published $< 6$ hours ago with a single source is marked `unverified_yet` with a `recheck_at` timestamp—never falsely contradicted.

### 2. Semantic Story Clustering (Daily Digest)
- Eliminates arbitrary pairings: Anchors to top breaking events and uses dense vector proximity (`embedding <=> anchor_vec < 0.42`) to pair genuine sibling articles covering the *same story*.
- **Single-Anchor Fallback**: If no sibling exists within 7 days, gracefully executes single-story deep verification against Tavily consensus without 500 errors or mismatched pairings.

### 3. "RSS-less Newsroom Monitoring" (`site_watcher.py`)
- Corporate newsrooms and technical research labs (e.g. OpenAI, DeepMind, TSMC) frequently publish major breaking news without standard RSS feeds.
- FeedProof continuously maps site URL hierarchies via `tavily_map()`, computes URL diffs against PostgreSQL, extracts fresh articles with `tavily_extract()`, and injects them directly into the Laya triage pipeline.

### 4. Real-time Tavily Telemetry (`GET /api/tavily/stats`)
- Transparently tracks every Tavily operation: endpoint distribution (`search`, `extract`, `map`, `crawl`, `research`), purpose breakdown, latency (ms), and credit savings from in-memory and PostgreSQL caching.

### 5. Architectural Evolution: Deprecation of V0 Scraper
- **V0 Prototype (`legacy/scraping_algorithm.py`)**: Initially developed as a 1,480-line multi-source scraper (DuckDuckGo, Yahoo RSS, Reddit, HN). In real-world operation, it suffered from >60% yield loss, anti-bot Cloudflare challenges, and brittle DOM selectors.
- **V1 Architecture**: Entirely deprecated brittle scrapers in favor of Tavily's unified news intelligence API as the sole, resilient ingestion and verification backbone.

---

## 🏗️ Architecture & Stack

- **Inference Runtime**: Nebius Token Factory (NVIDIA Nemotron 70B & Llama 3.1 8B)
- **Edge Decision SLM**: Laya (`convaiinnovations/laya`) for sub-100ms macro/micro classification
- **Web Intelligence**: Tavily Search, Extract, Map, Crawl, Research (`tavily-python>=0.8.5`)
- **Vector Database**: PostgreSQL 16 + `pgvector` (HNSW cosine indexing)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384 float dimensions)
- **Backend**: FastAPI (Python 3.11+) + `psycopg-pool`

---

## 🚀 Quickstart

### 1. Database Setup
```bash
docker-compose up -d
```

### 2. Environment Setup
```bash
cp .env.example .env
# Configure your TAVILY_API_KEY, INGESTION_SECRET, and AI_API_KEY in .env
```

### 3. Install Dependencies
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 4. Run V1 Database Migration
```bash
python migrate_v1_tables.py
```

### 5. Start API Server & Workers
```bash
# Terminal 1: FastAPI Gateway
uvicorn main:app --reload --port 8000

# Terminal 2: Asynchronous AI Enrichment Worker
python worker.py

# Terminal 3: Autonomous Tavily News Ingestion
python ingest_tavily.py --once

# Terminal 4: RSS-less Newsroom Watcher (Tavily Map & Extract)
python site_watcher.py --once
```

---

## 🧪 Testing & Verification Suites

```bash
# 1. Tavily Multi-Endpoint Integration Suite (Search, Extract, Map, Crawl, Research, Telemetry):
python tests/test_tavily_integration.py

# 2. Evidence Verifier, Span Grounding & Origin Analysis Suite:
python tests/test_evidence_and_origins.py

# 3. Content Sanitizer 4-Pass Cleaner Suite:
python tests/test_content_cleaner.py

# 4. Ingestion, Shingle Dedup & Date Validation Suite:
python tests/test_ingestion_and_cleaner.py

# 5. Master 9-Tier End-to-End Regression Suite:
python tests/comprehensive_test_evaluation.py
```
