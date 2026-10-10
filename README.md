# Agentic News Intelligence Platform 🚀

A high-performance news aggregation, asynchronous AI enrichment, and multi-agent synthesis platform built with **FastAPI**, **PostgreSQL** (`pgvector`), and modern LLM reasoning runtimes.

---

## 🌟 Key Features

### 1. Ingestion & Asynchronous Enrichment Pipeline
- **Fast Deduplication Ingest**: `/api/internal/ingest` normalizes incoming articles and deduplicates using SHA-256 URL hashes.
- **Background Worker Engine** (`worker.py`): Decoupled polling service that classifies articles into macro categories (`Tech`, `AI`, `Hardware`, `Automotive`, `Gaming`, `Finance`, `Science`), generates specific micro-tags, and produces 2-sentence empirical summaries (TL;DR).

### 2. Multi-Tier AI Agent Architecture
- **Tier 1 — Intent-Driven Search** (`/api/search`): Analyzes natural language search queries to infer category and tag intents, queried against PostgreSQL GIN indexes with full-text search fallback.
- **Tier 2 — Grounded Article Copilot** (`/api/articles/{id}/chat`): Streams real-time grounded conversational answers strictly anchored to article content with anti-hallucination guardrails.
- **Tier 3 — Multi-Source Consensus & Discrepancy Agent** (`/api/intelligence/synthesize`): Cross-examines multiple news sources on a topic to extract:
  1. **Shared Consensus** (Points agreed on by all outlets)
  2. **Framing Discrepancies & Contradictions** (Where outlets spin, diverge, or omit facts)
  3. **Singular / Unverified Claims**

### 3. Social & Community Layer
- **Paragraph-Anchored Annotations**: Read and post contextual comments tied to specific selected article text.
- **Frictionless Identity**: Zero-friction avatar and profile assignment for immediate collaboration.

---

## 🏗️ Architecture & Stack

- **Framework**: FastAPI (Python 3.11+)
- **Database**: PostgreSQL 16 with `pgvector` & `uuid-ossp`
- **Driver / Pooling**: `psycopg` 3 + `psycopg-pool`
- **AI Runtimes**: Compatible with LM Studio, Ollama, OpenAI, and Nebius Cloud

---

## 🚀 Quickstart

### 1. Database Setup
```bash
docker-compose up -d
```

### 2. Environment Setup
```bash
cp .env.example .env
# Edit .env with your local or cloud LLM credentials
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

### 4. Run API & Worker
Start the API server:
```bash
uvicorn main:app --reload --port 8000
```

Start the background enrichment worker in a separate terminal:
```bash
python worker.py
```

---

## 🧪 Testing & Verification

- **Evaluation Suite**:
  ```bash
  python comprehensive_test_evaluation.py
  ```
- **Copilot Grounding Demo**:
  ```bash
  python test_copilot_demo.py
  ```
- **Seed Ingestion**:
  ```bash
  python test_seed.py
  ```
