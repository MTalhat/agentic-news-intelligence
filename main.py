"""
Agentic News Intelligence — Primary FastAPI API Gateway.

Modularized into domain-specific APIRouter packages under `routers/`:
  - routers.articles      : Feed retrieval, article detail, HMAC ingestion
  - routers.search        : 3-Tier Hybrid RRF search & Live Web Horizon
  - routers.intelligence  : Grounded Copilot SSE, Multi-Source Synthesis SSE,
                            Daily Digest, Verifiable Evidence Receipts & Tavily Ops
  - routers.users         : User onboarding, preferences, EMA interactions & personalized feed
  - routers.comments      : Contextual paragraph annotations & taxonomy dictionary
"""

from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from routers import (
    articles_router,
    comments_router,
    intelligence_router,
    search_router,
    users_router,
)

# Re-export core helpers for backward compatibility with direct test imports
from routers.deps import INGESTION_SECRET, parse_vector, validate_uuid
from routers.intelligence import (
    _compute_deterministic_cluster_id,
    _upsert_story_cluster,
    prepare_synthesis_context,
)
from routers.search import execute_hybrid_search, natural_search
from routers.users import get_personalized_feed_data


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from migrate_v1_tables import run_migration
        run_migration()
    except Exception:
        pass
    yield


app = FastAPI(title="Intelligence Newsroom API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register domain routers
app.include_router(articles_router)
app.include_router(search_router)
app.include_router(intelligence_router)
app.include_router(users_router)
app.include_router(comments_router)

__all__ = [
    "app",
    "INGESTION_SECRET",
    "parse_vector",
    "validate_uuid",
    "execute_hybrid_search",
    "natural_search",
    "_compute_deterministic_cluster_id",
    "_upsert_story_cluster",
    "prepare_synthesis_context",
    "get_personalized_feed_data",
]
