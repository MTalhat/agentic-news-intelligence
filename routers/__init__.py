from routers.articles import router as articles_router
from routers.search import router as search_router
from routers.intelligence import router as intelligence_router
from routers.users import router as users_router
from routers.comments import router as comments_router

__all__ = [
    "articles_router",
    "search_router",
    "intelligence_router",
    "users_router",
    "comments_router",
]
