from fastapi import APIRouter

from db import get_db
from schemas import CommentCreate
from taxonomy import get_taxonomy_hierarchy
from routers.deps import validate_uuid

router = APIRouter(tags=["Social Annotations & Taxonomy"])


@router.get("/api/taxonomy")
def get_taxonomy():
    """Returns official hierarchy of Macro categories and their partitioned micro tags.
    Powers the frontend onboarding modal with zero cross-domain tag contamination."""
    return get_taxonomy_hierarchy()


@router.post("/api/articles/{article_id}/comments")
def post_comment(article_id: str, c: CommentCreate):
    valid_article_id = validate_uuid(article_id, "article_id")
    valid_user_id = validate_uuid(c.user_id, "user_id")
    with get_db() as cur:
        cur.execute("""
            INSERT INTO article_comments (article_id, user_id, selected_text, comment_text)
            VALUES (%s, %s, %s, %s)
            RETURNING id, created_at;
        """, (valid_article_id, valid_user_id, c.selected_text, c.comment_text))
        res = cur.fetchone()
        return {"id": str(res[0]), "created_at": res[1]}


@router.get("/api/articles/{article_id}/comments")
def get_comments(article_id: str):
    valid_id = validate_uuid(article_id, "article_id")
    with get_db() as cur:
        cur.execute("""
            SELECT c.id, c.selected_text, c.comment_text, c.created_at, u.username, u.avatar_url
            FROM article_comments c
            JOIN users u ON c.user_id = u.id
            WHERE c.article_id = %s
            ORDER BY c.created_at ASC;
        """, (valid_id,))
        rows = cur.fetchall()
        return [
            {"id": str(r[0]), "selected_text": r[1], "comment": r[2], "created_at": r[3], "username": r[4], "avatar_url": r[5]}
            for r in rows
        ]
