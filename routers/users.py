from typing import Dict, Any
import numpy as np
from fastapi import APIRouter, HTTPException

from db import get_db
from embedder import get_embedder
from schemas import UserLoginRequest, UserPreferencesUpdate, UserInteractionRequest
from taxonomy import MACRO_CATEGORIES, clean_micro_tags
from routers.deps import validate_uuid, parse_vector

router = APIRouter(tags=["Users & Personalization"])


def get_personalized_feed_data(user_id: str, page: int = 1, page_size: int = 20) -> Dict[str, Any]:
    valid_user_id = validate_uuid(user_id, "user_id")
    offset = (page - 1) * page_size

    with get_db() as cur:
        cur.execute("""
            SELECT onboarding_completed, preferred_macro_tags, preferred_micro_tags, interest_embedding
            FROM users WHERE id = %s;
        """, (valid_user_id,))
        user_row = cur.fetchone()
        if not user_row:
            raise HTTPException(status_code=404, detail="User not found")

        onboarding_done, preferred_macros, preferred_micros, user_interest_emb = user_row
        preferred_macros = preferred_macros or []
        preferred_micros = preferred_micros or []

        # Cold-start guardrail: If onboarding incomplete or preferred_macro_tags empty, return chronological fallback
        if not onboarding_done or not preferred_macros:
            cur.execute("""
                SELECT id, source_feed, title, original_url, author, published_at, macro_tag, macro_tags, micro_tags, tldr
                FROM articles a
                WHERE a.is_processed = TRUE 
                  AND a.macro_tag IS NOT NULL 
                  AND a.macro_tag != 'Off-Topic'
                  AND (a.on_topic_score IS NULL OR a.on_topic_score >= 0.40)
                  AND NOT EXISTS (
                      SELECT 1 FROM user_article_interactions uai_d
                      WHERE uai_d.user_id = %s
                        AND uai_d.article_id = a.id
                        AND uai_d.interaction_type IN ('dislike', 'skip')
                  )
                ORDER BY a.published_at DESC LIMIT %s OFFSET %s;
            """, (valid_user_id, page_size, offset))
            fallback_rows = cur.fetchall()
            items = [
                {
                    "id": str(r[0]), "source": r[1], "title": r[2], "url": r[3],
                    "author": r[4], "published_at": r[5], "macro_tag": r[6],
                    "macro_tags": r[7] if r[7] else ([r[6]] if r[6] else []),
                    "micro_tags": r[8] or [], "tldr": r[9],
                    "personal_score": 50,
                    "is_liked": False,
                    "is_disliked": False
                }
                for r in fallback_rows
            ]
            return {
                "user_id": valid_user_id,
                "is_fallback": True,
                "fallback_reason": "Onboarding not completed or preferences empty",
                "preferred_macro_tags": preferred_macros,
                "preferred_micro_tags": preferred_micros,
                "total": len(items),
                "page": page,
                "page_size": page_size,
                "articles": items
            }

        # Multi-factor zero-leakage personalized ranking (excluding disliked/skipped articles)
        interest_vec_str = None
        user_vec = parse_vector(user_interest_emb)
        if user_vec is not None:
            interest_vec_str = str(user_vec.tolist())

        SQL_PERSONALIZED = """
        SELECT 
            a.id, a.source_feed, a.title, a.original_url, a.author, a.published_at, 
            a.macro_tag, a.macro_tags, a.micro_tags, a.tldr,
            (
                LEAST(40, 15 * cardinality(ARRAY(
                    SELECT unnest(a.micro_tags) 
                    INTERSECT 
                    SELECT unnest(%(preferred_micros)s::text[])
                ))) +
                (CASE WHEN a.macro_tag = ANY(%(preferred_macros)s) THEN 20 ELSE 0 END) +
                (CASE 
                    WHEN %(interest_vec)s::text IS NOT NULL AND a.embedding IS NOT NULL 
                    THEN ROUND(25.0 * GREATEST(0.0, 1.0 - (a.embedding <=> %(interest_vec)s::vector)))
                    ELSE 0 
                 END) +
                ROUND(15.0 * EXP(-EXTRACT(EPOCH FROM (NOW() - a.published_at)) / 259200.0))
            )::integer AS personal_score,
            EXISTS (
                SELECT 1 FROM user_article_interactions uai 
                WHERE uai.user_id = %(user_id)s 
                  AND uai.article_id = a.id 
                  AND uai.interaction_type = 'like'
            ) AS is_liked
        FROM articles a
        WHERE a.is_processed = TRUE 
          AND a.macro_tag IS NOT NULL 
          AND a.macro_tag != 'Off-Topic'
          AND (a.on_topic_score IS NULL OR a.on_topic_score >= 0.40)
          AND (a.macro_tag = ANY(%(preferred_macros)s) OR a.macro_tags && %(preferred_macros)s::text[])
          AND NOT EXISTS (
              SELECT 1 FROM user_article_interactions uai_d
              WHERE uai_d.user_id = %(user_id)s
                AND uai_d.article_id = a.id
                AND uai_d.interaction_type IN ('dislike', 'skip')
          )
        ORDER BY personal_score DESC, a.published_at DESC
        LIMIT %(limit)s OFFSET %(offset)s;
        """

        cur.execute(SQL_PERSONALIZED, {
            "preferred_macros": preferred_macros,
            "preferred_micros": preferred_micros,
            "interest_vec": interest_vec_str,
            "user_id": valid_user_id,
            "limit": page_size,
            "offset": offset
        })
        ranked_rows = cur.fetchall()

        items = [
            {
                "id": str(r[0]), "source": r[1], "title": r[2], "url": r[3],
                "author": r[4], "published_at": r[5], "macro_tag": r[6],
                "macro_tags": r[7] if r[7] else ([r[6]] if r[6] else []),
                "micro_tags": r[8] or [], "tldr": r[9],
                "personal_score": r[10],
                "is_liked": bool(r[11]),
                "is_disliked": False
            }
            for r in ranked_rows
        ]

        return {
            "user_id": valid_user_id,
            "is_fallback": False,
            "preferred_macro_tags": preferred_macros,
            "preferred_micro_tags": preferred_micros,
            "total": len(items),
            "page": page,
            "page_size": page_size,
            "articles": items
        }


@router.get("/api/feed/personalized")
def personalized_feed(user_id: str, page: int = 1, page_size: int = 20):
    """Zero-leakage personalized feed ranked by user preferences, GIN micro tags, and semantic vectors."""
    return get_personalized_feed_data(user_id=user_id, page=page, page_size=page_size)


@router.post("/api/articles/{article_id}/interactions")
def toggle_interaction(article_id: str, req: UserInteractionRequest):
    """
    Toggles like, dislike, skip, bookmark, or read state for an article.
    Feeds user interaction context directly into both the 384d interest_embedding
    (EMA attraction on 'like', negative EMA repulsion on 'dislike'/'skip') and
    the user's preferred_micro_tags / preferred_macro_tags profile.
    """
    valid_art_id = validate_uuid(article_id, "article_id")
    valid_user_id = validate_uuid(req.user_id, "user_id")
    itype = req.interaction_type.strip().lower()
    valid_types = ("like", "dislike", "skip", "bookmark", "read")
    if itype not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid interaction_type. Must be one of: {', '.join(valid_types)}."
        )

    with get_db() as cur:
        cur.execute("""
            SELECT id FROM user_article_interactions
            WHERE user_id = %s AND article_id = %s AND interaction_type = %s;
        """, (valid_user_id, valid_art_id, itype))
        existing = cur.fetchone()

        if existing:
            cur.execute("""
                DELETE FROM user_article_interactions
                WHERE user_id = %s AND article_id = %s AND interaction_type = %s;
            """, (valid_user_id, valid_art_id, itype))
            return {
                "status": "removed",
                "active": False,
                "interaction_type": itype,
                "article_id": valid_art_id,
                "user_id": valid_user_id
            }

        # Remove mutually exclusive opposite signals (like vs dislike/skip)
        if itype == "like":
            cur.execute("""
                DELETE FROM user_article_interactions
                WHERE user_id = %s AND article_id = %s AND interaction_type IN ('dislike', 'skip');
            """, (valid_user_id, valid_art_id))
        elif itype in ("dislike", "skip"):
            cur.execute("""
                DELETE FROM user_article_interactions
                WHERE user_id = %s AND article_id = %s AND interaction_type = 'like';
            """, (valid_user_id, valid_art_id))

        cur.execute("""
            INSERT INTO user_article_interactions (user_id, article_id, interaction_type)
            VALUES (%s, %s, %s);
        """, (valid_user_id, valid_art_id, itype))

        updated_micros = None
        updated_macros = None
        new_vec_str = None

        # Dynamic EMA Vector & Tag Context Learning on 'like' and 'dislike'/'skip'
        if itype in ("like", "dislike", "skip"):
            cur.execute("""
                SELECT embedding, macro_tag, micro_tags
                FROM articles WHERE id = %s;
            """, (valid_art_id,))
            art_row = cur.fetchone()
            art_emb = art_row[0] if art_row else None
            art_macro = art_row[1] if art_row else None
            art_micros = (art_row[2] or []) if art_row else []

            cur.execute("""
                SELECT interest_embedding, preferred_macro_tags, preferred_micro_tags
                FROM users WHERE id = %s;
            """, (valid_user_id,))
            user_row = cur.fetchone()
            user_emb = user_row[0] if user_row else None
            curr_macros = list(user_row[1] or []) if user_row else []
            curr_micros = list(user_row[2] or []) if user_row else []

            art_vec = parse_vector(art_emb)
            user_vec = parse_vector(user_emb)

            new_vec_str = None
            if art_vec is not None:
                if itype == "like":
                    # Positive EMA attraction (+0.15 toward liked article)
                    if user_vec is None:
                        new_vec = art_vec.tolist()
                    else:
                        blended = 0.85 * user_vec + 0.15 * art_vec
                        norm = float(np.linalg.norm(blended))
                        new_vec = (blended / norm if norm > 0 else blended).tolist()
                    new_vec_str = str(new_vec)
                else:
                    # Negative EMA repulsion (-0.12 away from disliked/skipped article)
                    if user_vec is not None:
                        repelled = user_vec - 0.12 * art_vec
                        norm = float(np.linalg.norm(repelled))
                        new_vec = (repelled / norm if norm > 0 else user_vec).tolist()
                        new_vec_str = str(new_vec)

            if itype == "like":
                if art_macro and art_macro != "Off-Topic" and art_macro not in curr_macros:
                    curr_macros.append(art_macro)
                for mt in art_micros[:2]:
                    if mt and mt != "off_topic" and mt not in curr_micros:
                        curr_micros.insert(0, mt)
                curr_micros = curr_micros[:15]
            else:
                # On dislike/skip, prune disliked article's micro-tags if user has other tags
                disliked_set = set(art_micros[:2])
                remaining = [t for t in curr_micros if t not in disliked_set]
                if remaining:
                    curr_micros = remaining

            updated_macros = curr_macros
            updated_micros = curr_micros

            if new_vec_str is not None:
                cur.execute("""
                    UPDATE users 
                    SET interest_embedding = %s::vector,
                        preferred_macro_tags = %s,
                        preferred_micro_tags = %s,
                        onboarding_completed = TRUE
                    WHERE id = %s;
                """, (new_vec_str, updated_macros, updated_micros, valid_user_id))
            else:
                cur.execute("""
                    UPDATE users 
                    SET preferred_macro_tags = %s,
                        preferred_micro_tags = %s
                    WHERE id = %s;
                """, (updated_macros, updated_micros, valid_user_id))

        return {
            "status": "recorded",
            "active": True,
            "interaction_type": itype,
            "article_id": valid_art_id,
            "user_id": valid_user_id,
            "ema_updated": bool(itype in ("like", "dislike", "skip") and new_vec_str is not None),
            "updated_preferred_macro_tags": updated_macros,
            "updated_preferred_micro_tags": updated_micros
        }


@router.post("/api/users/login")
def user_quick_login(req: UserLoginRequest):
    raw_name = req.username or req.name or ""
    username = raw_name.strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username cannot be empty")
    with get_db() as cur:
        cur.execute("""
            SELECT id, username, avatar_url, onboarding_completed, preferred_macro_tags, preferred_micro_tags 
            FROM users WHERE username = %s;
        """, (username,))
        row = cur.fetchone()
        if row:
            return {
                "id": str(row[0]), 
                "username": row[1], 
                "avatar_url": row[2],
                "onboarding_completed": bool(row[3]) if row[3] is not None else False,
                "preferred_macro_tags": row[4] or [],
                "preferred_micro_tags": row[5] or []
            }
        avatar = f"https://avatar.iran.liara.run/public/boy?username={username}"
        cur.execute("""
            INSERT INTO users (username, avatar_url, onboarding_completed) 
            VALUES (%s, %s, FALSE)
            RETURNING id, username, avatar_url, onboarding_completed, preferred_macro_tags, preferred_micro_tags;
        """, (username, avatar))
        new_row = cur.fetchone()
        return {
            "id": str(new_row[0]), 
            "username": new_row[1], 
            "avatar_url": new_row[2],
            "onboarding_completed": bool(new_row[3]),
            "preferred_macro_tags": new_row[4] or [],
            "preferred_micro_tags": new_row[5] or []
        }


@router.put("/api/users/{user_id}/preferences")
def update_user_preferences(user_id: str, prefs: UserPreferencesUpdate):
    """
    Saves user-selected macro/micro preferences from onboarding modal.
    Synthesizes clean natural language sentence and generates 384d dense interest vector.
    """
    valid_user_id = validate_uuid(user_id, "user_id")

    clean_macros = []
    for m in prefs.preferred_macro_tags:
        for canonical in MACRO_CATEGORIES:
            if canonical.lower() == m.strip().lower() and canonical not in clean_macros:
                clean_macros.append(canonical)
    if not clean_macros and prefs.preferred_macro_tags:
        clean_macros = [MACRO_CATEGORIES[0]]

    clean_micros = clean_micro_tags(prefs.preferred_micro_tags, max_tags=12)

    title_macros = [m.title() for m in clean_macros]
    title_micros = [tag.replace("_", " ").title() for tag in clean_micros]
    interest_text = f"Technical news and updates in {', '.join(title_macros)}, specifically focused on {', '.join(title_micros)}."

    embedder = get_embedder()
    interest_vector = embedder.embed_text(interest_text)

    with get_db() as cur:
        cur.execute("""
            UPDATE users 
            SET preferred_macro_tags = %s,
                preferred_micro_tags = %s,
                onboarding_completed = TRUE,
                interest_embedding = %s::vector
            WHERE id = %s
            RETURNING id, username, avatar_url, onboarding_completed, preferred_macro_tags, preferred_micro_tags;
        """, (clean_macros, clean_micros, str(interest_vector), valid_user_id))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="User not found")
        return {
            "id": str(row[0]),
            "username": row[1],
            "avatar_url": row[2],
            "onboarding_completed": row[3],
            "preferred_macro_tags": row[4] or [],
            "preferred_micro_tags": row[5] or [],
            "interest_summary": interest_text
        }


@router.get("/api/users/{user_id}/preferences")
def get_user_preferences(user_id: str):
    valid_user_id = validate_uuid(user_id, "user_id")
    with get_db() as cur:
        cur.execute("""
            SELECT id, username, avatar_url, onboarding_completed, preferred_macro_tags, preferred_micro_tags
            FROM users WHERE id = %s;
        """, (valid_user_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="User not found")
        return {
            "id": str(row[0]),
            "username": row[1],
            "avatar_url": row[2],
            "onboarding_completed": bool(row[3]),
            "preferred_macro_tags": row[4] or [],
            "preferred_micro_tags": row[5] or []
        }


@router.get("/api/users")
def get_all_users():
    with get_db() as cur:
        cur.execute("SELECT id, username, avatar_url, created_at FROM users ORDER BY created_at ASC;")
        rows = cur.fetchall()
        return [
            {"id": str(r[0]), "username": r[1], "avatar_url": r[2], "created_at": r[3]}
            for r in rows
        ]
