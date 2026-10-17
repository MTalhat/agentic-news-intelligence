import os
import uuid
from typing import Optional
import numpy as np
from dotenv import load_dotenv
from fastapi import HTTPException

load_dotenv()

INGESTION_SECRET = os.getenv("INGESTION_SECRET")
if not INGESTION_SECRET:
    raise RuntimeError("INGESTION_SECRET environment variable must be set!")


def parse_vector(val) -> Optional[np.ndarray]:
    """Safely parses pgvector string/list/tuple/ndarray into a float32 NumPy array."""
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        return val.astype(np.float32)
    if isinstance(val, (list, tuple)):
        return np.array(val, dtype=np.float32)
    if isinstance(val, str):
        cleaned = val.strip().strip("[]()")
        if not cleaned:
            return None
        try:
            return np.array([float(x.strip()) for x in cleaned.split(",") if x.strip()], dtype=np.float32)
        except Exception:
            return None
    return np.array(val, dtype=np.float32)


def validate_uuid(raw_id: str, field_name: str = "article_id") -> str:
    """Sanitizes and validates a UUID string.
    Automatically strips accidental enclosing curly braces ({...}), quotes, and whitespace.
    Raises HTTPException(400) if not a valid UUID format."""
    if not raw_id:
        raise HTTPException(status_code=400, detail=f"{field_name} cannot be empty.")
    cleaned = str(raw_id).strip("{} \t\r\n\"'")
    try:
        return str(uuid.UUID(cleaned))
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid UUID format for {field_name}: '{raw_id}'. Expected valid 32-hex UUID."
        )
