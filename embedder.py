"""Local GPU/CPU Embedding Engine using sentence-transformers/all-MiniLM-L6-v2.

Provides sub-10ms 384-dimensional dense vector embeddings for hybrid semantic search
and article enrichment, with automatic cache check and offline execution.
"""

import os
import logging
from typing import List

# Suppress verbose Hugging Face hub & tokenizers warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logger = logging.getLogger("embedder")

_embedder_instance = None


class TextEmbedder:
    """Singleton wrapper for local sentence-transformers model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        import torch
        from sentence_transformers import SentenceTransformer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name

        # Safe loading pattern:
        # First attempt to load with offline mode enabled.
        # If not cached locally, allow online download once, then lock to offline mode.
        try:
            os.environ["HF_HUB_OFFLINE"] = "1"
            logger.info("Loading %s in offline mode on %s...", self.model_name, self.device)
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Loaded %s from local cache on %s.", self.model_name, self.device)
        except Exception as e:
            logger.info("Local cache miss or offline load failed (%s). Downloading model...", e)
            os.environ.pop("HF_HUB_OFFLINE", None)
            self.model = SentenceTransformer(self.model_name, device=self.device)
            os.environ["HF_HUB_OFFLINE"] = "1"
            logger.info("Successfully downloaded and cached %s on %s.", self.model_name, self.device)

    def embed_text(self, text: str) -> List[float]:
        """Embeds a single string into a 384-dimensional float vector."""
        if not text or not text.strip():
            text = "empty"
        vec = self.model.encode(text.strip(), convert_to_numpy=True, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embeds a batch of strings efficiently into 384-dimensional vectors."""
        clean_texts = [t.strip() if t and t.strip() else "empty" for t in texts]
        vecs = self.model.encode(clean_texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.tolist()


def get_embedder() -> TextEmbedder:
    """Returns the singleton instance of TextEmbedder."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = TextEmbedder()
    return _embedder_instance
