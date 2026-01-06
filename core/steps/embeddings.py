"""
Event embedding generation.

Extracted from experiments/04_2_embed_events.py for production pipeline.
"""

import numpy as np
from loguru import logger

from core.models import get_embedder
from core.state import PipelineState

# =============================================================================
# CONFIGURATION
# =============================================================================

BATCH_SIZE = 32


# =============================================================================
# EMBEDDING FUNCTIONS
# =============================================================================


def create_event_text(event: dict) -> str:
    """Create text representation of event for embedding."""
    parts = []

    # Title
    if title := event.get("title"):
        parts.append(title)

    # Description (truncated)
    if desc := event.get("description"):
        parts.append(desc[:500])

    # Market questions
    for market in event.get("markets", [])[:3]:
        if question := market.get("question"):
            parts.append(question)

    return " ".join(parts)


def embed_events(
    events: list[dict],
    state: PipelineState | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Generate embeddings for events.

    Args:
        events: List of NLP-prepared events
        state: Pipeline state (optional, for incremental append)

    Returns:
        Tuple of (embeddings array, event_ids list)
    """
    model = get_embedder()

    # Create texts for embedding
    texts = []
    event_ids = []

    for event in events:
        event_id = event.get("id")
        if not event_id:
            continue

        text = create_event_text(event)
        if text.strip():
            texts.append(text)
            event_ids.append(event_id)

    if not texts:
        logger.warning("No texts to embed")
        return np.array([]), []

    logger.info(f"Embedding {len(texts)} events...")

    # Encode in batches
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    embeddings = np.array(embeddings)
    logger.info(f"Generated embeddings with shape {embeddings.shape}")

    # Optionally append to state
    if state:
        state.append_embeddings(embeddings, event_ids)
        logger.info("Appended embeddings to state")

    return embeddings, event_ids


def get_similar_events(
    query_embedding: np.ndarray,
    all_embeddings: np.ndarray,
    event_ids: list[str],
    top_k: int = 10,
    threshold: float = 0.5,
) -> list[tuple[str, float]]:
    """
    Find similar events using cosine similarity.

    Args:
        query_embedding: Query vector (1D or 2D)
        all_embeddings: All event embeddings
        event_ids: Corresponding event IDs
        top_k: Number of results
        threshold: Minimum similarity threshold

    Returns:
        List of (event_id, similarity_score) tuples
    """
    if len(all_embeddings) == 0:
        return []

    # Ensure query is 2D
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    # Compute cosine similarities (embeddings are normalized)
    similarities = np.dot(all_embeddings, query_embedding.T).flatten()

    # Get top-k above threshold
    indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in indices:
        sim = similarities[idx]
        if sim >= threshold:
            results.append((event_ids[idx], float(sim)))

    return results
