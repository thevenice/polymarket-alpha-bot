"""
Generate semantic embeddings for Polymarket events.

Pipeline: 02_prepare_nlp_data -> 03_2_dedupe_entities -> 03_3_normalize_entities -> [04_2_embed_events]

This script:
1. Reads events from 02_prepare_nlp_data output
2. Loads extracted entities per event from 03_2_dedupe_entities
3. Loads canonical mappings from 03_3_normalize_entities
4. Creates enriched text: title + entities grouped by type
5. Generates embeddings using BGE-base model
6. Saves embeddings + entity sets per event (for Jaccard similarity)

Quality improvements over simple title+tags approach:
- Uses extracted entities instead of noisy tags (removed "Breaking News", "2025 Predictions", etc.)
- Maps entity variants to canonical names ("Trump" -> "donald trump")
- Groups entities by semantic type (politician, country, organization, etc.)
- Outputs entity sets for hybrid similarity computation (embedding + Jaccard)

Uses sentence-transformers for embedding generation.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base data directory
DATA_DIR = Path(__file__).parent.parent / "data"

# Input: events from 02_prepare_nlp_data
INPUT_EVENTS_DIR = DATA_DIR / "02_prepare_nlp_data"
INPUT_EVENTS_FOLDER: str | None = None  # None = latest

# Input: entities per event from 03_2_dedupe_entities
INPUT_ENTITIES_DIR = DATA_DIR / "03_2_dedupe_entities"
INPUT_ENTITIES_FOLDER: str | None = None  # None = latest

# Input: canonical mappings from 03_3_normalize_entities
INPUT_NORMALIZE_DIR = DATA_DIR / "03_3_normalize_entities"
INPUT_NORMALIZE_FOLDER: str | None = None  # None = latest

# Output: writes to 04_2_embed_events folder
SCRIPT_OUTPUT_DIR = DATA_DIR / "04_2_embed_events"

# Model settings
MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM = 768
NORMALIZE_EMBEDDINGS = True  # L2 normalize for cosine similarity

# Processing settings
BATCH_SIZE = 32  # Events per batch for encoding
MIN_ENTITY_CONFIDENCE = 0.5  # Minimum confidence to include entity

# Entity type ordering (for consistent text generation)
ENTITY_TYPE_ORDER = [
    "POLITICIAN",
    "COUNTRY",
    "POLITICAL_PARTY",
    "GOVERNMENT_AGENCY",
    "INTERNATIONAL_ORGANIZATION",
    "STATE",
    "POLITICAL_EVENT",
    "LEGISLATION",
]

# Logging
LOG_LEVEL = logging.INFO

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def find_latest_run_folder(script_dir: Path) -> Path | None:
    """Find the most recent run folder."""
    if not script_dir.exists():
        return None
    run_folders = [f for f in script_dir.iterdir() if f.is_dir()]
    if not run_folders:
        return None
    return max(run_folders, key=lambda f: f.stat().st_mtime)


def build_variant_to_canonical(merge_mappings: list[dict]) -> dict[str, str]:
    """Build lookup from variant text to canonical name."""
    lookup = {}
    for mapping in merge_mappings:
        canonical = mapping["canonical"]
        for variant in mapping["variants"]:
            lookup[variant.lower()] = canonical
    return lookup


def get_canonical_name(text: str, variant_lookup: dict[str, str]) -> str:
    """Get canonical name for entity text, or return normalized text."""
    normalized = text.lower().strip()
    return variant_lookup.get(normalized, normalized)


def create_enriched_text(
    event: dict,
    event_entities: list[dict],
    variant_lookup: dict[str, str],
) -> tuple[str, set[str]]:
    """
    Create enriched text for embedding and return entity set.

    Returns:
        tuple: (enriched_text, set_of_canonical_entities)
    """
    title = event.get("title", "").strip()

    # Group entities by type, map to canonical names
    entities_by_type: dict[str, set[str]] = defaultdict(set)
    all_entities: set[str] = set()

    for ent in event_entities:
        if ent.get("confidence", 0) < MIN_ENTITY_CONFIDENCE:
            continue

        text = ent.get("text", "")
        label = ent.get("label", "OTHER")

        canonical = get_canonical_name(text, variant_lookup)
        entities_by_type[label].add(canonical)
        all_entities.add(canonical)

    # Build text with entities grouped by type
    parts = [title]

    for entity_type in ENTITY_TYPE_ORDER:
        if entity_type in entities_by_type:
            entities = sorted(entities_by_type[entity_type])
            type_label = entity_type.lower().replace("_", " ")
            parts.append(f"{type_label}: {', '.join(entities)}")

    # Add any remaining types not in ORDER
    for entity_type, entities in entities_by_type.items():
        if entity_type not in ENTITY_TYPE_ORDER:
            entities_list = sorted(entities)
            type_label = entity_type.lower().replace("_", " ")
            parts.append(f"{type_label}: {', '.join(entities_list)}")

    enriched_text = ". ".join(parts)
    return enriched_text, all_entities


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    start_time = datetime.now(timezone.utc)

    logger.info("=" * 60)
    logger.info("Event Embedding Generation (Entity-Enriched)")
    logger.info("=" * 60)

    # Find input folders
    events_folder = (
        INPUT_EVENTS_DIR / INPUT_EVENTS_FOLDER
        if INPUT_EVENTS_FOLDER
        else find_latest_run_folder(INPUT_EVENTS_DIR)
    )
    entities_folder = (
        INPUT_ENTITIES_DIR / INPUT_ENTITIES_FOLDER
        if INPUT_ENTITIES_FOLDER
        else find_latest_run_folder(INPUT_ENTITIES_DIR)
    )
    normalize_folder = (
        INPUT_NORMALIZE_DIR / INPUT_NORMALIZE_FOLDER
        if INPUT_NORMALIZE_FOLDER
        else find_latest_run_folder(INPUT_NORMALIZE_DIR)
    )

    if not events_folder or not events_folder.exists():
        logger.error(f"Events folder not found: {events_folder}")
        return
    if not entities_folder or not entities_folder.exists():
        logger.error(f"Entities folder not found: {entities_folder}")
        return
    if not normalize_folder or not normalize_folder.exists():
        logger.error(f"Normalize folder not found: {normalize_folder}")
        return

    logger.info(f"Events folder: {events_folder}")
    logger.info(f"Entities folder: {entities_folder}")
    logger.info(f"Normalize folder: {normalize_folder}")

    # Load events
    events_file = events_folder / "nlp_events.json"
    logger.info(f"Loading events from: {events_file}")
    with open(events_file, encoding="utf-8") as f:
        events_data = json.load(f)
    events = events_data.get("events", [])
    events_by_id = {e["id"]: e for e in events}
    logger.info(f"Loaded {len(events)} events")

    # Load entities by event
    entities_file = entities_folder / "entities_by_source.json"
    logger.info(f"Loading entities from: {entities_file}")
    with open(entities_file, encoding="utf-8") as f:
        entities_data = json.load(f)
    entities_by_event = entities_data.get("by_event", {})
    logger.info(f"Loaded entities for {len(entities_by_event)} events")

    # Load merge mappings for canonical names
    mappings_file = normalize_folder / "merge_mappings.json"
    logger.info(f"Loading mappings from: {mappings_file}")
    with open(mappings_file, encoding="utf-8") as f:
        mappings_data = json.load(f)
    variant_lookup = build_variant_to_canonical(mappings_data.get("merges", []))
    logger.info(f"Loaded {len(variant_lookup)} variant mappings")

    # Load model
    logger.info(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    logger.info(
        f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}"
    )

    # Prepare texts for embedding
    logger.info("Preparing enriched event texts...")
    event_ids = []
    event_texts = []
    event_entity_sets: dict[
        str, list[str]
    ] = {}  # event_id -> list of canonical entities

    events_with_entities = 0
    events_without_entities = 0

    for event in events:
        event_id = event.get("id")
        if not event_id:
            continue

        # Get entities for this event
        event_ents_data = entities_by_event.get(event_id, {})
        event_ents = event_ents_data.get("event_entities", [])

        # Also include market entities if present
        market_ents = event_ents_data.get("market_entities", {})
        for market_id, ments in market_ents.items():
            event_ents.extend(ments)

        # Create enriched text
        enriched_text, entity_set = create_enriched_text(
            event, event_ents, variant_lookup
        )

        if entity_set:
            events_with_entities += 1
        else:
            events_without_entities += 1

        event_ids.append(event_id)
        event_texts.append(enriched_text)
        event_entity_sets[event_id] = sorted(entity_set)

    logger.info(f"Prepared {len(event_texts)} texts for embedding")
    logger.info(f"  Events with entities: {events_with_entities}")
    logger.info(f"  Events without entities: {events_without_entities}")

    # Generate embeddings
    logger.info(f"Generating embeddings (batch_size={BATCH_SIZE})...")
    embeddings = model.encode(
        event_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        convert_to_numpy=True,
    )

    logger.info(f"Generated embeddings shape: {embeddings.shape}")

    # Verify normalization
    norms = np.linalg.norm(embeddings, axis=1)
    logger.info(
        f"Vector norms - min: {norms.min():.4f}, max: {norms.max():.4f}, mean: {norms.mean():.4f}"
    )

    # Prepare output folder
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_folder = SCRIPT_OUTPUT_DIR / timestamp
    output_folder.mkdir(parents=True, exist_ok=True)

    # Save embeddings as NumPy binary
    embeddings_file = output_folder / "embeddings.npy"
    np.save(embeddings_file, embeddings.astype(np.float32))
    logger.info(f"Saved: {embeddings_file}")

    # Save metadata JSON (event_id -> index mapping + event info + entities)
    metadata = {
        "_meta": {
            "description": "Entity-enriched event embeddings metadata",
            "source_events": str(events_file),
            "source_entities": str(entities_file),
            "source_mappings": str(mappings_file),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": MODEL_NAME,
            "embedding_dim": EMBEDDING_DIM,
            "normalized": NORMALIZE_EMBEDDINGS,
            "min_entity_confidence": MIN_ENTITY_CONFIDENCE,
        },
        "id_to_index": {eid: idx for idx, eid in enumerate(event_ids)},
        "index_to_id": event_ids,
        "events": [
            {
                "id": eid,
                "title": events_by_id.get(eid, {}).get("title", ""),
                "text_embedded": event_texts[idx],
                "entities": event_entity_sets.get(eid, []),
            }
            for idx, eid in enumerate(event_ids)
        ],
    }

    metadata_file = output_folder / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: {metadata_file}")

    # Save entity sets separately for efficient Jaccard computation
    entity_sets_file = output_folder / "entity_sets.json"
    with open(entity_sets_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "_meta": {
                    "description": "Canonical entity sets per event for Jaccard similarity",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                "entity_sets": event_entity_sets,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info(f"Saved: {entity_sets_file}")

    # Save summary
    end_time = datetime.now(timezone.utc)

    # Compute entity coverage stats
    entity_counts = [len(event_entity_sets.get(eid, [])) for eid in event_ids]

    summary = {
        "run_info": {
            "started_at": start_time.isoformat(),
            "completed_at": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "input_events": str(events_file),
            "input_entities": str(entities_file),
            "input_mappings": str(mappings_file),
            "output_folder": str(output_folder),
        },
        "model": {
            "name": MODEL_NAME,
            "embedding_dim": EMBEDDING_DIM,
            "normalized": NORMALIZE_EMBEDDINGS,
        },
        "embeddings": {
            "total_events": len(events),
            "embedded_events": len(event_ids),
            "shape": list(embeddings.shape),
            "dtype": str(embeddings.dtype),
            "file_size_mb": round(embeddings_file.stat().st_size / (1024 * 1024), 2),
        },
        "entities": {
            "events_with_entities": events_with_entities,
            "events_without_entities": events_without_entities,
            "min_entities_per_event": min(entity_counts) if entity_counts else 0,
            "max_entities_per_event": max(entity_counts) if entity_counts else 0,
            "avg_entities_per_event": round(sum(entity_counts) / len(entity_counts), 2)
            if entity_counts
            else 0,
            "min_entity_confidence": MIN_ENTITY_CONFIDENCE,
        },
        "vector_stats": {
            "norm_min": float(norms.min()),
            "norm_max": float(norms.max()),
            "norm_mean": float(norms.mean()),
        },
        "sample_texts": event_texts[:5],
    }

    summary_file = output_folder / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: {summary_file}")

    # Print summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Events embedded: {len(event_ids)}")
    logger.info(f"Embedding shape: {embeddings.shape}")
    logger.info(f"Events with entities: {events_with_entities}")
    logger.info(f"Avg entities/event: {summary['entities']['avg_entities_per_event']}")
    logger.info(f"File size: {summary['embeddings']['file_size_mb']} MB")
    logger.info(f"Duration: {(end_time - start_time).total_seconds():.1f}s")
    logger.info("-" * 40)
    logger.info("Sample enriched texts:")
    for text in event_texts[:3]:
        logger.info(f"  {text[:100]}...")
    logger.info("-" * 40)
    logger.info(f"Output: {output_folder}")


if __name__ == "__main__":
    main()
