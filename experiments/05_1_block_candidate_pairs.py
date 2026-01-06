"""
Fast candidate pair blocking using FAISS HNSW with multi-hop expansion.

Pipeline: 05_0_enrich_quality + 04_2_embed_events -> [05_1_block_candidate_pairs] -> 05_2_classify_structural

This script:
1. Loads quality-enriched events from 05_0_enrich_quality
2. Loads embeddings from 04_2_embed_events
3. Builds FAISS index for fast similarity search
4. Builds entity inverted index for multi-hop expansion
5. Finds direct candidates via FAISS (top-k neighbors)
6. Expands candidates via shared entities (multi-hop)
7. Computes hybrid scores (embedding + entity Jaccard)
8. Outputs blocked candidate pairs for classification
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "data"

# Inputs
INPUT_QUALITY_DIR = DATA_DIR / "05_0_enrich_quality"
INPUT_EMBEDDINGS_DIR = DATA_DIR / "04_2_embed_events"
INPUT_RUN_FOLDER: str | None = None  # None = use latest

# Output
SCRIPT_OUTPUT_DIR = DATA_DIR / "05_1_block_candidate_pairs"

# FAISS parameters
FAISS_TOP_K = 50  # Neighbors to retrieve per event

# Hybrid scoring weights
EMBEDDING_WEIGHT = 0.7
ENTITY_JACCARD_WEIGHT = 0.3

# Blocking threshold
BLOCKING_THRESHOLD = 0.35

# Multi-hop expansion settings
ENABLE_MULTI_HOP = True
MAX_MULTI_HOP_PER_EVENT = 20
MULTI_HOP_MIN_ENTITY_OVERLAP = 1

# Logging
LOG_LEVEL = logging.INFO

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class CandidatePair:
    """A candidate pair for relation classification."""

    event_id_a: str
    event_id_b: str
    title_a: str
    title_b: str
    embedding_similarity: float
    entity_jaccard: float
    hybrid_score: float
    shared_entities: list[str] = field(default_factory=list)
    blocking_method: str = "direct"
    blocking_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "event_id_a": self.event_id_a,
            "event_id_b": self.event_id_b,
            "title_a": self.title_a,
            "title_b": self.title_b,
            "embedding_similarity": round(self.embedding_similarity, 6),
            "entity_jaccard": round(self.entity_jaccard, 6),
            "hybrid_score": round(self.hybrid_score, 6),
            "shared_entities": self.shared_entities,
            "blocking_method": self.blocking_method,
            "blocking_reasons": self.blocking_reasons,
        }


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


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def build_entity_inverted_index(events: list[dict]) -> dict[str, set[str]]:
    """Build inverted index: entity -> set of event IDs."""
    index: dict[str, set[str]] = defaultdict(set)

    for event in events:
        event_id = event["id"]
        entities = event.get("entities", [])
        for entity in entities:
            index[entity].add(event_id)

    return dict(index)


def find_shared_entities(entities_a: list[str], entities_b: list[str]) -> list[str]:
    """Find entities shared between two events."""
    set_a = set(entities_a) if entities_a else set()
    set_b = set(entities_b) if entities_b else set()
    return sorted(set_a & set_b)


def expand_candidates_multi_hop(
    event_id: str,
    direct_candidates: list[tuple[str, float]],
    entity_index: dict[str, set[str]],
    events_by_id: dict[str, dict],
    embeddings: np.ndarray,
    id_to_idx: dict[str, int],
) -> list[tuple[str, float, list[str]]]:
    """
    Expand candidates via shared entities (multi-hop).

    Returns: [(event_id, embedding_similarity, hop_path), ...]
    """
    if not ENABLE_MULTI_HOP:
        return []

    event = events_by_id.get(event_id)
    if not event:
        return []

    event_entities = set(event.get("entities", []))
    if not event_entities:
        return []

    direct_ids = {eid for eid, _ in direct_candidates}
    expanded = []

    for candidate_id, _ in direct_candidates:
        candidate = events_by_id.get(candidate_id)
        if not candidate:
            continue

        candidate_entities = set(candidate.get("entities", []))
        shared_with_candidate = event_entities & candidate_entities

        if len(shared_with_candidate) < MULTI_HOP_MIN_ENTITY_OVERLAP:
            continue

        for entity in candidate_entities:
            if entity not in entity_index:
                continue

            for hop_event_id in entity_index[entity]:
                if hop_event_id in direct_ids or hop_event_id == event_id:
                    continue

                if hop_event_id not in id_to_idx or event_id not in id_to_idx:
                    continue

                idx_a = id_to_idx[event_id]
                idx_b = id_to_idx[hop_event_id]
                sim = float(np.dot(embeddings[idx_a], embeddings[idx_b]))

                hop_path = [event_id, candidate_id, hop_event_id]
                expanded.append((hop_event_id, sim, hop_path))

    # Deduplicate and sort by similarity
    seen = set()
    unique_expanded = []
    for hop_id, sim, path in sorted(expanded, key=lambda x: -x[1]):
        if hop_id not in seen:
            seen.add(hop_id)
            unique_expanded.append((hop_id, sim, path))
            if len(unique_expanded) >= MAX_MULTI_HOP_PER_EVENT:
                break

    return unique_expanded


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Main entry point."""
    start_time = datetime.now(timezone.utc)
    logger.info("Starting 05_1_block_candidate_pairs")

    # Determine input folders
    if INPUT_RUN_FOLDER:
        quality_folder = INPUT_QUALITY_DIR / INPUT_RUN_FOLDER
        embeddings_folder = INPUT_EMBEDDINGS_DIR / INPUT_RUN_FOLDER
    else:
        quality_folder = find_latest_run_folder(INPUT_QUALITY_DIR)
        embeddings_folder = find_latest_run_folder(INPUT_EMBEDDINGS_DIR)

    if not quality_folder or not quality_folder.exists():
        logger.error(f"Quality folder not found: {quality_folder}")
        return
    if not embeddings_folder or not embeddings_folder.exists():
        logger.error(f"Embeddings folder not found: {embeddings_folder}")
        return

    logger.info(f"Loading from: {quality_folder}")
    logger.info(f"Loading embeddings from: {embeddings_folder}")

    # Load quality-enriched events
    with open(quality_folder / "quality_enriched_events.json", encoding="utf-8") as f:
        quality_data = json.load(f)
    events = quality_data.get("events", [])
    logger.info(f"Loaded {len(events)} quality-enriched events")

    # Load embeddings
    embeddings = np.load(embeddings_folder / "embeddings.npy")
    logger.info(f"Loaded embeddings: {embeddings.shape}")

    # Load metadata for ID mapping
    with open(embeddings_folder / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    id_to_idx = metadata.get("id_to_index", {})
    idx_to_id = {v: k for k, v in id_to_idx.items()}
    logger.info(f"Loaded ID mappings for {len(id_to_idx)} events")

    # Build events lookup
    events_by_id = {e["id"]: e for e in events}

    # Build FAISS index
    logger.info("Building FAISS index...")
    # Normalize embeddings for inner product = cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / np.maximum(norms, 1e-10)

    dimension = normalized_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(normalized_embeddings)
    logger.info(f"FAISS index built with {index.ntotal} vectors")

    # Build entity inverted index
    logger.info("Building entity inverted index...")
    entity_index = build_entity_inverted_index(events)
    logger.info(f"Entity index has {len(entity_index)} entities")

    # Find candidates
    logger.info(f"Finding top-{FAISS_TOP_K} candidates per event...")
    distances, indices = index.search(normalized_embeddings, FAISS_TOP_K + 1)

    # Build candidate pairs
    logger.info("Building candidate pairs with hybrid scoring...")
    pairs_seen: set[tuple[str, str]] = set()
    candidate_pairs: list[CandidatePair] = []

    stats = {
        "direct_pairs": 0,
        "multi_hop_pairs": 0,
        "total_before_threshold": 0,
        "total_after_threshold": 0,
    }

    for i, event_id in enumerate(idx_to_id.values()):
        event = events_by_id.get(event_id)
        if not event:
            continue

        event_entities = set(event.get("entities", []))

        # Process direct candidates
        for j in range(len(indices[i])):
            neighbor_idx = indices[i][j]
            if neighbor_idx == i or neighbor_idx < 0:
                continue

            neighbor_id = idx_to_id.get(neighbor_idx)
            if neighbor_id is None:
                continue

            pair_key = tuple(sorted([event_id, neighbor_id]))
            if pair_key in pairs_seen:
                continue
            pairs_seen.add(pair_key)

            neighbor = events_by_id.get(neighbor_id)
            if not neighbor:
                continue

            neighbor_entities = set(neighbor.get("entities", []))

            emb_sim = float(distances[i][j])
            entity_jaccard = jaccard_similarity(event_entities, neighbor_entities)
            hybrid_score = (
                EMBEDDING_WEIGHT * emb_sim + ENTITY_JACCARD_WEIGHT * entity_jaccard
            )

            stats["total_before_threshold"] += 1

            if hybrid_score >= BLOCKING_THRESHOLD:
                shared = find_shared_entities(
                    event.get("entities", []), neighbor.get("entities", [])
                )

                # Determine blocking reasons
                reasons = []
                if emb_sim >= 0.80:
                    reasons.append("high_similarity")
                if shared:
                    reasons.append("shared_entity")

                candidate_pairs.append(
                    CandidatePair(
                        event_id_a=pair_key[0],
                        event_id_b=pair_key[1],
                        title_a=events_by_id[pair_key[0]]["title"],
                        title_b=events_by_id[pair_key[1]]["title"],
                        embedding_similarity=emb_sim,
                        entity_jaccard=entity_jaccard,
                        hybrid_score=hybrid_score,
                        shared_entities=shared,
                        blocking_method="direct",
                        blocking_reasons=reasons,
                    )
                )
                stats["direct_pairs"] += 1

        # Multi-hop expansion
        if ENABLE_MULTI_HOP:
            direct_candidates = [
                (idx_to_id.get(indices[i][j]), float(distances[i][j]))
                for j in range(len(indices[i]))
                if indices[i][j] != i
                and indices[i][j] >= 0
                and idx_to_id.get(indices[i][j])
            ]

            expanded = expand_candidates_multi_hop(
                event_id,
                direct_candidates,
                entity_index,
                events_by_id,
                normalized_embeddings,
                id_to_idx,
            )

            for hop_id, emb_sim, hop_path in expanded:
                pair_key = tuple(sorted([event_id, hop_id]))
                if pair_key in pairs_seen:
                    continue
                pairs_seen.add(pair_key)

                hop_event = events_by_id.get(hop_id)
                if not hop_event:
                    continue

                hop_entities = set(hop_event.get("entities", []))
                entity_jaccard = jaccard_similarity(event_entities, hop_entities)
                hybrid_score = (
                    EMBEDDING_WEIGHT * emb_sim + ENTITY_JACCARD_WEIGHT * entity_jaccard
                )

                stats["total_before_threshold"] += 1

                if hybrid_score >= BLOCKING_THRESHOLD:
                    shared = find_shared_entities(
                        event.get("entities", []), hop_event.get("entities", [])
                    )
                    candidate_pairs.append(
                        CandidatePair(
                            event_id_a=pair_key[0],
                            event_id_b=pair_key[1],
                            title_a=events_by_id[pair_key[0]]["title"],
                            title_b=events_by_id[pair_key[1]]["title"],
                            embedding_similarity=emb_sim,
                            entity_jaccard=entity_jaccard,
                            hybrid_score=hybrid_score,
                            shared_entities=shared,
                            blocking_method="multi_hop",
                            blocking_reasons=["multi_hop_entity"],
                        )
                    )
                    stats["multi_hop_pairs"] += 1

    stats["total_after_threshold"] = len(candidate_pairs)

    # Sort by hybrid score descending
    candidate_pairs.sort(key=lambda p: -p.hybrid_score)

    logger.info(f"Found {len(candidate_pairs)} candidate pairs")
    logger.info(f"  Direct pairs: {stats['direct_pairs']}")
    logger.info(f"  Multi-hop pairs: {stats['multi_hop_pairs']}")

    # Create output folder
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    output_folder = SCRIPT_OUTPUT_DIR / timestamp
    output_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output folder: {output_folder}")

    # Save candidate pairs (using 'pairs' key for backward compatibility)
    pairs_output = {
        "_meta": {
            "description": "Candidate event pairs for relation classification",
            "created_at": start_time.isoformat(),
            "source_quality": str(quality_folder),
            "source_embeddings": str(embeddings_folder),
            "blocking_method": "faiss_hybrid",
        },
        "pairs": [p.to_dict() for p in candidate_pairs],
    }
    with open(output_folder / "candidate_pairs.json", "w", encoding="utf-8") as f:
        json.dump(pairs_output, f, indent=2, ensure_ascii=False)
    logger.info("Saved candidate_pairs.json")

    # Save entity inverted index
    index_output = {
        "_meta": {
            "description": "Entity to event IDs inverted index",
            "created_at": start_time.isoformat(),
        },
        "index": {k: sorted(v) for k, v in entity_index.items()},
    }
    with open(output_folder / "entity_inverted_index.json", "w", encoding="utf-8") as f:
        json.dump(index_output, f, indent=2, ensure_ascii=False)
    logger.info("Saved entity_inverted_index.json")

    # Compute reduction statistics
    n_events = len(events)
    total_possible_pairs = n_events * (n_events - 1) // 2
    blocked_pairs = len(candidate_pairs)
    reduction_ratio = (
        blocked_pairs / total_possible_pairs if total_possible_pairs > 0 else 0
    )

    # Save summary
    end_time = datetime.now(timezone.utc)
    summary = {
        "run_info": {
            "started_at": start_time.isoformat(),
            "completed_at": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "input_quality_folder": str(quality_folder),
            "input_embeddings_folder": str(embeddings_folder),
            "output_folder": str(output_folder),
        },
        "configuration": {
            "faiss_top_k": FAISS_TOP_K,
            "embedding_weight": EMBEDDING_WEIGHT,
            "entity_jaccard_weight": ENTITY_JACCARD_WEIGHT,
            "blocking_threshold": BLOCKING_THRESHOLD,
            "enable_multi_hop": ENABLE_MULTI_HOP,
            "max_multi_hop_per_event": MAX_MULTI_HOP_PER_EVENT,
        },
        "statistics": {
            "total_events": n_events,
            "total_entities_in_index": len(entity_index),
            "total_possible_pairs": total_possible_pairs,
            "pairs_evaluated": stats["total_before_threshold"],
            "pairs_after_threshold": stats["total_after_threshold"],
            "direct_pairs": stats["direct_pairs"],
            "multi_hop_pairs": stats["multi_hop_pairs"],
            "reduction_ratio": round(reduction_ratio, 6),
            "reduction_percent": round((1 - reduction_ratio) * 100, 2),
            "avg_hybrid_score": (
                sum(p.hybrid_score for p in candidate_pairs) / len(candidate_pairs)
                if candidate_pairs
                else 0
            ),
        },
    }
    with open(output_folder / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Saved summary.json")

    logger.info(f"Completed in {summary['run_info']['duration_seconds']:.2f}s")


if __name__ == "__main__":
    main()
