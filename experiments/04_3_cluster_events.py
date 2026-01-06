"""
Cluster events by multiple dimensions.

Pipeline: 04_1_extract_event_semantics + 04_2_embed_events + 03_3_normalize_entities -> 04_3_cluster_events

Dimensions created:
1. CONCEPT: Normalized event titles (removing dates, numbers, thresholds)
2. TOPIC_CLUSTER: HDBSCAN clustering on embeddings, auto-labeled by top entities
3. Entity facets: One dimension per entity type (POLITICIAN, COUNTRY, etc.)
"""

import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "data"

# Inputs
INPUT_SEMANTICS_DIR = DATA_DIR / "04_1_extract_event_semantics"
INPUT_EMBEDDINGS_DIR = DATA_DIR / "04_2_embed_events"
INPUT_NORMALIZE_DIR = DATA_DIR / "03_3_normalize_entities"
INPUT_RUN_FOLDER: str | None = None  # Use latest if None

# Output
SCRIPT_OUTPUT_DIR = DATA_DIR / "04_3_cluster_events"

# Concept extraction settings
CONCEPT_MIN_GROUP_SIZE = 2  # Minimum events to form a concept group
CONCEPT_USE_ENTITIES = True  # Include top entity in concept name for disambiguation

# Entity facet settings
MIN_ENTITY_FREQUENCY = 3  # Entity must appear in >= N events to form a group

# HDBSCAN settings
HDBSCAN_MIN_CLUSTER_SIZE = 10
HDBSCAN_MIN_SAMPLES = 5
TOPIC_LABEL_TOP_K = 3  # Use top K entities to auto-label topic

# Logging
LOG_LEVEL = logging.INFO

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# HELPERS
# =============================================================================


def find_latest_run_folder(script_dir: Path) -> Path | None:
    """Find the most recent run folder in a script output directory."""
    if not script_dir.exists():
        return None
    run_folders = [f for f in script_dir.iterdir() if f.is_dir()]
    if not run_folders:
        return None
    return max(run_folders, key=lambda f: f.stat().st_mtime)


# =============================================================================
# CONCEPT EXTRACTION
# =============================================================================


def extract_concept_from_title(title: str) -> str:
    """
    Extract the core concept from an event title by normalizing away parameters.

    This removes:
    - Specific dates (December 16 - December 23, January 2026, etc.)
    - Numeric ranges/thresholds (420-439, 0-19, 5%, 10%, etc.)
    - Years (2025, 2026, etc.)
    - Ordinal indicators (1st, 2nd, 3rd, etc.)

    Examples:
    - "Elon Musk # tweets December 16 - December 23, 2025?" -> "elon musk tweets"
    - "Will Elon Musk post 420-439 tweets from Dec 23 to Dec 30?" -> "will elon musk post tweets from to"
    - "Trump approval rating above 50% by March 2025?" -> "trump approval rating above by"
    - "Ukraine signs peace deal with Russia in 2025?" -> "ukraine signs peace deal with russia in"
    """
    text = title.lower()

    # Remove date ranges like "December 16 - December 23" or "Dec 23 - Jan 2"
    months = (
        r"(?:january|february|march|april|may|june|july|august|september|october|"
        r"november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
    )
    # Date range: "Month DD - Month DD" or "Month DD to Month DD"
    text = re.sub(
        rf"{months}\s*\d{{1,2}}\s*[-\u2013\u2014to]+\s*{months}\s*\d{{1,2}}",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Single date: "Month DD" or "DD Month"
    text = re.sub(
        rf"{months}\s*\d{{1,2}}|\d{{1,2}}\s*{months}",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Month + year: "January 2026", "Q1 2025"
    text = re.sub(
        rf"{months}\s*,?\s*\d{{4}}|q[1-4]\s*\d{{4}}",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Just month names
    text = re.sub(rf"\b{months}\b", "", text, flags=re.IGNORECASE)

    # Remove years (4-digit numbers that look like years)
    text = re.sub(r"\b20[2-3]\d\b", "", text)

    # Remove numeric ranges: "420-439", "0-19", "220-239"
    text = re.sub(r"\b\d+\s*[-\u2013\u2014]\s*\d+\b", "", text)

    # Remove standalone numbers and percentages: "50%", "10", "#"
    text = re.sub(r"\b\d+%?|\#", "", text)

    # Remove ordinals: "1st", "2nd", "3rd", "4th", etc.
    text = re.sub(r"\b\d+(?:st|nd|rd|th)\b", "", text, flags=re.IGNORECASE)

    # Remove time-related words that are parameters
    text = re.sub(r"\b(?:by|before|after|in|on|from|to|through|until)\s+$", "", text)

    # Clean up: remove extra spaces and punctuation
    text = re.sub(r"[,;:?!.]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    # Remove trailing prepositions
    text = re.sub(r"\s+(?:by|in|on|from|to|at)$", "", text)

    return text


def build_concept_groups(
    events: list[dict],
    entity_sets: dict[str, list[str]] | None = None,
) -> list[dict]:
    """
    Build concept groups by extracting and normalizing concepts from event titles.

    Events with the same normalized concept are grouped together.

    Args:
        events: List of event dicts with 'id' and 'title'
        entity_sets: Optional dict of event_id -> [entities] for disambiguation

    Returns:
        List of concept group dicts with members, concept_template, etc.
    """
    if entity_sets is None:
        entity_sets = {}

    # Extract concept for each event
    concept_to_events: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for event in events:
        event_id = event["id"]
        title = event.get("title", "")
        concept = extract_concept_from_title(title)

        if concept:  # Skip empty concepts
            concept_to_events[concept].append((event_id, title))

    # If using entities for disambiguation, create entity-qualified concepts
    if CONCEPT_USE_ENTITIES and entity_sets:
        refined_concepts: dict[str, list[tuple[str, str]]] = defaultdict(list)

        for concept, event_list in concept_to_events.items():
            # Check if all events share a common entity
            if len(event_list) >= 2:
                # Find entities common to all events in this concept
                all_entity_sets = [
                    set(entity_sets.get(eid, [])) for eid, _ in event_list
                ]
                if all_entity_sets:
                    common_entities = (
                        set.intersection(*all_entity_sets) if all_entity_sets else set()
                    )

                    # Use the first common entity (if any) to qualify the concept
                    if common_entities:
                        # Pick the most specific entity (shortest name usually)
                        primary_entity = min(common_entities, key=len)
                        qualified_concept = f"{primary_entity}:{concept}"
                        refined_concepts[qualified_concept].extend(event_list)
                    else:
                        refined_concepts[concept].extend(event_list)
                else:
                    refined_concepts[concept].extend(event_list)
            else:
                refined_concepts[concept].extend(event_list)

        concept_to_events = refined_concepts

    # Build output groups
    groups = []
    for concept, event_list in sorted(
        concept_to_events.items(), key=lambda x: -len(x[1])
    ):
        if len(event_list) < CONCEPT_MIN_GROUP_SIZE:
            continue

        members = [eid for eid, _ in event_list]
        titles = [title for _, title in event_list]

        # Find shared entities across all events in this concept
        shared_entities: set[str] = set()
        if entity_sets:
            all_entity_sets = [set(entity_sets.get(eid, [])) for eid in members]
            if all_entity_sets:
                shared_entities = set.intersection(*all_entity_sets)

        groups.append(
            {
                "concept_template": concept,
                "members": sorted(members),
                "titles": titles,
                "shared_entities": sorted(shared_entities),
                "size": len(members),
            }
        )

    # Sort by size and assign IDs
    groups.sort(key=lambda x: -x["size"])
    for i, g in enumerate(groups, 1):
        g["group_id"] = f"concept_{i}"

    return groups


# =============================================================================
# EMBEDDING CLUSTERING
# =============================================================================


def cluster_embeddings(
    embeddings: np.ndarray,
    event_ids: list[str],
    entity_sets: dict[str, list[str]],
) -> list[dict]:
    """
    Cluster embeddings using HDBSCAN and auto-label clusters.

    Returns: list of cluster groups with auto-generated labels
    """
    logger.info(f"Clustering {len(embeddings)} embeddings with HDBSCAN...")

    clusterer = HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
    )

    labels = clusterer.fit_predict(embeddings)

    # Count clusters (excluding noise label -1)
    unique_labels = set(labels)
    n_clusters = len([lbl for lbl in unique_labels if lbl >= 0])
    n_noise = sum(1 for lbl in labels if lbl == -1)

    logger.info(f"Found {n_clusters} clusters, {n_noise} noise points")

    # Build cluster groups
    clusters: dict[int, list[str]] = defaultdict(list)
    for idx, label in enumerate(labels):
        if label >= 0:  # Skip noise
            clusters[label].append(event_ids[idx])

    # Auto-label clusters by most frequent entities
    cluster_groups = []
    for cluster_id, member_ids in sorted(clusters.items()):
        # Count entity frequency in this cluster
        entity_counts: dict[str, int] = defaultdict(int)
        for event_id in member_ids:
            for entity in entity_sets.get(event_id, []):
                entity_counts[entity] += 1

        # Get top entities for label
        top_entities = sorted(entity_counts.items(), key=lambda x: -x[1])[
            :TOPIC_LABEL_TOP_K
        ]

        auto_label = (
            " + ".join(e[0] for e in top_entities)
            if top_entities
            else f"cluster_{cluster_id}"
        )

        cluster_groups.append(
            {
                "group_id": f"topic_{cluster_id}",
                "name": auto_label,
                "members": member_ids,
                "size": len(member_ids),
                "top_entities": [
                    {"entity": e, "count": c} for e, c in top_entities[:5]
                ],
            }
        )

    return sorted(cluster_groups, key=lambda x: -x["size"])


# =============================================================================
# ENTITY FACETS
# =============================================================================


def extract_entity_facets(
    events: list[dict],
    entities_normalized: list[dict],
) -> dict[str, dict[str, list[str]]]:
    """
    Extract entity-based facets from normalized entities.

    Returns: {entity_type: {entity_value: [event_ids]}}
    """
    # Build entity label lookup
    entity_to_label: dict[str, str] = {}
    variant_to_canonical: dict[str, str] = {}

    for ent in entities_normalized:
        canonical = ent["canonical_name"].lower()
        label = ent["label"]
        entity_to_label[canonical] = label

        for variant in ent.get("variants", []):
            variant_to_canonical[variant.lower()] = canonical

    # Collect entity mentions per event from semantics
    type_entity_events: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )

    for event in events:
        event_id = event["id"]
        semantics = event.get("semantics", {})

        # Extract entities from semantics fields
        for field in ["subject_entity", "object_entity"]:
            entity_text = semantics.get(field)
            if entity_text:
                entity_text = entity_text.lower()
                # Normalize via variant lookup
                canonical = variant_to_canonical.get(entity_text, entity_text)
                label = entity_to_label.get(canonical)
                if label:
                    type_entity_events[label][canonical].add(event_id)

    # Filter by minimum frequency and convert to lists
    result: dict[str, dict[str, list[str]]] = {}

    for entity_type, entities in type_entity_events.items():
        filtered_entities = {}
        for entity_value, event_ids in entities.items():
            if len(event_ids) >= MIN_ENTITY_FREQUENCY:
                filtered_entities[entity_value] = sorted(event_ids)

        if filtered_entities:
            result[entity_type] = filtered_entities

    return result


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    start_time = datetime.now(timezone.utc)

    logger.info("=" * 60)
    logger.info("04_3_cluster_events - Multi-Dimensional Event Clustering")
    logger.info("=" * 60)

    # Find input folders
    semantics_folder = (
        INPUT_SEMANTICS_DIR / INPUT_RUN_FOLDER
        if INPUT_RUN_FOLDER
        else find_latest_run_folder(INPUT_SEMANTICS_DIR)
    )
    embeddings_folder = find_latest_run_folder(INPUT_EMBEDDINGS_DIR)
    normalize_folder = find_latest_run_folder(INPUT_NORMALIZE_DIR)

    if not semantics_folder:
        raise FileNotFoundError(f"No run folder found in {INPUT_SEMANTICS_DIR}")
    if not embeddings_folder:
        raise FileNotFoundError(f"No run folder found in {INPUT_EMBEDDINGS_DIR}")
    if not normalize_folder:
        raise FileNotFoundError(f"No run folder found in {INPUT_NORMALIZE_DIR}")

    logger.info(f"Semantics: {semantics_folder}")
    logger.info(f"Embeddings: {embeddings_folder}")
    logger.info(f"Normalize: {normalize_folder}")

    # Load event semantics
    semantics_file = semantics_folder / "event_semantics.json"
    with open(semantics_file, encoding="utf-8") as f:
        semantics_data = json.load(f)
    events = semantics_data.get("events", [])
    logger.info(f"Loaded {len(events)} events from semantics")

    # Load embeddings and metadata
    embeddings_file = embeddings_folder / "embeddings.npy"
    metadata_file = embeddings_folder / "metadata.json"
    entity_sets_file = embeddings_folder / "entity_sets.json"

    embeddings = np.load(embeddings_file)
    with open(metadata_file, encoding="utf-8") as f:
        metadata = json.load(f)

    # Build index_to_id list from id_to_index
    id_to_index = metadata.get("id_to_index", {})
    index_to_id = [""] * len(id_to_index)
    for event_id, idx in id_to_index.items():
        index_to_id[idx] = event_id

    logger.info(f"Loaded embeddings: {embeddings.shape}")

    # Load entity sets
    entity_sets: dict[str, list[str]] = {}
    if entity_sets_file.exists():
        with open(entity_sets_file, encoding="utf-8") as f:
            es_data = json.load(f)
        entity_sets = es_data.get("entity_sets", {})
        logger.info(f"Loaded entity sets for {len(entity_sets)} events")

    # Load normalized entities
    entities_file = normalize_folder / "entities_normalized.json"
    with open(entities_file, encoding="utf-8") as f:
        entities_data = json.load(f)
    entities_normalized = entities_data.get("entities", [])
    logger.info(f"Loaded {len(entities_normalized)} normalized entities")

    # =========================================================================
    # BUILD DIMENSIONS
    # =========================================================================

    dimensions: dict[str, list[dict]] = {}

    # Dimension 1: CONCEPT
    logger.info("Building CONCEPT dimension...")
    concept_groups = build_concept_groups(events, entity_sets)
    concept_dimension = []
    for cg in concept_groups:
        concept_dimension.append(
            {
                "group_id": cg["group_id"],
                "name": cg["concept_template"][:80],
                "concept_template": cg["concept_template"],
                "members": cg["members"],
                "size": cg["size"],
                "shared_entities": cg.get("shared_entities", []),
                "sample_titles": cg["titles"][:3],
            }
        )
    dimensions["CONCEPT"] = concept_dimension
    logger.info(f"  CONCEPT: {len(concept_dimension)} groups")

    # Dimension 2: TOPIC_CLUSTER
    logger.info("Building TOPIC_CLUSTER dimension...")
    topic_clusters = cluster_embeddings(embeddings, index_to_id, entity_sets)
    dimensions["TOPIC_CLUSTER"] = topic_clusters
    logger.info(f"  TOPIC_CLUSTER: {len(topic_clusters)} groups")

    # Dimensions 3-N: Entity facets
    logger.info("Extracting entity-based facets...")
    entity_facets = extract_entity_facets(events, entities_normalized)

    for entity_type, entities in sorted(entity_facets.items()):
        dimension_groups = []
        for entity_value, member_ids in sorted(
            entities.items(), key=lambda x: -len(x[1])
        ):
            group_id = f"{entity_type.lower()}_{entity_value.replace(' ', '_')}"
            dimension_groups.append(
                {
                    "group_id": group_id,
                    "name": entity_value,
                    "members": member_ids,
                    "size": len(member_ids),
                }
            )

        dimensions[entity_type] = dimension_groups
        logger.info(f"  {entity_type}: {len(dimension_groups)} groups")

    # =========================================================================
    # BUILD EVENT MEMBERSHIPS
    # =========================================================================

    logger.info("Building event memberships...")

    # Reverse index: event_id -> {dimension: [group_ids]}
    event_memberships: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for dimension_name, groups in dimensions.items():
        for group in groups:
            group_id = group["group_id"]
            for member_id in group["members"]:
                event_memberships[member_id][dimension_name].append(group_id)

    # Convert to regular dict
    event_memberships = {k: dict(v) for k, v in event_memberships.items()}

    # =========================================================================
    # COMPUTE STATISTICS
    # =========================================================================

    dimension_stats = {}
    for dim_name, groups in dimensions.items():
        sizes = [g["size"] for g in groups]
        dimension_stats[dim_name] = {
            "num_groups": len(groups),
            "total_memberships": sum(sizes),
            "avg_size": round(sum(sizes) / len(sizes), 2) if sizes else 0,
            "max_size": max(sizes) if sizes else 0,
            "min_size": min(sizes) if sizes else 0,
        }

    # Events by number of dimensions they belong to
    membership_depth = defaultdict(int)
    for event_id, dims in event_memberships.items():
        membership_depth[len(dims)] += 1

    # =========================================================================
    # SAVE OUTPUT
    # =========================================================================

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_folder = SCRIPT_OUTPUT_DIR / timestamp
    output_folder.mkdir(parents=True, exist_ok=True)

    # Save event_clusters.json
    clusters_output = {
        "_meta": {
            "description": "Multi-dimensional event clustering",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_semantics": str(semantics_folder),
            "source_embeddings": str(embeddings_folder),
            "source_normalize": str(normalize_folder),
        },
        "dimensions": dimensions,
    }

    clusters_file = output_folder / "event_clusters.json"
    with open(clusters_file, "w", encoding="utf-8") as f:
        json.dump(clusters_output, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: {clusters_file}")

    # Save event_memberships.json
    memberships_output = {
        "_meta": {
            "description": "Event to group memberships across dimensions",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "events": [
            {
                "id": e["id"],
                "title": e["title"],
                "memberships": event_memberships.get(e["id"], {}),
                "num_dimensions": len(event_memberships.get(e["id"], {})),
            }
            for e in events
        ],
    }

    memberships_file = output_folder / "event_memberships.json"
    with open(memberships_file, "w", encoding="utf-8") as f:
        json.dump(memberships_output, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: {memberships_file}")

    # Save summary.json
    end_time = datetime.now(timezone.utc)

    summary = {
        "run_info": {
            "started_at": start_time.isoformat(),
            "completed_at": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "output_folder": str(output_folder),
        },
        "configuration": {
            "concept_min_group_size": CONCEPT_MIN_GROUP_SIZE,
            "concept_use_entities": CONCEPT_USE_ENTITIES,
            "min_entity_frequency": MIN_ENTITY_FREQUENCY,
            "hdbscan_min_cluster_size": HDBSCAN_MIN_CLUSTER_SIZE,
            "hdbscan_min_samples": HDBSCAN_MIN_SAMPLES,
            "topic_label_top_k": TOPIC_LABEL_TOP_K,
        },
        "input": {
            "total_events": len(events),
            "embedding_shape": list(embeddings.shape),
            "normalized_entities": len(entities_normalized),
        },
        "output": {
            "num_dimensions": len(dimensions),
            "dimension_names": list(dimensions.keys()),
            "total_groups": sum(len(g) for g in dimensions.values()),
            "events_with_memberships": len(event_memberships),
        },
        "dimension_stats": dimension_stats,
        "membership_depth_distribution": dict(sorted(membership_depth.items())),
    }

    summary_file = output_folder / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: {summary_file}")

    # =========================================================================
    # PRINT SUMMARY
    # =========================================================================

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total events: {len(events)}")
    logger.info(f"Dimensions created: {len(dimensions)}")
    logger.info(f"Total groups: {summary['output']['total_groups']}")
    logger.info("-" * 40)

    for dim_name, stats in dimension_stats.items():
        logger.info(
            f"{dim_name}: {stats['num_groups']} groups, avg size: {stats['avg_size']}"
        )

    logger.info("-" * 40)
    logger.info("Membership depth (events by # of dimensions):")
    for depth, count in sorted(membership_depth.items()):
        logger.info(f"  {depth} dimensions: {count} events")

    logger.info("-" * 40)
    logger.info(f"Duration: {(end_time - start_time).total_seconds():.1f}s")
    logger.info(f"Output: {output_folder}")


if __name__ == "__main__":
    main()
