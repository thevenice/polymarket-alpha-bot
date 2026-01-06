"""
Extract relations between entities using GLiNER2 with entity constraints.

Extracts 28 relation types from event/market texts using GLiNER2, then validates
each relation against normalized entities from the pipeline. This constraint-based
approach dramatically improves precision over naive extraction.

Validation filters:
1. Entity matching: Head and tail must match known normalized entities
2. Type constraints: Relation must satisfy head_type → tail_type rules
   (e.g., "president of" requires POLITICIAN → COUNTRY)
3. Self-reference filter: Head and tail must be different entities

Pipeline Position: 02_prepare_nlp_data + 03_3_normalize_entities → 03_4_extract_relations

Input:
    From: data/02_prepare_nlp_data/<timestamp>/
        - nlp_events.json: Events with text fields for relation extraction
    From: data/03_3_normalize_entities/<timestamp>/
        - entities_normalized.json: Canonical entity names and labels for validation

Output:
    To: data/03_4_extract_relations/<timestamp>/
    Files:
        - relations_raw.json: All validated relations with confidence scores
        - relations_by_event.json: Relations grouped by source event
        - relations_unique.json: Aggregated unique (head, relation, tail) triplets
        - summary.json: Statistics, acceptance rate, filter breakdown, top relations

Runtime: ~4 seconds/text on CPU (~4000 texts = ~4-5 hours)
Model: fastino/gliner2-base-v1 (GLiNER2)

Relation Types (28 total):
    Leadership (8): president_of, prime_minister_of, leader_of, member_of, governor_of, represents, candidate_in, head_of
    Conflict (5): in_conflict_with, party_to, controls, claims, ally_of
    Legal (4): charged_with, defendant_in, decided_by, prosecuted_by
    Policy (4): implemented_by, targets, proposed_by, signed_by
    Agreement (2): signatory_to, negotiated_by
    Location (2): located_in, part_of
    Affiliation (3): affiliated_with, opponent_of, endorses

Configuration:
    - CONFIDENCE_THRESHOLD: Minimum relation confidence (0.5)
    - MAX_EVENTS: Limit events for testing (None = process all)
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from gliner2 import GLiNER2

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_NLP_DIR = DATA_DIR / "02_prepare_nlp_data"
INPUT_NLP_RUN_FOLDER: str | None = None  # None for latest
INPUT_ENTITIES_DIR = DATA_DIR / "03_3_normalize_entities"
INPUT_ENTITIES_RUN_FOLDER: str | None = None  # None for latest
SCRIPT_OUTPUT_DIR = DATA_DIR / "03_4_extract_relations"

GLINER2_MODEL = "fastino/gliner2-base-v1"
CONFIDENCE_THRESHOLD = 0.5
MAX_EVENTS: int | None = 400  # None for full run, set to small number for testing

# Relation types with semantic descriptions and entity type constraints
# Format: {relation_name: {"description": ..., "head_types": [...], "tail_types": [...]}}
# Organized by category. Total: 28 relation types.
RELATION_SCHEMA: dict[str, dict] = {
    # =========================================================================
    # LEADERSHIP & POLITICAL ROLES (8 types)
    # =========================================================================
    "president of": {
        "description": "person who is president or head of state of a country",
        "head_types": ["POLITICIAN"],
        "tail_types": ["COUNTRY"],
    },
    "prime minister of": {
        "description": "person who is prime minister of a country",
        "head_types": ["POLITICIAN"],
        "tail_types": ["COUNTRY"],
    },
    "leader of": {
        "description": "person who leads a country, party, organization, or armed group",
        "head_types": ["POLITICIAN", "PUBLIC_FIGURE"],
        "tail_types": [
            "COUNTRY",
            "POLITICAL_PARTY",
            "INTERNATIONAL_ORGANIZATION",
            "ARMED_GROUP",
        ],
    },
    "member of": {
        "description": "person who is a member of a political party or organization",
        "head_types": ["POLITICIAN"],
        "tail_types": ["POLITICAL_PARTY", "INTERNATIONAL_ORGANIZATION"],
    },
    "governor of": {
        "description": "person who is governor of a state or region",
        "head_types": ["POLITICIAN"],
        "tail_types": ["STATE"],
    },
    "represents": {
        "description": "politician who represents a state or constituency",
        "head_types": ["POLITICIAN"],
        "tail_types": ["STATE", "COUNTRY"],
    },
    "candidate in": {
        "description": "person running in a political event or election",
        "head_types": ["POLITICIAN"],
        "tail_types": ["POLITICAL_EVENT"],
    },
    "head of": {
        "description": "person who heads an agency, organization, or company",
        "head_types": ["POLITICIAN", "PUBLIC_FIGURE"],
        "tail_types": ["GOVERNMENT_AGENCY", "INTERNATIONAL_ORGANIZATION", "COMPANY"],
    },
    # =========================================================================
    # CONFLICT & MILITARY RELATIONS (5 types)
    # =========================================================================
    "in conflict with": {
        "description": "country or armed group in conflict or war with another",
        "head_types": ["COUNTRY", "ARMED_GROUP"],
        "tail_types": ["COUNTRY", "ARMED_GROUP"],
    },
    "party to": {
        "description": "country or group involved in a military conflict",
        "head_types": ["COUNTRY", "ARMED_GROUP"],
        "tail_types": ["MILITARY_CONFLICT"],
    },
    "controls": {
        "description": "country or group that controls a territory or city",
        "head_types": ["COUNTRY", "ARMED_GROUP"],
        "tail_types": ["TERRITORY", "CITY"],
    },
    "claims": {
        "description": "country that claims sovereignty over territory",
        "head_types": ["COUNTRY"],
        "tail_types": ["TERRITORY"],
    },
    "ally of": {
        "description": "country or organization allied with another",
        "head_types": ["COUNTRY", "INTERNATIONAL_ORGANIZATION", "POLITICAL_PARTY"],
        "tail_types": ["COUNTRY", "INTERNATIONAL_ORGANIZATION", "POLITICAL_PARTY"],
    },
    # =========================================================================
    # LEGAL RELATIONS (4 types)
    # =========================================================================
    "charged with": {
        "description": "person facing criminal charges or accusations",
        "head_types": ["POLITICIAN", "PUBLIC_FIGURE"],
        "tail_types": ["CRIMINAL_CHARGE"],
    },
    "defendant in": {
        "description": "person who is defendant in a legal case",
        "head_types": ["POLITICIAN", "PUBLIC_FIGURE"],
        "tail_types": ["LEGAL_CASE"],
    },
    "decided by": {
        "description": "case or legislation decided or ruled on by a court",
        "head_types": ["LEGAL_CASE", "LEGISLATION"],
        "tail_types": ["COURT"],
    },
    "prosecuted by": {
        "description": "person or case prosecuted by government agency",
        "head_types": ["POLITICIAN", "PUBLIC_FIGURE", "LEGAL_CASE"],
        "tail_types": ["GOVERNMENT_AGENCY", "COURT"],
    },
    # =========================================================================
    # POLICY & GOVERNMENT ACTIONS (4 types)
    # =========================================================================
    "implemented by": {
        "description": "policy or sanction implemented by politician, agency, or country",
        "head_types": ["GOVERNMENT_POLICY", "SANCTION"],
        "tail_types": ["POLITICIAN", "GOVERNMENT_AGENCY", "COUNTRY"],
    },
    "targets": {
        "description": "policy or sanction targeting a country, person, or company",
        "head_types": ["GOVERNMENT_POLICY", "SANCTION"],
        "tail_types": [
            "COUNTRY",
            "POLITICIAN",
            "PUBLIC_FIGURE",
            "COMPANY",
            "ARMED_GROUP",
        ],
    },
    "proposed by": {
        "description": "legislation proposed by politician or party",
        "head_types": ["LEGISLATION"],
        "tail_types": ["POLITICIAN", "POLITICAL_PARTY"],
    },
    "signed by": {
        "description": "legislation or agreement signed by politician",
        "head_types": ["LEGISLATION", "AGREEMENT"],
        "tail_types": ["POLITICIAN"],
    },
    # =========================================================================
    # AGREEMENT & TREATY RELATIONS (2 types)
    # =========================================================================
    "signatory to": {
        "description": "country that is a signatory to an agreement or treaty",
        "head_types": ["COUNTRY"],
        "tail_types": ["AGREEMENT"],
    },
    "negotiated by": {
        "description": "agreement or treaty negotiated by politician",
        "head_types": ["AGREEMENT"],
        "tail_types": ["POLITICIAN"],
    },
    # =========================================================================
    # LOCATION RELATIONS (2 types)
    # =========================================================================
    "located in": {
        "description": "entity located in a country, state, or territory",
        "head_types": [
            "CITY",
            "TERRITORY",
            "GOVERNMENT_AGENCY",
            "POLITICAL_EVENT",
            "COMPANY",
        ],
        "tail_types": ["COUNTRY", "STATE", "TERRITORY"],
    },
    "part of": {
        "description": "state, territory, or city that is part of a country",
        "head_types": ["STATE", "TERRITORY", "CITY"],
        "tail_types": ["COUNTRY"],
    },
    # =========================================================================
    # AFFILIATION & OPPOSITION (3 types)
    # =========================================================================
    "affiliated with": {
        "description": "person or entity associated with a party, country, or organization",
        "head_types": [
            "POLITICIAN",
            "PUBLIC_FIGURE",
            "NEWS_SOURCE",
            "ARMED_GROUP",
            "COMPANY",
        ],
        "tail_types": ["POLITICAL_PARTY", "COUNTRY", "INTERNATIONAL_ORGANIZATION"],
    },
    "opponent of": {
        "description": "politician, party, or country opposing another",
        "head_types": ["POLITICIAN", "POLITICAL_PARTY", "COUNTRY"],
        "tail_types": ["POLITICIAN", "POLITICAL_PARTY", "COUNTRY"],
    },
    "endorses": {
        "description": "politician, party, or news source endorsing another politician",
        "head_types": ["POLITICIAN", "POLITICAL_PARTY", "NEWS_SOURCE", "PUBLIC_FIGURE"],
        "tail_types": ["POLITICIAN", "POLITICAL_PARTY", "LEGISLATION"],
    },
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# MAIN LOGIC
# =============================================================================


def find_latest_run_folder(script_dir: Path) -> Path | None:
    """Find the most recent run folder."""
    if not script_dir.exists():
        return None
    folders = [f for f in script_dir.iterdir() if f.is_dir()]
    return max(folders, key=lambda f: f.stat().st_mtime) if folders else None


def load_events(nlp_dir: Path) -> list[dict]:
    """Load NLP events."""
    nlp_folder = (
        nlp_dir / INPUT_NLP_RUN_FOLDER
        if INPUT_NLP_RUN_FOLDER
        else find_latest_run_folder(nlp_dir)
    )

    if not nlp_folder:
        raise FileNotFoundError("NLP folder not found")

    nlp_file = nlp_folder / "nlp_events.json"
    if not nlp_file.exists():
        raise FileNotFoundError(f"NLP events file not found: {nlp_file}")

    logger.info(f"Loading NLP events from: {nlp_file}")
    with open(nlp_file, encoding="utf-8") as f:
        nlp_data = json.load(f)

    events = nlp_data.get("events", [])
    logger.info(f"Loaded {len(events)} events")
    return events


def load_normalized_entities(
    entities_dir: Path,
) -> tuple[dict[str, tuple[str, str]], dict[str, set[str]]]:
    """
    Load normalized entities and build lookup structures.

    Returns:
        entity_lookup: {lowercase_variant: (canonical_name, label)}
        entities_by_label: {label: {canonical_name1, canonical_name2, ...}}
    """
    entities_folder = (
        entities_dir / INPUT_ENTITIES_RUN_FOLDER
        if INPUT_ENTITIES_RUN_FOLDER
        else find_latest_run_folder(entities_dir)
    )

    if not entities_folder:
        raise FileNotFoundError("Entities folder not found")

    entities_file = entities_folder / "entities_normalized.json"
    if not entities_file.exists():
        raise FileNotFoundError(f"Entities file not found: {entities_file}")

    logger.info(f"Loading normalized entities from: {entities_file}")
    with open(entities_file, encoding="utf-8") as f:
        entities_data = json.load(f)

    entities = entities_data.get("entities", [])
    logger.info(f"Loaded {len(entities)} normalized entities")

    # Build lookup: variant -> (canonical, label)
    entity_lookup: dict[str, tuple[str, str]] = {}
    entities_by_label: dict[str, set[str]] = defaultdict(set)

    for entity in entities:
        canonical = entity["canonical_name"]
        label = entity["label"]
        entities_by_label[label].add(canonical)

        # Map all variants to canonical
        for variant in entity.get("variants", [canonical]):
            entity_lookup[variant.lower().strip()] = (canonical, label)

    logger.info(f"Built lookup with {len(entity_lookup)} variant mappings")
    return entity_lookup, dict(entities_by_label)


def extract_texts(events: list[dict]) -> list[dict]:
    """Extract all text snippets from events."""
    texts = []
    for event in events:
        event_id = event.get("id", "unknown")
        seen = set()

        for field in ["title", "description"]:
            if text := event.get(field, "").strip():
                if text not in seen and len(text) > 20:
                    seen.add(text)
                    texts.append(
                        {
                            "text": text,
                            "source_type": f"event_{field}",
                            "source_id": event_id,
                            "event_id": event_id,
                        }
                    )

        for market in event.get("markets", []):
            if text := market.get("question", "").strip():
                if text not in seen and len(text) > 20:
                    seen.add(text)
                    texts.append(
                        {
                            "text": text,
                            "source_type": "market_question",
                            "source_id": market.get("id", "unknown"),
                            "event_id": event_id,
                        }
                    )

    return texts


def match_entity(
    text: str,
    entity_lookup: dict[str, tuple[str, str]],
) -> tuple[str, str] | None:
    """
    Try to match extracted text to a known entity.

    Returns (canonical_name, label) if found, None otherwise.
    """
    normalized = text.lower().strip()

    # Direct match
    if normalized in entity_lookup:
        return entity_lookup[normalized]

    # Try without common suffixes/prefixes
    for suffix in ["'s", "'s"]:
        if normalized.endswith(suffix):
            base = normalized[: -len(suffix)]
            if base in entity_lookup:
                return entity_lookup[base]

    return None


def validate_relation(
    head_label: str,
    tail_label: str,
    relation_type: str,
) -> bool:
    """Check if relation satisfies entity type constraints."""
    schema = RELATION_SCHEMA.get(relation_type)
    if not schema:
        return False

    head_ok = head_label in schema.get("head_types", [])
    tail_ok = tail_label in schema.get("tail_types", [])

    return head_ok and tail_ok


def process_texts(
    model: GLiNER2,
    texts: list[dict],
    entity_lookup: dict[str, tuple[str, str]],
) -> tuple[list[dict], dict]:
    """
    Extract relations from texts using GLiNER2 individual extraction.

    Note: batch_extract_relations becomes very slow with many relation types (28+),
    so we use individual extraction which is faster in practice.

    Returns:
        relations: List of validated relation dicts
        filter_stats: Statistics about filtering
    """
    relations = []
    relation_labels = {k: v["description"] for k, v in RELATION_SCHEMA.items()}
    total = len(texts)

    filter_stats = {
        "total_extracted": 0,
        "no_head_match": 0,
        "no_tail_match": 0,
        "self_reference": 0,
        "type_constraint_fail": 0,
        "accepted": 0,
    }

    logger.info(f"Processing {total} texts...")
    logger.info(f"Extracting {len(relation_labels)} relation types")

    for idx, item in enumerate(texts):
        try:
            result = model.extract_relations(
                item["text"],
                relation_labels,
                threshold=CONFIDENCE_THRESHOLD,
                include_confidence=True,
            )

            # Result format: {'relation_extraction': {rel_type: [{'head': {...}, 'tail': {...}}]}}
            rel_data = result.get("relation_extraction", {})
            for rel_type, rel_list in rel_data.items():
                for rel in rel_list:
                    filter_stats["total_extracted"] += 1

                    head = rel.get("head", {})
                    tail = rel.get("tail", {})
                    head_text = head.get("text", "").strip()
                    tail_text = tail.get("text", "").strip()

                    if not head_text or not tail_text:
                        continue

                    # Match to known entities
                    head_match = match_entity(head_text, entity_lookup)
                    if not head_match:
                        filter_stats["no_head_match"] += 1
                        continue

                    tail_match = match_entity(tail_text, entity_lookup)
                    if not tail_match:
                        filter_stats["no_tail_match"] += 1
                        continue

                    head_canonical, head_label = head_match
                    tail_canonical, tail_label = tail_match

                    # Skip self-references
                    if head_canonical == tail_canonical:
                        filter_stats["self_reference"] += 1
                        continue

                    # Validate entity type constraints
                    if not validate_relation(head_label, tail_label, rel_type):
                        filter_stats["type_constraint_fail"] += 1
                        continue

                    filter_stats["accepted"] += 1
                    relations.append(
                        {
                            "head": head_canonical,
                            "head_label": head_label,
                            "head_raw": head_text,
                            "head_confidence": round(head.get("confidence", 0), 4),
                            "relation": rel_type,
                            "tail": tail_canonical,
                            "tail_label": tail_label,
                            "tail_raw": tail_text,
                            "tail_confidence": round(tail.get("confidence", 0), 4),
                            "event_id": item["event_id"],
                            "source_type": item["source_type"],
                            "source_id": item["source_id"],
                            "source_text": item["text"][:500],
                        }
                    )

        except Exception as e:
            logger.warning(f"Error processing text {idx}: {e}")
            continue

        if (idx + 1) % 50 == 0 or idx == total - 1:
            logger.info(
                f"  Processed {idx + 1}/{total} texts - "
                f"{filter_stats['accepted']} accepted / {filter_stats['total_extracted']} extracted"
            )

    return relations, filter_stats


def group_relations_by_event(relations: list[dict]) -> dict[str, list[dict]]:
    """Group relations by event_id."""
    by_event: dict[str, list[dict]] = defaultdict(list)
    for rel in relations:
        by_event[rel["event_id"]].append(rel)
    return dict(by_event)


def aggregate_unique_relations(relations: list[dict]) -> list[dict]:
    """Aggregate unique relation triplets with counts."""
    counts: dict[tuple, dict] = {}

    for rel in relations:
        key = (rel["head"], rel["relation"], rel["tail"])

        if key not in counts:
            counts[key] = {
                "head": rel["head"],
                "head_label": rel["head_label"],
                "relation": rel["relation"],
                "tail": rel["tail"],
                "tail_label": rel["tail_label"],
                "count": 0,
                "head_confidences": [],
                "tail_confidences": [],
                "event_ids": set(),
            }

        counts[key]["count"] += 1
        counts[key]["head_confidences"].append(rel["head_confidence"])
        counts[key]["tail_confidences"].append(rel["tail_confidence"])
        counts[key]["event_ids"].add(rel["event_id"])

    result = []
    for data in counts.values():
        avg_head_conf = sum(data["head_confidences"]) / len(data["head_confidences"])
        avg_tail_conf = sum(data["tail_confidences"]) / len(data["tail_confidences"])
        result.append(
            {
                "head": data["head"],
                "head_label": data["head_label"],
                "relation": data["relation"],
                "tail": data["tail"],
                "tail_label": data["tail_label"],
                "count": data["count"],
                "avg_confidence": round((avg_head_conf + avg_tail_conf) / 2, 4),
                "event_count": len(data["event_ids"]),
            }
        )

    return sorted(result, key=lambda x: (-x["count"], -x["avg_confidence"]))


def compute_stats(
    relations: list[dict],
    unique_relations: list[dict],
    texts_count: int,
    filter_stats: dict,
) -> dict:
    """Compute summary statistics."""
    relation_counts = defaultdict(int)
    head_label_counts = defaultdict(int)
    tail_label_counts = defaultdict(int)

    for rel in relations:
        relation_counts[rel["relation"]] += 1
        head_label_counts[rel["head_label"]] += 1
        tail_label_counts[rel["tail_label"]] += 1

    return {
        "total_relations": len(relations),
        "unique_relations": len(unique_relations),
        "texts_processed": texts_count,
        "events_with_relations": len(set(r["event_id"] for r in relations)),
        "filter_stats": filter_stats,
        "acceptance_rate": round(
            filter_stats["accepted"] / max(filter_stats["total_extracted"], 1), 4
        ),
        "by_relation_type": dict(sorted(relation_counts.items(), key=lambda x: -x[1])),
        "by_head_label": dict(sorted(head_label_counts.items(), key=lambda x: -x[1])),
        "by_tail_label": dict(sorted(tail_label_counts.items(), key=lambda x: -x[1])),
        "top_20_relations": [
            {
                "head": r["head"],
                "head_label": r["head_label"],
                "relation": r["relation"],
                "tail": r["tail"],
                "tail_label": r["tail_label"],
                "count": r["count"],
                "avg_confidence": r["avg_confidence"],
            }
            for r in unique_relations[:20]
        ],
    }


def main() -> None:
    start_time = datetime.now(timezone.utc)
    logger.info("=" * 60)
    logger.info("Polymarket Relation Extraction (GLiNER2 + Entity Constraints)")
    logger.info("=" * 60)

    # Load input data
    try:
        events = load_events(INPUT_NLP_DIR)
        entity_lookup, entities_by_label = load_normalized_entities(INPUT_ENTITIES_DIR)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    logger.info(f"Entity types available: {list(entities_by_label.keys())}")
    logger.info(
        f"Entity counts: {', '.join(f'{k}={len(v)}' for k, v in entities_by_label.items())}"
    )

    if MAX_EVENTS:
        events = events[:MAX_EVENTS]
        logger.info(f"Limited to {MAX_EVENTS} events")

    # Load model
    logger.info(f"Loading GLiNER2 model: {GLINER2_MODEL}")
    model = GLiNER2.from_pretrained(GLINER2_MODEL)

    # Extract texts
    texts = extract_texts(events)
    logger.info(f"Extracted {len(texts)} texts from {len(events)} events")

    # Process with entity constraints
    relations, filter_stats = process_texts(model, texts, entity_lookup)
    logger.info(f"Extracted {len(relations)} valid relations")
    logger.info(
        f"Filter stats: {filter_stats['accepted']}/{filter_stats['total_extracted']} accepted "
        f"({filter_stats['acceptance_rate'] if 'acceptance_rate' in filter_stats else round(filter_stats['accepted'] / max(filter_stats['total_extracted'], 1), 2):.1%})"
    )

    # Aggregate results
    by_event_relations = group_relations_by_event(relations)
    unique_relations = aggregate_unique_relations(relations)
    stats = compute_stats(relations, unique_relations, len(texts), filter_stats)

    # Output
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_folder = SCRIPT_OUTPUT_DIR / timestamp
    output_folder.mkdir(parents=True, exist_ok=True)

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gliner2_model": GLINER2_MODEL,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "relation_schema": RELATION_SCHEMA,
        "entity_constrained": True,
    }

    outputs = [
        (
            "relations_raw.json",
            {
                "_meta": {
                    **meta,
                    "description": "All extracted and validated relations",
                },
                "relations": relations,
            },
        ),
        (
            "relations_by_event.json",
            {
                "_meta": {**meta, "description": "Relations grouped by event"},
                "by_event": by_event_relations,
            },
        ),
        (
            "relations_unique.json",
            {
                "_meta": {**meta, "description": "Aggregated unique relations"},
                "relations": unique_relations,
            },
        ),
    ]

    for filename, data in outputs:
        with open(output_folder / filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved: {filename}")

    end_time = datetime.now(timezone.utc)
    summary = {
        "run_info": {
            "started_at": start_time.isoformat(),
            "completed_at": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "output_folder": str(output_folder),
        },
        "configuration": {
            "gliner2_model": GLINER2_MODEL,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "max_events": MAX_EVENTS,
            "relation_types": list(RELATION_SCHEMA.keys()),
            "entity_constrained": True,
        },
        "statistics": stats,
    }

    with open(output_folder / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Saved: summary.json")

    # Summary
    logger.info("=" * 60)
    logger.info(
        f"Events: {len(events)} | Texts: {len(texts)} | "
        f"Relations: {len(relations)} | Unique: {len(unique_relations)}"
    )
    logger.info(
        f"Acceptance rate: {stats['acceptance_rate']:.1%} "
        f"({filter_stats['accepted']}/{filter_stats['total_extracted']})"
    )
    logger.info(
        f"Duration: {(end_time - start_time).total_seconds():.1f}s | Output: {output_folder}"
    )


if __name__ == "__main__":
    main()
