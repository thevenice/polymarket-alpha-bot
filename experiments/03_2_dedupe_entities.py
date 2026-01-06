"""
Deduplicate and cluster entities from GLiNER2 extraction.

4-stage processing pipeline:
1. Per-event exact-match dedupe: Normalize text, keep highest confidence per (text, label) per event
2. Cross-event fuzzy clustering: RapidFuzz matching within same label (freq >= 2 entities only)
3. Apply canonical forms: Add text_canonical field mapping variants to cluster representative
4. Aggregate unique entities: Weighted label selection using (count × avg_confidence) scoring

Pipeline Position: 03_1_extract_entities → 03_2_dedupe_entities → 03_3_normalize_entities

Input:
    From: data/03_1_extract_entities/<timestamp>/
    Files:
        - entities_raw.json: All extracted entities with confidence scores

Output:
    To: data/03_2_dedupe_entities/<timestamp>/
    Files:
        - entities_raw.json: Deduplicated entities (one per text+label per event)
        - entities_by_source.json: Deduplicated entities grouped by event/market
        - entities_unique.json: Aggregated unique entities with occurrence counts
        - fuzzy_merge_mappings.json: Fuzzy clustering results and similarity thresholds
        - summary.json: Deduplication statistics and reduction metrics

Runtime: ~1-2 minutes (CPU-bound fuzzy string matching)

Configuration:
    - FUZZY_MIN_FREQUENCY: Minimum occurrences for fuzzy matching (2)
    - FUZZY_THRESHOLDS: Label-specific similarity thresholds (88-95%)
      STATE/POLITICAL_EVENT: 95% (strict - prevent wrong merges)
      POLITICIAN: 88% (looser - allow surname variations)
      COUNTRY: 92% (moderate - handle abbreviations)
"""

import json
import logging
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_SCRIPT_DIR = DATA_DIR / "03_1_extract_entities"
INPUT_RUN_FOLDER: str | None = None
SCRIPT_OUTPUT_DIR = DATA_DIR / "03_2_dedupe_entities"
FUZZY_MIN_FREQUENCY = 2

# Label-specific fuzzy matching thresholds
FUZZY_THRESHOLDS = {
    "STATE": 0.95,  # Strict - congressional districts differ by numbers
    "POLITICAL_EVENT": 0.95,  # Strict - different elections shouldn't merge
    "POLITICIAN": 0.88,  # Looser - allow surname variations
    "COUNTRY": 0.92,  # Moderate - abbreviations like US/USA
    "DEFAULT": 0.90,  # Fallback for other labels
}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# HELPERS
# =============================================================================


def find_latest_run(script_dir: Path) -> Path | None:
    """Find most recent run folder."""
    if not script_dir.exists():
        return None
    folders = [f for f in script_dir.iterdir() if f.is_dir()]
    return max(folders, key=lambda f: f.stat().st_mtime) if folders else None


def normalize(text: str) -> str:
    """Normalize text: lowercase, NFC unicode, collapse whitespace."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text.lower().strip()))


def save_json(path: Path, data: dict) -> None:
    """Save JSON with standard formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: {path}")


# =============================================================================
# CORE LOGIC
# =============================================================================


def dedupe_per_event(entities: list[dict]) -> list[dict]:
    """Keep one entity per (normalized_text, label) per event (highest confidence)."""
    by_event: dict[str, list[dict]] = defaultdict(list)
    for e in entities:
        by_event[e["event_id"]].append(e)

    result = []
    for event_ents in by_event.values():
        seen: dict[tuple[str, str], dict] = {}
        for e in event_ents:
            key = (normalize(e["text"]), e["label"])
            if key not in seen or e.get("confidence", 0) > seen[key].get(
                "confidence", 0
            ):
                seen[key] = e
        result.extend(seen.values())

    logger.info(f"Dedupe: {len(entities)} -> {len(result)} entities")
    return result


def fuzzy_cluster(entities: list[dict]) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Cluster similar entities using fuzzy matching (within same label).

    Uses label-specific thresholds from FUZZY_THRESHOLDS config.
    """
    from rapidfuzz import fuzz

    # Count frequencies
    freq: dict[str, int] = defaultdict(int)
    for e in entities:
        freq[normalize(e["text"])] += 1

    # Group by label (only frequent entities)
    by_label: dict[str, set[str]] = defaultdict(set)
    for e in entities:
        key = normalize(e["text"])
        if freq[key] >= FUZZY_MIN_FREQUENCY:
            by_label[e["label"]].add(key)

    # Union-Find with path compression
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: str, y: str) -> None:
        parent[find(x)] = find(y)

    # Match within each label using label-specific thresholds
    merges = 0
    merges_by_label: dict[str, int] = defaultdict(int)
    for label, texts in by_label.items():
        threshold = FUZZY_THRESHOLDS.get(label, FUZZY_THRESHOLDS["DEFAULT"])
        texts_list = sorted(texts)
        for i, t1 in enumerate(texts_list):
            for t2 in texts_list[i + 1 :]:
                if fuzz.ratio(t1, t2) / 100.0 >= threshold:
                    union(t1, t2)
                    merges += 1
                    merges_by_label[label] += 1

    logger.info(f"Fuzzy: {merges} merges (by label: {dict(merges_by_label)})")

    # Build clusters and merge map
    clusters: dict[str, list[str]] = defaultdict(list)
    for x in parent:
        clusters[find(x)].append(x)

    merge_map: dict[str, str] = {}
    canonical_clusters: dict[str, list[str]] = {}
    for members in clusters.values():
        if len(members) > 1:
            canonical = max(members, key=lambda x: (freq[x], -len(x)))
            canonical_clusters[canonical] = sorted(members)
            for m in members:
                merge_map[m] = canonical

    return merge_map, canonical_clusters


def apply_canonical(entities: list[dict], merge_map: dict[str, str]) -> list[dict]:
    """Add text_canonical field to all entities."""
    result = []
    for e in entities:
        e = e.copy()
        norm = normalize(e["text"])
        e["text_canonical"] = merge_map.get(norm, norm)
        result.append(e)
    return result


def aggregate_unique(entities: list[dict]) -> list[dict]:
    """Aggregate entities by text_canonical with weighted label selection."""
    groups: dict[str, dict] = {}

    for e in entities:
        key = e.get("text_canonical") or normalize(e["text"])
        if key not in groups:
            groups[key] = {
                "text_normalized": key,
                "variations": [],
                "label_data": defaultdict(list),
                "events": set(),
                "markets": set(),
                "confidences": [],
            }
        g = groups[key]
        g["variations"].append(e["text"])
        g["label_data"][e["label"]].append(e.get("confidence", 0.0))
        g["confidences"].append(e.get("confidence", 0.0))
        g["events"].add(e["event_id"])
        if e.get("source_type", "").startswith("market_"):
            g["markets"].add(e["source_id"])

    # Build output
    result = []
    for key, g in groups.items():
        # Weighted label: score = count * avg_confidence
        label_scores = {
            lbl: len(confs) * (sum(confs) / len(confs) if confs else 0)
            for lbl, confs in g["label_data"].items()
        }
        label = max(label_scores, key=label_scores.get)
        label_counts = {lbl: len(confs) for lbl, confs in g["label_data"].items()}
        total = sum(label_counts.values())

        result.append(
            {
                "text_normalized": key,
                "label": label,
                "label_confidence": round(label_counts[label] / total, 3)
                if total
                else 1.0,
                "count": total,
                "variations": sorted(set(g["variations"])),
                "label_counts": label_counts,
                "event_count": len(g["events"]),
                "market_count": len(g["markets"]),
                "avg_confidence": round(
                    sum(g["confidences"]) / len(g["confidences"]), 3
                )
                if g["confidences"]
                else 0,
            }
        )

    return sorted(result, key=lambda x: -x["count"])


def group_by_source(entities: list[dict]) -> dict:
    """Group entities by event, separating event vs market sources."""
    by_event: dict[str, dict] = defaultdict(
        lambda: {"event_entities": [], "market_entities": defaultdict(list)}
    )
    for e in entities:
        if e.get("source_type", "").startswith("event_"):
            by_event[e["event_id"]]["event_entities"].append(e)
        else:
            by_event[e["event_id"]]["market_entities"][e["source_id"]].append(e)

    return {
        eid: {
            "event_entities": d["event_entities"],
            "market_entities": dict(d["market_entities"]),
        }
        for eid, d in by_event.items()
    }


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    start = datetime.now(timezone.utc)
    logger.info("Entity Deduplication")

    # Validate input directory exists
    if not INPUT_SCRIPT_DIR.exists():
        logger.error(
            f"Input directory does not exist: {INPUT_SCRIPT_DIR}\n"
            f"Expected output from 03_1_extract_entities. Run that script first."
        )
        raise FileNotFoundError(f"Input directory not found: {INPUT_SCRIPT_DIR}")

    # Find input
    input_folder = (
        INPUT_SCRIPT_DIR / INPUT_RUN_FOLDER
        if INPUT_RUN_FOLDER
        else find_latest_run(INPUT_SCRIPT_DIR)
    )
    if not input_folder:
        logger.error(
            f"No run folders found in: {INPUT_SCRIPT_DIR}\n"
            f"Run 03_1_extract_entities first to generate entity data."
        )
        raise FileNotFoundError(f"No run folders in: {INPUT_SCRIPT_DIR}")

    input_file = input_folder / "entities_raw.json"
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Load
    with open(input_file, encoding="utf-8") as f:
        raw_entities = json.load(f).get("entities", [])
    logger.info(
        f"Loaded {len(raw_entities)} entities from {len(set(e['event_id'] for e in raw_entities))} events"
    )

    # Stage 1-3: Dedupe, fuzzy cluster, apply canonical
    deduped = dedupe_per_event(raw_entities)
    merge_map, clusters = fuzzy_cluster(deduped)
    deduped = apply_canonical(deduped, merge_map)

    # Stage 4: Aggregate
    unique = aggregate_unique(deduped)
    logger.info(f"Unique entities: {len(unique)}")

    # Prepare output
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = SCRIPT_OUTPUT_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = lambda desc: {
        "_meta": {
            "description": desc,
            "source": str(input_file),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    }

    # Save outputs
    save_json(
        out_dir / "entities_raw.json",
        {
            **meta("Deduplicated entities (one per text+label per event)"),
            "_meta": {
                **meta("")["_meta"],
                "description": "Deduplicated entities (one per text+label per event)",
                "original_count": len(raw_entities),
                "deduped_count": len(deduped),
            },
            "entities": deduped,
        },
    )

    save_json(
        out_dir / "entities_by_source.json",
        {
            **meta("Deduplicated entities grouped by event"),
            "by_event": group_by_source(deduped),
        },
    )

    save_json(
        out_dir / "entities_unique.json",
        {**meta("Aggregated unique entities after deduplication"), "entities": unique},
    )

    save_json(
        out_dir / "fuzzy_merge_mappings.json",
        {
            "_meta": {
                **meta("")["_meta"],
                "description": "Fuzzy merge mappings from similar entity clustering",
                "thresholds": FUZZY_THRESHOLDS,
                "min_frequency": FUZZY_MIN_FREQUENCY,
            },
            "merge_map": merge_map,
            "clusters": clusters,
        },
    )

    # Statistics
    label_counts = defaultdict(int)
    for e in deduped:
        label_counts[e["label"]] += 1
    confs = [e.get("confidence", 0.0) for e in deduped]

    end = datetime.now(timezone.utc)
    save_json(
        out_dir / "summary.json",
        {
            "run_info": {
                "started_at": start.isoformat(),
                "completed_at": end.isoformat(),
                "duration_seconds": (end - start).total_seconds(),
                "input_file": str(input_file),
                "output_folder": str(out_dir),
            },
            "deduplication": {
                "input_entities": len(raw_entities),
                "output_entities": len(deduped),
                "removed": len(raw_entities) - len(deduped),
                "reduction_pct": round(100 * (1 - len(deduped) / len(raw_entities)), 1)
                if raw_entities
                else 0,
            },
            "fuzzy_matching": {
                "thresholds": FUZZY_THRESHOLDS,
                "min_frequency": FUZZY_MIN_FREQUENCY,
                "merge_mappings": len(merge_map),
                "clusters": len(clusters),
            },
            "statistics": {
                "total_extractions": len(deduped),
                "unique_entities": len(unique),
                "events_with_entities": len(set(e["event_id"] for e in deduped)),
                "confidence_stats": {
                    "avg": round(sum(confs) / len(confs), 3) if confs else 0,
                    "min": round(min(confs), 3) if confs else 0,
                    "max": round(max(confs), 3) if confs else 0,
                },
                "by_label": dict(sorted(label_counts.items(), key=lambda x: -x[1])),
                "top_50_entities": [
                    {
                        "text": e["text_normalized"],
                        "label": e["label"],
                        "count": e["count"],
                        "avg_confidence": e["avg_confidence"],
                    }
                    for e in unique[:50]
                ],
            },
        },
    )

    logger.info(
        f"Done: {len(raw_entities)} -> {len(deduped)} entities, {len(unique)} unique"
    )
    logger.info(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
