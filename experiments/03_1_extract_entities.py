"""
Extract Named Entities from Polymarket events using GLiNER2 zero-shot NER.

Processes event titles, descriptions, and market questions through GLiNER2
model to extract 28 entity types across categories: people, organizations,
places, conflicts, legal actions, policies, events, and outcome states.

Note: Uses individual extraction (not batch) because batch_extract_entities
becomes extremely slow with 20+ entity types.

Pipeline Position: 02_prepare_nlp_data → 03_1_extract_entities → 03_2_dedupe_entities

Input:
    From: data/02_prepare_nlp_data/<timestamp>/
    Files:
        - nlp_events.json: Events with text fields (title, description, questions)

Output:
    To: data/03_1_extract_entities/<timestamp>/
    Files:
        - entities_raw.json: All extracted entities with source context and confidence
        - entities_by_source.json: Entities grouped by event and market
        - entities_unique.json: Deduplicated entities with occurrence counts
        - summary.json: Statistics (counts by label, confidence distribution)

Runtime: ~5 seconds/text on CPU (~800 events, ~4000 texts = ~5-6 hours)
Model: fastino/gliner2-base-v1 (GLiNER2, 205M parameters)

Entity Types (28 total):
    People: POLITICIAN, PUBLIC_FIGURE
    Organizations: POLITICAL_PARTY, GOVERNMENT_AGENCY, COURT, INTERNATIONAL_ORGANIZATION, ARMED_GROUP, COMPANY
    Places: COUNTRY, TERRITORY, STATE, CITY
    Conflicts: MILITARY_EQUIPMENT
    Legal: CRIMINAL_CHARGE, LEGAL_CASE, LEGISLATION, AGREEMENT
    Policy: GOVERNMENT_POLICY, SANCTION
    Events: POLITICAL_EVENT, ECONOMIC_INDICATOR, NEWS_SOURCE
    Outcome States: ECONOMIC_CRISIS, CONFLICT_OUTCOME, PEACE_OUTCOME, REGIME_CHANGE, MARKET_SHOCK, POLICY_SHIFT

Configuration:
    - CONFIDENCE_THRESHOLD: Minimum extraction confidence (0.6)
    - MIN_ENTITY_LENGTH: Minimum text length to keep (3 chars)
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
INPUT_SCRIPT_DIR = DATA_DIR / "02_prepare_nlp_data"
INPUT_RUN_FOLDER: str | None = None  # Specific timestamp folder, or None for latest
SCRIPT_OUTPUT_DIR = DATA_DIR / "03_1_extract_entities"

GLINER2_MODEL = "fastino/gliner2-base-v1"
CONFIDENCE_THRESHOLD = 0.6
MIN_ENTITY_LENGTH = 3
MAX_EVENTS: int | None = 400  # None for full run, set to small number for testing

# Entity labels with concise descriptions for fast extraction.
# Total: 28 entity types organized by category (military_conflict is in Outcome States).
ENTITY_LABELS: dict[str, str] = {
    # PEOPLE
    "politician": "Elected/appointed officials: presidents, senators, governors, ministers, judges",
    "public figure": "Non-politicians: business leaders, celebrities, activists, criminals",
    # ORGANIZATIONS
    "political party": "Political parties: Democratic Party, Republican Party, Labour, BJP",
    "government agency": "Government departments and agencies: FBI, CIA, DOJ, EPA, Pentagon",
    "court": "Courts and tribunals: Supreme Court, ICC, District Court",
    "international organization": "International bodies: NATO, UN, EU, WHO, IMF, G7",
    "armed group": "Militant/paramilitary groups: Hamas, Hezbollah, Taliban, Wagner Group",
    "company": "Corporations: Tesla, SpaceX, Meta, Google, Apple, OpenAI",
    # PLACES
    "country": "Sovereign nations: United States, China, Russia, Ukraine, Israel",
    "territory": "Disputed territories: Crimea, Gaza Strip, West Bank, Taiwan, Kosovo",
    "state": "Subnational regions: California, Texas, Ontario, Bavaria, Scotland",
    "city": "Cities: Kyiv, Moscow, Jerusalem, Tehran, Beijing, Damascus",
    # CONFLICTS & MILITARY
    "military conflict": "Wars and conflicts: Russia-Ukraine War, Israel-Hamas War, Sudan Civil War",
    "military equipment": "Weapons systems: ATACMS, HIMARS, Patriot missile, F-35, Iron Dome",
    # LEGAL & POLITICAL ACTIONS
    "criminal charge": "Legal charges: indictment, impeachment, fraud, treason, bribery",
    "legal case": "Court cases: Trump trial, Roe v. Wade, Citizens United",
    "legislation": "Laws and bills: Affordable Care Act, 25th Amendment, PATRIOT Act",
    "agreement": "Treaties and accords: JCPOA, Paris Agreement, Abraham Accords, ceasefire",
    # GOVERNMENT ACTIONS
    "government policy": "Executive policies: travel ban, deportation policy, tariff policy",
    "sanction": "Sanctions and embargoes: Russia sanctions, SWIFT ban, arms embargo",
    # EVENTS & METRICS
    "political event": "Elections, summits, referendums: 2024 Election, G7 Summit, Brexit",
    "economic indicator": "Economic metrics: inflation rate, GDP, unemployment, CPI",
    "news source": "Media outlets: CNN, Fox News, NYT, Reuters, BBC, Bloomberg",
    # OUTCOME STATES (6 types for causal reasoning)
    "economic crisis": "Economic downturns and financial distress: recession, depression, inflation spike, stagflation, market crash",
    "conflict outcome": "Armed conflict states and military actions: war, invasion, escalation, attack, bombing, military operation",
    "peace outcome": "Conflict resolution: ceasefire, peace treaty, withdrawal, de-escalation, diplomatic resolution",
    "regime change": "Political transitions: coup, resignation, impeachment, overthrow, assassination, succession",
    "market shock": "Financial market events: stock crash, currency crisis, commodity spike, liquidity crisis",
    "policy shift": "Major policy changes: sanctions, tariffs, trade war, regulation change, ban, embargo",
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


def extract_texts(events: list[dict]) -> list[dict]:
    """Extract all text snippets from events and markets."""
    texts = []
    for event in events:
        event_id = event.get("id", "unknown")
        seen = set()

        for field in ["title", "description"]:
            if text := event.get(field, "").strip():
                if text not in seen:
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
                if text not in seen:
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


def process_texts(model: GLiNER2, texts: list[dict]) -> list[dict]:
    """Extract entities from texts using GLiNER2 individual extraction.

    Note: batch_extract_entities becomes very slow with many entity types (23+),
    so we use individual extraction which is faster in practice.
    """
    entities = []
    total = len(texts)

    logger.info(f"Processing {total} texts...")

    for idx, item in enumerate(texts):
        result = model.extract_entities(
            item["text"],
            ENTITY_LABELS,
            threshold=CONFIDENCE_THRESHOLD,
            include_confidence=True,
        )

        # Result format: {'entities': {label: [{'text': ..., 'confidence': ...}]}}
        for label, ents in result.get("entities", {}).items():
            for ent in ents:
                text = (
                    ent["text"].strip() if isinstance(ent, dict) else str(ent).strip()
                )
                conf = ent.get("confidence", 1.0) if isinstance(ent, dict) else 1.0

                if len(text) >= MIN_ENTITY_LENGTH:
                    entities.append(
                        {
                            "text": text,
                            "label": label.upper().replace(" ", "_"),
                            "source_type": item["source_type"],
                            "source_id": item["source_id"],
                            "event_id": item["event_id"],
                            "context": item["text"],
                            "confidence": conf,
                        }
                    )

        if (idx + 1) % 50 == 0 or idx == total - 1:
            logger.info(f"  Processed {idx + 1}/{total} texts")

    return entities


def aggregate_entities(entities: list[dict]) -> list[dict]:
    """Aggregate entities into unique entries."""
    unique: dict[str, dict] = {}

    for ent in entities:
        key = ent["text"].lower().strip()
        if key not in unique:
            unique[key] = {
                "text_normalized": key,
                "label": ent["label"],
                "count": 0,
                "variations": [],
                "label_counts": defaultdict(int),
                "source_events": set(),
                "source_markets": set(),
                "confidences": [],
            }

        u = unique[key]
        u["count"] += 1
        u["variations"].append(ent["text"])
        u["confidences"].append(ent["confidence"])
        u["label_counts"][ent["label"]] += 1
        u["source_events"].add(ent["event_id"])
        if ent["source_type"].startswith("market_"):
            u["source_markets"].add(ent["source_id"])

    # Finalize
    result = []
    for u in unique.values():
        u["label"] = max(u["label_counts"], key=u["label_counts"].get)
        u["avg_confidence"] = round(sum(u["confidences"]) / len(u["confidences"]), 3)
        result.append(
            {
                "text_normalized": u["text_normalized"],
                "label": u["label"],
                "count": u["count"],
                "variations": sorted(set(u["variations"])),
                "label_counts": dict(u["label_counts"]),
                "event_count": len(u["source_events"]),
                "market_count": len(u["source_markets"]),
                "avg_confidence": u["avg_confidence"],
            }
        )

    return sorted(result, key=lambda x: -x["count"])


def group_by_source(entities: list[dict]) -> dict:
    """Group entities by event and market."""
    by_event: dict[str, dict] = defaultdict(
        lambda: {"event_entities": [], "market_entities": defaultdict(list)}
    )

    for ent in entities:
        if ent["source_type"].startswith("event_"):
            by_event[ent["event_id"]]["event_entities"].append(ent)
        else:
            by_event[ent["event_id"]]["market_entities"][ent["source_id"]].append(ent)

    return {
        k: {
            "event_entities": v["event_entities"],
            "market_entities": dict(v["market_entities"]),
        }
        for k, v in by_event.items()
    }


def compute_stats(
    entities: list[dict], unique: list[dict], texts_count: int, events_count: int
) -> dict:
    """Compute summary statistics."""
    label_counts = defaultdict(int)
    source_counts = defaultdict(int)
    confidences = []

    for e in entities:
        label_counts[e["label"]] += 1
        source_counts[e["source_type"]] += 1
        confidences.append(e["confidence"])

    return {
        "total_extractions": len(entities),
        "unique_entities": len(unique),
        "texts_processed": texts_count,
        "events_processed": events_count,
        "confidence_stats": {
            "avg": round(sum(confidences) / len(confidences), 3) if confidences else 0,
            "min": round(min(confidences), 3) if confidences else 0,
            "max": round(max(confidences), 3) if confidences else 0,
        },
        "by_label": dict(sorted(label_counts.items(), key=lambda x: -x[1])),
        "by_source_type": dict(sorted(source_counts.items(), key=lambda x: -x[1])),
        "top_50_entities": [
            {
                "text": e["text_normalized"],
                "label": e["label"],
                "count": e["count"],
                "avg_confidence": e["avg_confidence"],
            }
            for e in unique[:50]
        ],
    }


def main() -> None:
    start_time = datetime.now(timezone.utc)
    logger.info("=" * 60)
    logger.info("Polymarket Entity Extraction (GLiNER2)")
    logger.info("=" * 60)

    # Find input
    input_folder = (
        INPUT_SCRIPT_DIR / INPUT_RUN_FOLDER
        if INPUT_RUN_FOLDER
        else find_latest_run_folder(INPUT_SCRIPT_DIR)
    )
    if not input_folder or not input_folder.exists():
        logger.error(f"Input folder not found: {input_folder}")
        return

    input_file = input_folder / "nlp_events.json"
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return

    logger.info(f"Input: {input_file}")

    with open(input_file, encoding="utf-8") as f:
        events = json.load(f).get("events", [])

    if MAX_EVENTS:
        events = events[:MAX_EVENTS]
        logger.info(f"Limited to {MAX_EVENTS} events")

    logger.info(f"Loaded {len(events)} events")

    # Process
    logger.info(f"Loading model: {GLINER2_MODEL}")
    model = GLiNER2.from_pretrained(GLINER2_MODEL)

    texts = extract_texts(events)
    logger.info(f"Extracted {len(texts)} texts")

    entities = process_texts(model, texts)
    logger.info(f"Extracted {len(entities)} entities")

    unique = aggregate_entities(entities)
    logger.info(f"Unique entities: {len(unique)}")

    by_source = group_by_source(entities)
    stats = compute_stats(entities, unique, len(texts), len(events))

    # Output
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_folder = SCRIPT_OUTPUT_DIR / timestamp
    output_folder.mkdir(parents=True, exist_ok=True)

    meta = {
        "source": str(input_file),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gliner2_model": GLINER2_MODEL,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }

    for filename, data in [
        (
            "entities_raw.json",
            {"_meta": {**meta, "entity_labels": ENTITY_LABELS}, "entities": entities},
        ),
        ("entities_by_source.json", {"_meta": meta, "by_event": by_source}),
        ("entities_unique.json", {"_meta": meta, "entities": unique}),
    ]:
        with open(output_folder / filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved: {filename}")

    end_time = datetime.now(timezone.utc)
    summary = {
        "run_info": {
            "started_at": start_time.isoformat(),
            "completed_at": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "input_file": str(input_file),
            "output_folder": str(output_folder),
        },
        "configuration": {
            "gliner2_model": GLINER2_MODEL,
            "entity_labels": ENTITY_LABELS,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "min_entity_length": MIN_ENTITY_LENGTH,
            "max_events": MAX_EVENTS,
        },
        "statistics": stats,
    }

    with open(output_folder / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Saved: summary.json")

    # Summary
    logger.info("=" * 60)
    logger.info(
        f"Events: {len(events)} | Texts: {len(texts)} | Entities: {len(entities)} | Unique: {len(unique)}"
    )
    logger.info(
        f"Duration: {(end_time - start_time).total_seconds():.1f}s | Output: {output_folder}"
    )


if __name__ == "__main__":
    main()
