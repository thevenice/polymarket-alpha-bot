"""
Entity extraction, deduplication, and normalization.

Combines logic from:
- experiments/03_1_extract_entities.py
- experiments/03_2_dedupe_entities.py
- experiments/03_3_normalize_entities.py

For production pipeline with incremental support.
"""

from collections import defaultdict

from loguru import logger
from rapidfuzz import fuzz

from core.models import get_gliner, get_llm_client
from core.state import PipelineState

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIDENCE_THRESHOLD = 0.6
MIN_ENTITY_LENGTH = 3

# Fuzzy matching thresholds
FUZZY_THRESHOLD = 85
FUZZY_MIN_FREQUENCY = 2

# Entity labels for GLiNER2
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
    "military conflict": "Wars and conflicts: Russia-Ukraine War, Israel-Hamas War",
    "military equipment": "Weapons systems: ATACMS, HIMARS, Patriot missile, F-35",
    # LEGAL & POLITICAL ACTIONS
    "criminal charge": "Legal charges: indictment, impeachment, fraud, treason",
    "legal case": "Court cases: Trump trial, Roe v. Wade, Citizens United",
    "legislation": "Laws and bills: Affordable Care Act, 25th Amendment",
    "agreement": "Treaties and accords: JCPOA, Paris Agreement, Abraham Accords",
    # GOVERNMENT ACTIONS
    "government policy": "Executive policies: travel ban, deportation policy, tariff",
    "sanction": "Sanctions and embargoes: Russia sanctions, SWIFT ban",
    # EVENTS & METRICS
    "political event": "Elections, summits, referendums: 2024 Election, G7 Summit",
    "economic indicator": "Economic metrics: inflation rate, GDP, unemployment",
    "news source": "Media outlets: CNN, Fox News, NYT, Reuters, BBC",
    # OUTCOME STATES
    "economic crisis": "Economic downturns: recession, depression, inflation spike",
    "conflict outcome": "Armed conflict states: war, invasion, escalation, attack",
    "peace outcome": "Conflict resolution: ceasefire, peace treaty, withdrawal",
    "regime change": "Political transitions: coup, resignation, impeachment",
    "market shock": "Financial market events: stock crash, currency crisis",
    "policy shift": "Major policy changes: sanctions, tariffs, trade war",
}


# =============================================================================
# TEXT EXTRACTION
# =============================================================================


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


# =============================================================================
# ENTITY EXTRACTION (GLiNER2)
# =============================================================================


def extract_entities_from_texts(texts: list[dict]) -> list[dict]:
    """
    Extract entities from texts using GLiNER2.

    Uses singleton model loader to avoid reloading.
    """
    model = get_gliner()
    entities = []
    total = len(texts)

    logger.info(f"Extracting entities from {total} texts...")

    for idx, item in enumerate(texts):
        result = model.extract_entities(
            item["text"],
            ENTITY_LABELS,
            threshold=CONFIDENCE_THRESHOLD,
            include_confidence=True,
        )

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

        if (idx + 1) % 100 == 0:
            logger.debug(f"  Processed {idx + 1}/{total} texts")

    logger.info(f"Extracted {len(entities)} raw entities")
    return entities


# =============================================================================
# AGGREGATION
# =============================================================================


def aggregate_entities(entities: list[dict]) -> list[dict]:
    """Aggregate entities into unique entries."""
    unique: dict[str, dict] = {}

    for ent in entities:
        key = ent["text"].lower().strip()
        if key not in unique:
            unique[key] = {
                "text": ent["text"],
                "text_normalized": key,
                "label": ent["label"],
                "count": 0,
                "variations": [],
                "label_counts": defaultdict(int),
                "source_events": set(),
                "confidences": [],
            }

        u = unique[key]
        u["count"] += 1
        u["variations"].append(ent["text"])
        u["confidences"].append(ent["confidence"])
        u["label_counts"][ent["label"]] += 1
        u["source_events"].add(ent["event_id"])

    # Finalize
    result = []
    for u in unique.values():
        u["label"] = max(u["label_counts"], key=u["label_counts"].get)
        u["avg_confidence"] = round(sum(u["confidences"]) / len(u["confidences"]), 3)
        result.append(
            {
                "text": u["text"],
                "text_normalized": u["text_normalized"],
                "label": u["label"],
                "count": u["count"],
                "variations": sorted(set(u["variations"])),
                "event_count": len(u["source_events"]),
                "avg_confidence": u["avg_confidence"],
                "source_events": list(u["source_events"]),
            }
        )

    # Sort by frequency
    result.sort(key=lambda x: x["count"], reverse=True)
    return result


# =============================================================================
# DEDUPLICATION (Fuzzy Matching)
# =============================================================================


def dedupe_entities_fuzzy(
    entities: list[dict],
    existing_mappings: dict[str, str] | None = None,
) -> tuple[list[dict], dict[str, str]]:
    """
    Deduplicate entities using fuzzy matching.

    Args:
        entities: List of aggregated entities
        existing_mappings: Existing raw->canonical mappings from state

    Returns:
        Tuple of (deduped_entities, new_mappings)
    """
    existing_mappings = existing_mappings or {}
    new_mappings: dict[str, str] = {}

    # Group by label for more accurate matching
    by_label: dict[str, list[dict]] = defaultdict(list)
    for ent in entities:
        by_label[ent["label"]].append(ent)

    # Process each label group
    deduped = []
    for label, group in by_label.items():
        # Sort by count (most frequent first = canonical)
        group.sort(key=lambda x: x["count"], reverse=True)

        canonical_map: dict[str, str] = {}  # normalized -> canonical

        for ent in group:
            normalized = ent["text_normalized"]

            # Check if already mapped
            if normalized in existing_mappings:
                ent["canonical"] = existing_mappings[normalized]
                continue

            # Check for fuzzy match with existing canonicals
            best_match = None
            best_score = 0

            for canonical in canonical_map:
                score = fuzz.ratio(normalized, canonical)
                if score > best_score and score >= FUZZY_THRESHOLD:
                    best_score = score
                    best_match = canonical

            if best_match:
                # Map to existing canonical
                ent["canonical"] = canonical_map[best_match]
                new_mappings[normalized] = canonical_map[best_match]
            else:
                # This is a new canonical
                ent["canonical"] = ent["text"]
                canonical_map[normalized] = ent["text"]

        deduped.extend(group)

    logger.info(
        f"Fuzzy dedup: {len(entities)} -> {len(deduped)}, {len(new_mappings)} new mappings"
    )
    return deduped, new_mappings


# =============================================================================
# NORMALIZATION (LLM-based)
# =============================================================================


async def normalize_entities_llm(
    entities: list[dict],
    state: PipelineState,
    batch_size: int = 20,
) -> list[dict]:
    """
    Normalize entities using LLM for better canonical forms.

    Only processes entities that need normalization (high frequency, ambiguous).
    """
    llm = get_llm_client()

    # Filter entities that need normalization
    to_normalize = [
        e
        for e in entities
        if e["count"] >= FUZZY_MIN_FREQUENCY and len(e.get("variations", [])) > 1
    ]

    if not to_normalize:
        logger.info("No entities need LLM normalization")
        return entities

    logger.info(f"Normalizing {len(to_normalize)} entities with LLM...")

    # Process in batches
    for i in range(0, len(to_normalize), batch_size):
        batch = to_normalize[i : i + batch_size]

        for ent in batch:
            variations = ent.get("variations", [ent["text"]])[:5]
            prompt = f"""Given these variations of the same entity:
{", ".join(variations)}

Return ONLY the most standard/canonical form. Just the name, nothing else."""

            try:
                messages = [{"role": "user", "content": prompt}]
                response = await llm.complete(messages, temperature=0.0)
                canonical = response.strip().strip('"').strip("'")

                if canonical and len(canonical) < 100:
                    ent["canonical_llm"] = canonical
                    state.add_entity_mappings(
                        {ent["text_normalized"]: canonical}, mapping_type="llm"
                    )
            except Exception as e:
                logger.warning(f"LLM normalization failed for {ent['text']}: {e}")

    return entities


# =============================================================================
# MAIN FUNCTION
# =============================================================================


def extract_and_process_entities(
    events: list[dict],
    state: PipelineState,
) -> list[dict]:
    """
    Full entity extraction pipeline.

    1. Extract texts from events
    2. Extract entities using GLiNER2
    3. Aggregate and deduplicate
    4. Apply existing mappings from state

    Args:
        events: NLP-prepared events
        state: Pipeline state for mappings

    Returns:
        List of processed entities with canonical forms
    """
    # Extract texts
    texts = extract_texts(events)
    logger.info(f"Extracted {len(texts)} text segments")

    # Extract entities
    raw_entities = extract_entities_from_texts(texts)

    # Aggregate
    aggregated = aggregate_entities(raw_entities)
    logger.info(f"Aggregated to {len(aggregated)} unique entities")

    # Dedupe with fuzzy matching
    existing_mappings = state.get_entity_mappings()
    deduped, new_mappings = dedupe_entities_fuzzy(aggregated, existing_mappings)

    # Save new mappings to state
    if new_mappings:
        state.add_entity_mappings(new_mappings, mapping_type="fuzzy")

    # Save entities to state
    state.add_entities(deduped)

    return deduped


def get_entities_by_event(entities: list[dict]) -> dict[str, list[dict]]:
    """Group entities by event ID."""
    by_event: dict[str, list[dict]] = defaultdict(list)
    for ent in entities:
        for event_id in ent.get("source_events", []):
            by_event[event_id].append(ent)
    return dict(by_event)
