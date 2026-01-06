"""
Normalize entities using LLM to merge duplicates and filter noise.

Uses LLM to identify:
1. Duplicate groups: Same real-world entity with different surface forms
   (e.g., "Trump" + "Donald Trump" → "donald trump")
2. Noise entities: Generic fragments that aren't real entities
   (e.g., "state", "federal", "country")

Conservative merging rules enforce high precision - only 100% certain merges.
Propagates canonical names to by-source data for full event-entity traceability.

Pipeline Position: 03_2_dedupe_entities → 03_3_normalize_entities → 03_4_extract_relations

Input:
    From: data/03_2_dedupe_entities/<timestamp>/
    Files:
        - entities_unique.json: Aggregated unique entities with counts
        - entities_by_source.json: Entities grouped by event/market

Output:
    To: data/03_3_normalize_entities/<timestamp>/
    Files:
        - entities_normalized.json: Clean entities with canonical names and labels
        - entities_noise.json: Filtered noise entities with removal reasons
        - merge_mappings.json: Duplicate group mappings with merge reasons
        - entities_by_source_normalized.json: Event/market entities with canonical names
        - summary.json: Normalization statistics, top merged groups

Runtime: ~10 minutes for ~2500 entities (LLM API-bound)
Model: xiaomi/mimo-v2-flash:free (via OpenRouter)

Configuration:
    - LLM_MODEL: Model for entity resolution
    - BATCH_SIZE: Entities per LLM request (50)
    - MAX_RETRIES: API retry attempts (3)
"""

import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

# =============================================================================
# CONFIGURATION
# =============================================================================

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_SCRIPT_DIR = DATA_DIR / "03_2_dedupe_entities"
INPUT_RUN_FOLDER: str | None = None
SCRIPT_OUTPUT_DIR = DATA_DIR / "03_3_normalize_entities"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "xiaomi/mimo-v2-flash:free"

BATCH_SIZE = 50
MAX_RETRIES = 3
REQUEST_TIMEOUT = 60.0

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# LLM PROMPTS
# =============================================================================

SYSTEM_PROMPT = """You are an expert at entity resolution for political/geopolitical entities.

CONTEXT: Entities extracted from Polymarket political prediction markets.

Your task: identify duplicates and noise entities.

CRITICAL RULES:
- Only merge if 100% CERTAIN they're the SAME entity
- NEVER merge different countries (Ukraine ≠ United Kingdom, China ≠ Taiwan)
- Valid country merges: abbreviations only (US/USA/United States, UK/United Kingdom)
- NEVER merge country with adjective (China ≠ Chinese, Russia ≠ Russian)
- Politicians: only merge if FIRST NAME matches (John Sununu ≠ Chris Sununu)
- Events: NEVER merge different locations (GA-14 ≠ NJ-11, Texas ≠ Florida)
- NOISE: Generic fragments only ("state", "country", "federal")
- KEEP: Acronyms (CIA, FBI, NATO), titles, roles, events, legislation

BE CONSERVATIVE. When in doubt, do NOT merge.
Output valid JSON only, no markdown."""

USER_PROMPT = """Analyze these {label} entities:

{entities_list}

Respond with JSON:
{{"duplicate_groups": [{{"canonical": "name", "variants": ["v1", "v2"], "reason": "why"}}],
  "noise_entities": [{{"name": "entity", "reason": "why noise"}}],
  "clean_entities": ["entity1", "entity2"]}}"""

# =============================================================================
# MAIN LOGIC
# =============================================================================

total_tokens = 0


def llm_complete(prompt: str, client: httpx.Client) -> str:
    """Send completion request with retries."""
    global total_tokens
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                json={"model": LLM_MODEL, "messages": messages, "temperature": 0.1},
            )
            resp.raise_for_status()
            data = resp.json()
            total_tokens += data.get("usage", {}).get("total_tokens", 0)
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"LLM error (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 * (attempt + 1))
    return ""


def parse_llm_response(response: str) -> dict:
    """Parse JSON from LLM response, handling markdown wrapping."""
    response = response.strip()
    if response.startswith("```"):
        response = response.split("\n", 1)[1]
    if response.endswith("```"):
        response = response.rsplit("```", 1)[0]
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        return {"duplicate_groups": [], "noise_entities": [], "clean_entities": []}


def process_entities(
    entities: list[dict], client: httpx.Client
) -> tuple[list[dict], list[dict], dict, dict]:
    """Process all entities through LLM, return (clean, noise, stats, variant_map)."""
    by_label: dict[str, list[dict]] = defaultdict(list)
    for e in entities:
        by_label[e["label"]].append(e)

    clean, noise = [], []
    stats = {"by_label": {}, "merged_count": 0}
    variant_to_canonical: dict[
        str, tuple[str, str]
    ] = {}  # variant -> (canonical, label)

    for label, label_ents in by_label.items():
        logger.info(f"Processing {len(label_ents)} {label} entities...")
        lookup = {e["text_normalized"]: e for e in label_ents}
        processed = set()

        # Process in batches
        all_groups, all_noise, all_clean = [], [], []
        for i in range(0, len(label_ents), BATCH_SIZE):
            batch = label_ents[i : i + BATCH_SIZE]
            ent_list = "\n".join(
                f'- "{e["text_normalized"]}" ({e["count"]})'
                for e in sorted(batch, key=lambda x: -x["count"])
            )
            result = parse_llm_response(
                llm_complete(
                    USER_PROMPT.format(label=label, entities_list=ent_list), client
                )
            )
            all_groups.extend(result.get("duplicate_groups", []))
            all_noise.extend(result.get("noise_entities", []))
            all_clean.extend(result.get("clean_entities", []))

        # Process duplicate groups
        for group in all_groups:
            canonical = group["canonical"].lower().strip()
            variants = [canonical] + [
                v.lower().strip() for v in group.get("variants", [])
            ]
            matching = [
                lookup[v] for v in variants if v in lookup and v not in processed
            ]
            for v in variants:
                processed.add(v)
            if matching:
                total_count = sum(e["count"] for e in matching)
                matched_variants = [e["text_normalized"] for e in matching]
                clean.append(
                    {
                        "canonical_name": canonical,
                        "label": label,
                        "variants": matched_variants,
                        "count": total_count,
                        "avg_confidence": round(
                            sum(e["avg_confidence"] * e["count"] for e in matching)
                            / total_count,
                            3,
                        ),
                        "merged": len(matching) > 1,
                        "merge_reason": group.get("reason"),
                    }
                )
                stats["merged_count"] += len(matching) - 1
                # Build variant->canonical mapping
                for v in matched_variants:
                    variant_to_canonical[v] = (canonical, label)

        # Process noise
        for n in all_noise:
            name = n["name"].lower().strip()
            if name in lookup and name not in processed:
                processed.add(name)
                noise.append(
                    {
                        "name": name,
                        "label": label,
                        "count": lookup[name]["count"],
                        "reason": n.get("reason"),
                    }
                )

        # Process clean + unprocessed
        for name in all_clean + [n for n in lookup if n not in processed]:
            name = name.lower().strip() if isinstance(name, str) else name
            if name in lookup and name not in processed:
                processed.add(name)
                e = lookup[name]
                clean.append(
                    {
                        "canonical_name": name,
                        "label": label,
                        "variants": [name],
                        "count": e["count"],
                        "avg_confidence": round(e["avg_confidence"], 3),
                        "merged": False,
                        "merge_reason": None,
                    }
                )
                # Build variant->canonical mapping (self-mapping for non-merged)
                variant_to_canonical[name] = (name, label)

        stats["by_label"][label] = {
            "input": len(label_ents),
            "output": len([c for c in clean if c["label"] == label]),
            "noise": len([n for n in noise if n["label"] == label]),
        }

    clean.sort(key=lambda x: -x["count"])
    noise.sort(key=lambda x: -x["count"])
    return clean, noise, stats, variant_to_canonical


def normalize_entity(
    entity: dict, variant_map: dict[str, tuple[str, str]], noise_set: set[str]
) -> dict | None:
    """Apply normalization mapping to a single entity, return None if noise."""
    text_canonical = entity.get(
        "text_canonical", entity.get("text", "").lower().strip()
    )
    if text_canonical in noise_set:
        return None
    canonical_name, label = variant_map.get(
        text_canonical, (text_canonical, entity.get("label", "UNKNOWN"))
    )
    return {
        "canonical_name": canonical_name,
        "label": label,
        "original_text": entity.get("text", text_canonical),
    }


def normalize_by_source(
    by_source_data: dict,
    variant_map: dict[str, tuple[str, str]],
    noise_entities: list[dict],
) -> dict:
    """Apply normalization mappings to entities_by_source data."""
    noise_set = {n["name"] for n in noise_entities}
    normalized = {}

    for event_id, event_data in by_source_data.get("by_event", {}).items():
        # Normalize event-level entities
        event_entities = []
        for entity in event_data.get("event_entities", []):
            normalized_entity = normalize_entity(entity, variant_map, noise_set)
            if normalized_entity:
                event_entities.append(normalized_entity)

        # Normalize market-level entities
        market_entities = {}
        for market_id, market_ents in event_data.get("market_entities", {}).items():
            market_normalized = []
            for entity in market_ents:
                normalized_entity = normalize_entity(entity, variant_map, noise_set)
                if normalized_entity:
                    market_normalized.append(normalized_entity)
            if market_normalized:
                market_entities[market_id] = market_normalized

        if event_entities or market_entities:
            normalized[event_id] = {
                "event_entities": event_entities,
                "market_entities": market_entities,
            }

    return normalized


def main() -> None:
    start_time = datetime.now(timezone.utc)
    logger.info("Entity Normalization via LLM")

    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY not set")
        return

    # Find input
    if INPUT_RUN_FOLDER:
        input_folder = INPUT_SCRIPT_DIR / INPUT_RUN_FOLDER
    else:
        folders = (
            [f for f in INPUT_SCRIPT_DIR.iterdir() if f.is_dir()]
            if INPUT_SCRIPT_DIR.exists()
            else []
        )
        input_folder = (
            max(folders, key=lambda f: f.stat().st_mtime) if folders else None
        )

    if (
        not input_folder
        or not (input_file := input_folder / "entities_unique.json").exists()
    ):
        logger.error(f"Input not found: {input_folder}")
        return

    by_source_file = input_folder / "entities_by_source.json"
    if not by_source_file.exists():
        logger.error(f"entities_by_source.json not found: {by_source_file}")
        return

    logger.info(f"Input: {input_file}")
    with open(input_file, encoding="utf-8") as f:
        entities = json.load(f).get("entities", [])
    logger.info(f"Loaded {len(entities)} entities")

    logger.info(f"Input: {by_source_file}")
    with open(by_source_file, encoding="utf-8") as f:
        by_source_data = json.load(f)
    logger.info(
        f"Loaded {len(by_source_data.get('by_event', {}))} events from by_source"
    )

    # Process
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        clean, noise, stats, variant_map = process_entities(entities, client)

    logger.info(f"Clean: {len(clean)}, Noise: {len(noise)}, Tokens: {total_tokens}")

    # Apply normalization to by_source data
    by_source_normalized = normalize_by_source(by_source_data, variant_map, noise)
    logger.info(f"Normalized {len(by_source_normalized)} events in by_source")

    # Output
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_folder = SCRIPT_OUTPUT_DIR / timestamp
    output_folder.mkdir(parents=True, exist_ok=True)
    now_iso = datetime.now(timezone.utc).isoformat()

    def save(name: str, data: dict):
        with open(output_folder / name, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved: {output_folder / name}")

    save(
        "entities_normalized.json",
        {
            "_meta": {
                "description": "Normalized entities",
                "source": str(input_file),
                "created_at": now_iso,
                "model": LLM_MODEL,
            },
            "entities": clean,
        },
    )

    save(
        "entities_noise.json",
        {
            "_meta": {
                "description": "Noise entities",
                "source": str(input_file),
                "created_at": now_iso,
            },
            "entities": noise,
        },
    )

    merged = [e for e in clean if e["merged"]]
    save(
        "merge_mappings.json",
        {
            "_meta": {"description": "Entity merge mappings", "created_at": now_iso},
            "merges": [
                {
                    "canonical": e["canonical_name"],
                    "variants": e["variants"],
                    "label": e["label"],
                    "reason": e["merge_reason"],
                }
                for e in merged
            ],
        },
    )

    save(
        "entities_by_source_normalized.json",
        {
            "_meta": {
                "description": "Normalized entities grouped by event with canonical names",
                "source_unique": str(input_file),
                "source_by_event": str(by_source_file),
                "created_at": now_iso,
                "model": LLM_MODEL,
            },
            "by_event": by_source_normalized,
        },
    )

    end_time = datetime.now(timezone.utc)
    save(
        "summary.json",
        {
            "run_info": {
                "started_at": start_time.isoformat(),
                "completed_at": end_time.isoformat(),
                "duration_seconds": (end_time - start_time).total_seconds(),
                "input_file": str(input_file),
                "input_by_source_file": str(by_source_file),
                "output_folder": str(output_folder),
                "model": LLM_MODEL,
                "total_tokens": total_tokens,
            },
            "normalization": {
                "input_entities": len(entities),
                "output_entities": len(clean),
                "noise_entities": len(noise),
                "merged_groups": len(merged),
                "total_merged_variants": sum(len(e["variants"]) for e in merged),
            },
            "by_source_normalization": {
                "input_events": len(by_source_data.get("by_event", {})),
                "output_events": len(by_source_normalized),
                "variant_mappings": len(variant_map),
            },
            "by_label": stats["by_label"],
            "top_merged": [
                {
                    "canonical": e["canonical_name"],
                    "variants": e["variants"],
                    "count": e["count"],
                    "reason": e["merge_reason"],
                }
                for e in sorted(merged, key=lambda x: -x["count"])[:20]
            ],
        },
    )

    logger.info(
        f"Done in {(end_time - start_time).total_seconds():.1f}s | Output: {output_folder}"
    )


if __name__ == "__main__":
    main()
