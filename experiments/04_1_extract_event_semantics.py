"""
Extract structured semantics from event titles.

Parses event titles into structured semantic components using a hybrid approach:
- Extract timeframe with regex (dates, years, quarters)
- Extract thresholds with regex (numbers, percentages)
- Use LLM for: event_type, polarity, subject/predicate/object, outcome_states

Pipeline: 02_prepare_nlp_data + 03_3_normalize_entities -> 04_1_extract_event_semantics -> 04_2_embed_events

Input:
    From: data/02_prepare_nlp_data/<timestamp>/
    Files:
        - nlp_events.json: Events with titles and descriptions

    From: data/03_3_normalize_entities/<timestamp>/
    Files:
        - entities_normalized.json: Normalized entities for entity matching

Output:
    To: data/04_1_extract_event_semantics/<timestamp>/
    Files:
        - event_semantics.json: Structured semantic parsing of events
        - summary.json: Statistics and run information

Runtime: ~5-10 minutes for ~800 events (LLM API-bound)
Model: xiaomi/mimo-v2-flash:free (via OpenRouter)

Configuration:
    - LLM_MODEL: Model for semantic extraction
    - BATCH_SIZE: Events per LLM request (10)
    - MAX_RETRIES: API retry attempts (3)
"""

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

# =============================================================================
# CONFIGURATION
# =============================================================================

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_NLP_DIR = DATA_DIR / "02_prepare_nlp_data"
INPUT_ENTITIES_DIR = DATA_DIR / "03_3_normalize_entities"
INPUT_NLP_RUN_FOLDER: str | None = None  # None for latest
INPUT_ENTITIES_RUN_FOLDER: str | None = None  # None for latest
SCRIPT_OUTPUT_DIR = DATA_DIR / "04_1_extract_event_semantics"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "xiaomi/mimo-v2-flash:free"

BATCH_SIZE = 10
MAX_RETRIES = 3
REQUEST_TIMEOUT = 120.0

EVENT_TYPES = ["OCCURRENCE", "THRESHOLD", "DEADLINE", "COMPARISON", "COUNT"]
POLARITIES = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
OUTCOME_STATES = [
    "economic_crisis",
    "military_conflict",
    "peace_outcome",
    "regime_change",
    "market_shock",
    "policy_shift",
]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# LLM PROMPTS
# =============================================================================

SYSTEM_PROMPT = """You are an expert at analyzing prediction market event titles and extracting structured semantic components.

CONTEXT: These are event titles from Polymarket political prediction markets. Each event asks about whether something will happen.

Your task: Extract semantic structure from event titles.

OUTPUT RULES:
- event_type: OCCURRENCE (binary yes/no), THRESHOLD (numeric comparison), DEADLINE (time-bounded), COMPARISON (relative), COUNT (enumeration)
- polarity: POSITIVE (good/constructive like peace, growth), NEGATIVE (bad/destructive like war, recession), NEUTRAL (no value judgment)
- subject_entity: The entity performing the action (lowercase, use canonical name if provided)
- predicate: The main action/event verb phrase
- object_entity: The entity receiving the action (lowercase, use canonical name if provided)
- outcome_states: List from [economic_crisis, military_conflict, peace_outcome, regime_change, market_shock, policy_shift]
- negation: true if the event asks about something NOT happening

Be precise and consistent. Output valid JSON only, no markdown."""

USER_PROMPT_TEMPLATE = """Analyze these prediction market events and extract structured semantics.

Events:
{events_text}

For each event, return JSON array with:
{{
  "id": "event_id",
  "event_type": "OCCURRENCE|THRESHOLD|DEADLINE|COMPARISON|COUNT",
  "polarity": "POSITIVE|NEGATIVE|NEUTRAL",
  "subject_entity": "entity or null",
  "predicate": "main action phrase",
  "object_entity": "entity or null",
  "outcome_states": ["list of applicable states"],
  "negation": false
}}

Return a JSON array with one object per event. Use the exact event IDs provided."""

# =============================================================================
# REGEX PATTERNS FOR TIMEFRAME AND THRESHOLD EXTRACTION
# =============================================================================

# Year patterns: "in 2025", "by 2025", "2025?"
YEAR_PATTERN = re.compile(
    r"\b(?:in|by|before|after|during)?\s*(\d{4})\b\??", re.IGNORECASE
)

# Date patterns: "by March 2025", "June 30, 2026", "Q1 2025"
MONTH_NAMES = r"(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
DATE_PATTERN = re.compile(
    rf"\b(?:by|before|on)?\s*(?:{MONTH_NAMES})\s*\d{{1,2}}?,?\s*\d{{4}}\b",
    re.IGNORECASE,
)
MONTH_YEAR_PATTERN = re.compile(
    rf"\b(?:by|before|in)?\s*(?:{MONTH_NAMES})\s*\d{{4}}\b", re.IGNORECASE
)
QUARTER_PATTERN = re.compile(r"\bQ([1-4])\s*(\d{4})\b", re.IGNORECASE)

# Threshold patterns: "750,000 or more", "less than 250,000", "50%", "above 100"
THRESHOLD_PATTERN = re.compile(
    r"\b(?:more than|less than|at least|over|under|above|below|exceed|reach)?\s*"
    r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*"
    r"(?:or more|or less|or fewer|\+|%|percent|people|million|billion|trillion)?\b",
    re.IGNORECASE,
)
PERCENTAGE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", re.IGNORECASE)
RANGE_PATTERN = re.compile(r"(\d{1,3}(?:,\d{3})*)-(\d{1,3}(?:,\d{3})*)", re.IGNORECASE)

# =============================================================================
# MAIN LOGIC
# =============================================================================

total_tokens = 0


def find_latest_run_folder(script_dir: Path) -> Path | None:
    """Find the most recent run folder."""
    if not script_dir.exists():
        return None
    folders = [f for f in script_dir.iterdir() if f.is_dir()]
    return max(folders, key=lambda f: f.stat().st_mtime) if folders else None


def parse_number(text: str) -> int | float | None:
    """Parse a number string, handling commas and suffixes."""
    text = text.replace(",", "").strip()
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return None


def extract_timeframe(title: str, end_date: str | None = None) -> dict | None:
    """Extract timeframe from event title using regex."""
    result = {}

    # Try quarter pattern first
    quarter_match = QUARTER_PATTERN.search(title)
    if quarter_match:
        quarter = int(quarter_match.group(1))
        year = int(quarter_match.group(2))
        quarter_end_months = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
        result = {
            "type": "deadline",
            "end_date": f"{year}-{quarter_end_months[quarter]}",
            "year": year,
            "quarter": quarter,
        }
        return result

    # Try full date pattern
    date_match = DATE_PATTERN.search(title)
    if date_match:
        date_str = date_match.group(0)
        # Parse the date string
        year_in_date = YEAR_PATTERN.search(date_str)
        if year_in_date:
            year = int(year_in_date.group(1))
            result = {
                "type": "deadline",
                "end_date": f"{year}-12-31",  # Simplified
                "year": year,
            }
            return result

    # Try month-year pattern
    month_year_match = MONTH_YEAR_PATTERN.search(title)
    if month_year_match:
        match_text = month_year_match.group(0)
        year_in_match = YEAR_PATTERN.search(match_text)
        if year_in_match:
            year = int(year_in_match.group(1))
            result = {
                "type": "deadline",
                "end_date": f"{year}-12-31",
                "year": year,
            }
            return result

    # Try year pattern
    year_match = YEAR_PATTERN.search(title)
    if year_match:
        year = int(year_match.group(1))
        if 2020 <= year <= 2035:  # Reasonable year range
            result = {
                "type": "deadline",
                "end_date": f"{year}-12-31",
                "year": year,
            }
            return result

    # Fall back to end_date from event if available
    if end_date:
        try:
            dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            return {
                "type": "deadline",
                "end_date": dt.strftime("%Y-%m-%d"),
                "year": dt.year,
            }
        except (ValueError, AttributeError):
            pass

    return None


def extract_threshold(title: str) -> dict | None:
    """Extract threshold/condition from event title using regex."""
    # Check for percentage
    pct_match = PERCENTAGE_PATTERN.search(title)
    if pct_match:
        value = parse_number(pct_match.group(1))
        if value is not None:
            operator = ">="
            if "less than" in title.lower() or "under" in title.lower():
                operator = "<"
            elif "more than" in title.lower() or "over" in title.lower():
                operator = ">"
            return {
                "type": "threshold",
                "operator": operator,
                "value": value,
                "unit": "percent",
            }

    # Check for range
    range_match = RANGE_PATTERN.search(title)
    if range_match:
        low = parse_number(range_match.group(1))
        high = parse_number(range_match.group(2))
        if low is not None and high is not None:
            return {
                "type": "range",
                "low": low,
                "high": high,
                "unit": "count",
            }

    # Check for threshold numbers
    threshold_match = THRESHOLD_PATTERN.search(title)
    if threshold_match:
        value = parse_number(threshold_match.group(1))
        if value is not None and value > 100:  # Likely a count, not a year
            operator = ">="
            title_lower = title.lower()
            if (
                "less than" in title_lower
                or "under" in title_lower
                or "fewer" in title_lower
            ):
                operator = "<"
            elif (
                "more than" in title_lower
                or "over" in title_lower
                or "exceed" in title_lower
            ):
                operator = ">"
            elif "or more" in title_lower or "at least" in title_lower:
                operator = ">="
            elif "or less" in title_lower or "or fewer" in title_lower:
                operator = "<="

            unit = "count"
            if "people" in title_lower:
                unit = "people"
            elif "million" in title_lower:
                unit = "million"
            elif "billion" in title_lower:
                unit = "billion"

            return {
                "type": "threshold",
                "operator": operator,
                "value": value,
                "unit": unit,
            }

    return None


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
        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP error (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 * (attempt + 1))
        except Exception as e:
            logger.warning(f"LLM error (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 * (attempt + 1))
    return "[]"


def parse_llm_response(response: str) -> list[dict]:
    """Parse JSON array from LLM response, handling markdown wrapping."""
    response = response.strip()
    if response.startswith("```"):
        response = response.split("\n", 1)[1]
    if response.endswith("```"):
        response = response.rsplit("```", 1)[0]
    response = response.strip()
    try:
        result = json.loads(response)
        if isinstance(result, list):
            return result
        return []
    except json.JSONDecodeError:
        # Try to find JSON array in response
        start = response.find("[")
        end = response.rfind("]")
        if start != -1 and end != -1:
            try:
                return json.loads(response[start : end + 1])
            except json.JSONDecodeError:
                pass
        return []


def build_entity_lookup(entities_data: dict) -> dict[str, str]:
    """Build a lookup from entity variants to canonical names."""
    lookup = {}
    for entity in entities_data.get("entities", []):
        canonical = entity.get("canonical_name", "").lower()
        for variant in entity.get("variants", []):
            lookup[variant.lower()] = canonical
        lookup[canonical] = canonical
    return lookup


def process_events_batch(
    events: list[dict], entity_lookup: dict[str, str], client: httpx.Client
) -> list[dict]:
    """Process a batch of events through LLM."""
    # Build prompt with event titles
    events_text = "\n".join(
        f'{i + 1}. [ID: {e["id"]}] "{e["title"]}"' for i, e in enumerate(events)
    )

    prompt = USER_PROMPT_TEMPLATE.format(events_text=events_text)
    response = llm_complete(prompt, client)
    llm_results = parse_llm_response(response)

    # Create lookup by ID
    llm_by_id = {str(r.get("id", "")): r for r in llm_results}

    results = []
    for event in events:
        event_id = event["id"]
        title = event["title"]
        end_date = event.get("endDate")

        # Get LLM result
        llm_result = llm_by_id.get(event_id, {})

        # Extract timeframe with regex
        timeframe = extract_timeframe(title, end_date)

        # Extract threshold/condition with regex
        condition = extract_threshold(title)

        # Normalize entities using lookup
        subject = llm_result.get("subject_entity")
        if subject and isinstance(subject, str):
            subject = entity_lookup.get(subject.lower(), subject.lower())

        obj = llm_result.get("object_entity")
        if obj and isinstance(obj, str):
            obj = entity_lookup.get(obj.lower(), obj.lower())

        # Validate event_type
        event_type = llm_result.get("event_type", "OCCURRENCE")
        if event_type not in EVENT_TYPES:
            event_type = "OCCURRENCE"

        # Validate polarity
        polarity = llm_result.get("polarity", "NEUTRAL")
        if polarity not in POLARITIES:
            polarity = "NEUTRAL"

        # Validate outcome_states
        outcome_states = llm_result.get("outcome_states", [])
        if not isinstance(outcome_states, list):
            outcome_states = []
        outcome_states = [s for s in outcome_states if s in OUTCOME_STATES]

        semantics = {
            "event_type": event_type,
            "polarity": polarity,
            "subject_entity": subject,
            "predicate": llm_result.get("predicate"),
            "object_entity": obj,
            "condition": condition,
            "timeframe": timeframe,
            "outcome_states": outcome_states,
            "negation": bool(llm_result.get("negation", False)),
        }

        results.append({"id": event_id, "title": title, "semantics": semantics})

    return results


def process_all_events(
    events: list[dict], entity_lookup: dict[str, str], client: httpx.Client
) -> list[dict]:
    """Process all events in batches."""
    results = []
    total = len(events)

    for i in range(0, total, BATCH_SIZE):
        batch = events[i : i + BATCH_SIZE]
        batch_results = process_events_batch(batch, entity_lookup, client)
        results.extend(batch_results)
        logger.info(f"Processed {min(i + BATCH_SIZE, total)}/{total} events")

    return results


def compute_statistics(events_semantics: list[dict]) -> dict:
    """Compute summary statistics."""
    stats = {
        "total_events": len(events_semantics),
        "by_event_type": {},
        "by_polarity": {},
        "events_with_conditions": 0,
        "events_with_timeframes": 0,
        "events_with_outcome_states": 0,
        "events_with_subject": 0,
        "events_with_object": 0,
        "events_with_negation": 0,
        "outcome_state_counts": {},
    }

    for event in events_semantics:
        sem = event.get("semantics", {})

        # Count event types
        event_type = sem.get("event_type", "UNKNOWN")
        stats["by_event_type"][event_type] = (
            stats["by_event_type"].get(event_type, 0) + 1
        )

        # Count polarities
        polarity = sem.get("polarity", "UNKNOWN")
        stats["by_polarity"][polarity] = stats["by_polarity"].get(polarity, 0) + 1

        # Count conditions
        if sem.get("condition"):
            stats["events_with_conditions"] += 1

        # Count timeframes
        if sem.get("timeframe"):
            stats["events_with_timeframes"] += 1

        # Count outcome states
        outcome_states = sem.get("outcome_states", [])
        if outcome_states:
            stats["events_with_outcome_states"] += 1
            for state in outcome_states:
                stats["outcome_state_counts"][state] = (
                    stats["outcome_state_counts"].get(state, 0) + 1
                )

        # Count subjects/objects
        if sem.get("subject_entity"):
            stats["events_with_subject"] += 1
        if sem.get("object_entity"):
            stats["events_with_object"] += 1

        # Count negations
        if sem.get("negation"):
            stats["events_with_negation"] += 1

    return stats


def main() -> None:
    start_time = datetime.now(timezone.utc)
    logger.info("=" * 60)
    logger.info("Event Semantics Extraction")
    logger.info("=" * 60)

    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY not set")
        return

    # Find NLP input
    input_nlp_folder = (
        INPUT_NLP_DIR / INPUT_NLP_RUN_FOLDER
        if INPUT_NLP_RUN_FOLDER
        else find_latest_run_folder(INPUT_NLP_DIR)
    )
    if not input_nlp_folder or not input_nlp_folder.exists():
        logger.error(f"NLP input folder not found: {input_nlp_folder}")
        return

    nlp_file = input_nlp_folder / "nlp_events.json"
    if not nlp_file.exists():
        logger.error(f"NLP events file not found: {nlp_file}")
        return

    logger.info(f"NLP Input: {nlp_file}")

    # Find entities input
    input_entities_folder = (
        INPUT_ENTITIES_DIR / INPUT_ENTITIES_RUN_FOLDER
        if INPUT_ENTITIES_RUN_FOLDER
        else find_latest_run_folder(INPUT_ENTITIES_DIR)
    )
    entities_file = None
    entity_lookup = {}

    if input_entities_folder and input_entities_folder.exists():
        entities_file = input_entities_folder / "entities_normalized.json"
        if entities_file.exists():
            logger.info(f"Entities Input: {entities_file}")
            with open(entities_file, encoding="utf-8") as f:
                entities_data = json.load(f)
            entity_lookup = build_entity_lookup(entities_data)
            logger.info(f"Loaded {len(entity_lookup)} entity mappings")
        else:
            logger.warning(f"Entities file not found: {entities_file}")
            entities_file = None
    else:
        logger.warning(f"Entities folder not found: {input_entities_folder}")

    # Load NLP events
    with open(nlp_file, encoding="utf-8") as f:
        nlp_data = json.load(f)
    events = nlp_data.get("events", [])
    logger.info(f"Loaded {len(events)} events")

    # Process events
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        events_semantics = process_all_events(events, entity_lookup, client)

    logger.info(f"Processed {len(events_semantics)} events, Tokens: {total_tokens}")

    # Compute statistics
    stats = compute_statistics(events_semantics)

    # Create output
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_folder = SCRIPT_OUTPUT_DIR / timestamp
    output_folder.mkdir(parents=True, exist_ok=True)
    now_iso = datetime.now(timezone.utc).isoformat()

    # Save event semantics
    output_data = {
        "_meta": {
            "description": "Structured semantic parsing of events",
            "created_at": now_iso,
            "model": LLM_MODEL,
            "source_nlp": str(nlp_file),
            "source_entities": str(entities_file) if entities_file else None,
        },
        "events": events_semantics,
    }

    output_file = output_folder / "event_semantics.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: {output_file}")

    # Save summary
    end_time = datetime.now(timezone.utc)
    summary = {
        "run_info": {
            "started_at": start_time.isoformat(),
            "completed_at": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "input_nlp_file": str(nlp_file),
            "input_entities_file": str(entities_file) if entities_file else None,
            "output_folder": str(output_folder),
            "model": LLM_MODEL,
            "total_tokens": total_tokens,
        },
        "configuration": {
            "batch_size": BATCH_SIZE,
            "event_types": EVENT_TYPES,
            "polarities": POLARITIES,
            "outcome_states": OUTCOME_STATES,
        },
        "statistics": stats,
    }

    summary_file = output_folder / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: {summary_file}")

    # Log summary
    logger.info("=" * 60)
    logger.info(f"Total events: {stats['total_events']}")
    logger.info(f"By event type: {stats['by_event_type']}")
    logger.info(f"By polarity: {stats['by_polarity']}")
    logger.info(f"Events with conditions: {stats['events_with_conditions']}")
    logger.info(f"Events with outcome states: {stats['events_with_outcome_states']}")
    logger.info(
        f"Duration: {(end_time - start_time).total_seconds():.1f}s | Output: {output_folder}"
    )


if __name__ == "__main__":
    main()
