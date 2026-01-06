"""
Prepare NLP-ready data from fetched Polymarket events.

Extracts text fields (titles, descriptions, market questions) from events,
filtering placeholder entries ("Person A/B/C") and applying data quality filters.
Produces a simplified JSON structure optimized for downstream NLP processing.

Pipeline Position: 01_fetch_events → 02_prepare_nlp_data → 03_1_extract_entities

Input:
    From: data/01_fetch_events/<timestamp>/
    Files:
        - events.json: Raw events with nested markets from API

Output:
    To: data/02_prepare_nlp_data/<timestamp>/
    Files:
        - nlp_events.json: Simplified events structure containing:
          {id, title, description, endDate, tags, markets[{id, question, outcomes}]}

Runtime: <1 minute (pure data transformation, no external calls)

Configuration:
    - FILTER_PLACEHOLDERS: Remove "Person A/B/C" placeholder markets (True)
    - FILTER_EMPTY_MARKETS: Remove events with no markets (True)
    - FILTER_EMPTY_TITLES: Remove events with empty/missing titles (True)
    - FILTER_EMPTY_DESCRIPTIONS: Remove events with empty/missing descriptions (True)
    - EVENT_FIELDS: Fields to extract from events
    - MARKET_FIELDS: Fields to extract from markets
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_SCRIPT_DIR = DATA_DIR / "01_fetch_events"
INPUT_RUN_FOLDER: str | None = None  # Specific folder or None for latest
SCRIPT_OUTPUT_DIR = DATA_DIR / "02_prepare_nlp_data"
OUTPUT_FILENAME = "nlp_events.json"

# Filter placeholder markets like "Person J" (Polymarket placeholders for late entries)
FILTER_PLACEHOLDERS = True
PLACEHOLDER_PATTERN = re.compile(r"^Person [A-Z]$")

# Data quality filters (filter out incomplete/rubbish events)
FILTER_EMPTY_MARKETS = True  # Events with no markets
FILTER_EMPTY_TITLES = True  # Events with empty/missing title
FILTER_EMPTY_DESCRIPTIONS = True  # Events with empty/missing description

# Fields to extract
EVENT_FIELDS = ["id", "title", "description", "endDate"]
MARKET_FIELDS = ["id", "question", "outcomes", "active", "closed", "endDate"]
INCLUDE_TAGS = True
INCLUDE_STATS = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# DATA PROCESSING
# =============================================================================


def get_input_folder() -> Path | None:
    """Get input folder (specified or latest)."""
    if INPUT_RUN_FOLDER:
        return INPUT_SCRIPT_DIR / INPUT_RUN_FOLDER
    if not INPUT_SCRIPT_DIR.exists():
        return None
    folders = [f for f in INPUT_SCRIPT_DIR.iterdir() if f.is_dir()]
    return max(folders, key=lambda f: f.stat().st_mtime) if folders else None


def is_placeholder(market: dict[str, Any]) -> bool:
    """Check if market is a placeholder."""
    name = market.get("groupItemTitle") or market.get("outcome") or ""
    return bool(PLACEHOLDER_PATTERN.match(str(name)))


def process_market(market: dict[str, Any]) -> dict[str, Any]:
    """Extract fields from market, parsing outcomes JSON if needed."""
    result = {f: market.get(f) for f in MARKET_FIELDS}
    if isinstance(result.get("outcomes"), str):
        try:
            result["outcomes"] = json.loads(result["outcomes"])
        except json.JSONDecodeError:
            pass
    return result


def process_event(event: dict[str, Any]) -> dict[str, Any]:
    """Extract fields from event and its markets."""
    result = {f: event.get(f) for f in EVENT_FIELDS}
    if INCLUDE_TAGS and "tags" in event:
        result["tags"] = [t.get("label") or t.get("slug") for t in event["tags"]]

    markets = event.get("markets", [])
    if FILTER_PLACEHOLDERS:
        markets = [m for m in markets if not is_placeholder(m)]

    result["markets"] = [process_market(m) for m in markets]

    if INCLUDE_STATS:
        result["_stats"] = {
            "market_count": len(markets),
            "active_markets": sum(1 for m in markets if m.get("active")),
            "closed_markets": sum(1 for m in markets if m.get("closed")),
        }
    return result


def compute_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary statistics."""
    all_markets = [m for e in events for m in e.get("markets", [])]
    return {
        "total_events": len(events),
        "total_markets": len(all_markets),
        "active_events": sum(1 for e in events if e.get("active")),
        "closed_events": sum(1 for e in events if e.get("closed")),
        "active_markets": sum(1 for m in all_markets if m.get("active")),
        "closed_markets": sum(1 for m in all_markets if m.get("closed")),
        "events_with_multiple_markets": sum(
            1 for e in events if len(e.get("markets", [])) > 1
        ),
        "fields_extracted": {
            "event": EVENT_FIELDS + (["tags"] if INCLUDE_TAGS else []),
            "market": MARKET_FIELDS,
        },
    }


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Main entry point."""
    logger.info("Polymarket NLP Data Preparation")

    # Find input
    input_folder = get_input_folder()
    if not input_folder or not input_folder.exists():
        logger.error(f"Input folder not found: {input_folder}")
        return

    events_file = input_folder / "events.json"
    if not events_file.exists():
        logger.error(f"Events file not found: {events_file}")
        return

    logger.info(f"Loading from: {events_file}")
    with open(events_file, encoding="utf-8") as f:
        events = json.load(f)
    logger.info(f"Loaded {len(events)} events")

    # Process
    processed = [process_event(e) for e in events]

    # Apply data quality filters
    initial_count = len(processed)
    filter_stats = {"empty_markets": 0, "empty_titles": 0, "empty_descriptions": 0}

    if FILTER_EMPTY_MARKETS:
        before = len(processed)
        processed = [e for e in processed if e["markets"]]
        filter_stats["empty_markets"] = before - len(processed)

    if FILTER_EMPTY_TITLES:
        before = len(processed)
        processed = [e for e in processed if e.get("title") and str(e["title"]).strip()]
        filter_stats["empty_titles"] = before - len(processed)

    if FILTER_EMPTY_DESCRIPTIONS:
        before = len(processed)
        processed = [
            e
            for e in processed
            if e.get("description") and str(e["description"]).strip()
        ]
        filter_stats["empty_descriptions"] = before - len(processed)

    filtered_total = initial_count - len(processed)
    logger.info(
        f"Filtered {filtered_total} events: "
        f"{filter_stats['empty_markets']} no markets, "
        f"{filter_stats['empty_titles']} empty titles, "
        f"{filter_stats['empty_descriptions']} empty descriptions"
    )

    # Build output
    summary = compute_summary(events)
    summary["quality_filtering"] = {
        "enabled": {
            "empty_markets": FILTER_EMPTY_MARKETS,
            "empty_titles": FILTER_EMPTY_TITLES,
            "empty_descriptions": FILTER_EMPTY_DESCRIPTIONS,
            "placeholders": FILTER_PLACEHOLDERS,
        },
        "filtered_counts": filter_stats,
        "events_before_filtering": initial_count,
        "events_after_filtering": len(processed),
    }

    # Track input source explicitly for pipeline traceability
    input_run_timestamp = input_folder.name
    run_timestamp = datetime.now(timezone.utc).isoformat()

    output = {
        "_meta": {
            "source": str(events_file),
            "input_run": input_run_timestamp,
            "created_at": run_timestamp,
            "description": "NLP-ready Polymarket events data",
            "placeholder_filtering": FILTER_PLACEHOLDERS,
        },
        "_summary": summary,
        "events": processed,
    }

    # Save
    output_folder = SCRIPT_OUTPUT_DIR / input_run_timestamp
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / OUTPUT_FILENAME
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Save separate summary.json for pipeline traceability
    summary_output = {
        "script": "02_prepare_nlp_data",
        "run_at": run_timestamp,
        "input": {
            "script": "01_fetch_events",
            "run": input_run_timestamp,
            "file": str(events_file),
            "events_count": len(events),
        },
        "output": {
            "folder": str(output_folder),
            "file": str(output_file),
            "events_count": len(processed),
        },
        "processing": summary,
    }
    summary_file = output_folder / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_output, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(processed)} events to: {output_file}")
    logger.info(f"Markets: {summary['total_markets']} total")


if __name__ == "__main__":
    main()
