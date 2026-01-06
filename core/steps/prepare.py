"""
Prepare NLP-ready data from events.

Extracted from experiments/02_prepare_nlp_data.py for production pipeline.
"""

import json
import re
from typing import Any

from loguru import logger

# =============================================================================
# CONFIGURATION
# =============================================================================

# Filter placeholder markets like "Person J"
FILTER_PLACEHOLDERS = True
PLACEHOLDER_PATTERN = re.compile(r"^Person [A-Z]$")

# Data quality filters
FILTER_EMPTY_MARKETS = True
FILTER_EMPTY_TITLES = True
FILTER_EMPTY_DESCRIPTIONS = True

# Fields to extract
EVENT_FIELDS = ["id", "title", "description", "endDate"]
MARKET_FIELDS = ["id", "question", "outcomes", "active", "closed", "endDate"]


# =============================================================================
# PROCESSING FUNCTIONS
# =============================================================================


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

    # Include tags
    if "tags" in event:
        result["tags"] = [t.get("label") or t.get("slug") for t in event["tags"]]

    # Filter and process markets
    markets = event.get("markets", [])
    if FILTER_PLACEHOLDERS:
        markets = [m for m in markets if not is_placeholder(m)]

    result["markets"] = [process_market(m) for m in markets]

    # Add stats
    result["_stats"] = {
        "market_count": len(result["markets"]),
        "active_markets": sum(1 for m in result["markets"] if m.get("active")),
    }

    return result


# =============================================================================
# MAIN FUNCTION
# =============================================================================


def prepare_nlp_data(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Prepare events for NLP processing.

    - Extract relevant fields
    - Filter placeholder markets
    - Apply quality filters

    Args:
        events: Raw events from API

    Returns:
        Filtered and processed events ready for NLP
    """
    logger.info(f"Preparing NLP data from {len(events)} events...")

    # Process all events
    processed = [process_event(e) for e in events]
    initial_count = len(processed)

    # Apply quality filters
    filter_stats = {
        "empty_markets": 0,
        "empty_titles": 0,
        "empty_descriptions": 0,
    }

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

    logger.info(f"Prepared {len(processed)} events for NLP")
    return processed


# =============================================================================
# TEXT EXTRACTION
# =============================================================================


def extract_texts_for_ner(events: list[dict[str, Any]]) -> list[dict[str, str]]:
    """
    Extract text fields for NER processing.

    Returns list of dicts with:
    - event_id
    - source_type: 'title', 'description', 'market_question'
    - text: The actual text to process
    """
    texts = []

    for event in events:
        event_id = event["id"]

        # Event title
        if event.get("title"):
            texts.append(
                {
                    "event_id": event_id,
                    "source_type": "title",
                    "text": event["title"],
                }
            )

        # Event description
        if event.get("description"):
            texts.append(
                {
                    "event_id": event_id,
                    "source_type": "description",
                    "text": event["description"],
                }
            )

        # Market questions
        for market in event.get("markets", []):
            if market.get("question"):
                texts.append(
                    {
                        "event_id": event_id,
                        "market_id": market["id"],
                        "source_type": "market_question",
                        "text": market["question"],
                    }
                )

    logger.debug(f"Extracted {len(texts)} text segments for NER")
    return texts
