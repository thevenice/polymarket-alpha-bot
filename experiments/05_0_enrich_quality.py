"""
Enrich events with quality flags and detect negation pairs for blocking.

Pipeline: 02_prepare_nlp_data + 04_2_embed_events -> [05_0_enrich_quality] -> 05_1_block_candidate_pairs

This script:
1. Loads events from 02_prepare_nlp_data
2. Loads entity sets from 04_2_embed_events
3. Detects negation tokens in titles (win/lose, pass/fail, etc.)
4. Assigns quality flags (META_MARKET, NO_ENTITIES, EXPIRED)
5. Normalizes titles for concept matching
6. Pre-identifies MUTUALLY_EXCLUSIVE pairs via negation detection
7. Outputs enriched events ready for blocking stage
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "data"

# Inputs
INPUT_EVENTS_DIR = DATA_DIR / "02_prepare_nlp_data"
INPUT_ENTITY_SETS_DIR = DATA_DIR / "04_2_embed_events"
INPUT_RUN_FOLDER: str | None = None  # None = use latest

# Output
SCRIPT_OUTPUT_DIR = DATA_DIR / "05_0_enrich_quality"

# Negation detection patterns
# Pairs of (positive, negative) words that indicate mutual exclusivity
NEGATION_PAIRS = [
    ("win", "lose"),
    ("wins", "loses"),
    ("winning", "losing"),
    ("won", "lost"),
    ("pass", "fail"),
    ("passes", "fails"),
    ("passing", "failing"),
    ("passed", "failed"),
    ("rise", "fall"),
    ("rises", "falls"),
    ("rising", "falling"),
    ("rose", "fell"),
    ("above", "below"),
    ("over", "under"),
    ("more", "less"),
    ("increase", "decrease"),
    ("increases", "decreases"),
    ("increasing", "decreasing"),
    ("increased", "decreased"),
    ("gain", "drop"),
    ("gains", "drops"),
    ("up", "down"),
    ("yes", "no"),
    ("will", "won't"),
    ("can", "can't"),
    ("join", "leave"),
    ("joins", "leaves"),
    ("approve", "reject"),
    ("approves", "rejects"),
    ("approved", "rejected"),
    ("accept", "deny"),
    ("accepts", "denies"),
    ("accepted", "denied"),
    ("positive", "negative"),
    ("higher", "lower"),
    ("before", "after"),
    ("start", "end"),
    ("starts", "ends"),
    ("begin", "finish"),
    ("begins", "finishes"),
    ("enter", "exit"),
    ("enters", "exits"),
    ("support", "oppose"),
    ("supports", "opposes"),
    ("supporting", "opposing"),
]

# Quality flags
META_MARKET_KEYWORDS = ["polymarket", "this market", "resolution", "resolve to"]
MIN_DESCRIPTION_LENGTH = 20

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
class QualityEnrichedEvent:
    """Event enriched with quality information."""

    id: str
    title: str
    description: str
    end_date: str
    entities: list[str] = field(default_factory=list)
    quality_flags: list[str] = field(default_factory=list)
    negation_tokens: list[str] = field(default_factory=list)
    title_normalized: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "end_date": self.end_date,
            "entities": self.entities,
            "quality_flags": self.quality_flags,
            "negation_tokens": self.negation_tokens,
            "title_normalized": self.title_normalized,
        }


@dataclass
class NegationPair:
    """Pre-identified mutually exclusive pair based on negation."""

    event_id_a: str
    event_id_b: str
    title_a: str
    title_b: str
    negation_a: str
    negation_b: str
    shared_base: str  # The base concept they share

    def to_dict(self) -> dict:
        return {
            "event_id_a": self.event_id_a,
            "event_id_b": self.event_id_b,
            "title_a": self.title_a,
            "title_b": self.title_b,
            "negation_a": self.negation_a,
            "negation_b": self.negation_b,
            "shared_base": self.shared_base,
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


def detect_negation_tokens(title: str) -> list[str]:
    """Detect negation-indicating words in a title."""
    title_lower = title.lower()
    words = set(re.findall(r"\b\w+\b", title_lower))

    detected = []
    for pos, neg in NEGATION_PAIRS:
        if pos in words:
            detected.append(pos)
        if neg in words:
            detected.append(neg)

    return detected


def normalize_title_for_concept(title: str) -> str:
    """
    Normalize title by removing dates, numbers, thresholds.

    This helps identify events about the same concept with different parameters.
    """
    text = title.lower()

    # Remove date ranges like "December 16 - December 23"
    months = (
        r"(?:january|february|march|april|may|june|july|august|september|october|"
        r"november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
    )
    text = re.sub(
        rf"{months}\s*\d{{1,2}}\s*[-–—to]+\s*{months}\s*\d{{1,2}}",
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

    # Remove years (4-digit numbers starting with 19 or 20)
    text = re.sub(r"\b(?:19|20)\d{2}\b", "", text)

    # Remove numeric ranges like "420-439", "0-19"
    text = re.sub(r"\b\d+\s*[-–—]\s*\d+\b", "", text)

    # Remove dollar amounts like "$100k", "$1M", "$50B"
    text = re.sub(r"\$\s*\d+\.?\d*\s*[kKmMbBtT]?\b", "", text)

    # Remove percentages like "50%", "5.5%"
    text = re.sub(r"\d+\.?\d*\s*%", "", text)

    # Remove standalone numbers
    text = re.sub(r"\b\d+\.?\d*\b", "", text)

    # Remove ordinals like "1st", "2nd", "3rd", "4th"
    text = re.sub(r"\b\d+(?:st|nd|rd|th)\b", "", text, flags=re.IGNORECASE)

    # Clean up multiple spaces and punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def compute_quality_flags(event: dict, entities: list[str]) -> list[str]:
    """
    Compute quality flags for an event.

    Flags:
    - META_MARKET: References Polymarket or resolution mechanics
    - NO_ENTITIES: No entities extracted
    - SHORT_DESCRIPTION: Very short description
    - EXPIRED: End date in the past
    """
    flags = []

    # Check for meta-market
    title_desc = (event.get("title", "") + " " + event.get("description", "")).lower()
    if any(kw in title_desc for kw in META_MARKET_KEYWORDS):
        flags.append("META_MARKET")

    # Check for no entities
    if not entities:
        flags.append("NO_ENTITIES")

    # Check for short description
    description = event.get("description", "")
    if len(description) < MIN_DESCRIPTION_LENGTH:
        flags.append("SHORT_DESCRIPTION")

    # Check if expired
    end_date_str = event.get("endDate", "")
    if end_date_str:
        try:
            end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
            if end_date < datetime.now(timezone.utc):
                flags.append("EXPIRED")
        except (ValueError, TypeError):
            pass

    return flags


def find_negation_pairs(events: list[QualityEnrichedEvent]) -> list[NegationPair]:
    """
    Find pairs of events that are likely MUTUALLY_EXCLUSIVE based on negation.

    Looks for events with:
    1. High title similarity (after normalization)
    2. One has a "positive" negation token, other has corresponding "negative"
    """
    # Build lookup from normalized title to events
    title_to_events: dict[str, list[QualityEnrichedEvent]] = {}
    for event in events:
        if event.title_normalized:
            key = event.title_normalized
            if key not in title_to_events:
                title_to_events[key] = []
            title_to_events[key].append(event)

    # Build negation token to its pair
    neg_to_pair: dict[str, str] = {}
    for pos, neg in NEGATION_PAIRS:
        neg_to_pair[pos] = neg
        neg_to_pair[neg] = pos

    negation_pairs = []

    # For each group of events with same normalized title
    for norm_title, group_events in title_to_events.items():
        if len(group_events) < 2:
            continue

        # Check all pairs within this group
        for i, event_a in enumerate(group_events):
            for event_b in group_events[i + 1 :]:
                # Check if they have complementary negation tokens
                tokens_a = set(event_a.negation_tokens)
                tokens_b = set(event_b.negation_tokens)

                for token_a in tokens_a:
                    complement = neg_to_pair.get(token_a)
                    if complement and complement in tokens_b:
                        negation_pairs.append(
                            NegationPair(
                                event_id_a=event_a.id,
                                event_id_b=event_b.id,
                                title_a=event_a.title,
                                title_b=event_b.title,
                                negation_a=token_a,
                                negation_b=complement,
                                shared_base=norm_title,
                            )
                        )
                        break

    return negation_pairs


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Main entry point."""
    start_time = datetime.now(timezone.utc)
    logger.info("Starting 05_0_enrich_quality")

    # Determine input folders
    if INPUT_RUN_FOLDER:
        events_folder = INPUT_EVENTS_DIR / INPUT_RUN_FOLDER
        entity_sets_folder = INPUT_ENTITY_SETS_DIR / INPUT_RUN_FOLDER
    else:
        events_folder = find_latest_run_folder(INPUT_EVENTS_DIR)
        entity_sets_folder = find_latest_run_folder(INPUT_ENTITY_SETS_DIR)

    if not events_folder or not events_folder.exists():
        logger.error(f"Events folder not found: {events_folder}")
        return
    if not entity_sets_folder or not entity_sets_folder.exists():
        logger.error(f"Entity sets folder not found: {entity_sets_folder}")
        return

    logger.info(f"Loading events from: {events_folder}")
    logger.info(f"Loading entity sets from: {entity_sets_folder}")

    # Load events
    events_file = events_folder / "nlp_events.json"
    with open(events_file, encoding="utf-8") as f:
        events_data = json.load(f)
    events_list = events_data.get("events", [])
    logger.info(f"Loaded {len(events_list)} events")

    # Load entity sets
    entity_sets_file = entity_sets_folder / "entity_sets.json"
    with open(entity_sets_file, encoding="utf-8") as f:
        entity_sets_data = json.load(f)
    entity_sets = entity_sets_data.get("entity_sets", {})
    logger.info(f"Loaded entity sets for {len(entity_sets)} events")

    # Process events
    enriched_events: list[QualityEnrichedEvent] = []
    stats = {
        "total_events": len(events_list),
        "events_with_negation": 0,
        "events_with_flags": 0,
        "flag_counts": {},
    }

    for event in events_list:
        event_id = str(event.get("id", ""))
        title = event.get("title", "")
        description = event.get("description", "")
        end_date = event.get("endDate", "")
        entities = entity_sets.get(event_id, [])

        # Detect negation tokens
        negation_tokens = detect_negation_tokens(title)
        if negation_tokens:
            stats["events_with_negation"] += 1

        # Normalize title
        title_normalized = normalize_title_for_concept(title)

        # Compute quality flags
        quality_flags = compute_quality_flags(event, entities)
        if quality_flags:
            stats["events_with_flags"] += 1
            for flag in quality_flags:
                stats["flag_counts"][flag] = stats["flag_counts"].get(flag, 0) + 1

        enriched_events.append(
            QualityEnrichedEvent(
                id=event_id,
                title=title,
                description=description,
                end_date=end_date,
                entities=entities,
                quality_flags=quality_flags,
                negation_tokens=negation_tokens,
                title_normalized=title_normalized,
            )
        )

    logger.info(f"Processed {len(enriched_events)} events")
    logger.info(f"Events with negation tokens: {stats['events_with_negation']}")
    logger.info(f"Events with quality flags: {stats['events_with_flags']}")

    # Find negation pairs
    negation_pairs = find_negation_pairs(enriched_events)
    logger.info(f"Found {len(negation_pairs)} potential MUTUALLY_EXCLUSIVE pairs")
    stats["negation_pairs_found"] = len(negation_pairs)

    # Create output folder
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    output_folder = SCRIPT_OUTPUT_DIR / timestamp
    output_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output folder: {output_folder}")

    # Save enriched events
    enriched_output = {
        "_meta": {
            "description": "Quality-enriched events for relations pipeline",
            "created_at": start_time.isoformat(),
            "source_events": str(events_file),
            "source_entity_sets": str(entity_sets_file),
        },
        "events": [e.to_dict() for e in enriched_events],
    }
    with open(
        output_folder / "quality_enriched_events.json", "w", encoding="utf-8"
    ) as f:
        json.dump(enriched_output, f, indent=2, ensure_ascii=False)
    logger.info("Saved quality_enriched_events.json")

    # Save negation pairs
    negation_output = {
        "_meta": {
            "description": "Pre-identified MUTUALLY_EXCLUSIVE candidates via negation",
            "created_at": start_time.isoformat(),
        },
        "pairs": [p.to_dict() for p in negation_pairs],
    }
    with open(output_folder / "negation_pairs.json", "w", encoding="utf-8") as f:
        json.dump(negation_output, f, indent=2, ensure_ascii=False)
    logger.info("Saved negation_pairs.json")

    # Save summary
    end_time = datetime.now(timezone.utc)
    summary = {
        "run_info": {
            "started_at": start_time.isoformat(),
            "completed_at": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "input_events_folder": str(events_folder),
            "input_entity_sets_folder": str(entity_sets_folder),
            "output_folder": str(output_folder),
        },
        "configuration": {
            "negation_pairs_count": len(NEGATION_PAIRS),
            "meta_market_keywords": META_MARKET_KEYWORDS,
            "min_description_length": MIN_DESCRIPTION_LENGTH,
        },
        "statistics": stats,
    }
    with open(output_folder / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Saved summary.json")

    logger.info(f"Completed in {summary['run_info']['duration_seconds']:.2f}s")


if __name__ == "__main__":
    main()
