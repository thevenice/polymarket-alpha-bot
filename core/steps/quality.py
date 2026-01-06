"""
Quality enrichment for events.

Extracted from experiments/05_0_enrich_quality.py for production pipeline.

Features:
- Detects negation tokens in titles (win/lose, pass/fail, etc.)
- Assigns quality flags (META_MARKET, NO_ENTITIES, EXPIRED)
- Normalizes titles for concept matching
- Pre-identifies MUTUALLY_EXCLUSIVE pairs via negation detection
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from loguru import logger

# =============================================================================
# CONFIGURATION
# =============================================================================

# Negation detection patterns - pairs of (positive, negative) words
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

META_MARKET_KEYWORDS = ["polymarket", "this market", "resolution", "resolve to"]
MIN_DESCRIPTION_LENGTH = 20


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
    shared_base: str

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
    Helps identify events about the same concept with different parameters.
    """
    text = title.lower()

    # Remove date ranges
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
    # Single date
    text = re.sub(
        rf"{months}\s*\d{{1,2}}|\d{{1,2}}\s*{months}",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Month + year
    text = re.sub(
        rf"{months}\s*,?\s*\d{{4}}|q[1-4]\s*\d{{4}}",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Just month names
    text = re.sub(rf"\b{months}\b", "", text, flags=re.IGNORECASE)

    # Remove years
    text = re.sub(r"\b(?:19|20)\d{2}\b", "", text)

    # Remove numeric ranges
    text = re.sub(r"\b\d+\s*[-–—]\s*\d+\b", "", text)

    # Remove dollar amounts
    text = re.sub(r"\$\s*\d+\.?\d*\s*[kKmMbBtT]?\b", "", text)

    # Remove percentages
    text = re.sub(r"\d+\.?\d*\s*%", "", text)

    # Remove standalone numbers
    text = re.sub(r"\b\d+\.?\d*\b", "", text)

    # Remove ordinals
    text = re.sub(r"\b\d+(?:st|nd|rd|th)\b", "", text, flags=re.IGNORECASE)

    # Clean up
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def compute_quality_flags(event: dict, entities: list[str]) -> list[str]:
    """Compute quality flags for an event."""
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
    """Find pairs of events that are likely MUTUALLY_EXCLUSIVE based on negation."""
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

    for norm_title, group_events in title_to_events.items():
        if len(group_events) < 2:
            continue

        for i, event_a in enumerate(group_events):
            for event_b in group_events[i + 1 :]:
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
# MAIN FUNCTIONS
# =============================================================================


def enrich_events_quality(
    events: list[dict],
    entity_sets: dict[str, list[str]],
) -> tuple[list[QualityEnrichedEvent], list[NegationPair]]:
    """
    Enrich events with quality information.

    Args:
        events: NLP-prepared events
        entity_sets: Dict mapping event_id to list of entity names

    Returns:
        Tuple of (enriched_events, negation_pairs)
    """
    enriched_events: list[QualityEnrichedEvent] = []

    stats = {
        "events_with_negation": 0,
        "events_with_flags": 0,
        "flag_counts": {},
    }

    for event in events:
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

    # Find negation pairs
    negation_pairs = find_negation_pairs(enriched_events)

    logger.info(f"Quality enrichment: {len(enriched_events)} events")
    logger.info(f"  - Events with negation tokens: {stats['events_with_negation']}")
    logger.info(f"  - Negation pairs found: {len(negation_pairs)}")
    if stats["flag_counts"]:
        logger.info(f"  - Quality flags: {stats['flag_counts']}")

    return enriched_events, negation_pairs


def get_negation_pair_set(negation_pairs: list[NegationPair]) -> set[tuple[str, str]]:
    """Convert negation pairs to a set of (id_a, id_b) tuples for fast lookup."""
    result = set()
    for pair in negation_pairs:
        a, b = sorted([pair.event_id_a, pair.event_id_b])
        result.add((a, b))
    return result
