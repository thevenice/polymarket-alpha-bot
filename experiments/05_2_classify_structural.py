"""
Rule-based structural relation classification for event pairs.

This script applies deterministic rules to classify structural relations
between events. NO LLM calls - pure rule-based classification.

Detects:
- THRESHOLD_VARIANT: Same concept, different numeric threshold
- TIMEFRAME_VARIANT: Same concept, different deadline
- HIERARCHICAL: One event contains/implies another
- SERIES_MEMBER: Same series, different index
- MUTUALLY_EXCLUSIVE: Cannot both be true

Input:
- data/05_1_block_candidate_pairs/<latest>/candidate_pairs.json
- data/04_1_extract_event_semantics/<latest>/event_semantics.json

Output:
- data/05_2_classify_structural/<timestamp>/structural_relations.json
- data/05_2_classify_structural/<timestamp>/summary.json

Pipeline: 05_1_block_candidate_pairs -> [05_2_classify_structural] -> 05_3_classify_causal
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
INPUT_PAIRS_DIR = DATA_DIR / "05_1_block_candidate_pairs"
INPUT_SEMANTICS_DIR = DATA_DIR / "04_1_extract_event_semantics"
INPUT_QUALITY_DIR = DATA_DIR / "05_0_enrich_quality"
INPUT_RUN_FOLDER: str | None = None  # None = use latest

# Output
SCRIPT_OUTPUT_DIR = DATA_DIR / "05_2_classify_structural"

# Logging
LOG_LEVEL = logging.INFO

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# PATTERNS FOR STRUCTURAL DETECTION
# =============================================================================

THRESHOLD_PATTERNS = [
    r"(\d+\.?\d*)\s*%",  # Percentages: 50%, 5.5%
    r"\$\s*(\d+\.?\d*)\s*([kKmMbBtT])?",  # Dollar amounts: $100k, $1M
    r"(\d+\.?\d*)\s*(million|billion|thousand|trillion)",  # Large numbers
    r"(above|below|over|under|more than|less than|at least|>\s*|<\s*)\s*(\d+)",  # Comparisons
]

TIMEFRAME_PATTERNS = [
    r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s*\d{4}",  # January 31, 2026
    r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}",  # January 2026
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{4}",
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",  # Month only
    r"\b(march|april|may|june)\b",  # Common month references
    r"q[1-4]\s*\d{4}",  # Q1 2025
    r"20[2-3]\d",  # Years 2020-2039
    r"(week|day|month)\s*\d+",  # Week 1, Day 5
    r"by\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}",  # by January 31
]

SERIES_PATTERNS = [
    r"(week|day|episode|part|round|phase)\s*(\d+)",
    r"#\s*(\d+)",  # Tweet #420
    r"\[(\d+)\s*[-–]\s*(\d+)\]",  # [420-439]
]

# Patterns for detecting district/state codes in series (TX-34, NY-19, etc.)
DISTRICT_PATTERN = r"([A-Z]{2})-(\d{1,2})"


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class StructuralRelation:
    """A rule-based structural relation between two events."""

    source_id: str
    target_id: str
    title_a: str
    title_b: str
    relation_type: str
    relation_level: int
    direction: str  # "forward", "reverse", "bidirectional", "none"
    confidence: float
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "title_a": self.title_a,
            "title_b": self.title_b,
            "relation_type": self.relation_type,
            "relation_level": self.relation_level,
            "direction": self.direction,
            "confidence": round(self.confidence, 4),
            "evidence": self.evidence,
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


def extract_thresholds(text: str) -> list[tuple[str, float]]:
    """Extract numeric thresholds from text."""
    thresholds = []
    text_lower = text.lower()

    for pattern in THRESHOLD_PATTERNS:
        for match in re.finditer(pattern, text_lower):
            try:
                groups = match.groups()
                value = float(groups[0].replace(",", ""))
                # Handle multipliers (k, m, b)
                if len(groups) > 1 and groups[1]:
                    multiplier = groups[1].lower()
                    if multiplier in ("k", "thousand"):
                        value *= 1000
                    elif multiplier in ("m", "million"):
                        value *= 1_000_000
                    elif multiplier in ("b", "billion"):
                        value *= 1_000_000_000
                thresholds.append((match.group(), value))
            except (ValueError, IndexError):
                continue

    return thresholds


def extract_timeframes(text: str) -> list[str]:
    """Extract timeframe references from text."""
    timeframes = []
    text_lower = text.lower()

    for pattern in TIMEFRAME_PATTERNS:
        for match in re.finditer(pattern, text_lower, re.IGNORECASE):
            timeframes.append(match.group().strip())

    return timeframes


def extract_series_indicators(text: str) -> list[tuple[str, int]]:
    """Extract series indicators (week 1, day 5, etc.)."""
    indicators = []
    text_lower = text.lower()

    for pattern in SERIES_PATTERNS:
        for match in re.finditer(pattern, text_lower):
            try:
                groups = match.groups()
                if len(groups) >= 2:
                    indicators.append((groups[0], int(groups[1])))
                elif len(groups) == 1:
                    indicators.append(("number", int(groups[0])))
            except (ValueError, IndexError):
                continue

    return indicators


def detect_series_member(title_a: str, title_b: str) -> tuple[bool, dict]:
    """
    Detect if two events are members of the same series.

    Series patterns (checked in order):
    1. District codes: "TX-34" vs "TX-35" (same state, different district)
    2. Sequential date ranges: "Dec 16-23" -> "Dec 23-30" (endpoints connect)
    3. Bracketed number ranges: "[420-439]" -> "[440-459]" (adjacent ranges)
    4. Explicit sequence markers: "Week 1" -> "Week 2"
    5. Quarterly sequences: "Q1 2025" -> "Q2 2025"

    Returns (is_series, evidence_dict)
    """
    title_a_lower = title_a.lower()
    title_b_lower = title_b.lower()

    # Pattern 0: District codes (TX-34 vs TX-35, NY-19 vs NY-22)
    districts_a = re.findall(DISTRICT_PATTERN, title_a)
    districts_b = re.findall(DISTRICT_PATTERN, title_b)

    if districts_a and districts_b:
        state_a, num_a = districts_a[0][0], int(districts_a[0][1])
        state_b, num_b = districts_b[0][0], int(districts_b[0][1])

        if state_a == state_b and num_a != num_b:
            # Same state, different district number - this is a series
            return True, {
                "pattern": "district_series",
                "state": state_a,
                "district_a": num_a,
                "district_b": num_b,
            }

    # Pattern 1: Sequential date ranges (endpoints connect)
    date_range_re = r"(\w+)\s+(\d{1,2})\s*[-–—]\s*(?:(\w+)\s+)?(\d{1,2})"
    ranges_a = re.findall(date_range_re, title_a_lower)
    ranges_b = re.findall(date_range_re, title_b_lower)

    if ranges_a and ranges_b:
        try:
            start_a, end_a = int(ranges_a[0][1]), int(ranges_a[0][3])
            start_b, end_b = int(ranges_b[0][1]), int(ranges_b[0][3])

            # Check if they're sequential (end of A == start of B or vice versa)
            if end_a == start_b or end_b == start_a:
                return True, {
                    "pattern": "sequential_date_range",
                    "range_a": f"{ranges_a[0][0]} {start_a}-{end_a}",
                    "range_b": f"{ranges_b[0][0]} {start_b}-{end_b}",
                    "connection": "endpoint_shared",
                }

            # Also check for adjacent ranges (end_a + 1 == start_b)
            if abs(end_a - start_b) <= 1 or abs(end_b - start_a) <= 1:
                return True, {
                    "pattern": "adjacent_date_range",
                    "range_a": f"{ranges_a[0][0]} {start_a}-{end_a}",
                    "range_b": f"{ranges_b[0][0]} {start_b}-{end_b}",
                    "connection": "adjacent",
                }
        except (ValueError, IndexError):
            pass

    # Pattern 2: Bracketed number ranges [N-M] -> [M+1-O] or [N-M] -> [M-O]
    bracket_re = r"\[(\d+)\s*[-–—]\s*(\d+)\]"
    brackets_a = re.findall(bracket_re, title_a)
    brackets_b = re.findall(bracket_re, title_b)

    if brackets_a and brackets_b:
        try:
            start_a, end_a = int(brackets_a[0][0]), int(brackets_a[0][1])
            start_b, end_b = int(brackets_b[0][0]), int(brackets_b[0][1])

            # Check for adjacent or overlapping ranges
            if abs(end_a - start_b) <= 1 or abs(end_b - start_a) <= 1:
                return True, {
                    "pattern": "bracketed_sequence",
                    "range_a": f"[{start_a}-{end_a}]",
                    "range_b": f"[{start_b}-{end_b}]",
                }
        except (ValueError, IndexError):
            pass

    # Pattern 3: Explicit sequence markers (week N, part N, etc.)
    seq_re = r"\b(week|day|part|round|phase|episode|chapter)\s*#?\s*(\d+)\b"
    seq_a = re.findall(seq_re, title_a_lower)
    seq_b = re.findall(seq_re, title_b_lower)

    if seq_a and seq_b:
        # Check if same series type with different numbers
        type_a, num_a = seq_a[0][0], int(seq_a[0][1])
        type_b, num_b = seq_b[0][0], int(seq_b[0][1])

        if type_a == type_b and num_a != num_b:
            return True, {
                "pattern": "explicit_sequence",
                "series_type": type_a,
                "position_a": num_a,
                "position_b": num_b,
            }

    # Pattern 4: Quarterly sequences (Q1 2025 -> Q2 2025)
    quarter_re = r"\b(q[1-4])\s*(\d{4})\b"
    quarters_a = re.findall(quarter_re, title_a_lower)
    quarters_b = re.findall(quarter_re, title_b_lower)

    if quarters_a and quarters_b:
        q_a, year_a = quarters_a[0]
        q_b, year_b = quarters_b[0]

        if year_a == year_b and q_a != q_b:
            return True, {
                "pattern": "quarterly_sequence",
                "quarter_a": f"{q_a.upper()} {year_a}",
                "quarter_b": f"{q_b.upper()} {year_b}",
            }

    return False, {}


def normalize_for_comparison(text: str) -> str:
    """Normalize text for comparison by removing dates, numbers, punctuation."""
    text = text.lower()
    # Remove dates
    months = r"(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
    text = re.sub(rf"{months}\s*\d{{0,2}},?\s*\d{{4}}", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b20[2-3]\d\b", "", text)
    # Remove numbers
    text = re.sub(r"\$?\d+\.?\d*[kKmMbB%]?", "", text)
    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =============================================================================
# STRUCTURAL CLASSIFICATION RULES
# =============================================================================


def classify_structural(
    pair: dict,
    semantics_by_id: dict,
    negation_pairs: dict[tuple[str, str], dict] | None = None,
) -> StructuralRelation | None:
    """
    Apply structural rules for deterministic classification.

    Returns StructuralRelation if a rule matches, None otherwise.
    Confidence is always 1.0 for rule-based classifications.
    """
    title_a = pair.get("title_a", "")
    title_b = pair.get("title_b", "")
    event_id_a = pair["event_id_a"]
    event_id_b = pair["event_id_b"]
    same_subject = pair.get("same_subject", False)
    embedding_similarity = pair.get("embedding_similarity", 0.0)

    # =========================================================================
    # RULE 0: MUTUALLY_EXCLUSIVE from pre-computed negation pairs (05_0)
    # =========================================================================
    if negation_pairs:
        pair_key = tuple(sorted([event_id_a, event_id_b]))
        if pair_key in negation_pairs:
            neg_info = negation_pairs[pair_key]
            return StructuralRelation(
                source_id=event_id_a,
                target_id=event_id_b,
                title_a=title_a,
                title_b=title_b,
                relation_type="MUTUALLY_EXCLUSIVE",
                relation_level=2,
                direction="bidirectional",
                confidence=1.0,
                evidence={
                    "rule": "negation_pair_precomputed",
                    "negation_a": neg_info.get("negation_a"),
                    "negation_b": neg_info.get("negation_b"),
                    "shared_base": neg_info.get("shared_base"),
                },
            )

    sem_a = semantics_by_id.get(event_id_a, {})
    sem_b = semantics_by_id.get(event_id_b, {})

    # Normalize titles for comparison
    norm_a = normalize_for_comparison(title_a)
    norm_b = normalize_for_comparison(title_b)

    # Extract parametric patterns
    thresholds_a = extract_thresholds(title_a)
    thresholds_b = extract_thresholds(title_b)
    timeframes_a = extract_timeframes(title_a)
    timeframes_b = extract_timeframes(title_b)
    series_a = extract_series_indicators(title_a)
    series_b = extract_series_indicators(title_b)

    # =========================================================================
    # RULE 1: SERIES_MEMBER (check first - handles sequential date ranges)
    # =========================================================================
    is_series, series_evidence = detect_series_member(title_a, title_b)
    if is_series:
        return StructuralRelation(
            source_id=event_id_a,
            target_id=event_id_b,
            title_a=title_a,
            title_b=title_b,
            relation_type="SERIES_MEMBER",
            relation_level=1,
            direction="none",
            confidence=1.0,
            evidence={"rule": "series_member_detection", **series_evidence},
        )

    # Fallback: old series detection for explicit markers (week N, part N)
    if series_a and series_b:
        types_a = {s[0] for s in series_a}
        types_b = {s[0] for s in series_b}
        if types_a & types_b:  # Shared series type
            nums_a = {s[1] for s in series_a}
            nums_b = {s[1] for s in series_b}
            if nums_a != nums_b:  # Different numbers
                return StructuralRelation(
                    source_id=event_id_a,
                    target_id=event_id_b,
                    title_a=title_a,
                    title_b=title_b,
                    relation_type="SERIES_MEMBER",
                    relation_level=1,
                    direction="none",
                    confidence=1.0,
                    evidence={
                        "rule": "series_member_explicit",
                        "series_a": [f"{s[0]}_{s[1]}" for s in series_a],
                        "series_b": [f"{s[0]}_{s[1]}" for s in series_b],
                    },
                )

    # SERIES_MEMBER: same_subject with very high similarity and different district codes
    # This catches cases like "TX-34 House Election" vs "TX-35 House Election" (different states)
    # or "NY-19 House Election" vs "NY-22 House Election"
    if same_subject and embedding_similarity >= 0.90:
        # Check for different state-district patterns in the same series
        district_pattern = r"([A-Z]{2})-(\d{1,2})"
        districts_a = re.findall(district_pattern, title_a)
        districts_b = re.findall(district_pattern, title_b)

        if districts_a and districts_b:
            # If they have district codes and they're different, it's a series
            if districts_a != districts_b:
                return StructuralRelation(
                    source_id=event_id_a,
                    target_id=event_id_b,
                    title_a=title_a,
                    title_b=title_b,
                    relation_type="SERIES_MEMBER",
                    relation_level=1,
                    direction="none",
                    confidence=1.0,
                    evidence={
                        "rule": "district_series_same_subject",
                        "district_a": f"{districts_a[0][0]}-{districts_a[0][1]}",
                        "district_b": f"{districts_b[0][0]}-{districts_b[0][1]}",
                        "embedding_similarity": embedding_similarity,
                    },
                )

    # SERIES_MEMBER: Election series (Governor, Senate, House elections)
    # Pairs like "Nebraska Governor Republican Primary" vs "New Hampshire Governor Republican Primary"
    if same_subject and embedding_similarity >= 0.80:
        election_keywords = [
            "governor",
            "senate",
            "house",
            "election",
            "primary",
            "winner",
        ]
        title_a_lower = title_a.lower()
        title_b_lower = title_b.lower()

        # Check if both are election-related
        has_election_a = any(kw in title_a_lower for kw in election_keywords)
        has_election_b = any(kw in title_b_lower for kw in election_keywords)

        if has_election_a and has_election_b:
            # Extract state names or abbreviations
            states_pattern = r"\b(alabama|alaska|arizona|arkansas|california|colorado|connecticut|delaware|florida|georgia|hawaii|idaho|illinois|indiana|iowa|kansas|kentucky|louisiana|maine|maryland|massachusetts|michigan|minnesota|mississippi|missouri|montana|nebraska|nevada|new hampshire|new jersey|new mexico|new york|north carolina|north dakota|ohio|oklahoma|oregon|pennsylvania|rhode island|south carolina|south dakota|tennessee|texas|utah|vermont|virginia|washington|west virginia|wisconsin|wyoming)\b"
            states_a = re.findall(states_pattern, title_a_lower)
            states_b = re.findall(states_pattern, title_b_lower)

            if states_a and states_b and states_a != states_b:
                return StructuralRelation(
                    source_id=event_id_a,
                    target_id=event_id_b,
                    title_a=title_a,
                    title_b=title_b,
                    relation_type="SERIES_MEMBER",
                    relation_level=1,
                    direction="none",
                    confidence=1.0,
                    evidence={
                        "rule": "election_series_different_states",
                        "state_a": states_a[0] if states_a else None,
                        "state_b": states_b[0] if states_b else None,
                        "embedding_similarity": embedding_similarity,
                    },
                )

    # =========================================================================
    # RULE 2: THRESHOLD_VARIANT (same concept, different thresholds)
    # =========================================================================
    # Use same_subject + high similarity as proxy for "same concept"
    is_same_concept = norm_a == norm_b or (
        same_subject and embedding_similarity >= 0.90
    )

    if thresholds_a and thresholds_b and is_same_concept:
        # Same concept with different thresholds
        return StructuralRelation(
            source_id=event_id_a,
            target_id=event_id_b,
            title_a=title_a,
            title_b=title_b,
            relation_type="THRESHOLD_VARIANT",
            relation_level=1,
            direction="bidirectional",
            confidence=1.0,
            evidence={
                "rule": "same_concept_different_threshold",
                "threshold_a": [t[1] for t in thresholds_a],
                "threshold_b": [t[1] for t in thresholds_b],
            },
        )

    # Semantics-based THRESHOLD_VARIANT: COUNT event paired with THRESHOLD event
    if sem_a and sem_b:
        sem_a_data = sem_a.get("semantics", {})
        sem_b_data = sem_b.get("semantics", {})
        type_a = sem_a_data.get("event_type")
        type_b = sem_b_data.get("event_type")
        subj_a = sem_a_data.get("subject_entity")
        subj_b = sem_b_data.get("subject_entity")
        pred_a = sem_a_data.get("predicate")
        pred_b = sem_b_data.get("predicate")

        # Two THRESHOLD events with same subject/predicate but different thresholds
        if type_a == "THRESHOLD" and type_b == "THRESHOLD":
            if subj_a == subj_b and pred_a == pred_b:
                cond_a = sem_a_data.get("condition", {})
                cond_b = sem_b_data.get("condition", {})
                val_a = cond_a.get("value") if cond_a else None
                val_b = cond_b.get("value") if cond_b else None
                if val_a is not None and val_b is not None and val_a != val_b:
                    return StructuralRelation(
                        source_id=event_id_a,
                        target_id=event_id_b,
                        title_a=title_a,
                        title_b=title_b,
                        relation_type="THRESHOLD_VARIANT",
                        relation_level=1,
                        direction="bidirectional",
                        confidence=1.0,
                        evidence={
                            "rule": "semantics_threshold_variant",
                            "threshold_a": val_a,
                            "threshold_b": val_b,
                            "subject": subj_a,
                            "predicate": pred_a,
                        },
                    )

    # =========================================================================
    # RULE 3: TIMEFRAME_VARIANT (same concept, different timeframe)
    # =========================================================================
    # Relaxed matching: same_subject + high similarity indicates same concept
    if timeframes_a and timeframes_b and is_same_concept:
        if set(timeframes_a) != set(timeframes_b):
            return StructuralRelation(
                source_id=event_id_a,
                target_id=event_id_b,
                title_a=title_a,
                title_b=title_b,
                relation_type="TIMEFRAME_VARIANT",
                relation_level=1,
                direction="bidirectional",
                confidence=1.0,
                evidence={
                    "rule": "same_concept_different_timeframe",
                    "timeframes_a": timeframes_a,
                    "timeframes_b": timeframes_b,
                },
            )

    # Additional TIMEFRAME_VARIANT: same_subject with different "by X" dates
    if same_subject and embedding_similarity >= 0.85:
        # Check for "by [date]" pattern differences
        by_date_pattern = r"by\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}"
        by_dates_a = re.findall(by_date_pattern, title_a.lower())
        by_dates_b = re.findall(by_date_pattern, title_b.lower())

        if by_dates_a and by_dates_b and by_dates_a != by_dates_b:
            return StructuralRelation(
                source_id=event_id_a,
                target_id=event_id_b,
                title_a=title_a,
                title_b=title_b,
                relation_type="TIMEFRAME_VARIANT",
                relation_level=1,
                direction="bidirectional",
                confidence=1.0,
                evidence={
                    "rule": "same_subject_different_deadline",
                    "deadline_a": by_dates_a,
                    "deadline_b": by_dates_b,
                    "embedding_similarity": embedding_similarity,
                },
            )

    # TIMEFRAME_VARIANT: "in [month]?" pattern (e.g., "Fed decision in March?" vs "in April?")
    if same_subject and embedding_similarity >= 0.90:
        in_month_pattern = r"\bin\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b"
        in_months_a = re.findall(in_month_pattern, title_a.lower())
        in_months_b = re.findall(in_month_pattern, title_b.lower())

        if in_months_a and in_months_b and in_months_a != in_months_b:
            return StructuralRelation(
                source_id=event_id_a,
                target_id=event_id_b,
                title_a=title_a,
                title_b=title_b,
                relation_type="TIMEFRAME_VARIANT",
                relation_level=1,
                direction="bidirectional",
                confidence=1.0,
                evidence={
                    "rule": "same_subject_different_month",
                    "month_a": in_months_a,
                    "month_b": in_months_b,
                    "embedding_similarity": embedding_similarity,
                },
            )

    # Semantics-based TIMEFRAME_VARIANT
    if sem_a and sem_b:
        sem_a_data = sem_a.get("semantics", {})
        sem_b_data = sem_b.get("semantics", {})
        subj_a = sem_a_data.get("subject_entity")
        subj_b = sem_b_data.get("subject_entity")
        pred_a = sem_a_data.get("predicate")
        pred_b = sem_b_data.get("predicate")
        tf_a = sem_a_data.get("timeframe", {})
        tf_b = sem_b_data.get("timeframe", {})

        if subj_a == subj_b and pred_a == pred_b and subj_a and pred_a:
            end_a = tf_a.get("end_date") if tf_a else None
            end_b = tf_b.get("end_date") if tf_b else None
            if end_a and end_b and end_a != end_b:
                return StructuralRelation(
                    source_id=event_id_a,
                    target_id=event_id_b,
                    title_a=title_a,
                    title_b=title_b,
                    relation_type="TIMEFRAME_VARIANT",
                    relation_level=1,
                    direction="bidirectional",
                    confidence=1.0,
                    evidence={
                        "rule": "semantics_timeframe_variant",
                        "end_date_a": end_a,
                        "end_date_b": end_b,
                        "subject": subj_a,
                        "predicate": pred_a,
                    },
                )

    # =========================================================================
    # RULE 4: HIERARCHICAL (COUNT contains THRESHOLD, or entity subset)
    # =========================================================================
    if sem_a and sem_b:
        sem_a_data = sem_a.get("semantics", {})
        sem_b_data = sem_b.get("semantics", {})
        type_a = sem_a_data.get("event_type")
        type_b = sem_b_data.get("event_type")
        subj_a = sem_a_data.get("subject_entity")
        subj_b = sem_b_data.get("subject_entity")
        pred_a = sem_a_data.get("predicate")
        pred_b = sem_b_data.get("predicate")

        # COUNT event encompasses THRESHOLD event
        if type_a == "COUNT" and type_b == "THRESHOLD":
            if subj_a == subj_b and pred_a == pred_b:
                return StructuralRelation(
                    source_id=event_id_a,
                    target_id=event_id_b,
                    title_a=title_a,
                    title_b=title_b,
                    relation_type="HIERARCHICAL",
                    relation_level=1,
                    direction="forward",  # COUNT -> THRESHOLD
                    confidence=1.0,
                    evidence={
                        "rule": "count_question_contains_threshold",
                        "subject": subj_a,
                        "predicate": pred_a,
                        "threshold_value": (sem_b_data.get("condition") or {}).get(
                            "value"
                        ),
                    },
                )
        elif type_b == "COUNT" and type_a == "THRESHOLD":
            if subj_a == subj_b and pred_a == pred_b:
                return StructuralRelation(
                    source_id=event_id_a,
                    target_id=event_id_b,
                    title_a=title_a,
                    title_b=title_b,
                    relation_type="HIERARCHICAL",
                    relation_level=1,
                    direction="reverse",  # THRESHOLD <- COUNT
                    confidence=1.0,
                    evidence={
                        "rule": "count_question_contains_threshold",
                        "subject": subj_b,
                        "predicate": pred_b,
                        "threshold_value": (sem_a_data.get("condition") or {}).get(
                            "value"
                        ),
                    },
                )

    # Entity-based HIERARCHICAL: one event's entities are subset of the other
    shared_entities = pair.get("shared_entities", [])
    if shared_entities:
        # If we have shared entities but one title is more specific
        # (contains more entity mentions), it's hierarchical
        entities_in_a = set(shared_entities)
        entities_in_b = set(shared_entities)

        # Count additional entity-like words in each title
        words_a = set(norm_a.split())
        words_b = set(norm_b.split())

        # If one has significantly more content words, it's more specific
        if len(words_a) > len(words_b) * 1.5 and len(words_b) > 2:
            return StructuralRelation(
                source_id=event_id_a,
                target_id=event_id_b,
                title_a=title_a,
                title_b=title_b,
                relation_type="HIERARCHICAL",
                relation_level=1,
                direction="forward",  # A is more specific
                confidence=1.0,
                evidence={
                    "rule": "specificity_hierarchy",
                    "words_a": len(words_a),
                    "words_b": len(words_b),
                    "shared_entities": shared_entities,
                },
            )
        elif len(words_b) > len(words_a) * 1.5 and len(words_a) > 2:
            return StructuralRelation(
                source_id=event_id_a,
                target_id=event_id_b,
                title_a=title_a,
                title_b=title_b,
                relation_type="HIERARCHICAL",
                relation_level=1,
                direction="reverse",  # B is more specific
                confidence=1.0,
                evidence={
                    "rule": "specificity_hierarchy",
                    "words_a": len(words_a),
                    "words_b": len(words_b),
                    "shared_entities": shared_entities,
                },
            )

    # =========================================================================
    # RULE 5: MUTUALLY_EXCLUSIVE (bracket markets, explicit negation)
    # =========================================================================
    # Check for explicit negation patterns
    negation_patterns = [
        (r"\bnot\b", r"(?<!\bnot\b)"),  # "not X" vs "X"
        (r"\bno\b", r"(?<!\bno\b)"),  # "no X" vs "X"
        (r"\bwon't\b", r"\bwill\b"),  # "won't" vs "will"
        (r"\bfail\b", r"\bsucceed\b"),  # "fail" vs "succeed"
    ]

    # Simple negation check: one has "not" and the other doesn't
    has_negation_a = "not " in title_a.lower() or " no " in title_a.lower()
    has_negation_b = "not " in title_b.lower() or " no " in title_b.lower()

    if has_negation_a != has_negation_b and norm_a == norm_b:
        return StructuralRelation(
            source_id=event_id_a,
            target_id=event_id_b,
            title_a=title_a,
            title_b=title_b,
            relation_type="MUTUALLY_EXCLUSIVE",
            relation_level=2,
            direction="bidirectional",
            confidence=1.0,
            evidence={
                "rule": "negation_pattern",
                "negation_a": has_negation_a,
                "negation_b": has_negation_b,
            },
        )

    # Check semantics-based negation
    if sem_a and sem_b:
        sem_a_data = sem_a.get("semantics", {})
        sem_b_data = sem_b.get("semantics", {})
        neg_a = sem_a_data.get("negation", False)
        neg_b = sem_b_data.get("negation", False)
        subj_a = sem_a_data.get("subject_entity")
        subj_b = sem_b_data.get("subject_entity")
        pred_a = sem_a_data.get("predicate")
        pred_b = sem_b_data.get("predicate")

        if neg_a != neg_b and subj_a == subj_b and pred_a == pred_b:
            return StructuralRelation(
                source_id=event_id_a,
                target_id=event_id_b,
                title_a=title_a,
                title_b=title_b,
                relation_type="MUTUALLY_EXCLUSIVE",
                relation_level=2,
                direction="bidirectional",
                confidence=1.0,
                evidence={
                    "rule": "semantics_negation",
                    "negation_a": neg_a,
                    "negation_b": neg_b,
                    "subject": subj_a,
                    "predicate": pred_a,
                },
            )

    # =========================================================================
    # RULE 6: SERIES_MEMBER fallback for high similarity pairs
    # =========================================================================
    # High embedding similarity (>=0.83) indicates structural similarity
    if embedding_similarity >= 0.83:
        # Check if titles differ in only a few words
        words_a = set(title_a.lower().split())
        words_b = set(title_b.lower().split())
        common = words_a & words_b
        diff_a = words_a - words_b
        diff_b = words_b - words_a

        # If most words are shared and only 1-3 differ, it's likely a series variant
        if len(common) >= 2 and len(diff_a) <= 3 and len(diff_b) <= 3:
            return StructuralRelation(
                source_id=event_id_a,
                target_id=event_id_b,
                title_a=title_a,
                title_b=title_b,
                relation_type="SERIES_MEMBER",
                relation_level=1,
                direction="none",
                confidence=1.0,
                evidence={
                    "rule": "high_similarity_word_overlap",
                    "common_words": len(common),
                    "diff_words_a": list(diff_a)[:5],
                    "diff_words_b": list(diff_b)[:5],
                    "embedding_similarity": embedding_similarity,
                },
            )

    # Same subject with moderate similarity (>=0.80)
    if same_subject and embedding_similarity >= 0.80:
        words_a = set(title_a.lower().split())
        words_b = set(title_b.lower().split())
        common = words_a & words_b
        diff_a = words_a - words_b
        diff_b = words_b - words_a

        if len(common) >= 2 and len(diff_a) <= 3 and len(diff_b) <= 3:
            return StructuralRelation(
                source_id=event_id_a,
                target_id=event_id_b,
                title_a=title_a,
                title_b=title_b,
                relation_type="SERIES_MEMBER",
                relation_level=1,
                direction="none",
                confidence=1.0,
                evidence={
                    "rule": "same_subject_moderate_similarity",
                    "common_words": len(common),
                    "diff_words_a": list(diff_a)[:5],
                    "diff_words_b": list(diff_b)[:5],
                    "embedding_similarity": embedding_similarity,
                },
            )

    return None


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Main entry point."""
    start_time = datetime.now(timezone.utc)
    logger.info("Starting 05_2_classify_structural")

    # Determine input folders
    if INPUT_RUN_FOLDER:
        pairs_folder = INPUT_PAIRS_DIR / INPUT_RUN_FOLDER
        semantics_folder = INPUT_SEMANTICS_DIR / INPUT_RUN_FOLDER
        quality_folder = INPUT_QUALITY_DIR / INPUT_RUN_FOLDER
    else:
        pairs_folder = find_latest_run_folder(INPUT_PAIRS_DIR)
        semantics_folder = find_latest_run_folder(INPUT_SEMANTICS_DIR)
        quality_folder = find_latest_run_folder(INPUT_QUALITY_DIR)

    if not pairs_folder or not pairs_folder.exists():
        logger.error(f"Candidate pairs folder not found: {pairs_folder}")
        return
    if not semantics_folder or not semantics_folder.exists():
        logger.error(f"Semantics folder not found: {semantics_folder}")
        return

    logger.info(f"Loading candidate pairs from: {pairs_folder}")
    logger.info(f"Loading semantics from: {semantics_folder}")

    # Load candidate pairs
    with open(pairs_folder / "candidate_pairs.json", encoding="utf-8") as f:
        pairs_data = json.load(f)
    pairs = pairs_data.get("pairs", [])
    logger.info(f"Loaded {len(pairs)} candidate pairs")

    # Load event semantics
    with open(semantics_folder / "event_semantics.json", encoding="utf-8") as f:
        semantics_data = json.load(f)
    events_semantics = semantics_data.get("events", [])
    semantics_by_id = {e["id"]: e for e in events_semantics}
    logger.info(f"Loaded semantics for {len(semantics_by_id)} events")

    # Load pre-computed negation pairs from 05_0 (optional)
    negation_pairs: dict[tuple[str, str], dict] = {}
    if quality_folder and quality_folder.exists():
        negation_file = quality_folder / "negation_pairs.json"
        if negation_file.exists():
            with open(negation_file, encoding="utf-8") as f:
                negation_data = json.load(f)
            for pair_info in negation_data.get("pairs", []):
                key = tuple(sorted([pair_info["event_id_a"], pair_info["event_id_b"]]))
                negation_pairs[key] = pair_info
            logger.info(f"Loaded {len(negation_pairs)} pre-computed negation pairs")
    else:
        logger.info("No quality folder found, skipping negation pairs")

    # Classify pairs
    classified_relations: list[StructuralRelation] = []
    unclassified_pairs: list[dict] = []

    for pair in pairs:
        result = classify_structural(pair, semantics_by_id, negation_pairs)
        if result:
            classified_relations.append(result)
        else:
            unclassified_pairs.append(
                {
                    "event_id_a": pair["event_id_a"],
                    "event_id_b": pair["event_id_b"],
                    "reason": "no_structural_rule_matched",
                }
            )

    # Compute statistics
    classification_rate = len(classified_relations) / len(pairs) if pairs else 0
    relation_type_counts: dict[str, int] = {}
    for rel in classified_relations:
        rt = rel.relation_type
        relation_type_counts[rt] = relation_type_counts.get(rt, 0) + 1

    logger.info(
        f"Classified: {len(classified_relations)} / {len(pairs)} ({classification_rate:.2%})"
    )
    logger.info(f"Unclassified: {len(unclassified_pairs)}")
    logger.info(f"Relation types: {relation_type_counts}")

    # Create output folder
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    output_folder = SCRIPT_OUTPUT_DIR / timestamp
    output_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output folder: {output_folder}")

    # Save structural_relations.json
    relations_output = {
        "_meta": {
            "description": "Rule-based structural relations",
            "created_at": start_time.isoformat(),
            "classification_method": "deterministic_rules",
            "source_pairs": str(pairs_folder),
            "source_semantics": str(semantics_folder),
        },
        "relations": [r.to_dict() for r in classified_relations],
        "unclassified_pairs": unclassified_pairs,
    }
    with open(output_folder / "structural_relations.json", "w", encoding="utf-8") as f:
        json.dump(relations_output, f, indent=2, ensure_ascii=False)
    logger.info("Saved structural_relations.json")

    # Save summary.json
    end_time = datetime.now(timezone.utc)
    summary = {
        "run_info": {
            "started_at": start_time.isoformat(),
            "completed_at": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "input_pairs_folder": str(pairs_folder),
            "input_semantics_folder": str(semantics_folder),
            "output_folder": str(output_folder),
        },
        "statistics": {
            "input_pairs": len(pairs),
            "classified": len(classified_relations),
            "unclassified": len(unclassified_pairs),
            "classification_rate": round(classification_rate, 4),
            "by_relation_type": relation_type_counts,
        },
    }
    with open(output_folder / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Saved summary.json")

    logger.info(f"Completed in {summary['run_info']['duration_seconds']:.2f}s")


if __name__ == "__main__":
    main()
