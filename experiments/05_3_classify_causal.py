"""
LLM-based causal classification for event pairs.

Pipeline: 05_2_classify_structural -> 05_3_classify_causal -> 05_4_build_relation_graph

This script processes unclassified pairs from 05_2_classify_structural and uses
an LLM to detect causal relationships:
- DIRECT_CAUSE: A directly causes B (P(B|A) > 80%)
- ENABLING_CONDITION: A makes B possible (P(B|A) 50-80%)
- INHIBITING_CONDITION: A prevents/reduces B
- REQUIRES: B cannot happen without A
- CORRELATED: Co-occur without causation
- INDEPENDENT: No meaningful connection

Pair prioritization: Shared outcome_states first, opposite polarity, same subject.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

# =============================================================================
# CONFIGURATION
# =============================================================================

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data"

# Input directories
INPUT_STRUCTURAL_DIR = DATA_DIR / "05_2_classify_structural"
INPUT_SEMANTICS_DIR = DATA_DIR / "04_1_extract_event_semantics"
INPUT_RUN_FOLDER: str | None = None  # None = use latest

# Output directory
SCRIPT_OUTPUT_DIR = DATA_DIR / "05_3_classify_causal"

# OpenRouter API settings
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "xiaomi/mimo-v2-flash:free"

# LLM settings
LLM_BATCH_SIZE = 5  # Pairs per prompt
MAX_LLM_PAIRS = 500  # Limit for initial test (can increase later)
MAX_RETRIES = 3
RETRY_DELAY = 2.0
REQUEST_TIMEOUT = 90.0
REQUEST_DELAY = 0.5  # Delay between requests (free tier)

# Logging
LOG_LEVEL = logging.INFO

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# CAUSAL RELATION TYPES
# =============================================================================

CAUSAL_RELATION_TYPES = {
    "DIRECT_CAUSE": {"level": 3, "p_b_given_a_range": (0.80, 0.95)},
    "ENABLING_CONDITION": {"level": 3, "p_b_given_a_range": (0.50, 0.80)},
    "INHIBITING_CONDITION": {"level": 3, "p_b_given_a_range": (0.05, 0.25)},
    "REQUIRES": {"level": 3, "p_b_given_a_range": (0.00, 0.00)},
    "CORRELATED": {"level": 3, "p_b_given_a_range": (0.30, 0.70)},
    "INDEPENDENT": {"level": 2, "p_b_given_a_range": None},
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class CausalRelation:
    """A classified causal relation between two events."""

    source_id: str
    target_id: str
    title_a: str
    title_b: str
    relation_type: str
    relation_level: int
    direction: str
    confidence: float
    reasoning: str
    implied_conditional: dict = field(default_factory=dict)

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
            "reasoning": self.reasoning,
            "implied_conditional": self.implied_conditional,
        }


@dataclass
class IndependentPair:
    """A pair classified as independent (no causal connection)."""

    event_id_a: str
    event_id_b: str
    title_a: str
    title_b: str
    relation_type: str = "INDEPENDENT"
    confidence: float = 0.9
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "event_id_a": self.event_id_a,
            "event_id_b": self.event_id_b,
            "title_a": self.title_a,
            "title_b": self.title_b,
            "relation_type": self.relation_type,
            "confidence": round(self.confidence, 4),
            "reasoning": self.reasoning,
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


def prioritize_pairs(
    pairs: list[dict], semantics_by_id: dict
) -> list[tuple[int, dict]]:
    """
    Order pairs by likelihood of causal relationship.
    Higher score = more likely causal.
    """
    scored = []
    for pair in pairs:
        score = 0
        sem_a = semantics_by_id.get(pair["event_id_a"], {}).get("semantics", {})
        sem_b = semantics_by_id.get(pair["event_id_b"], {}).get("semantics", {})

        # Shared outcome states (most important)
        states_a = set(sem_a.get("outcome_states", []))
        states_b = set(sem_b.get("outcome_states", []))
        if states_a and states_b and states_a & states_b:
            score += 10

        # Opposite polarity (inhibiting?)
        pol_a = sem_a.get("polarity")
        pol_b = sem_b.get("polarity")
        if pol_a and pol_b and pol_a != pol_b:
            score += 5

        # Same subject entity
        subj_a = sem_a.get("subject_entity")
        subj_b = sem_b.get("subject_entity")
        if subj_a and subj_b and subj_a == subj_b:
            score += 3

        # Same predicate (related actions)
        pred_a = sem_a.get("predicate")
        pred_b = sem_b.get("predicate")
        if pred_a and pred_b and pred_a == pred_b:
            score += 2

        scored.append((score, pair))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


# =============================================================================
# LLM CLIENT
# =============================================================================


class LLMClient:
    """Client for OpenRouter API."""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.client = httpx.Client(timeout=REQUEST_TIMEOUT)
        self.total_tokens = 0

    def close(self) -> None:
        self.client.close()

    def complete(self, prompt: str, system: str | None = None) -> str:
        """Send completion request to LLM."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.post(
                    f"{OPENROUTER_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": 0.1,
                    },
                )
                response.raise_for_status()
                data = response.json()

                if "usage" in data:
                    self.total_tokens += data["usage"].get("total_tokens", 0)

                return data["choices"][0]["message"]["content"]

            except httpx.HTTPStatusError as e:
                logger.warning(f"HTTP error (attempt {attempt + 1}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    raise
            except Exception as e:
                logger.warning(f"Error (attempt {attempt + 1}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    raise

        return ""


# =============================================================================
# LLM PROMPTS
# =============================================================================

LLM_SYSTEM_PROMPT = """You are an expert geopolitical analyst assessing causal relationships between prediction market events.

For each pair of events, analyze the causal relationship and determine:
1. The relationship type
2. The direction of causation
3. Your confidence level
4. The implied conditional probabilities

RELATION TYPES:
- DIRECT_CAUSE: A directly causes B (P(B|A) > 80%). Example: Nuclear war -> Recession
- ENABLING_CONDITION: A makes B possible/more likely (P(B|A) 50-80%). Example: Ceasefire -> Peace treaty
- INHIBITING_CONDITION: A prevents or reduces likelihood of B. Example: Nuclear detonation -> Ceasefire (inhibits)
- REQUIRES: B cannot happen without A (P(B|not A) = 0). Example: NATO Article 5 -> US troops in Europe
- CORRELATED: A and B co-occur but no clear causal direction. Example: Oil spike <-> Inflation
- INDEPENDENT: No meaningful connection between A and B.

RULES:
- Be conservative. Only assign causal relations when the mechanism is clear.
- Consider the direction: does A cause B, B cause A, or bidirectional?
- For INHIBITING_CONDITION, P(B|A) should be LOW (event A reduces probability of B)
- Output valid JSON only, no other text."""


def build_llm_batch_prompt(pairs: list[dict], semantics_by_id: dict) -> str:
    """Build prompt for batch LLM classification."""
    prompt_parts = ["Classify the causal relationships for these event pairs:\n"]

    for i, pair in enumerate(pairs):
        sem_a = semantics_by_id.get(pair["event_id_a"], {})
        sem_b = semantics_by_id.get(pair["event_id_b"], {})
        semantics_a = sem_a.get("semantics", {})
        semantics_b = sem_b.get("semantics", {})

        prompt_parts.append(f"\n=== PAIR {i + 1} ===")
        prompt_parts.append(f'Event A: "{sem_a.get("title", "N/A")}"')
        prompt_parts.append(f"  - Type: {semantics_a.get('event_type', 'N/A')}")
        prompt_parts.append(f"  - Entities: {semantics_a.get('subject_entity', 'N/A')}")
        prompt_parts.append(
            f"  - Outcome states: {semantics_a.get('outcome_states', [])}"
        )

        prompt_parts.append(f'Event B: "{sem_b.get("title", "N/A")}"')
        prompt_parts.append(f"  - Type: {semantics_b.get('event_type', 'N/A')}")
        prompt_parts.append(f"  - Entities: {semantics_b.get('subject_entity', 'N/A')}")
        prompt_parts.append(
            f"  - Outcome states: {semantics_b.get('outcome_states', [])}"
        )

    prompt_parts.append("\n\nOutput JSON array with one object per pair:")
    prompt_parts.append("""[
  {
    "pair": 1,
    "relation_type": "DIRECT_CAUSE|ENABLING_CONDITION|INHIBITING_CONDITION|REQUIRES|CORRELATED|INDEPENDENT",
    "direction": "forward|reverse|bidirectional",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of causal mechanism",
    "P_B_given_A": 0.0-1.0,
    "P_B_given_not_A": 0.0-1.0
  },
  ...
]""")

    return "\n".join(prompt_parts)


def parse_llm_response(response: str, num_pairs: int) -> list[dict]:
    """Parse LLM JSON response."""
    import re

    # Try to extract JSON array from response
    try:
        start = response.find("[")
        end = response.rfind("]") + 1
        if start >= 0 and end > start:
            json_str = response[start:end]
            results = json.loads(json_str)
            return results
    except json.JSONDecodeError:
        pass

    # Try to parse as individual JSON objects
    results = []
    for i in range(num_pairs):
        try:
            pattern = rf'\{{"pair":\s*{i + 1}[^}}]+\}}'
            match = re.search(pattern, response)
            if match:
                results.append(json.loads(match.group()))
        except (json.JSONDecodeError, AttributeError):
            continue

    return results


# =============================================================================
# CAUSAL CLASSIFICATION
# =============================================================================


def classify_causal_pairs(
    pairs: list[dict],
    semantics_by_id: dict,
    llm_client: LLMClient,
    cache: dict,
) -> tuple[list[CausalRelation], list[IndependentPair]]:
    """
    Classify pairs using LLM with batching and caching.

    Returns (causal_relations, independent_pairs)
    """
    causal_relations: list[CausalRelation] = []
    independent_pairs: list[IndependentPair] = []
    pairs_to_classify: list[dict] = []

    # Check cache first
    for pair in pairs:
        cache_key = f"{pair['event_id_a']}|{pair['event_id_b']}"
        if cache_key in cache:
            cached = cache[cache_key]
            sem_a = semantics_by_id.get(pair["event_id_a"], {})
            sem_b = semantics_by_id.get(pair["event_id_b"], {})
            title_a = sem_a.get("title", "")
            title_b = sem_b.get("title", "")

            if cached["relation_type"] == "INDEPENDENT":
                independent_pairs.append(
                    IndependentPair(
                        event_id_a=pair["event_id_a"],
                        event_id_b=pair["event_id_b"],
                        title_a=title_a,
                        title_b=title_b,
                        confidence=cached["confidence"],
                        reasoning=cached.get("reasoning", ""),
                    )
                )
            else:
                rel_info = CAUSAL_RELATION_TYPES.get(
                    cached["relation_type"], {"level": 3}
                )
                causal_relations.append(
                    CausalRelation(
                        source_id=pair["event_id_a"],
                        target_id=pair["event_id_b"],
                        title_a=title_a,
                        title_b=title_b,
                        relation_type=cached["relation_type"],
                        relation_level=rel_info["level"],
                        direction=cached.get("direction", "forward"),
                        confidence=cached["confidence"],
                        reasoning=cached.get("reasoning", ""),
                        implied_conditional={
                            "P_B_given_A": cached.get("P_B_given_A", 0.5),
                            "P_B_given_not_A": cached.get("P_B_given_not_A", 0.5),
                        },
                    )
                )
        else:
            pairs_to_classify.append(pair)

    logger.info(
        f"Cache: {len(pairs) - len(pairs_to_classify)} cached, "
        f"{len(pairs_to_classify)} to classify"
    )

    # Process in batches
    total_batches = (len(pairs_to_classify) + LLM_BATCH_SIZE - 1) // LLM_BATCH_SIZE
    for batch_idx, batch_start in enumerate(
        range(0, len(pairs_to_classify), LLM_BATCH_SIZE)
    ):
        batch = pairs_to_classify[batch_start : batch_start + LLM_BATCH_SIZE]
        logger.info(f"Processing batch {batch_idx + 1}/{total_batches}")

        try:
            prompt = build_llm_batch_prompt(batch, semantics_by_id)
            response = llm_client.complete(prompt, LLM_SYSTEM_PROMPT)
            parsed = parse_llm_response(response, len(batch))

            for i, pair in enumerate(batch):
                sem_a = semantics_by_id.get(pair["event_id_a"], {})
                sem_b = semantics_by_id.get(pair["event_id_b"], {})
                title_a = sem_a.get("title", "")
                title_b = sem_b.get("title", "")

                if i < len(parsed):
                    result = parsed[i]
                    relation_type = result.get("relation_type", "INDEPENDENT")

                    # Validate relation type
                    if relation_type not in CAUSAL_RELATION_TYPES:
                        relation_type = "INDEPENDENT"

                    confidence = float(result.get("confidence", 0.5))
                    p_b_given_a = float(result.get("P_B_given_A", 0.5))
                    p_b_given_not_a = float(result.get("P_B_given_not_A", 0.5))
                    reasoning = result.get("reasoning", "")
                    direction = result.get("direction", "forward")

                    # Update cache
                    cache_key = f"{pair['event_id_a']}|{pair['event_id_b']}"
                    cache[cache_key] = {
                        "relation_type": relation_type,
                        "direction": direction,
                        "confidence": confidence,
                        "reasoning": reasoning,
                        "P_B_given_A": p_b_given_a,
                        "P_B_given_not_A": p_b_given_not_a,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }

                    if relation_type == "INDEPENDENT":
                        independent_pairs.append(
                            IndependentPair(
                                event_id_a=pair["event_id_a"],
                                event_id_b=pair["event_id_b"],
                                title_a=title_a,
                                title_b=title_b,
                                confidence=confidence,
                                reasoning=reasoning,
                            )
                        )
                    else:
                        rel_info = CAUSAL_RELATION_TYPES[relation_type]
                        causal_relations.append(
                            CausalRelation(
                                source_id=pair["event_id_a"],
                                target_id=pair["event_id_b"],
                                title_a=title_a,
                                title_b=title_b,
                                relation_type=relation_type,
                                relation_level=rel_info["level"],
                                direction=direction,
                                confidence=confidence,
                                reasoning=reasoning,
                                implied_conditional={
                                    "P_B_given_A": p_b_given_a,
                                    "P_B_given_not_A": p_b_given_not_a,
                                },
                            )
                        )
                else:
                    # LLM didn't return result for this pair, mark as independent
                    cache_key = f"{pair['event_id_a']}|{pair['event_id_b']}"
                    cache[cache_key] = {
                        "relation_type": "INDEPENDENT",
                        "direction": "none",
                        "confidence": 0.5,
                        "reasoning": "LLM response missing",
                        "P_B_given_A": 0.5,
                        "P_B_given_not_A": 0.5,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    independent_pairs.append(
                        IndependentPair(
                            event_id_a=pair["event_id_a"],
                            event_id_b=pair["event_id_b"],
                            title_a=title_a,
                            title_b=title_b,
                            confidence=0.5,
                            reasoning="LLM response missing",
                        )
                    )

            # Rate limiting delay
            time.sleep(REQUEST_DELAY)

        except Exception as e:
            logger.error(f"LLM batch error: {e}")
            # Mark all pairs in failed batch as independent
            for pair in batch:
                sem_a = semantics_by_id.get(pair["event_id_a"], {})
                sem_b = semantics_by_id.get(pair["event_id_b"], {})
                independent_pairs.append(
                    IndependentPair(
                        event_id_a=pair["event_id_a"],
                        event_id_b=pair["event_id_b"],
                        title_a=sem_a.get("title", ""),
                        title_b=sem_b.get("title", ""),
                        confidence=0.5,
                        reasoning=f"LLM error: {e}",
                    )
                )
            continue

    return causal_relations, independent_pairs


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Main entry point."""
    start_time = datetime.now(timezone.utc)
    logger.info("Starting 05_3_classify_causal")

    # Check API key
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY not set")
        return

    # Determine input folders
    if INPUT_RUN_FOLDER:
        structural_folder = INPUT_STRUCTURAL_DIR / INPUT_RUN_FOLDER
        semantics_folder = INPUT_SEMANTICS_DIR / INPUT_RUN_FOLDER
    else:
        structural_folder = find_latest_run_folder(INPUT_STRUCTURAL_DIR)
        semantics_folder = find_latest_run_folder(INPUT_SEMANTICS_DIR)

    if not structural_folder or not structural_folder.exists():
        logger.error(f"Structural folder not found: {structural_folder}")
        return
    if not semantics_folder or not semantics_folder.exists():
        logger.error(f"Semantics folder not found: {semantics_folder}")
        return

    logger.info(f"Loading structural relations from: {structural_folder}")
    logger.info(f"Loading event semantics from: {semantics_folder}")

    # Load structural relations (for unclassified_pairs)
    structural_file = structural_folder / "structural_relations.json"
    with open(structural_file, encoding="utf-8") as f:
        structural_data = json.load(f)

    unclassified_pairs = structural_data.get("unclassified_pairs", [])
    logger.info(f"Loaded {len(unclassified_pairs)} unclassified pairs")

    # Load event semantics
    semantics_file = semantics_folder / "event_semantics.json"
    with open(semantics_file, encoding="utf-8") as f:
        semantics_data = json.load(f)

    semantics_by_id = {e["id"]: e for e in semantics_data.get("events", [])}
    logger.info(f"Loaded {len(semantics_by_id)} event semantics")

    # Prioritize pairs
    logger.info("Prioritizing pairs by causal likelihood...")
    scored_pairs = prioritize_pairs(unclassified_pairs, semantics_by_id)

    # Limit to MAX_LLM_PAIRS
    if len(scored_pairs) > MAX_LLM_PAIRS:
        logger.info(f"Limiting from {len(scored_pairs)} to {MAX_LLM_PAIRS} pairs")
        scored_pairs = scored_pairs[:MAX_LLM_PAIRS]

    # Extract just the pairs (drop scores)
    pairs_to_process = [pair for _, pair in scored_pairs]

    # Load or initialize cache
    llm_cache: dict = {}

    # Initialize LLM client
    llm_client = LLMClient(OPENROUTER_API_KEY, LLM_MODEL)

    try:
        # Classify pairs
        causal_relations, independent_pairs = classify_causal_pairs(
            pairs_to_process, semantics_by_id, llm_client, llm_cache
        )

        logger.info(f"Causal relations found: {len(causal_relations)}")
        logger.info(f"Independent pairs: {len(independent_pairs)}")
        logger.info(f"Total LLM tokens: {llm_client.total_tokens}")

    finally:
        llm_client.close()

    # Create output folder
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    output_folder = SCRIPT_OUTPUT_DIR / timestamp
    output_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output folder: {output_folder}")

    # Count relation types
    relation_type_counts: dict[str, int] = {}
    for rel in causal_relations:
        rt = rel.relation_type
        relation_type_counts[rt] = relation_type_counts.get(rt, 0) + 1

    # Save causal relations
    causal_output = {
        "_meta": {
            "description": "LLM-classified causal relations",
            "created_at": start_time.isoformat(),
            "model": LLM_MODEL,
            "total_tokens": llm_client.total_tokens,
            "source_structural": str(structural_folder),
            "source_semantics": str(semantics_folder),
        },
        "relations": [r.to_dict() for r in causal_relations],
        "independent_pairs": [p.to_dict() for p in independent_pairs],
    }

    with open(output_folder / "causal_relations.json", "w", encoding="utf-8") as f:
        json.dump(causal_output, f, indent=2, ensure_ascii=False)
    logger.info("Saved causal_relations.json")

    # Save LLM cache
    with open(output_folder / "llm_cache.json", "w", encoding="utf-8") as f:
        json.dump({"cache": llm_cache}, f, indent=2, ensure_ascii=False)
    logger.info("Saved llm_cache.json")

    # Save summary
    end_time = datetime.now(timezone.utc)
    summary = {
        "run_info": {
            "started_at": start_time.isoformat(),
            "completed_at": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "input_structural_folder": str(structural_folder),
            "input_semantics_folder": str(semantics_folder),
            "output_folder": str(output_folder),
        },
        "configuration": {
            "llm_model": LLM_MODEL,
            "llm_batch_size": LLM_BATCH_SIZE,
            "max_llm_pairs": MAX_LLM_PAIRS,
        },
        "statistics": {
            "total_unclassified_input": len(unclassified_pairs),
            "pairs_processed": len(pairs_to_process),
            "causal_relations_found": len(causal_relations),
            "independent_pairs": len(independent_pairs),
            "relation_type_counts": relation_type_counts,
            "total_tokens": llm_client.total_tokens,
        },
    }

    with open(output_folder / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Saved summary.json")

    # Final stats
    logger.info(f"Relation type distribution: {relation_type_counts}")
    logger.info(f"Completed in {summary['run_info']['duration_seconds']:.2f}s")


if __name__ == "__main__":
    main()
