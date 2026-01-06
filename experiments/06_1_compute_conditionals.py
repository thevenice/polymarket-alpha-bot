"""
Compute conditional probabilities for causal edges.

Pipeline: 05_4_build_relation_graph -> 06_1_compute_conditionals -> 06_2_detect_alpha

This script processes the relation graph and computes P(B|A) for all causal edges,
comparing model estimates against market-implied probabilities (which assume independence).
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_GRAPH_DIR = DATA_DIR / "05_4_build_relation_graph"
SCRIPT_OUTPUT_DIR = DATA_DIR / "06_1_compute_conditionals"

# Relation types to skip (structural, not causal)
STRUCTURAL_RELATION_TYPES = {
    "TIMEFRAME_VARIANT",
    "THRESHOLD_VARIANT",
    "HIERARCHICAL",
    "SERIES_MEMBER",
    "MUTUALLY_EXCLUSIVE",
    "INDEPENDENT",
}

# Causal relation type priors for P(B|A)
RELATION_TYPE_PRIORS = {
    "DIRECT_CAUSE": {"min": 0.80, "max": 0.95, "default": 0.88},
    "ENABLING_CONDITION": {"min": 0.50, "max": 0.80, "default": 0.65},
    "INHIBITING_CONDITION": {"min": 0.02, "max": 0.25, "default": 0.10},
    "REQUIRES": {"min": 0.00, "max": 0.00, "default": 0.00},
    "CORRELATED": {"min": 0.40, "max": 0.70, "default": 0.55},
}

# Logging configuration
LOG_LEVEL = logging.INFO

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_latest_input_dir(base_dir: Path) -> Path:
    """Get the most recent timestamped subdirectory."""
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in {base_dir}")
    return max(subdirs, key=lambda x: x.name)


def load_json(filepath: Path) -> dict:
    """Load JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def save_json(data: dict, filepath: Path) -> None:
    """Save data to JSON file with pretty formatting."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved: {filepath}")


def estimate_conditional(edge: dict) -> tuple[float, float, float]:
    """
    Estimate P(B|A) for a causal edge.

    Returns:
        tuple of (P_B_given_A, P_B_given_not_A, confidence)
    """
    rel_type = edge["relation_type"]
    priors = RELATION_TYPE_PRIORS.get(rel_type)

    if priors is None:
        # Unknown causal relation type, use neutral estimate
        return 0.5, 0.5, 0.5

    # Use LLM-provided estimate if available
    if "implied_conditional" in edge:
        impl = edge["implied_conditional"]
        p_b_given_a = impl.get("P_B_given_A", priors["default"])
        p_b_given_not_a = impl.get("P_B_given_not_A", 0.5)
        conf = edge.get("confidence", 0.7)
        return p_b_given_a, p_b_given_not_a, conf

    # Fall back to prior adjusted by confidence
    conf = edge.get("confidence", 0.7)
    base = priors["default"]

    # Higher confidence -> closer to prior extremes
    if rel_type == "INHIBITING_CONDITION":
        # For inhibiting, higher confidence means lower probability
        p_b_given_a = priors["min"] + (priors["default"] - priors["min"]) * (1 - conf)
        p_b_given_not_a = 0.5  # Neutral when A doesn't happen
    elif rel_type == "REQUIRES":
        # If B requires A, then P(B|not A) = 0
        p_b_given_a = priors["default"]
        p_b_given_not_a = 0.0
    else:
        # For positive causal relations, higher confidence means higher probability
        p_b_given_a = base + (priors["max"] - base) * conf
        p_b_given_not_a = base * 0.5  # Reduced probability when A doesn't happen

    return p_b_given_a, p_b_given_not_a, conf


def compute_market_implied(target_price: float) -> float:
    """
    Compute market-implied P(B|A) under independence assumption.

    Under independence: P(B|A) = P(B) = target_price
    """
    return target_price


def compute_chain_conditional(
    chain: dict, edge_conditionals: dict[str, float]
) -> tuple[float | None, float]:
    """
    Compute P(end|start) for a transitive chain.

    For chain A -> B -> C:
    P(C|A) approx= P(C|B) * P(B|A) (simplified chain rule)

    Returns:
        tuple of (chain_P_end_given_start, chain_confidence)
    """
    path = chain["path"]
    if len(path) < 2:
        return None, 0.0

    total_prob = 1.0
    total_confidence = chain.get("total_confidence", 0.5)

    for i in range(len(path) - 1):
        source, target = path[i], path[i + 1]
        edge_key = f"{source}|{target}"
        p_next = edge_conditionals.get(edge_key)

        if p_next is None:
            # Try reverse direction
            edge_key_rev = f"{target}|{source}"
            p_next = edge_conditionals.get(edge_key_rev)

        if p_next is None:
            # Edge not found, use neutral probability
            p_next = 0.5

        total_prob *= p_next

    return total_prob, total_confidence


# =============================================================================
# MAIN LOGIC
# =============================================================================


def process_edges(
    edges: list[dict], nodes_by_id: dict[str, dict]
) -> tuple[list[dict], dict[str, float]]:
    """
    Process all causal edges and compute conditional probabilities.

    Returns:
        tuple of (processed_edges, edge_conditionals_map)
    """
    processed_edges = []
    edge_conditionals = {}

    for edge in edges:
        rel_type = edge["relation_type"]

        # Skip structural relations
        if rel_type in STRUCTURAL_RELATION_TYPES:
            continue

        # Skip if relation type not in our priors
        if rel_type not in RELATION_TYPE_PRIORS:
            logger.warning(f"Unknown causal relation type: {rel_type}")
            continue

        source_id = edge["source"]
        target_id = edge["target"]

        source_node = nodes_by_id.get(source_id)
        target_node = nodes_by_id.get(target_id)

        if not source_node or not target_node:
            logger.warning(f"Missing node for edge {source_id} -> {target_id}")
            continue

        # Get prices
        source_price = source_node.get("current_price", 0.5)
        target_price = target_node.get("current_price", 0.5)

        # Estimate model conditional
        p_b_given_a, p_b_given_not_a, model_conf = estimate_conditional(edge)

        # Compute market-implied conditional (under independence)
        market_implied = compute_market_implied(target_price)

        # Compute delta
        conditional_delta = p_b_given_a - market_implied

        processed_edge = {
            "source_id": source_id,
            "target_id": target_id,
            "source_title": source_node.get("title", ""),
            "target_title": target_node.get("title", ""),
            "relation_type": rel_type,
            "source_price": source_price,
            "target_price": target_price,
            "model_P_B_given_A": round(p_b_given_a, 4),
            "model_P_B_given_not_A": round(p_b_given_not_a, 4),
            "model_confidence": round(model_conf, 4),
            "market_implied_P_B_given_A": round(market_implied, 4),
            "market_assumption": "independence",
            "conditional_delta": round(conditional_delta, 4),
            "notes": (
                "Market prices B as independent of A"
                if abs(conditional_delta) > 0.1
                else "Low delta - market and model roughly aligned"
            ),
        }

        processed_edges.append(processed_edge)

        # Store for transitive chain computation
        edge_key = f"{source_id}|{target_id}"
        edge_conditionals[edge_key] = p_b_given_a

    return processed_edges, edge_conditionals


def process_transitive_chains(
    chains: list[dict], edge_conditionals: dict[str, float]
) -> list[dict]:
    """Process transitive chains and compute chain probabilities."""
    processed_chains = []

    for chain in chains:
        # Skip structural chains
        chain_type = chain.get("chain_type", "")
        if chain_type == "structural_chain":
            continue

        # Skip chains that don't involve causal relations
        relations = chain.get("relations", [])
        has_causal = any(r in RELATION_TYPE_PRIORS for r in relations)
        if not has_causal:
            continue

        chain_prob, chain_conf = compute_chain_conditional(chain, edge_conditionals)

        if chain_prob is not None:
            processed_chain = {
                "path": chain["path"],
                "relations": relations,
                "chain_P_end_given_start": round(chain_prob, 4),
                "chain_confidence": round(chain_conf, 4),
            }
            processed_chains.append(processed_chain)

    return processed_chains


def compute_statistics(edges: list[dict]) -> dict:
    """Compute summary statistics for processed edges."""
    if not edges:
        return {"total_edges": 0}

    deltas = [e["conditional_delta"] for e in edges]
    by_type: dict[str, dict] = {}

    for e in edges:
        rt = e["relation_type"]
        if rt not in by_type:
            by_type[rt] = {"count": 0, "p_values": []}
        by_type[rt]["count"] += 1
        by_type[rt]["p_values"].append(e["model_P_B_given_A"])

    by_type_summary = {}
    for rt, data in by_type.items():
        avg_p = sum(data["p_values"]) / len(data["p_values"]) if data["p_values"] else 0
        by_type_summary[rt] = {
            "count": data["count"],
            "avg_P_B_given_A": round(avg_p, 4),
        }

    return {
        "total_edges": len(edges),
        "edges_with_conditionals": len(edges),
        "avg_conditional_delta": round(sum(deltas) / len(deltas), 4) if deltas else 0,
        "max_conditional_delta": round(max(deltas), 4) if deltas else 0,
        "min_conditional_delta": round(min(deltas), 4) if deltas else 0,
        "by_relation_type": by_type_summary,
    }


def main() -> None:
    """Main entry point."""
    logger.info("Starting conditional probability computation")

    # Find latest input directory
    input_dir = get_latest_input_dir(INPUT_GRAPH_DIR)
    logger.info(f"Using input directory: {input_dir}")

    # Load input data
    relation_graph = load_json(input_dir / "relation_graph.json")
    transitive_relations = load_json(input_dir / "transitive_relations.json")

    logger.info(
        f"Loaded {len(relation_graph.get('edges', []))} edges, "
        f"{len(transitive_relations.get('chains', []))} chains"
    )

    # Build node lookup
    nodes_by_id = {n["id"]: n for n in relation_graph.get("nodes", [])}
    logger.info(f"Built lookup for {len(nodes_by_id)} nodes")

    # Process edges
    processed_edges, edge_conditionals = process_edges(
        relation_graph.get("edges", []), nodes_by_id
    )
    logger.info(f"Processed {len(processed_edges)} causal edges")

    # Process transitive chains
    processed_chains = process_transitive_chains(
        transitive_relations.get("chains", []), edge_conditionals
    )
    logger.info(f"Processed {len(processed_chains)} transitive chains")

    # Compute statistics
    stats = compute_statistics(processed_edges)

    # Create output directory
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = SCRIPT_OUTPUT_DIR / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare output data
    conditional_probabilities = {
        "_meta": {
            "description": "Conditional probability estimates for causal edges",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "estimation_method": "relation_type_priors",
            "input_directory": str(input_dir),
        },
        "edges": processed_edges,
        "transitive_chains": processed_chains,
    }

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(input_dir),
        "statistics": stats,
        "configuration": {
            "relation_type_priors": RELATION_TYPE_PRIORS,
            "structural_types_skipped": list(STRUCTURAL_RELATION_TYPES),
        },
    }

    # Save outputs
    save_json(conditional_probabilities, output_dir / "conditional_probabilities.json")
    save_json(summary, output_dir / "summary.json")

    # Log summary
    logger.info("=== Summary ===")
    logger.info(f"Total causal edges processed: {stats['total_edges']}")
    logger.info(f"Avg conditional delta: {stats.get('avg_conditional_delta', 0):.4f}")
    logger.info(f"Max conditional delta: {stats.get('max_conditional_delta', 0):.4f}")
    logger.info(f"Transitive chains: {len(processed_chains)}")
    for rt, data in stats.get("by_relation_type", {}).items():
        logger.info(
            f"  {rt}: {data['count']} edges, avg P(B|A)={data['avg_P_B_given_A']:.2f}"
        )

    logger.info(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
