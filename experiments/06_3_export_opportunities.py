"""
Export alpha opportunities in UI-ready formats.

Pipeline: 06_2_detect_alpha + 05_4_build_relation_graph -> 06_3_export_opportunities -> UI

This script transforms alpha signals and relation graph data into multiple
UI-ready formats: opportunities feed, graph visualization, combination strategies,
and per-event exploration views.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_ALPHA_DIR = DATA_DIR / "06_2_detect_alpha"
INPUT_GRAPH_DIR = DATA_DIR / "05_4_build_relation_graph"
SCRIPT_OUTPUT_DIR = DATA_DIR / "06_3_export_opportunities"

POLYMARKET_BASE_URL = "https://polymarket.com/event"
MAX_TITLE_LENGTH = 40

# Minimum number of consequences to create a combination strategy
MIN_CONSEQUENCES_FOR_COMBO = 2

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
# DISPLAY FORMATTING FUNCTIONS
# =============================================================================


def format_price(price: float) -> str:
    """Format price as percentage string."""
    return f"{int(price * 100)}%"


def format_alpha(alpha: float) -> str:
    """Format alpha with sign."""
    sign = "+" if alpha > 0 else ""
    return f"{sign}{int(alpha * 100)}%"


def format_relation_type(rel_type: str) -> str:
    """Format relation type for display."""
    mapping = {
        "DIRECT_CAUSE": "Directly Causes",
        "ENABLING_CONDITION": "Enables",
        "INHIBITING_CONDITION": "Prevents",
        "REQUIRES": "Requires",
        "THRESHOLD_VARIANT": "Threshold Variant",
        "TIMEFRAME_VARIANT": "Timeframe Variant",
        "CORRELATED": "Correlated With",
        "HIERARCHICAL": "Hierarchical",
        "MUTUALLY_EXCLUSIVE": "Mutually Exclusive",
    }
    return mapping.get(rel_type, rel_type)


def truncate_title(title: str, max_len: int = MAX_TITLE_LENGTH) -> str:
    """Truncate title to max length with ellipsis."""
    return title[:max_len] + "..." if len(title) > max_len else title


def get_relation_direction_arrow(rel_type: str) -> str:
    """Get direction arrow based on relation type."""
    if rel_type in ["INHIBITING_CONDITION", "MUTUALLY_EXCLUSIVE"]:
        return "-|"
    return "->"


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


def generate_market_url(event_id: str) -> str:
    """Generate Polymarket URL for an event."""
    return f"{POLYMARKET_BASE_URL}/{event_id}"


# =============================================================================
# EXPORT: OPPORTUNITIES.JSON
# =============================================================================


def build_opportunities(signals: list[dict], per_event_relations: dict) -> list[dict]:
    """Build UI-ready opportunities list."""
    opportunities = []

    # Build index of signals by trigger event for related opportunities
    signals_by_trigger: dict[str, list[str]] = {}
    for signal in signals:
        trigger_id = signal["trigger_event"]["id"]
        if trigger_id not in signals_by_trigger:
            signals_by_trigger[trigger_id] = []
        signals_by_trigger[trigger_id].append(signal["signal_id"])

    for idx, signal in enumerate(signals, start=1):
        trigger = signal["trigger_event"]
        consequence = signal["consequence_event"]

        # Find related opportunities (same trigger)
        trigger_id = trigger["id"]
        related_opps = [
            sid
            for sid in signals_by_trigger.get(trigger_id, [])
            if sid != signal["signal_id"]
        ][:5]  # Limit to 5 related

        # Generate tags from relation type and context
        tags = generate_opportunity_tags(signal)

        # Compute expected return
        alpha_val = signal["alpha_signal"]
        market_price = consequence["current_price"]
        if alpha_val > 0 and market_price > 0:
            expected_return = f"{round(1 / market_price, 1)}x per dollar"
        elif alpha_val < 0 and market_price < 1:
            expected_return = f"{round(1 / (1 - market_price), 1)}x per dollar"
        else:
            expected_return = "N/A"

        opportunity = {
            "id": f"opp_{idx:03d}",
            "rank": idx,
            "trigger": {
                "event_id": trigger["id"],
                "title": trigger["title"],
                "price": trigger["current_price"],
                "price_display": format_price(trigger["current_price"]),
                "market_url": generate_market_url(trigger["id"]),
            },
            "consequence": {
                "event_id": consequence["id"],
                "title": consequence["title"],
                "price": consequence["current_price"],
                "price_display": format_price(consequence["current_price"]),
                "market_url": generate_market_url(consequence["id"]),
            },
            "relation": {
                "type": signal["relation_type"],
                "type_display": format_relation_type(signal["relation_type"]),
                "direction": get_relation_direction_arrow(signal["relation_type"]),
                "confidence": signal["confidence"],
                "confidence_display": format_price(signal["confidence"]),
            },
            "alpha": {
                "signal": signal["alpha_signal"],
                "signal_display": format_alpha(signal["alpha_signal"]),
                "direction": signal["strategy"]["action"],
                "confidence_adjusted": signal["confidence_adjusted_alpha"],
            },
            "strategy": {
                "summary": signal["strategy"]["description"],
                "detailed": generate_detailed_strategy(signal),
                "expected_return": expected_return,
            },
            "related_opportunities": related_opps,
            "tags": tags,
        }

        opportunities.append(opportunity)

    return opportunities


def generate_detailed_strategy(signal: dict) -> str:
    """Generate detailed strategy description."""
    trigger_title = signal["trigger_event"]["title"]
    consequence_title = signal["consequence_event"]["title"]
    consequence_price = signal["consequence_event"]["current_price"]
    model_prob = signal["model_conditional"]
    action = signal["strategy"]["action"]

    price_pct = format_price(consequence_price)
    model_pct = format_price(model_prob)

    if action == "BUY":
        return (
            f"If '{truncate_title(trigger_title, 50)}' resolves to YES, "
            f"immediately buy '{truncate_title(consequence_title, 50)}' at current price of {price_pct}. "
            f"Model predicts {model_pct} probability vs market's {price_pct}."
        )
    else:
        return (
            f"If '{truncate_title(trigger_title, 50)}' resolves to YES, "
            f"sell '{truncate_title(consequence_title, 50)}' at current price of {price_pct} "
            f"(or buy NO). Model predicts lower probability than market's {price_pct}."
        )


def generate_opportunity_tags(signal: dict) -> list[str]:
    """Generate tags for an opportunity based on its content."""
    tags = []

    # Relation type tag
    rel_type = signal["relation_type"]
    if rel_type in ["DIRECT_CAUSE", "ENABLING_CONDITION"]:
        tags.append("causal")
    elif rel_type == "INHIBITING_CONDITION":
        tags.append("inhibiting")
    elif rel_type == "CORRELATED":
        tags.append("correlation")

    # Alpha magnitude tag
    alpha = abs(signal["alpha_signal"])
    if alpha >= 0.5:
        tags.append("high-alpha")
    elif alpha >= 0.3:
        tags.append("medium-alpha")

    # Trigger probability tag
    trigger_price = signal["trigger_event"]["current_price"]
    if trigger_price < 0.1:
        tags.append("low-probability-trigger")

    # Content-based tags from titles
    titles = (
        signal["trigger_event"]["title"].lower()
        + signal["consequence_event"]["title"].lower()
    )

    if any(word in titles for word in ["russia", "ukraine", "war", "ceasefire"]):
        tags.append("geopolitical")
    if any(word in titles for word in ["trump", "biden", "election", "president"]):
        tags.append("politics")
    if any(word in titles for word in ["recession", "economy", "market", "stock"]):
        tags.append("economic")
    if any(word in titles for word in ["israel", "iran", "palestine", "hamas"]):
        tags.append("middle-east")

    return tags[:5]  # Limit to 5 tags


# =============================================================================
# EXPORT: EVENT_GRAPH_UI.JSON
# =============================================================================


def build_event_graph_ui(
    nodes: list[dict],
    signals: list[dict],
    per_event_relations: dict,
) -> dict:
    """Build Cytoscape-compatible graph format."""
    # Build node lookup
    node_lookup = {node["id"]: node for node in nodes}

    # Track which events have alpha signals
    events_with_alpha: dict[str, dict] = {}  # event_id -> {count, total_alpha}

    for signal in signals:
        for event_key in ["trigger_event", "consequence_event"]:
            event_id = signal[event_key]["id"]
            if event_id not in events_with_alpha:
                events_with_alpha[event_id] = {"count": 0, "total_alpha": 0}
            events_with_alpha[event_id]["count"] += 1
            events_with_alpha[event_id]["total_alpha"] += abs(signal["alpha_signal"])

    # Build nodes
    graph_nodes = []
    included_node_ids = set()

    for signal in signals:
        for event_key, node_type in [
            ("trigger_event", "trigger"),
            ("consequence_event", "consequence"),
        ]:
            event_id = signal[event_key]["id"]
            if event_id in included_node_ids:
                continue
            included_node_ids.add(event_id)

            event_data = signal[event_key]
            alpha_info = events_with_alpha.get(event_id, {"count": 0})

            node_classes = [node_type]
            if alpha_info["count"] >= 3:
                node_classes.append("high-connectivity")

            graph_node = {
                "data": {
                    "id": event_id,
                    "label": truncate_title(event_data["title"], 30),
                    "fullTitle": event_data["title"],
                    "price": event_data["current_price"],
                    "priceDisplay": format_price(event_data["current_price"]),
                    "hasAlpha": True,
                    "alphaCount": alpha_info["count"],
                    "nodeType": node_type,
                },
                "classes": " ".join(node_classes),
            }
            graph_nodes.append(graph_node)

    # Build edges from signals
    graph_edges = []
    edge_ids_seen = set()

    for signal in signals:
        source_id = signal["trigger_event"]["id"]
        target_id = signal["consequence_event"]["id"]
        edge_id = f"e_{source_id}_{target_id}"

        if edge_id in edge_ids_seen:
            continue
        edge_ids_seen.add(edge_id)

        rel_type = signal["relation_type"]
        alpha = signal["alpha_signal"]

        edge_classes = ["alpha-edge"]
        if rel_type in ["DIRECT_CAUSE", "ENABLING_CONDITION"]:
            edge_classes.append("causal")
        elif rel_type == "INHIBITING_CONDITION":
            edge_classes.append("inhibiting")

        if abs(alpha) >= 0.5:
            edge_classes.append("high-alpha")

        graph_edge = {
            "data": {
                "id": edge_id,
                "source": source_id,
                "target": target_id,
                "label": format_relation_type(rel_type).upper()[:10],
                "relationType": rel_type,
                "alpha": alpha,
                "alphaDisplay": format_alpha(alpha),
                "confidence": signal["confidence"],
            },
            "classes": " ".join(edge_classes),
        }
        graph_edges.append(graph_edge)

    return {
        "_meta": {
            "description": "Graph data for UI visualization",
            "format": "cytoscape-compatible",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "elements": {
            "nodes": graph_nodes,
            "edges": graph_edges,
        },
        "style": {
            "alpha_color_scale": {
                "min": "#ffffff",
                "max": "#ff0000",
            },
        },
    }


# =============================================================================
# EXPORT: COMBINATION_STRATEGIES.JSON
# =============================================================================


def build_combination_strategies(signals: list[dict]) -> list[dict]:
    """Build multi-leg trade ideas grouped by trigger."""
    # Group signals by trigger event
    by_trigger: dict[str, list[dict]] = {}

    for signal in signals:
        trigger_id = signal["trigger_event"]["id"]
        if trigger_id not in by_trigger:
            by_trigger[trigger_id] = []
        by_trigger[trigger_id].append(signal)

    combinations = []
    combo_counter = 0

    for trigger_id, trigger_signals in by_trigger.items():
        # Only create combinations if multiple consequences
        if len(trigger_signals) < MIN_CONSEQUENCES_FOR_COMBO:
            continue

        # Sort by alpha magnitude
        trigger_signals.sort(key=lambda s: abs(s["alpha_signal"]), reverse=True)

        combo_counter += 1
        trigger_event = trigger_signals[0]["trigger_event"]

        # Calculate combined metrics
        total_alpha = sum(s["alpha_signal"] for s in trigger_signals)
        avg_confidence = sum(s["confidence"] for s in trigger_signals) / len(
            trigger_signals
        )

        # Build consequence list
        consequence_events = []
        for sig in trigger_signals[:5]:  # Limit to top 5 consequences
            consequence_events.append(
                {
                    "event_id": sig["consequence_event"]["id"],
                    "title": truncate_title(sig["consequence_event"]["title"]),
                    "action": sig["strategy"]["action"],
                    "price": sig["consequence_event"]["current_price"],
                    "price_display": format_price(
                        sig["consequence_event"]["current_price"]
                    ),
                    "alpha": sig["alpha_signal"],
                    "alpha_display": format_alpha(sig["alpha_signal"]),
                }
            )

        # Generate combination name based on trigger
        combo_name = f"{truncate_title(trigger_event['title'], 30)} Cascade"

        combination = {
            "id": f"combo_{combo_counter:03d}",
            "name": combo_name,
            "description": f"Multiple consequences from: {trigger_event['title']}",
            "trigger_events": [
                {
                    "event_id": trigger_id,
                    "title": trigger_event["title"],
                    "condition": "YES",
                    "price": trigger_event["current_price"],
                    "price_display": format_price(trigger_event["current_price"]),
                }
            ],
            "consequence_events": consequence_events,
            "combined_expected_value": round(abs(total_alpha), 2),
            "total_alpha": round(total_alpha, 2),
            "total_alpha_display": format_alpha(total_alpha),
            "confidence": round(avg_confidence, 2),
            "consequence_count": len(trigger_signals),
            "execution_notes": (
                f"Monitor {truncate_title(trigger_event['title'], 30)}. "
                f"If trigger probability increases significantly, "
                f"prepare positions in {len(trigger_signals)} consequence markets."
            ),
        }

        combinations.append(combination)

    # Sort by total alpha
    combinations.sort(key=lambda c: abs(c["total_alpha"]), reverse=True)

    return combinations


# =============================================================================
# EXPORT: EXPLORE_VIEW.JSON
# =============================================================================


def build_explore_view(
    signals: list[dict],
    per_event_relations: dict,
    node_lookup: dict,
) -> dict:
    """Build per-event exploration data."""
    explore_events: dict[str, dict] = {}

    # First pass: collect all events from signals
    all_event_ids = set()
    for signal in signals:
        all_event_ids.add(signal["trigger_event"]["id"])
        all_event_ids.add(signal["consequence_event"]["id"])

    # Build signal indexes
    consequences_by_trigger: dict[str, list[dict]] = {}
    causes_by_consequence: dict[str, list[dict]] = {}

    for signal in signals:
        trigger_id = signal["trigger_event"]["id"]
        consequence_id = signal["consequence_event"]["id"]

        if trigger_id not in consequences_by_trigger:
            consequences_by_trigger[trigger_id] = []
        consequences_by_trigger[trigger_id].append(
            {
                "event_id": consequence_id,
                "title": signal["consequence_event"]["title"],
                "relation": signal["relation_type"],
                "alpha": signal["alpha_signal"],
                "hops": 1,
            }
        )

        if consequence_id not in causes_by_consequence:
            causes_by_consequence[consequence_id] = []
        causes_by_consequence[consequence_id].append(
            {
                "event_id": trigger_id,
                "title": signal["trigger_event"]["title"],
                "relation": signal["relation_type"],
                "alpha": signal["alpha_signal"],
                "hops": 1,
            }
        )

    # Build explore view for each event
    for event_id in all_event_ids:
        # Get event info from signals
        event_info = None
        for signal in signals:
            if signal["trigger_event"]["id"] == event_id:
                event_info = signal["trigger_event"]
                break
            if signal["consequence_event"]["id"] == event_id:
                event_info = signal["consequence_event"]
                break

        if not event_info:
            continue

        # Get per_event_relations data if available
        per_event_data = per_event_relations.get(event_id, {})

        # Collect variants from per_event_relations
        variants = []
        for variant in per_event_data.get("variants", []):
            variant_node = node_lookup.get(variant["event_id"], {})
            variants.append(
                {
                    "event_id": variant["event_id"],
                    "title": variant_node.get("title", "Unknown"),
                    "relation": variant.get("relation", "VARIANT"),
                }
            )

        # Calculate total alpha potential
        consequences = consequences_by_trigger.get(event_id, [])
        causes = causes_by_consequence.get(event_id, [])
        total_alpha = sum(abs(c["alpha"]) for c in consequences)
        total_alpha += sum(abs(c["alpha"]) for c in causes)

        explore_events[event_id] = {
            "title": event_info["title"],
            "price": event_info["current_price"],
            "price_display": format_price(event_info["current_price"]),
            "market_url": generate_market_url(event_id),
            "consequences": consequences,
            "causes": causes,
            "variants": variants,
            "related_by_entity": [],  # Would need entity data to populate
            "total_alpha_potential": round(total_alpha, 2),
            "consequence_count": len(consequences),
            "cause_count": len(causes),
        }

    return explore_events


# =============================================================================
# MAIN LOGIC
# =============================================================================


def main() -> None:
    """Main entry point."""
    logger.info("Starting export of UI-ready opportunities")

    # Find latest input directories
    alpha_dir = get_latest_input_dir(INPUT_ALPHA_DIR)
    graph_dir = get_latest_input_dir(INPUT_GRAPH_DIR)
    logger.info(f"Using alpha input: {alpha_dir}")
    logger.info(f"Using graph input: {graph_dir}")

    # Load input data
    alpha_signals_data = load_json(alpha_dir / "alpha_signals.json")
    alpha_ranked_data = load_json(alpha_dir / "alpha_ranked.json")
    relation_graph_data = load_json(graph_dir / "relation_graph.json")
    per_event_relations = load_json(graph_dir / "per_event_relations.json")

    signals = alpha_signals_data.get("signals", [])
    ranked_opportunities = alpha_ranked_data.get("ranked_opportunities", [])
    nodes = relation_graph_data.get("nodes", [])

    logger.info(f"Loaded {len(signals)} alpha signals")
    logger.info(f"Loaded {len(nodes)} graph nodes")

    # Build node lookup
    node_lookup = {node["id"]: node for node in nodes}

    # Sort signals by confidence-adjusted alpha (absolute value)
    sorted_signals = sorted(
        signals,
        key=lambda s: abs(s["confidence_adjusted_alpha"]),
        reverse=True,
    )

    # Build all exports
    logger.info("Building opportunities.json...")
    opportunities = build_opportunities(sorted_signals, per_event_relations)

    logger.info("Building event_graph_ui.json...")
    event_graph_ui = build_event_graph_ui(nodes, signals, per_event_relations)

    logger.info("Building combination_strategies.json...")
    combinations = build_combination_strategies(signals)

    logger.info("Building explore_view.json...")
    explore_events = build_explore_view(signals, per_event_relations, node_lookup)

    # Create output directory
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = SCRIPT_OUTPUT_DIR / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare output data
    opportunities_data = {
        "_meta": {
            "description": "UI-ready alpha opportunities",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "format_version": "1.0",
            "alpha_input_dir": str(alpha_dir),
            "graph_input_dir": str(graph_dir),
        },
        "opportunities": opportunities,
    }

    combination_strategies_data = {
        "_meta": {
            "description": "Multi-event combination strategies",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "combinations": combinations,
    }

    explore_view_data = {
        "_meta": {
            "description": "Per-event exploration data",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "events": explore_events,
    }

    run_summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "alpha_input_dir": str(alpha_dir),
        "graph_input_dir": str(graph_dir),
        "output_dir": str(output_dir),
        "statistics": {
            "opportunities_count": len(opportunities),
            "graph_nodes_count": len(event_graph_ui["elements"]["nodes"]),
            "graph_edges_count": len(event_graph_ui["elements"]["edges"]),
            "combinations_count": len(combinations),
            "explore_events_count": len(explore_events),
        },
    }

    # Save outputs
    save_json(opportunities_data, output_dir / "opportunities.json")
    save_json(event_graph_ui, output_dir / "event_graph_ui.json")
    save_json(combination_strategies_data, output_dir / "combination_strategies.json")
    save_json(explore_view_data, output_dir / "explore_view.json")
    save_json(run_summary, output_dir / "summary.json")

    # Log summary
    logger.info("=== Export Summary ===")
    logger.info(f"Opportunities: {len(opportunities)}")
    logger.info(f"Graph nodes: {len(event_graph_ui['elements']['nodes'])}")
    logger.info(f"Graph edges: {len(event_graph_ui['elements']['edges'])}")
    logger.info(f"Combinations: {len(combinations)}")
    logger.info(f"Explore events: {len(explore_events)}")
    logger.info(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
