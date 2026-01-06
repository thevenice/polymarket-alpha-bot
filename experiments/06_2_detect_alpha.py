"""
Detect alpha opportunities from model vs market divergence.

Pipeline: 06_1_compute_conditionals -> 06_2_detect_alpha -> 06_3_export_opportunities

This script computes alpha signals by comparing model conditional probabilities
against market-implied probabilities, generating trading strategies for mispriced
conditional relationships.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_CONDITIONALS_DIR = DATA_DIR / "06_1_compute_conditionals"
SCRIPT_OUTPUT_DIR = DATA_DIR / "06_2_detect_alpha"

# Alpha filtering thresholds
MIN_ALPHA_THRESHOLD = 0.10  # Minimum alpha to report
MIN_CONFIDENCE = 0.60  # Minimum model confidence
MIN_TRIGGER_PRICE = 0.02  # Trigger must have some probability
MAX_TRIGGER_PRICE = 0.80  # Trigger shouldn't be near-certain

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


# =============================================================================
# ALPHA COMPUTATION
# =============================================================================


def compute_alpha(edge: dict) -> dict:
    """
    Compute alpha signal from model vs market divergence.

    Alpha = P(B|A)_model - P(B|A)_market

    Under market independence assumption:
    P(B|A)_market = P(B) = target_price

    Returns:
        dict with alpha_signal, alpha_direction, confidence_adjusted_alpha
    """
    model_conditional = edge["model_P_B_given_A"]
    market_implied = edge["market_implied_P_B_given_A"]

    alpha_signal = model_conditional - market_implied
    confidence = edge.get("model_confidence", 0.7)

    return {
        "alpha_signal": round(alpha_signal, 4),
        "alpha_direction": (
            "BUY_CONSEQUENCE_IF_TRIGGER"
            if alpha_signal > 0
            else "SELL_CONSEQUENCE_IF_TRIGGER"
        ),
        "confidence_adjusted_alpha": round(alpha_signal * confidence, 4),
    }


def generate_strategy(edge: dict, alpha: dict) -> dict:
    """Generate human-readable trading strategy."""
    source_title = edge["source_title"]
    target_title = edge["target_title"]
    target_id = edge["target_id"]

    if alpha["alpha_signal"] > 0:
        # Model thinks B is more likely given A than market implies
        return {
            "description": f"If {source_title} resolves YES, buy {target_title} YES",
            "trigger_condition": "TRIGGER_YES",
            "action": "BUY",
            "target_market": target_id,
            "target_outcome": "YES",
        }
    else:
        # Model thinks B is less likely given A than market implies
        return {
            "description": f"If {source_title} resolves YES, sell {target_title} YES (or buy NO)",
            "trigger_condition": "TRIGGER_YES",
            "action": "SELL",
            "target_market": target_id,
            "target_outcome": "YES",
        }


def compute_expected_value(edge: dict, alpha: dict) -> dict:
    """
    Compute expected value of strategy assuming trigger resolves YES.

    EV of strategy:
    - If we buy consequence at market price P_market:
      - Win (1 - P_market) with probability P_model
      - Lose P_market with probability (1 - P_model)

    EV = P_model * (1 - P_market) - (1 - P_model) * P_market
       = P_model - P_market
       = alpha_signal

    Per dollar at risk:
    EV_per_dollar = alpha_signal / P_market
    """
    market_price = edge["target_price"]
    alpha_signal = alpha["alpha_signal"]

    if alpha_signal > 0:
        # For BUY, risk is market_price
        ev_per_dollar = alpha_signal / market_price if market_price > 0 else 0
    else:
        # For SELL, risk is (1 - market_price)
        risk = 1 - market_price
        ev_per_dollar = abs(alpha_signal) / risk if risk > 0 else 0

    return {
        "per_dollar_at_risk": round(ev_per_dollar, 2),
        "assuming_trigger_resolves_yes": True,
    }


def should_report_signal(edge: dict, alpha: dict) -> bool:
    """Check if signal passes minimum thresholds."""
    source_price = edge.get("source_price", 0)
    model_confidence = edge.get("model_confidence", 0)

    return (
        abs(alpha["alpha_signal"]) >= MIN_ALPHA_THRESHOLD
        and model_confidence >= MIN_CONFIDENCE
        and source_price >= MIN_TRIGGER_PRICE
        and source_price <= MAX_TRIGGER_PRICE
    )


# =============================================================================
# MAIN LOGIC
# =============================================================================


def process_edges(edges: list[dict]) -> list[dict]:
    """Process all edges and generate alpha signals."""
    signals = []
    signal_counter = 0

    for edge in edges:
        # Compute alpha
        alpha = compute_alpha(edge)

        # Filter by thresholds
        if not should_report_signal(edge, alpha):
            continue

        signal_counter += 1
        signal_id = f"alpha_{signal_counter:04d}"

        # Generate strategy
        strategy = generate_strategy(edge, alpha)

        # Compute expected value
        expected_value = compute_expected_value(edge, alpha)

        signal = {
            "signal_id": signal_id,
            "trigger_event": {
                "id": edge["source_id"],
                "title": edge["source_title"],
                "current_price": edge["source_price"],
            },
            "consequence_event": {
                "id": edge["target_id"],
                "title": edge["target_title"],
                "current_price": edge["target_price"],
            },
            "relation_type": edge["relation_type"],
            "direction": "forward",
            "model_conditional": edge["model_P_B_given_A"],
            "market_implied": edge["market_implied_P_B_given_A"],
            "alpha_signal": alpha["alpha_signal"],
            "alpha_direction": alpha["alpha_direction"],
            "confidence": edge.get("model_confidence", 0.7),
            "confidence_adjusted_alpha": alpha["confidence_adjusted_alpha"],
            "strategy": strategy,
            "expected_value": expected_value,
        }

        signals.append(signal)

    return signals


def compute_summary(signals: list[dict]) -> dict:
    """Compute summary statistics for signals."""
    if not signals:
        return {
            "total_signals": 0,
            "avg_alpha": 0,
            "signals_by_direction": {},
        }

    alpha_values = [s["alpha_signal"] for s in signals]
    by_direction: dict[str, int] = {}

    for s in signals:
        direction = s["alpha_direction"]
        by_direction[direction] = by_direction.get(direction, 0) + 1

    return {
        "total_signals": len(signals),
        "avg_alpha": round(sum(alpha_values) / len(alpha_values), 4),
        "min_alpha": round(min(alpha_values), 4),
        "max_alpha": round(max(alpha_values), 4),
        "signals_by_direction": by_direction,
    }


def generate_ranked_opportunities(signals: list[dict]) -> list[dict]:
    """Rank signals by confidence-adjusted alpha and return top opportunities."""
    # Sort by absolute confidence_adjusted_alpha descending
    sorted_signals = sorted(
        signals, key=lambda s: abs(s["confidence_adjusted_alpha"]), reverse=True
    )

    ranked = []
    for rank, signal in enumerate(sorted_signals, start=1):
        ranked.append(
            {
                "rank": rank,
                "signal_id": signal["signal_id"],
                "alpha_signal": signal["alpha_signal"],
                "confidence_adjusted_alpha": signal["confidence_adjusted_alpha"],
                "trigger_title": signal["trigger_event"]["title"],
                "consequence_title": signal["consequence_event"]["title"],
                "strategy_summary": f"{signal['strategy']['action']} {signal['consequence_event']['title'][:30]}... if trigger YES",
            }
        )

    return ranked


def main() -> None:
    """Main entry point."""
    logger.info("Starting alpha detection")

    # Find latest input directory
    input_dir = get_latest_input_dir(INPUT_CONDITIONALS_DIR)
    logger.info(f"Using input directory: {input_dir}")

    # Load input data
    conditional_data = load_json(input_dir / "conditional_probabilities.json")
    edges = conditional_data.get("edges", [])
    logger.info(f"Loaded {len(edges)} edges from conditional probabilities")

    # Process edges and generate signals
    signals = process_edges(edges)
    logger.info(f"Generated {len(signals)} alpha signals passing filters")

    # Compute summary statistics
    summary_stats = compute_summary(signals)

    # Generate ranked opportunities
    ranked_opportunities = generate_ranked_opportunities(signals)

    # Create output directory
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = SCRIPT_OUTPUT_DIR / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare output data
    alpha_signals = {
        "_meta": {
            "description": "Alpha signals from model vs market divergence",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_directory": str(input_dir),
            "min_alpha_threshold": MIN_ALPHA_THRESHOLD,
            "min_confidence": MIN_CONFIDENCE,
            "min_trigger_price": MIN_TRIGGER_PRICE,
            "max_trigger_price": MAX_TRIGGER_PRICE,
        },
        "signals": signals,
        "summary": summary_stats,
    }

    alpha_ranked = {
        "_meta": {
            "description": "Alpha signals ranked by confidence-adjusted alpha",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "ranked_opportunities": ranked_opportunities,
    }

    run_summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "configuration": {
            "min_alpha_threshold": MIN_ALPHA_THRESHOLD,
            "min_confidence": MIN_CONFIDENCE,
            "min_trigger_price": MIN_TRIGGER_PRICE,
            "max_trigger_price": MAX_TRIGGER_PRICE,
        },
        "statistics": summary_stats,
    }

    # Save outputs
    save_json(alpha_signals, output_dir / "alpha_signals.json")
    save_json(alpha_ranked, output_dir / "alpha_ranked.json")
    save_json(run_summary, output_dir / "summary.json")

    # Log summary
    logger.info("=== Summary ===")
    logger.info(f"Total signals: {summary_stats['total_signals']}")
    if summary_stats["total_signals"] > 0:
        logger.info(f"Avg alpha: {summary_stats['avg_alpha']:.4f}")
        logger.info(f"Min alpha: {summary_stats['min_alpha']:.4f}")
        logger.info(f"Max alpha: {summary_stats['max_alpha']:.4f}")
        logger.info(f"By direction: {summary_stats['signals_by_direction']}")

        # Show top 5 opportunities
        logger.info("=== Top 5 Opportunities ===")
        for opp in ranked_opportunities[:5]:
            logger.info(
                f"  #{opp['rank']}: alpha={opp['alpha_signal']:.2f}, "
                f"adj={opp['confidence_adjusted_alpha']:.2f} - {opp['strategy_summary']}"
            )

    logger.info(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
