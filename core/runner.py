"""
Production pipeline runner.

Single-process incremental pipeline that:
- Loads models once, keeps in memory
- Uses SQLite for O(1) state lookups
- Processes only new events (incremental mode)
- Merges results into accumulated _live/ state

Usage:
    from core.runner import run
    run()           # Incremental (default)
    run(full=True)  # Full reprocessing
"""

import asyncio
import json
from datetime import datetime, timezone

from loguru import logger

from core.models import get_llm_client, preload_models
from core.state import (
    GraphData,
    export_live_data,
    load_state,
)
from core.steps.alpha import run_alpha_detection
from core.steps.embeddings import embed_events
from core.steps.entities import extract_and_process_entities
from core.steps.fetch import extract_prices, fetch_events
from core.steps.prepare import prepare_nlp_data
from core.steps.relations import (
    block_candidate_pairs,
    build_relation_graph,
    classify_causal,
    classify_structural,
    merge_into_graph,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

MAX_LLM_PAIRS = 10000  # Limit LLM calls for causal classification


# =============================================================================
# MAIN RUNNER
# =============================================================================


async def run_async(full: bool = False) -> dict:
    """
    Run the pipeline asynchronously.

    Args:
        full: If True, reprocess everything. If False, incremental.

    Returns:
        Dict with run statistics
    """
    start_time = datetime.now(timezone.utc)
    logger.info(f"Starting pipeline run (mode: {'full' if full else 'incremental'})")

    # Load state
    state = load_state()

    if full:
        logger.warning("Full mode: resetting state...")
        state.reset()

    # Start run tracking
    run_id = state.start_run("full" if full else "refresh")

    try:
        # =====================================================================
        # STEP 1: Fetch all events from API
        # =====================================================================
        logger.info("Step 1: Fetching events from Polymarket API...")
        all_events = await fetch_events()
        logger.info(f"Fetched {len(all_events)} events")

        # =====================================================================
        # STEP 2: Identify new events
        # =====================================================================
        all_ids = [e["id"] for e in all_events]
        new_ids = state.get_new_ids(all_ids)
        new_events = [e for e in all_events if e["id"] in new_ids]

        logger.info(f"Total events: {len(all_events)}, New events: {len(new_events)}")

        # =====================================================================
        # STEP 3: Handle no new events case
        # =====================================================================
        if not new_events and not full:
            logger.info("No new events - updating prices only...")

            # Update prices in state
            prices = extract_prices(all_events)
            state.update_event_prices(prices)

            # Load existing graph and run alpha detection with new prices
            graph = state.get_graph()
            if graph.nodes:
                opportunities, summary = run_alpha_detection(
                    graph.to_dict(), all_events
                )
                export_live_data(state, all_events, opportunities)

                state.complete_run(run_id, len(all_events), 0, "completed")

                elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                logger.info(f"Price update complete in {elapsed:.1f}s")

                return {
                    "mode": "price_update",
                    "total_events": len(all_events),
                    "new_events": 0,
                    "opportunities": len(opportunities),
                    "elapsed_seconds": elapsed,
                }

            logger.warning("No existing graph - nothing to update")
            state.complete_run(run_id, 0, 0, "skipped")
            return {"mode": "skipped", "reason": "no_graph"}

        # =====================================================================
        # STEP 4: Preload models (do this before heavy processing)
        # =====================================================================
        logger.info("Loading ML models...")
        preload_models()

        # =====================================================================
        # STEP 5: Prepare NLP data for new events
        # =====================================================================
        logger.info("Step 5: Preparing NLP data...")
        nlp_events = prepare_nlp_data(new_events)
        logger.info(f"Prepared {len(nlp_events)} events for NLP")

        # =====================================================================
        # STEP 6: Extract entities
        # =====================================================================
        logger.info("Step 6: Extracting entities...")
        entities = extract_and_process_entities(nlp_events, state)
        logger.info(f"Extracted {len(entities)} entities")

        # =====================================================================
        # STEP 6.5: Extract event semantics
        # =====================================================================
        logger.info("Step 6.5: Extracting event semantics...")
        from core.steps.semantics import (
            extract_event_semantics,
            get_semantics_for_prioritization,
        )

        semantics_by_id = await extract_event_semantics(nlp_events, state)
        semantics_for_pairs = get_semantics_for_prioritization(semantics_by_id)
        logger.info(f"Extracted semantics for {len(semantics_by_id)} events")

        # =====================================================================
        # STEP 7: Generate embeddings
        # =====================================================================
        logger.info("Step 7: Generating embeddings...")
        new_embeddings, new_event_ids = embed_events(nlp_events, state)
        logger.info(f"Generated embeddings for {len(new_event_ids)} events")

        # =====================================================================
        # STEP 7.5: Enrich quality
        # =====================================================================
        logger.info("Step 7.5: Enriching quality...")
        from core.steps.quality import enrich_events_quality

        entity_sets = {e["id"]: e.get("entities", []) for e in nlp_events}
        _enriched_events, negation_pairs = enrich_events_quality(
            nlp_events, entity_sets
        )
        logger.info(f"Quality enriched: {len(negation_pairs)} negation pairs found")

        # =====================================================================
        # STEP 8: Find candidate pairs (new vs all)
        # =====================================================================
        logger.info("Step 8: Finding candidate pairs...")

        # Get all embeddings and event IDs
        all_embeddings, all_event_ids = state.get_embeddings()

        if all_embeddings is None or len(all_embeddings) == 0:
            # First run - use only new embeddings
            all_embeddings = new_embeddings
            all_event_ids = new_event_ids

        # Prepare all events lookup
        existing_events = state.get_all_events()
        all_events_for_pairs = existing_events + nlp_events

        candidate_pairs = block_candidate_pairs(
            new_events=nlp_events,
            all_events=all_events_for_pairs,
            new_embeddings=new_embeddings,
            all_embeddings=all_embeddings,
            all_event_ids=all_event_ids,
        )
        logger.info(f"Found {len(candidate_pairs)} candidate pairs")

        # =====================================================================
        # STEP 9: Classify structural relations
        # =====================================================================
        logger.info("Step 9: Classifying structural relations...")
        events_by_id = {e["id"]: e for e in all_events_for_pairs}
        structural_relations = classify_structural(candidate_pairs, events_by_id)

        # Add negation pairs as MUTUALLY_EXCLUSIVE (from quality enrichment)
        for pair in negation_pairs:
            structural_relations.append(
                {
                    "source_id": pair.event_id_a,
                    "target_id": pair.event_id_b,
                    "relation_type": "MUTUALLY_EXCLUSIVE",
                    "confidence": 0.9,
                    "classification_method": "negation_detection",
                }
            )

        logger.info(f"Found {len(structural_relations)} structural relations")

        # =====================================================================
        # STEP 10: Classify causal relations (LLM)
        # =====================================================================
        logger.info("Step 10: Classifying causal relations (LLM)...")

        # Filter pairs not already classified as structural
        structural_pairs = {
            (r["source_id"], r["target_id"]) for r in structural_relations
        }
        pairs_for_causal = [
            p
            for p in candidate_pairs
            if (p["event_a_id"], p["event_b_id"]) not in structural_pairs
            and (p["event_b_id"], p["event_a_id"]) not in structural_pairs
        ][:MAX_LLM_PAIRS]

        causal_relations = await classify_causal(
            pairs_for_causal,
            events_by_id,
            semantics_by_id=semantics_for_pairs,
            max_pairs=MAX_LLM_PAIRS,
        )
        logger.info(f"Found {len(causal_relations)} causal relations")

        # =====================================================================
        # STEP 11: Build/merge graph
        # =====================================================================
        logger.info("Step 11: Building relation graph...")

        # Get existing graph
        existing_graph = state.get_graph()

        if existing_graph.nodes and not full:
            # Incremental: merge into existing graph
            new_graph_nodes = [
                {
                    "id": e["id"],
                    "title": e.get("title", ""),
                    "current_price": extract_prices([e]).get(e["id"], 0.5),
                }
                for e in nlp_events
            ]
            new_graph_edges = [
                {
                    "source": r["source_id"],
                    "target": r["target_id"],
                    "relation_type": r["relation_type"],
                    "confidence": r.get("confidence", 0.5),
                }
                for r in structural_relations + causal_relations
            ]

            merged = merge_into_graph(
                existing_graph.to_dict(), new_graph_nodes, new_graph_edges
            )
            graph = GraphData.from_dict(merged)
        else:
            # Full: build new graph
            graph_dict = build_relation_graph(
                all_events_for_pairs,
                structural_relations,
                causal_relations,
            )
            graph = GraphData.from_dict(graph_dict)

        # Save graph
        state.save_graph(graph)
        logger.info(f"Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

        # =====================================================================
        # STEP 12: Alpha detection
        # =====================================================================
        logger.info("Step 12: Detecting alpha opportunities...")
        opportunities, summary = run_alpha_detection(graph.to_dict(), all_events)
        logger.info(f"Found {len(opportunities)} alpha opportunities")

        # =====================================================================
        # STEP 13: Save new events to state
        # =====================================================================
        logger.info("Step 13: Saving state...")
        state.add_events(nlp_events)

        # =====================================================================
        # STEP 14: Export to _live/
        # =====================================================================
        logger.info("Step 14: Exporting to _live/...")
        export_live_data(state, all_events, opportunities)

        # Complete run
        state.complete_run(run_id, len(all_events), len(new_events), "completed")

        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        logger.info(f"Pipeline complete in {elapsed:.1f}s")

        return {
            "mode": "full" if full else "incremental",
            "total_events": len(all_events),
            "new_events": len(new_events),
            "entities": len(entities),
            "structural_relations": len(structural_relations),
            "causal_relations": len(causal_relations),
            "graph_nodes": len(graph.nodes),
            "graph_edges": len(graph.edges),
            "opportunities": len(opportunities),
            "elapsed_seconds": elapsed,
        }

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        state.complete_run(run_id, 0, 0, "failed")
        raise

    finally:
        # Cleanup LLM client
        llm = get_llm_client()
        await llm.close()
        state.close()


def run(full: bool = False) -> dict:
    """
    Run the pipeline synchronously.

    Args:
        full: If True, reprocess everything. If False, incremental.

    Returns:
        Dict with run statistics
    """
    return asyncio.run(run_async(full))


# =============================================================================
# CLI ENTRY POINT
# =============================================================================


def main():
    """CLI entry point."""
    import sys

    full = "--full" in sys.argv or "-f" in sys.argv

    try:
        result = run(full=full)
        print(json.dumps(result, indent=2))
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
