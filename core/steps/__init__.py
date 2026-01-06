"""Pipeline steps as reusable functions."""

from core.steps.fetch import fetch_events, fetch_events_sync, extract_prices
from core.steps.prepare import prepare_nlp_data, extract_texts_for_ner
from core.steps.entities import (
    extract_and_process_entities,
    get_entities_by_event,
)
from core.steps.embeddings import embed_events, get_similar_events
from core.steps.relations import (
    block_candidate_pairs,
    classify_structural,
    classify_causal,
    build_relation_graph,
    merge_into_graph,
)
from core.steps.alpha import run_alpha_detection, compute_conditionals, detect_alpha

__all__ = [
    # Fetch
    "fetch_events",
    "fetch_events_sync",
    "extract_prices",
    # Prepare
    "prepare_nlp_data",
    "extract_texts_for_ner",
    # Entities
    "extract_and_process_entities",
    "get_entities_by_event",
    # Embeddings
    "embed_events",
    "get_similar_events",
    # Relations
    "block_candidate_pairs",
    "classify_structural",
    "classify_causal",
    "build_relation_graph",
    "merge_into_graph",
    # Alpha
    "run_alpha_detection",
    "compute_conditionals",
    "detect_alpha",
]
