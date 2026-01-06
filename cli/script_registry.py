"""Registry of all pipeline scripts with metadata."""

from dataclasses import dataclass


@dataclass
class ScriptInfo:
    """Metadata for a pipeline script."""

    step: str
    name: str
    file: str
    description: str
    inputs: list[str]  # List of step IDs this script depends on
    outputs: list[str]  # Output files produced
    config_vars: list[str]  # Configuration variables that can be overridden


# All pipeline scripts in execution order
SCRIPTS: dict[str, ScriptInfo] = {
    "01": ScriptInfo(
        step="01",
        name="fetch_events",
        file="01_fetch_events.py",
        description="Fetch active events from Polymarket API",
        inputs=[],
        outputs=["events.json", "tag_info.json", "summary.json"],
        config_vars=["TARGET_TAG_SLUG", "CLOSED", "PAGE_SIZE"],
    ),
    "02": ScriptInfo(
        step="02",
        name="prepare_nlp_data",
        file="02_prepare_nlp_data.py",
        description="Extract text fields and apply quality filters",
        inputs=["01"],
        outputs=["nlp_events.json", "summary.json"],
        config_vars=["FILTER_PLACEHOLDERS", "FILTER_EMPTY_MARKETS"],
    ),
    "03_1": ScriptInfo(
        step="03_1",
        name="extract_entities",
        file="03_1_extract_entities.py",
        description="Extract named entities using GLiNER2",
        inputs=["02"],
        outputs=[
            "entities_raw.json",
            "entities_by_source.json",
            "entities_unique.json",
            "summary.json",
        ],
        config_vars=["CONFIDENCE_THRESHOLD", "MAX_EVENTS"],
    ),
    "03_2": ScriptInfo(
        step="03_2",
        name="dedupe_entities",
        file="03_2_dedupe_entities.py",
        description="Deduplicate and fuzzy-cluster entities",
        inputs=["03_1"],
        outputs=[
            "entities_raw.json",
            "entities_by_source.json",
            "entities_unique.json",
            "fuzzy_merge_mappings.json",
            "summary.json",
        ],
        config_vars=["FUZZY_MIN_FREQUENCY"],
    ),
    "03_3": ScriptInfo(
        step="03_3",
        name="normalize_entities",
        file="03_3_normalize_entities.py",
        description="LLM-based entity normalization and noise filtering",
        inputs=["03_2"],
        outputs=[
            "entities_normalized.json",
            "entities_noise.json",
            "merge_mappings.json",
            "summary.json",
        ],
        config_vars=["LLM_MODEL", "BATCH_SIZE"],
    ),
    "03_4": ScriptInfo(
        step="03_4",
        name="extract_relations",
        file="03_4_extract_relations.py",
        description="Extract relations between entities using GLiNER2",
        inputs=["02", "03_3"],
        outputs=[
            "relations_raw.json",
            "relations_by_event.json",
            "relations_unique.json",
            "summary.json",
        ],
        config_vars=["CONFIDENCE_THRESHOLD", "MAX_EVENTS"],
    ),
    "04_1": ScriptInfo(
        step="04_1",
        name="extract_event_semantics",
        file="04_1_extract_event_semantics.py",
        description="Parse event titles into structured semantic components",
        inputs=["02", "03_3"],
        outputs=["event_semantics.json", "summary.json"],
        config_vars=["LLM_MODEL", "BATCH_SIZE"],
    ),
    "04_2": ScriptInfo(
        step="04_2",
        name="embed_events",
        file="04_2_embed_events.py",
        description="Generate semantic embeddings for events",
        inputs=["02", "03_2", "03_3"],
        outputs=["embeddings.npy", "metadata.json", "entity_sets.json", "summary.json"],
        config_vars=["MODEL", "BATCH_SIZE"],
    ),
    "04_3": ScriptInfo(
        step="04_3",
        name="cluster_events",
        file="04_3_cluster_events.py",
        description="Cluster events by concept, topic, and entity dimensions",
        inputs=["04_1", "04_2", "03_3"],
        outputs=["event_clusters.json", "event_memberships.json", "summary.json"],
        config_vars=["CONCEPT_MIN_GROUP_SIZE", "HDBSCAN_MIN_CLUSTER_SIZE"],
    ),
    "05_0": ScriptInfo(
        step="05_0",
        name="enrich_quality",
        file="05_0_enrich_quality.py",
        description="Add quality flags and detect negation pairs",
        inputs=["02", "04_2"],
        outputs=["quality_enriched_events.json", "negation_pairs.json", "summary.json"],
        config_vars=[],
    ),
    "05_1": ScriptInfo(
        step="05_1",
        name="block_candidate_pairs",
        file="05_1_block_candidate_pairs.py",
        description="Fast candidate pair blocking using FAISS",
        inputs=["05_0", "04_2"],
        outputs=["candidate_pairs.json", "entity_inverted_index.json", "summary.json"],
        config_vars=["FAISS_TOP_K", "BLOCKING_THRESHOLD"],
    ),
    "05_2": ScriptInfo(
        step="05_2",
        name="classify_structural",
        file="05_2_classify_structural.py",
        description="Rule-based structural relation classification",
        inputs=["05_1", "04_1", "05_0"],
        outputs=["structural_relations.json", "summary.json"],
        config_vars=[],
    ),
    "05_3": ScriptInfo(
        step="05_3",
        name="classify_causal",
        file="05_3_classify_causal.py",
        description="LLM-based causal relation classification",
        inputs=["05_2", "04_1"],
        outputs=["causal_relations.json", "summary.json"],
        config_vars=["LLM_MODEL", "LLM_BATCH_SIZE", "MAX_LLM_PAIRS"],
    ),
    "05_4": ScriptInfo(
        step="05_4",
        name="build_relation_graph",
        file="05_4_build_relation_graph.py",
        description="Build unified relation graph with transitive closure",
        inputs=["05_2", "05_3", "01", "04_1"],
        outputs=["relation_graph.json", "adjacency_list.json", "summary.json"],
        config_vars=["MAX_HOPS", "EXPORT_GEXF"],
    ),
    "06_1": ScriptInfo(
        step="06_1",
        name="compute_conditionals",
        file="06_1_compute_conditionals.py",
        description="Compute P(B|A) conditional probabilities",
        inputs=["05_4"],
        outputs=["conditional_probabilities.json", "summary.json"],
        config_vars=[],
    ),
    "06_2": ScriptInfo(
        step="06_2",
        name="detect_alpha",
        file="06_2_detect_alpha.py",
        description="Detect alpha opportunities from model vs market divergence",
        inputs=["06_1"],
        outputs=["alpha_signals.json", "summary.json"],
        config_vars=["MIN_ALPHA_THRESHOLD", "MIN_CONFIDENCE"],
    ),
    "06_3": ScriptInfo(
        step="06_3",
        name="export_opportunities",
        file="06_3_export_opportunities.py",
        description="Export alpha opportunities in UI-ready formats",
        inputs=["06_2", "05_4"],
        outputs=[
            "opportunities.json",
            "event_graph_ui.json",
            "combination_strategies.json",
            "summary.json",
        ],
        config_vars=["MAX_TITLE_LENGTH"],
    ),
}

# Execution order for the full pipeline
PIPELINE_ORDER = [
    "01",
    "02",
    "03_1",
    "03_2",
    "03_3",
    "03_4",
    "04_1",
    "04_2",
    "04_3",
    "05_0",
    "05_1",
    "05_2",
    "05_3",
    "05_4",
    "06_1",
    "06_2",
    "06_3",
]


def get_script(step: str) -> ScriptInfo:
    """Get script info by step ID."""
    if step not in SCRIPTS:
        raise ValueError(
            f"Unknown step: {step}. Valid steps: {', '.join(PIPELINE_ORDER)}"
        )
    return SCRIPTS[step]


def get_steps_in_range(from_step: str, to_step: str) -> list[str]:
    """Get all steps between from_step and to_step (inclusive)."""
    if from_step not in PIPELINE_ORDER:
        raise ValueError(f"Unknown from_step: {from_step}")
    if to_step not in PIPELINE_ORDER:
        raise ValueError(f"Unknown to_step: {to_step}")

    from_idx = PIPELINE_ORDER.index(from_step)
    to_idx = PIPELINE_ORDER.index(to_step)

    if from_idx > to_idx:
        raise ValueError(
            f"from_step ({from_step}) must come before to_step ({to_step})"
        )

    return PIPELINE_ORDER[from_idx : to_idx + 1]


def get_output_dir_name(step: str) -> str:
    """Get the output directory name for a step."""
    script = get_script(step)
    # Pattern: NN_name -> data/NN_name/
    return f"{script.step.replace('_', '_')}_{script.name.replace('_', '_')}"
