"""
Build unified relation graph from structural and causal relations.

Pipeline: 05_2_classify_structural + 05_3_classify_causal + 01_fetch_events + 04_1_extract_event_semantics
          -> 05_4_build_relation_graph

This script:
1. Loads structural relations from 05_2_classify_structural
2. Loads causal relations from 05_3_classify_causal
3. Loads event metadata from 01_fetch_events (for prices)
4. Loads event semantics from 04_1_extract_event_semantics (for entities/outcome_states)
5. Combines into unified relation graph
6. Builds adjacency list for quick lookups
7. Discovers transitive relation chains (up to max_hops=2)
8. Outputs per-event relation summaries
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "data"

# Input directories
INPUT_STRUCTURAL_DIR = DATA_DIR / "05_2_classify_structural"
INPUT_CAUSAL_DIR = DATA_DIR / "05_3_classify_causal"
INPUT_EVENTS_DIR = DATA_DIR / "01_fetch_events"
INPUT_SEMANTICS_DIR = DATA_DIR / "04_1_extract_event_semantics"

# Use latest run folder if not specified
INPUT_RUN_FOLDER: str | None = None

# Output
SCRIPT_OUTPUT_DIR = DATA_DIR / "05_4_build_relation_graph"

# Transitive relation settings
MAX_HOPS = 2
MIN_CONFIDENCE_TRANSITIVE = 0.3

# GEXF export for Gephi visualization
EXPORT_GEXF = True

# Causal relation types that can form transitive chains
CAUSAL_TYPES = {
    "DIRECT_CAUSE",
    "ENABLING_CONDITION",
    "INHIBITING_CONDITION",
    "REQUIRES",
    "CORRELATED",
}

# Transitive inference rules: (rel1, rel2) -> inferred_rel
TRANSITIVE_INFERENCE_RULES: dict[tuple[str, str], str] = {
    # Structural
    ("THRESHOLD_VARIANT", "THRESHOLD_VARIANT"): "THRESHOLD_VARIANT",
    ("TIMEFRAME_VARIANT", "TIMEFRAME_VARIANT"): "TIMEFRAME_VARIANT",
    ("SERIES_MEMBER", "SERIES_MEMBER"): "SERIES_MEMBER",
    ("HIERARCHICAL", "HIERARCHICAL"): "HIERARCHICAL",
    # Causal chains
    ("DIRECT_CAUSE", "DIRECT_CAUSE"): "INDIRECT_CAUSE",
    ("DIRECT_CAUSE", "ENABLING_CONDITION"): "ENABLING_CONDITION",
    ("ENABLING_CONDITION", "DIRECT_CAUSE"): "ENABLING_CONDITION",
    ("ENABLING_CONDITION", "ENABLING_CONDITION"): "ENABLING_CONDITION",
    ("REQUIRES", "REQUIRES"): "REQUIRES",
    ("REQUIRES", "ENABLING_CONDITION"): "ENABLING_CONDITION",
    ("DIRECT_CAUSE", "REQUIRES"): "REQUIRES",
}

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
class GraphNode:
    """Node in the relation graph."""

    id: str
    title: str
    entities: list[str] = field(default_factory=list)
    outcome_states: list[str] = field(default_factory=list)
    markets: list[dict] = field(default_factory=list)
    current_price: float | None = None
    end_date: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "entities": self.entities,
            "outcome_states": self.outcome_states,
            "markets": self.markets,
            "current_price": self.current_price,
            "end_date": self.end_date,
        }


@dataclass
class GraphEdge:
    """Edge in the relation graph."""

    source: str
    target: str
    relation_type: str
    relation_level: int
    direction: str
    confidence: float
    classification_source: str  # "structural" or "causal"
    implied_conditional: dict | None = None
    evidence: dict | None = None

    def to_dict(self) -> dict:
        result = {
            "source": self.source,
            "target": self.target,
            "relation_type": self.relation_type,
            "relation_level": self.relation_level,
            "direction": self.direction,
            "confidence": round(self.confidence, 4),
            "classification_source": self.classification_source,
        }
        if self.implied_conditional:
            result["implied_conditional"] = self.implied_conditional
        if self.evidence:
            result["evidence"] = self.evidence
        return result


@dataclass
class TransitiveChain:
    """A transitive causal chain."""

    path: list[str]
    relations: list[str]
    chain_type: str
    total_confidence: float
    implied_P_end_given_start: float | None = None

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "relations": self.relations,
            "chain_type": self.chain_type,
            "total_confidence": round(self.total_confidence, 4),
            "implied_P_end_given_start": (
                round(self.implied_P_end_given_start, 4)
                if self.implied_P_end_given_start
                else None
            ),
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


def get_primary_price(event: dict) -> float | None:
    """Extract the primary Yes price from an event."""
    markets = event.get("markets", [])
    if not markets:
        return None

    # Get the first market's Yes price
    first_market = markets[0]
    outcome_prices = first_market.get("outcomePrices", [])
    if outcome_prices:
        try:
            return float(outcome_prices[0])
        except (ValueError, TypeError, IndexError):
            pass
    return None


def extract_entities_from_semantics(semantics: dict) -> list[str]:
    """Extract entity list from event semantics."""
    entities = []
    if semantics.get("subject_entity"):
        entities.append(semantics["subject_entity"])
    if semantics.get("object_entity"):
        entities.append(semantics["object_entity"])
    return entities


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================


def build_nodes(
    events: list[dict], semantics_by_id: dict[str, dict]
) -> dict[str, GraphNode]:
    """Build graph nodes from events with metadata."""
    nodes: dict[str, GraphNode] = {}

    for event in events:
        event_id = str(event["id"])
        sem = semantics_by_id.get(event_id, {})
        sem_data = sem.get("semantics", {})

        # Extract markets with prices
        markets = []
        for m in event.get("markets", []):
            markets.append(
                {
                    "id": m.get("id", ""),
                    "question": m.get("question", ""),
                    "outcomes": m.get("outcomes", []),
                    "prices": m.get("outcomePrices", []),
                }
            )

        nodes[event_id] = GraphNode(
            id=event_id,
            title=event.get("title", ""),
            entities=extract_entities_from_semantics(sem_data),
            outcome_states=sem_data.get("outcome_states", []),
            markets=markets,
            current_price=get_primary_price(event),
            end_date=event.get("endDate"),
        )

    return nodes


def build_edges(
    structural_relations: list[dict], causal_relations: list[dict]
) -> list[GraphEdge]:
    """Combine structural and causal relations into unified edge list."""
    edges: list[GraphEdge] = []
    seen_pairs: set[tuple[str, str]] = set()

    # Add structural edges
    for rel in structural_relations:
        source_id = str(rel["source_id"])
        target_id = str(rel["target_id"])
        pair = (source_id, target_id)

        if pair not in seen_pairs:
            seen_pairs.add(pair)
            edges.append(
                GraphEdge(
                    source=source_id,
                    target=target_id,
                    relation_type=rel["relation_type"],
                    relation_level=rel.get("relation_level", 1),
                    direction=rel.get("direction", "none"),
                    confidence=rel.get("confidence", 1.0),
                    classification_source="structural",
                    evidence=rel.get("evidence"),
                )
            )

    # Add causal edges (skip INDEPENDENT)
    for rel in causal_relations:
        if rel["relation_type"] == "INDEPENDENT":
            continue

        source_id = str(rel["source_id"])
        target_id = str(rel["target_id"])
        pair = (source_id, target_id)

        if pair not in seen_pairs:
            seen_pairs.add(pair)
            edges.append(
                GraphEdge(
                    source=source_id,
                    target=target_id,
                    relation_type=rel["relation_type"],
                    relation_level=rel.get("relation_level", 3),
                    direction=rel.get("direction", "forward"),
                    confidence=rel.get("confidence", 0.5),
                    classification_source="causal",
                    implied_conditional=rel.get("implied_conditional"),
                )
            )

    return edges


def build_adjacency_list(edges: list[GraphEdge]) -> dict:
    """Build adjacency lists for quick lookups."""
    outgoing: dict[str, list[dict]] = defaultdict(list)
    incoming: dict[str, list[dict]] = defaultdict(list)

    for edge in edges:
        outgoing[edge.source].append(
            {
                "target": edge.target,
                "relation": edge.relation_type,
                "confidence": edge.confidence,
            }
        )
        incoming[edge.target].append(
            {
                "source": edge.source,
                "relation": edge.relation_type,
                "confidence": edge.confidence,
            }
        )

    return {"outgoing": dict(outgoing), "incoming": dict(incoming)}


def build_networkx_graph(
    nodes: dict[str, GraphNode], edges: list[GraphEdge]
) -> "nx.DiGraph":
    """Build NetworkX directed graph for export."""
    import networkx as nx

    G = nx.DiGraph()

    # Add nodes with attributes
    for node_id, node in nodes.items():
        G.add_node(
            node_id,
            title=node.title,
            entities=",".join(node.entities) if node.entities else "",
            outcome_states=",".join(node.outcome_states) if node.outcome_states else "",
            current_price=node.current_price if node.current_price else 0.0,
            end_date=node.end_date if node.end_date else "",
        )

    # Add edges with attributes
    for edge in edges:
        G.add_edge(
            edge.source,
            edge.target,
            relation_type=edge.relation_type,
            relation_level=edge.relation_level,
            direction=edge.direction,
            confidence=round(edge.confidence, 4),
            classification_source=edge.classification_source,
        )

    return G


# =============================================================================
# TRANSITIVE INFERENCE
# =============================================================================


def find_transitive_chains(
    edges: list[GraphEdge], max_hops: int = 2, min_confidence: float = 0.3
) -> list[TransitiveChain]:
    """
    Find A -> B -> C chains for causal cascade detection.

    Uses BFS to find paths up to max_hops, then checks if we can infer
    a transitive relation based on the inference rules.
    """
    # Build adjacency for traversal (only forward/bidirectional edges)
    graph: dict[str, list[tuple[str, str, float, dict | None]]] = defaultdict(list)
    for edge in edges:
        if edge.relation_type in CAUSAL_TYPES or edge.relation_type in {
            "THRESHOLD_VARIANT",
            "TIMEFRAME_VARIANT",
            "SERIES_MEMBER",
            "HIERARCHICAL",
        }:
            graph[edge.source].append(
                (
                    edge.target,
                    edge.relation_type,
                    edge.confidence,
                    edge.implied_conditional,
                )
            )

    chains: list[TransitiveChain] = []
    existing_direct = {(e.source, e.target) for e in edges}

    # Convert defaultdict to regular dict to avoid modification during iteration
    graph_dict = dict(graph)

    for start_node in graph_dict:
        # BFS from start_node
        # queue: (current_node, path_nodes, path_relations, cumulative_conf, cumulative_P)
        queue: list[tuple[str, list[str], list[str], float, float | None]] = [
            (start_node, [start_node], [], 1.0, None)
        ]

        while queue:
            current, path, rels, conf, p_given = queue.pop(0)

            if len(path) > max_hops + 1:
                continue

            for neighbor, rel_type, edge_conf, implied_cond in graph_dict.get(
                current, []
            ):
                if neighbor in path:
                    continue  # Avoid cycles

                new_path = path + [neighbor]
                new_rels = rels + [rel_type]
                new_conf = conf * edge_conf

                # Calculate cumulative conditional probability
                new_p = p_given
                if implied_cond and "P_B_given_A" in implied_cond:
                    if new_p is None:
                        new_p = implied_cond["P_B_given_A"]
                    else:
                        new_p = new_p * implied_cond["P_B_given_A"]

                # Record chain if length >= 3 (at least 2 hops) and not a direct edge
                if len(new_path) >= 3 and (start_node, neighbor) not in existing_direct:
                    if new_conf >= min_confidence:
                        # Determine chain type based on relations
                        if all(r in CAUSAL_TYPES for r in new_rels):
                            chain_type = "causal_cascade"
                        else:
                            chain_type = "structural_chain"

                        chains.append(
                            TransitiveChain(
                                path=new_path,
                                relations=new_rels,
                                chain_type=chain_type,
                                total_confidence=new_conf,
                                implied_P_end_given_start=new_p,
                            )
                        )

                # Continue BFS if we haven't reached max hops
                if len(new_path) <= max_hops + 1:
                    queue.append((neighbor, new_path, new_rels, new_conf, new_p))

    return chains


# =============================================================================
# PER-EVENT RELATIONS
# =============================================================================


def build_per_event_relations(
    nodes: dict[str, GraphNode],
    edges: list[GraphEdge],
    chains: list[TransitiveChain],
) -> dict:
    """Build per-event relation summaries."""
    per_event: dict[str, dict] = {}

    # Initialize for all nodes
    for node_id, node in nodes.items():
        per_event[node_id] = {
            "title": node.title,
            "direct_consequences": [],
            "direct_causes": [],
            "transitive_consequences": [],
            "variants": [],
            "related_count": 0,
        }

    # Process direct edges
    for edge in edges:
        source = edge.source
        target = edge.target

        if source not in per_event:
            per_event[source] = {
                "title": "",
                "direct_consequences": [],
                "direct_causes": [],
                "transitive_consequences": [],
                "variants": [],
                "related_count": 0,
            }
        if target not in per_event:
            per_event[target] = {
                "title": "",
                "direct_consequences": [],
                "direct_causes": [],
                "transitive_consequences": [],
                "variants": [],
                "related_count": 0,
            }

        # Determine if it's a causal or structural relation
        if edge.classification_source == "causal":
            p_given = None
            if edge.implied_conditional:
                p_given = edge.implied_conditional.get("P_B_given_A")

            per_event[source]["direct_consequences"].append(
                {
                    "event_id": target,
                    "relation": edge.relation_type,
                    "P_given": p_given,
                }
            )
            per_event[target]["direct_causes"].append(
                {
                    "event_id": source,
                    "relation": edge.relation_type,
                    "P_given": p_given,
                }
            )
        else:
            per_event[source]["variants"].append(
                {
                    "event_id": target,
                    "relation": edge.relation_type,
                }
            )
            per_event[target]["variants"].append(
                {
                    "event_id": source,
                    "relation": edge.relation_type,
                }
            )

    # Process transitive chains
    for chain in chains:
        if chain.chain_type == "causal_cascade" and len(chain.path) >= 3:
            start = chain.path[0]
            end = chain.path[-1]
            hops = len(chain.path) - 1

            if start in per_event:
                per_event[start]["transitive_consequences"].append(
                    {
                        "event_id": end,
                        "hops": hops,
                        "P_given": chain.implied_P_end_given_start,
                    }
                )

    # Count total related
    for event_id in per_event:
        per_event[event_id]["related_count"] = (
            len(per_event[event_id]["direct_consequences"])
            + len(per_event[event_id]["direct_causes"])
            + len(per_event[event_id]["transitive_consequences"])
            + len(per_event[event_id]["variants"])
        )

    return per_event


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Main entry point."""
    start_time = datetime.now(timezone.utc)
    logger.info("Starting 05_4_build_relation_graph")

    # Determine input folders
    if INPUT_RUN_FOLDER:
        structural_folder = INPUT_STRUCTURAL_DIR / INPUT_RUN_FOLDER
        causal_folder = INPUT_CAUSAL_DIR / INPUT_RUN_FOLDER
        events_folder = INPUT_EVENTS_DIR / INPUT_RUN_FOLDER
        semantics_folder = INPUT_SEMANTICS_DIR / INPUT_RUN_FOLDER
    else:
        structural_folder = find_latest_run_folder(INPUT_STRUCTURAL_DIR)
        causal_folder = find_latest_run_folder(INPUT_CAUSAL_DIR)
        events_folder = find_latest_run_folder(INPUT_EVENTS_DIR)
        semantics_folder = find_latest_run_folder(INPUT_SEMANTICS_DIR)

    # Validate input folders
    for name, folder in [
        ("structural", structural_folder),
        ("causal", causal_folder),
        ("events", events_folder),
        ("semantics", semantics_folder),
    ]:
        if not folder or not folder.exists():
            logger.error(f"{name} folder not found: {folder}")
            return

    logger.info(f"Loading structural from: {structural_folder}")
    logger.info(f"Loading causal from: {causal_folder}")
    logger.info(f"Loading events from: {events_folder}")
    logger.info(f"Loading semantics from: {semantics_folder}")

    # Load structural relations
    with open(structural_folder / "structural_relations.json", encoding="utf-8") as f:
        structural_data = json.load(f)
    structural_relations = structural_data.get("relations", [])
    logger.info(f"Loaded {len(structural_relations)} structural relations")

    # Load causal relations
    with open(causal_folder / "causal_relations.json", encoding="utf-8") as f:
        causal_data = json.load(f)
    causal_relations = causal_data.get("relations", [])
    logger.info(f"Loaded {len(causal_relations)} causal relations")

    # Load events
    with open(events_folder / "events.json", encoding="utf-8") as f:
        events = json.load(f)
    logger.info(f"Loaded {len(events)} events")

    # Load event semantics
    with open(semantics_folder / "event_semantics.json", encoding="utf-8") as f:
        semantics_data = json.load(f)
    semantics_list = semantics_data.get("events", [])
    semantics_by_id = {str(e["id"]): e for e in semantics_list}
    logger.info(f"Loaded semantics for {len(semantics_by_id)} events")

    # Build graph nodes
    logger.info("Building graph nodes...")
    nodes = build_nodes(events, semantics_by_id)
    logger.info(f"Built {len(nodes)} nodes")

    # Build graph edges
    logger.info("Building graph edges...")
    edges = build_edges(structural_relations, causal_relations)
    logger.info(f"Built {len(edges)} edges")

    # Build adjacency list
    logger.info("Building adjacency list...")
    adjacency = build_adjacency_list(edges)
    logger.info(
        f"Adjacency list: {len(adjacency['outgoing'])} sources, {len(adjacency['incoming'])} targets"
    )

    # Find transitive chains
    logger.info("Finding transitive chains...")
    chains = find_transitive_chains(
        edges, max_hops=MAX_HOPS, min_confidence=MIN_CONFIDENCE_TRANSITIVE
    )
    logger.info(f"Found {len(chains)} transitive chains")

    # Build per-event relations
    logger.info("Building per-event relations...")
    per_event = build_per_event_relations(nodes, edges, chains)
    events_with_relations = len(
        [e for e in per_event if per_event[e]["related_count"] > 0]
    )
    logger.info(f"Events with relations: {events_with_relations}")

    # Create output folder
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    output_folder = SCRIPT_OUTPUT_DIR / timestamp
    output_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output folder: {output_folder}")

    # Count classification sources
    source_counts = defaultdict(int)
    for edge in edges:
        source_counts[edge.classification_source] += 1

    # Save relation_graph.json
    graph_output = {
        "_meta": {
            "description": "Unified event relation graph",
            "created_at": start_time.isoformat(),
            "node_count": len(nodes),
            "edge_count": len(edges),
        },
        "nodes": [node.to_dict() for node in nodes.values()],
        "edges": [edge.to_dict() for edge in edges],
    }
    with open(output_folder / "relation_graph.json", "w", encoding="utf-8") as f:
        json.dump(graph_output, f, indent=2, ensure_ascii=False)
    logger.info("Saved relation_graph.json")

    # Export GEXF for Gephi visualization
    if EXPORT_GEXF:
        import networkx as nx

        logger.info("Building NetworkX graph for GEXF export...")
        G_nx = build_networkx_graph(nodes, edges)
        nx.write_gexf(G_nx, output_folder / "relation_graph.gexf")
        logger.info("Saved relation_graph.gexf")

    # Save adjacency_list.json
    adjacency_output = {
        "_meta": {
            "description": "Adjacency list for quick lookups",
            "created_at": start_time.isoformat(),
        },
        **adjacency,
    }
    with open(output_folder / "adjacency_list.json", "w", encoding="utf-8") as f:
        json.dump(adjacency_output, f, indent=2, ensure_ascii=False)
    logger.info("Saved adjacency_list.json")

    # Save transitive_relations.json
    transitive_output = {
        "_meta": {
            "description": "Inferred transitive causal chains",
            "created_at": start_time.isoformat(),
            "max_hops": MAX_HOPS,
        },
        "chains": [chain.to_dict() for chain in chains],
    }
    with open(output_folder / "transitive_relations.json", "w", encoding="utf-8") as f:
        json.dump(transitive_output, f, indent=2, ensure_ascii=False)
    logger.info("Saved transitive_relations.json")

    # Save per_event_relations.json
    per_event_output = {
        "_meta": {
            "description": "Per-event relation summaries",
            "created_at": start_time.isoformat(),
        },
        **per_event,
    }
    with open(output_folder / "per_event_relations.json", "w", encoding="utf-8") as f:
        json.dump(per_event_output, f, indent=2, ensure_ascii=False)
    logger.info("Saved per_event_relations.json")

    # Compute statistics
    relation_type_counts: dict[str, int] = defaultdict(int)
    for edge in edges:
        relation_type_counts[edge.relation_type] += 1

    causal_chain_count = len([c for c in chains if c.chain_type == "causal_cascade"])

    # Count nodes with current_price populated
    nodes_with_price = len([n for n in nodes.values() if n.current_price is not None])

    # Save summary
    end_time = datetime.now(timezone.utc)
    summary = {
        "run_info": {
            "started_at": start_time.isoformat(),
            "completed_at": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "input_structural_folder": str(structural_folder),
            "input_causal_folder": str(causal_folder),
            "input_events_folder": str(events_folder),
            "input_semantics_folder": str(semantics_folder),
            "output_folder": str(output_folder),
        },
        "configuration": {
            "max_hops": MAX_HOPS,
            "min_confidence_transitive": MIN_CONFIDENCE_TRANSITIVE,
        },
        "statistics": {
            "total_events": len(events),
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "structural_edges": source_counts.get("structural", 0),
            "causal_edges": source_counts.get("causal", 0),
            "transitive_chains_found": len(chains),
            "causal_cascade_chains": causal_chain_count,
            "events_with_relations": events_with_relations,
            "nodes_with_price": nodes_with_price,
            "relation_type_distribution": dict(relation_type_counts),
            "classification_source_distribution": dict(source_counts),
        },
    }
    with open(output_folder / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Saved summary.json")

    # Log summary
    logger.info(f"Total edges: {len(edges)}")
    logger.info(f"Structural edges: {source_counts.get('structural', 0)}")
    logger.info(f"Causal edges: {source_counts.get('causal', 0)}")
    logger.info(f"Transitive chains: {len(chains)}")
    logger.info(f"Nodes with price: {nodes_with_price}")
    logger.info(f"Completed in {summary['run_info']['duration_seconds']:.2f}s")


if __name__ == "__main__":
    main()
