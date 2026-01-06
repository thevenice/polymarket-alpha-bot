"""Individual script wrapper commands for fetching data."""

import typer
from rich.console import Console

from cli.utils import run_script_by_step

app = typer.Typer(help="Fetch data from sources")
console = Console()


@app.command("events")
def fetch_events(
    tag: str = typer.Option(
        "politics",
        "--tag",
        "-t",
        help="Tag slug to fetch (e.g., 'politics', 'crypto')",
    ),
    include_closed: bool = typer.Option(
        False,
        "--include-closed",
        "-c",
        help="Include closed events",
    ),
):
    """Fetch events from Polymarket API.

    Wrapper for script 01_fetch_events.py with optional configuration.

    Examples:
        poly fetch events                     # Fetch politics events
        poly fetch events --tag crypto        # Fetch crypto events
        poly fetch events --include-closed    # Include closed events
    """
    env_overrides = {}

    if tag != "politics":
        env_overrides["ALPHAPOLY_TARGET_TAG_SLUG"] = tag
        console.print(f"[dim]Using custom tag: {tag}[/]")

    if include_closed:
        env_overrides["ALPHAPOLY_CLOSED"] = "true"
        console.print("[dim]Including closed events[/]")

    # Note: env overrides require the scripts to read them
    # For now, we run with defaults (scripts don't read env vars yet)
    if env_overrides:
        console.print(
            "[yellow]Note: Custom parameters require script modifications to take effect[/]"
        )

    exit_code = run_script_by_step("01", env_overrides if env_overrides else None)

    if exit_code != 0:
        raise typer.Exit(exit_code)


@app.command("entities")
def extract_entities(
    max_events: int = typer.Option(
        None,
        "--max-events",
        "-n",
        help="Limit number of events to process (for testing)",
    ),
):
    """Extract entities from events using GLiNER2.

    Wrapper for script 03_1_extract_entities.py.

    Examples:
        poly fetch entities               # Process all events
        poly fetch entities --max-events 50  # Process first 50 events
    """
    env_overrides = {}

    if max_events is not None:
        env_overrides["ALPHAPOLY_MAX_EVENTS"] = str(max_events)
        console.print(f"[dim]Processing max {max_events} events[/]")

    if env_overrides:
        console.print(
            "[yellow]Note: Custom parameters require script modifications to take effect[/]"
        )

    exit_code = run_script_by_step("03_1", env_overrides if env_overrides else None)

    if exit_code != 0:
        raise typer.Exit(exit_code)


@app.command("relations")
def extract_relations(
    max_events: int = typer.Option(
        None,
        "--max-events",
        "-n",
        help="Limit number of events to process (for testing)",
    ),
):
    """Extract relations between entities using GLiNER2.

    Wrapper for script 03_4_extract_relations.py.

    Examples:
        poly fetch relations              # Process all events
        poly fetch relations --max-events 50  # Process first 50 events
    """
    env_overrides = {}

    if max_events is not None:
        env_overrides["ALPHAPOLY_MAX_EVENTS"] = str(max_events)
        console.print(f"[dim]Processing max {max_events} events[/]")

    if env_overrides:
        console.print(
            "[yellow]Note: Custom parameters require script modifications to take effect[/]"
        )

    exit_code = run_script_by_step("03_4", env_overrides if env_overrides else None)

    if exit_code != 0:
        raise typer.Exit(exit_code)
