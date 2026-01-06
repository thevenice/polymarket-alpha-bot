"""Pipeline run commands."""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cli.script_registry import PIPELINE_ORDER, SCRIPTS, get_steps_in_range
from cli.utils import run_script_by_step

app = typer.Typer(help="Run pipeline steps")
console = Console()


# =============================================================================
# PRODUCTION PIPELINE COMMANDS
# =============================================================================


@app.callback(invoke_without_command=True)
def run_default(
    ctx: typer.Context,
    full: bool = typer.Option(
        False,
        "--full",
        "-f",
        help="Force full reprocessing (clear state first)",
    ),
):
    """Run the production pipeline (incremental by default).

    This is the recommended way to run the pipeline:
    - First run: processes all events (slower)
    - Subsequent runs: processes only NEW events (fast!)

    Examples:
        poly run            # Incremental - process new events only
        poly run --full     # Full - reprocess everything from scratch
    """
    if ctx.invoked_subcommand is not None:
        return

    from core.runner import run as run_pipeline
    from core.state import load_state

    # Show current state
    state = load_state()
    stats = state.get_stats()
    state.close()

    mode = "full" if full else "incremental"
    console.print(
        Panel(
            f"[bold]Production Pipeline[/]\n\n"
            f"Mode: [cyan]{mode}[/]\n"
            f"Current state: {stats.total_events} events, {stats.total_edges} edges",
            title="Alphapoly",
        )
    )

    if full:
        console.print(
            "[yellow]Full mode: will reset state and reprocess all events[/]\n"
        )
    else:
        console.print("[green]Incremental mode: will process only new events[/]\n")

    try:
        result = run_pipeline(full=full)

        # Show results
        console.print("\n[bold green]Pipeline completed![/]\n")

        table = Table(title="Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Mode", result.get("mode", "unknown"))
        table.add_row("Total Events", str(result.get("total_events", 0)))
        table.add_row("New Events", str(result.get("new_events", 0)))
        table.add_row("Graph Nodes", str(result.get("graph_nodes", 0)))
        table.add_row("Graph Edges", str(result.get("graph_edges", 0)))
        table.add_row("Opportunities", str(result.get("opportunities", 0)))
        table.add_row("Time", f"{result.get('elapsed_seconds', 0):.1f}s")

        console.print(table)

    except Exception as e:
        console.print(f"[red]Pipeline failed:[/] {e}")
        raise typer.Exit(1)


@app.command("reset")
def run_reset(
    confirm: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt",
    ),
):
    """Reset pipeline state (clear all accumulated data).

    This will delete:
    - All processed events from state
    - Entity mappings and indexes
    - Knowledge graph
    - Embeddings

    After reset, run 'poly run --full' to rebuild from scratch.
    """
    if not confirm:
        confirm = typer.confirm(
            "This will delete all accumulated pipeline data. Continue?"
        )
        if not confirm:
            console.print("[yellow]Aborted[/]")
            raise typer.Exit(0)

    from core.state import load_state

    state = load_state()
    state.reset()
    state.close()

    console.print("[green]Pipeline state reset successfully[/]")
    console.print("[dim]Run 'poly run --full' to rebuild from scratch[/]")


@app.command("state")
def show_state():
    """Show current pipeline state and statistics."""
    from core.state import load_state

    state = load_state()
    stats = state.get_stats()
    last_run = state.get_last_run()
    state.close()

    table = Table(title="Pipeline State")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Total Events", str(stats.total_events))
    table.add_row("Total Entities", str(stats.total_entities))
    table.add_row("Total Edges", str(stats.total_edges))
    table.add_row("Last Full Run", stats.last_full_run or "-")
    table.add_row("Last Refresh", stats.last_refresh or "-")

    console.print(table)

    if last_run:
        console.print(
            f"\n[dim]Last run: {last_run.get('run_type', 'unknown')} - "
            f"{last_run.get('status', 'unknown')} "
            f"({last_run.get('new_events', 0)} new events)[/]"
        )


# =============================================================================
# SCRIPT-BASED PIPELINE COMMANDS (for development/debugging)
# =============================================================================


@app.command("pipeline")
def run_pipeline(
    from_step: str = typer.Option(
        "01",
        "--from-step",
        "-f",
        help="Start from this step (e.g., '01', '03_1')",
    ),
    to_step: str = typer.Option(
        "06_3",
        "--to-step",
        "-t",
        help="End at this step (e.g., '06_3')",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Show plan without executing",
    ),
):
    """Run pipeline from step A to step B.

    Examples:
        poly run pipeline                    # Full pipeline
        poly run pipeline --from-step 03_1   # Start from entity extraction
        poly run pipeline -f 05_0 -t 05_4    # Run only step 05 (relations)
    """
    try:
        steps = get_steps_in_range(from_step, to_step)
    except ValueError as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)

    # Show plan
    console.print(
        Panel(
            f"[bold]Pipeline Plan[/]\n\n"
            f"Steps: {from_step} → {to_step}\n"
            f"Total: {len(steps)} scripts to run",
            title="Alphapoly Pipeline",
        )
    )

    console.print("\n[bold]Scripts to execute:[/]")
    for i, step in enumerate(steps, 1):
        script = SCRIPTS[step]
        console.print(f"  {i}. [{step}] {script.name} - {script.description}")

    if dry_run:
        console.print("\n[yellow]Dry run - no scripts executed[/]")
        return

    console.print("\n[bold]Starting pipeline...[/]\n")

    # Execute each step
    failed_step = None
    for i, step in enumerate(steps, 1):
        script = SCRIPTS[step]
        console.print(f"\n[bold cyan]Step {i}/{len(steps)}:[/] {script.name}")
        console.print(f"[dim]{script.description}[/]\n")

        exit_code = run_script_by_step(step)

        if exit_code != 0:
            failed_step = step
            console.print(f"\n[red]Pipeline failed at step {step}[/]")
            break

    if failed_step:
        console.print(f"\n[red]Pipeline stopped at step {failed_step}[/]")
        console.print(f"[dim]To resume: poly run pipeline --from-step {failed_step}[/]")
        raise typer.Exit(1)

    console.print("\n[green]Pipeline completed successfully![/]")


@app.command("step")
def run_step(
    step: str = typer.Argument(..., help="Step to run (e.g., '01', '03_1', '06_3')"),
):
    """Run a single pipeline step.

    Examples:
        poly run step 01       # Fetch events
        poly run step 03_1     # Extract entities
        poly run step 06_3     # Export opportunities
    """
    if step not in SCRIPTS:
        console.print(f"[red]Error:[/] Unknown step: {step}")
        console.print(f"[dim]Valid steps: {', '.join(PIPELINE_ORDER)}[/]")
        raise typer.Exit(1)

    exit_code = run_script_by_step(step)

    if exit_code != 0:
        raise typer.Exit(exit_code)


@app.command("quick")
def run_quick():
    """Run quick refresh: fetch events and update opportunities.

    Runs only steps 01 (fetch) and 06_1-06_3 (alpha detection).
    Assumes entity/relation extraction is already done.
    """
    quick_steps = ["01", "06_1", "06_2", "06_3"]

    console.print("[bold]Quick refresh mode[/]")
    console.print(
        "Running: fetch events → compute conditionals → detect alpha → export\n"
    )

    for step in quick_steps:
        exit_code = run_script_by_step(step)
        if exit_code != 0:
            console.print(f"\n[red]Quick refresh failed at step {step}[/]")
            raise typer.Exit(exit_code)

    console.print("\n[green]Quick refresh completed![/]")
