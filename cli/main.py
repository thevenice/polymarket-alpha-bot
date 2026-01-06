"""Main CLI entry point for Alphapoly."""

import typer
from rich.console import Console
from rich.table import Table

from cli import __version__
from cli.commands import fetch, run, serve
from cli.script_registry import PIPELINE_ORDER, SCRIPTS
from cli.utils import find_latest_run, format_timestamp, get_all_runs, load_manifest

app = typer.Typer(
    name="poly",
    help="Alphapoly - Polymarket alpha detection pipeline CLI",
    no_args_is_help=True,
)
console = Console()

# Add subcommand groups
app.add_typer(run.app, name="run", help="Run pipeline steps")
app.add_typer(fetch.app, name="fetch", help="Fetch data from sources")
app.add_typer(serve.app, name="serve", help="Start API server")


@app.command()
def version():
    """Show version information."""
    console.print(f"[bold]Alphapoly CLI[/] v{__version__}")


@app.command()
def status():
    """Show pipeline status and latest runs."""
    manifest = load_manifest()

    table = Table(title="Pipeline Status", show_header=True)
    table.add_column("Step", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Latest Run", style="green")
    table.add_column("Status", style="yellow")

    for step in PIPELINE_ORDER:
        script = SCRIPTS[step]
        latest = find_latest_run(step)

        if latest:
            ts = format_timestamp(latest.name)
            status_str = "✓ Complete"
        else:
            ts = "-"
            status_str = "○ No runs"

        table.add_row(step, script.name, ts, status_str)

    console.print(table)

    # Show manifest info
    if manifest:
        console.print("\n[dim]Last updated runs in manifest.json:[/]")
        for name, info in sorted(manifest.items()):
            console.print(f"  {name}: {info.get('timestamp', 'unknown')}")


@app.command("list")
def list_scripts():
    """List all available pipeline scripts."""
    table = Table(title="Pipeline Scripts", show_header=True)
    table.add_column("Step", style="cyan", width=6)
    table.add_column("Name", style="white", width=25)
    table.add_column("Description", style="dim")
    table.add_column("Inputs", style="yellow", width=15)

    for step in PIPELINE_ORDER:
        script = SCRIPTS[step]
        inputs = ", ".join(script.inputs) if script.inputs else "-"
        table.add_row(step, script.name, script.description, inputs)

    console.print(table)
    console.print(
        "\n[dim]Use 'poly run pipeline --from-step STEP' to run from a specific step[/]"
    )


@app.command()
def runs(
    step: str = typer.Argument(None, help="Step to show runs for (e.g., '01', '03_1')"),
    limit: int = typer.Option(5, "--limit", "-n", help="Number of runs to show"),
):
    """Show pipeline run history."""
    all_runs = get_all_runs()

    if step:
        # Show runs for specific step
        from cli.script_registry import get_output_dir_name

        dir_name = get_output_dir_name(step)
        if dir_name not in all_runs:
            console.print(f"[yellow]No runs found for step {step}[/]")
            return

        console.print(f"[bold]Runs for step {step}:[/]")
        for run_info in all_runs[dir_name][:limit]:
            ts = format_timestamp(run_info["timestamp"])
            summary = run_info.get("summary", {})
            stats = summary.get("statistics", {}) if summary else {}
            console.print(f"  [green]{ts}[/] - {run_info['path']}")
            if stats:
                for key, value in list(stats.items())[:3]:
                    console.print(f"    {key}: {value}")
    else:
        # Show latest run for each step
        table = Table(title="Latest Runs", show_header=True)
        table.add_column("Step", style="cyan")
        table.add_column("Timestamp", style="green")
        table.add_column("Path", style="dim")

        for step_id in PIPELINE_ORDER:
            from cli.script_registry import get_output_dir_name

            dir_name = get_output_dir_name(step_id)
            if dir_name in all_runs and all_runs[dir_name]:
                run_info = all_runs[dir_name][0]
                ts = format_timestamp(run_info["timestamp"])
                table.add_row(step_id, ts, run_info["path"])
            else:
                table.add_row(step_id, "-", "-")

        console.print(table)


@app.callback()
def main():
    """Alphapoly CLI - Polymarket alpha detection pipeline."""
    pass


if __name__ == "__main__":
    app()
