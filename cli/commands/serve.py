"""API server commands."""

import subprocess

import typer
from rich.console import Console

from cli.config import API_HOST, API_PORT, PROJECT_ROOT

app = typer.Typer(help="Start API server")
console = Console()


@app.callback(invoke_without_command=True)
def serve(
    ctx: typer.Context,
    host: str = typer.Option(
        API_HOST,
        "--host",
        "-h",
        help="Host to bind to",
    ),
    port: int = typer.Option(
        API_PORT,
        "--port",
        "-p",
        help="Port to bind to",
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        "-r",
        help="Enable auto-reload for development",
    ),
):
    """Start the FastAPI server.

    The server provides:
    - REST API for pipeline data
    - WebSocket for live price updates
    - Pipeline status and control

    Examples:
        poly serve                    # Start on localhost:8000
        poly serve --port 3001        # Custom port
        poly serve --reload           # Development mode with auto-reload
    """
    if ctx.invoked_subcommand is not None:
        return

    console.print("[bold]Starting Alphapoly API server[/]")
    console.print(f"[dim]Host: {host}:{port}[/]")

    if reload:
        console.print("[yellow]Auto-reload enabled (development mode)[/]")

    console.print("\n[green]API endpoints:[/]")
    console.print(f"  - REST API:  http://{host}:{port}/docs")
    console.print(f"  - WebSocket: ws://{host}:{port}/prices/ws")
    console.print("\n[dim]Press Ctrl+C to stop[/]\n")

    # Build uvicorn command
    cmd = [
        "uvicorn",
        "server.main:app",
        "--host",
        host,
        "--port",
        str(port),
    ]

    if reload:
        cmd.append("--reload")

    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped[/]")
    except FileNotFoundError:
        console.print(
            "[red]Error: uvicorn not found. Run 'uv sync' to install dependencies.[/]"
        )
        raise typer.Exit(1)
