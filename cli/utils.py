"""CLI utility functions."""

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

from cli.config import DATA_DIR, MANIFEST_PATH, PROJECT_ROOT, SCRIPTS_DIR
from cli.script_registry import SCRIPTS, get_output_dir_name, get_script

console = Console()


def find_latest_run(step: str) -> Path | None:
    """Find the latest run directory for a script step.

    Args:
        step: The step ID (e.g., "01", "03_1")

    Returns:
        Path to the latest run directory, or None if no runs exist.
    """
    dir_name = get_output_dir_name(step)
    script_dir = DATA_DIR / dir_name

    if not script_dir.exists():
        return None

    # Get all timestamp directories, sorted descending
    runs = sorted(
        [d for d in script_dir.iterdir() if d.is_dir() and d.name[0].isdigit()],
        key=lambda p: p.name,
        reverse=True,
    )
    return runs[0] if runs else None


def find_latest_run_for_output(output_name: str) -> Path | None:
    """Find the latest run that contains a specific output file.

    Args:
        output_name: Name of the output directory (e.g., "01_fetch_events")

    Returns:
        Path to the latest run directory containing the output.
    """
    output_dir = DATA_DIR / output_name

    if not output_dir.exists():
        return None

    runs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir() and d.name[0].isdigit()],
        key=lambda p: p.name,
        reverse=True,
    )
    return runs[0] if runs else None


def load_manifest() -> dict:
    """Load the manifest file, or return empty dict if not exists."""
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def save_manifest(manifest: dict) -> None:
    """Save the manifest file."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def update_manifest(step: str, run_path: Path) -> None:
    """Update manifest with the latest run path for a step.

    Args:
        step: The step ID
        run_path: Path to the run directory
    """
    manifest = load_manifest()
    dir_name = get_output_dir_name(step)

    manifest[dir_name] = {
        "step": step,
        "latest": str(run_path),
        "timestamp": run_path.name,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    save_manifest(manifest)


def run_script(
    script_file: str,
    env_overrides: dict[str, str] | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess:
    """Execute a pipeline script using uv.

    Args:
        script_file: Name of the script file (e.g., "01_fetch_events.py")
        env_overrides: Optional environment variable overrides
        capture_output: If True, capture stdout/stderr

    Returns:
        CompletedProcess with result
    """
    script_path = SCRIPTS_DIR / script_file

    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    env = {**os.environ, **(env_overrides or {})}

    result = subprocess.run(
        ["uv", "run", str(script_path)],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=capture_output,
        text=True,
    )
    return result


def run_script_by_step(
    step: str,
    env_overrides: dict[str, str] | None = None,
) -> int:
    """Run a script by its step ID.

    Args:
        step: The step ID (e.g., "01", "03_1")
        env_overrides: Optional environment variable overrides

    Returns:
        Return code from the script
    """
    script = get_script(step)
    console.print(f"[bold blue]Running:[/] {script.step} - {script.description}")

    result = run_script(script.file, env_overrides)

    if result.returncode == 0:
        # Find the new run directory and update manifest
        latest = find_latest_run(step)
        if latest:
            update_manifest(step, latest)
            console.print(f"[green]✓[/] Completed: {latest}")
    else:
        console.print(f"[red]✗[/] Failed with exit code {result.returncode}")

    return result.returncode


def get_all_runs() -> dict[str, list[dict]]:
    """Get all pipeline runs organized by step.

    Returns:
        Dict mapping step names to list of run info dicts.
    """
    runs: dict[str, list[dict]] = {}

    for step, script in SCRIPTS.items():
        dir_name = get_output_dir_name(step)
        step_dir = DATA_DIR / dir_name

        if not step_dir.exists():
            continue

        step_runs = []
        for run_dir in sorted(step_dir.iterdir(), reverse=True):
            if not run_dir.is_dir() or not run_dir.name[0].isdigit():
                continue

            summary_path = run_dir / "summary.json"
            summary = None
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text())
                except json.JSONDecodeError:
                    pass

            step_runs.append(
                {
                    "path": str(run_dir),
                    "timestamp": run_dir.name,
                    "summary": summary,
                }
            )

        if step_runs:
            runs[dir_name] = step_runs

    return runs


def format_timestamp(ts: str) -> str:
    """Format a timestamp string for display."""
    try:
        dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return ts
