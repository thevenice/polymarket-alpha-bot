"""CLI configuration and paths."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "manifest.json"

# API settings
API_HOST = "localhost"
API_PORT = 8000
