"""Central project paths used by training, evaluation, and the UI."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"


def ensure_project_directories() -> None:
    """Create all generated-artifact directories if they do not exist."""
    for directory in (PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, DOCS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

