"""Repository-local data paths, with explicit environment overrides."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("pog", "pog_dense", "spotify", "spotify_sparse", "electronic", "food", "clothing", "dunnhumby")


def data_root():
    return Path(os.environ.get("SIMBA_DATA_DIR", PROJECT_ROOT / "data")).expanduser().resolve()


def dataset_dir(dataset, override=None):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    explicit = os.environ.get(override) if override else None
    path = Path(explicit).expanduser().resolve() if explicit else data_root() / dataset
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {dataset} data: {path}. See docs/DATA.md or run python -m simba.data_tools --help.")
    return str(path)
