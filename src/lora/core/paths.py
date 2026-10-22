"""Stable paths for application assets, independent of the launch directory."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
