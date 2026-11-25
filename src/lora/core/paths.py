"""Stable paths for application assets, independent of the launch directory."""

from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _PACKAGE_ROOT.parents[1]
PROJECT_ROOT = _SOURCE_ROOT if (_SOURCE_ROOT / "config" / "lora.yaml").is_file() else Path.home() / ".lora"


def default_config_path(name: str) -> Path:
    checkout = _SOURCE_ROOT / "config" / name
    return checkout if checkout.is_file() else _PACKAGE_ROOT / "_defaults" / name


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
