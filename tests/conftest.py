from pathlib import Path

import pytest


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent

@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_lora.db"

@pytest.fixture
def sample_config_path(project_root: Path) -> Path:
    return project_root / "config" / "lora.yaml"
