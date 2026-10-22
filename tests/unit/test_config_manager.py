from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lora.config.manager import ConfigManager, LoraUserConfig


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")


def test_config_manager_initialization(tmp_path: Path):
    manager = ConfigManager(base_dir=tmp_path)
    paths = manager.paths

    assert paths.base_dir == tmp_path
    assert paths.skills_dir == tmp_path / "skills"
    assert paths.agents_dir == tmp_path / "agents"
    assert paths.memory_dir == tmp_path / "memory"
    assert paths.notebooks_dir == tmp_path / "notebooks"
    assert paths.outputs_dir == Path.home() / "LORA" / "outputs"
    assert paths.config_file == tmp_path / "config.yaml"
    assert paths.mcp_file == tmp_path / "mcp.json"
    assert paths.user_profile_file == tmp_path / "memory" / "user_profile.md"


def test_ensure_ecosystem_creates_directories_and_defaults(tmp_path: Path):
    manager = ConfigManager(base_dir=tmp_path)
    manager.ensure_ecosystem(apply_acls=False)

    # Check directories exist
    assert (tmp_path / "skills").is_dir()
    assert (tmp_path / "agents").is_dir()
    assert (tmp_path / "memory").is_dir()
    assert (tmp_path / "notebooks").is_dir()
    assert not (tmp_path / "outputs").exists()
    assert (Path.home() / "LORA" / "outputs").is_dir()

    # Check default files exist
    assert (tmp_path / "config.yaml").is_file()
    assert (tmp_path / "mcp.json").is_file()
    assert (tmp_path / "memory" / "user_profile.md").is_file()

    # Verify config loads cleanly
    cfg = manager.load_config()
    assert isinstance(cfg, LoraUserConfig)
    assert cfg.ui.ghost_mode_opacity == 0.35
    assert cfg.security.offline_only is False


def test_load_and_save_config(tmp_path: Path):
    manager = ConfigManager(base_dir=tmp_path)
    manager.ensure_ecosystem(apply_acls=False)

    cfg = manager.load_config()
    # Update config via model_copy
    updated_cfg = cfg.model_copy(
        update={
            "ui": cfg.ui.model_copy(update={"ghost_mode_opacity": 0.5}),
            "security": cfg.security.model_copy(update={"offline_only": True}),
        }
    )
    manager.save_config(updated_cfg)

    reloaded = manager.load_config()
    assert reloaded.ui.ghost_mode_opacity == 0.5
    assert reloaded.security.offline_only is True


def test_apply_ntfs_acls_invokes_icacls(tmp_path: Path):
    manager = ConfigManager(base_dir=tmp_path)
    manager.ensure_ecosystem(apply_acls=False)

    with patch("sys.platform", "win32"), patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        success = manager.apply_ntfs_acls()
        assert success is True
        assert mock_run.called
        cmd_args = mock_run.call_args[0][0]
        assert "icacls.exe" in cmd_args[0].lower() or "icacls" in cmd_args[0].lower()


def test_apply_ntfs_acls_handles_failure(tmp_path: Path):
    manager = ConfigManager(base_dir=tmp_path)
    with patch("sys.platform", "win32"), patch("subprocess.run") as mock_run:
        mock_run.side_effect = Exception("Permission denied")
        success = manager.apply_ntfs_acls()
        assert success is False
