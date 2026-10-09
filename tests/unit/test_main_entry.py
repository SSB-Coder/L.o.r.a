from pathlib import Path
from unittest.mock import MagicMock, patch

from lora.main import build_argument_parser, init_platform


def test_main_argument_parser():
    parser = build_argument_parser()
    args = parser.parse_args(["--headless", "--no-audio", "--port", "8765"])
    assert args.headless is True
    assert args.no_audio is True
    assert args.port == 8765


def test_init_platform_initializes_ecosystem(tmp_path: Path):
    with patch("lora.main.ConfigManager") as mock_cfg_mgr:
        instance = MagicMock()
        mock_cfg_mgr.return_value = instance

        cfg_mgr, cfg = init_platform(base_dir=tmp_path)
        assert cfg_mgr is not None
        assert instance.ensure_ecosystem.called
