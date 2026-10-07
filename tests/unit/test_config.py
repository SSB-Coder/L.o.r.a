from pathlib import Path

from lora.core.config import (
    ConfigManager,
    LoraConfig,
    PolicyConfig,
    load_config,
)


def test_load_lora_config(project_root: Path):
    config_file = project_root / "config" / "lora.yaml"
    cfg = load_config(config_file)
    assert isinstance(cfg, LoraConfig)
    assert cfg.models_dir == "C:\\Models"
    assert cfg.models.writer.name == "spark-x25-4b-q4"
    assert cfg.models.writer.num_ctx == 8192
    assert cfg.models.reader.num_ctx == 32768
    assert cfg.gateway.desktop_bind == "127.0.0.1:8765"
    assert cfg.gateway.mobile_bind == "192.168.50.10:8765"
    assert cfg.gateway.mobile.max_tier == 0
    assert cfg.gateway.mobile.max_run_mode == "standard"
    assert cfg.grounding.reranker == "bge-reranker-base"

def test_load_policy_config(project_root: Path):
    policy_file = project_root / "config" / "policy.yaml"
    mgr = ConfigManager()
    policy = mgr.load_policy(policy_file)
    assert isinstance(policy, PolicyConfig)
    assert len(policy.deny_rules) > 0
    # Check that executable extensions are denied
    assert any(".exe" in (rule.path_extensions or []) for rule in policy.deny_rules)
    # Check that delete operations are denied
    assert any(rule.tool_match == "*delete*" for rule in policy.deny_rules)

def test_policy_evaluates_hard_deny_first(project_root: Path):
    policy_file = project_root / "config" / "policy.yaml"
    mgr = ConfigManager()
    policy = mgr.load_policy(policy_file)

    # Deletion should always be hard denied regardless of origin
    res_desktop = policy.evaluate_action(tool="file_delete", origin="desktop", tier=3)
    assert res_desktop.allowed is False
    assert "permanently forbidden" in res_desktop.reason

    # Executable extension should always be hard denied
    res_exe = policy.evaluate_path(path="C:\\Users\\test\\malicious.exe", origin="desktop")
    assert res_exe.allowed is False
    assert "strictly forbidden" in res_exe.reason

    # Mobile origin driving desktop tier 1 action should be denied
    res_mobile = policy.evaluate_action(tool="open_app", origin="mobile", tier=1)
    assert res_mobile.allowed is False

    # Mobile origin doing tier 0 research should be allowed
    res_mobile_research = policy.evaluate_action(tool="web_search", origin="mobile", tier=0)
    assert res_mobile_research.allowed is True
