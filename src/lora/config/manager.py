import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class EcosystemPaths(BaseModel):
    model_config = ConfigDict(frozen=True)

    base_dir: Path
    skills_dir: Path
    agents_dir: Path
    memory_dir: Path
    notebooks_dir: Path
    outputs_dir: Path
    config_file: Path
    mcp_file: Path
    user_profile_file: Path

    @classmethod
    def from_base(cls, base_dir: Path) -> "EcosystemPaths":
        return cls(
            base_dir=base_dir,
            skills_dir=base_dir / "skills",
            agents_dir=base_dir / "agents",
            memory_dir=base_dir / "memory",
            notebooks_dir=base_dir / "notebooks",
            outputs_dir=Path.home() / "LORA" / "outputs",
            config_file=base_dir / "config.yaml",
            mcp_file=base_dir / "mcp.json",
            user_profile_file=base_dir / "memory" / "user_profile.md",
        )


class ModelsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    ollama_url: str = "http://127.0.0.1:11434"
    default_model: str = "hf.co/XHToken/Spark-X2.6-4B-GGUF:Q4_K_M"
    supervisor_model: str = "granite4.2:8b"
    verifier_model: str = "ornith-1.5:9b"


class VoiceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    tts_voice: str = "af_sky"
    wake_phrase: str = "Awaken LORA"
    vad_level: int = 2
    sound_effects_enabled: bool = True


class UiConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    ghost_mode_opacity: float = 0.35
    theme: str = "dark_aqueous"
    stay_on_top: bool = True
    snap_margin_px: int = 24


class SecurityConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    offline_only: bool = False
    require_human_confirmation: bool = False
    allowed_lan_subnets: tuple[str, ...] = ("192.168.", "10.", "172.16.")


class LoraUserConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: str = "1.0.0"
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    ui: UiConfig = Field(default_factory=UiConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)


DEFAULT_MCP_JSON = {
    "mcpServers": {
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:\\Users"],
            "disabled": True,
        }
    }
}

DEFAULT_USER_PROFILE_MD = """# User Profile & Preferences
*Last Updated: 2026-10-09*

## Interaction Preferences
- Style: Clear, casual, concise; technical depth when requested.
- Research depth: Thorough with multi-source citation verification.
- Tone: Casual friend and dependable butler.

## Memory & Context
- Default project root: C:\\Thought\\Projects\\LORA
- Operating system: Windows 11
"""


class ConfigManager:
    """Manages the ~/.lora/ directory ecosystem, user configuration, and NTFS ACL security."""

    def __init__(self, base_dir: Path | None = None) -> None:
        if base_dir is None:
            self._base_dir = Path.home() / ".lora"
        else:
            self._base_dir = Path(base_dir)

        self._paths = EcosystemPaths.from_base(self._base_dir)

    @property
    def paths(self) -> EcosystemPaths:
        return self._paths

    def ensure_ecosystem(self, apply_acls: bool = True) -> None:
        """Creates the ~/.lora directory hierarchy and default files if they do not exist."""
        # Create directories
        for d in (
            self._paths.base_dir,
            self._paths.skills_dir,
            self._paths.agents_dir,
            self._paths.memory_dir,
            self._paths.notebooks_dir,
            self._paths.outputs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

        # Create config.yaml if missing
        if not self._paths.config_file.is_file():
            default_cfg = LoraUserConfig()
            self.save_config(default_cfg)

        # Create mcp.json if missing
        if not self._paths.mcp_file.is_file():
            self._paths.mcp_file.write_text(
                json.dumps(DEFAULT_MCP_JSON, indent=2), encoding="utf-8"
            )

        # Create user_profile.md if missing
        if not self._paths.user_profile_file.is_file():
            self._paths.user_profile_file.write_text(
                DEFAULT_USER_PROFILE_MD, encoding="utf-8"
            )

        # Seed default starter skills if skills_dir is empty
        default_skill_dir = self._paths.skills_dir / "deep_research_brief"
        default_skill_yaml = default_skill_dir / "skill.yaml"
        if not any(self._paths.skills_dir.iterdir()):
            default_skill_dir.mkdir(parents=True, exist_ok=True)
            default_skill_yaml.write_text(
                yaml.dump(
                    {
                        "name": "deep_research_brief",
                        "description": "Multi-source verified research briefing with executive summary",
                        "triggers": ["research brief", "executive briefing", "deep briefing"],
                        "required_tools": ["web_search", "fetch_page"],
                        "system_prompt_addon": "Start with a 2-sentence executive summary, then list verified findings with inline [S1] citations.",
                        "default_depth": "deep_with_refinement",
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

        if apply_acls and sys.platform == "win32":
            self.apply_ntfs_acls()

    def load_config(self) -> LoraUserConfig:
        """Loads and parses the user config from config.yaml."""
        if not self._paths.config_file.is_file():
            return LoraUserConfig()

        try:
            with open(self._paths.config_file, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return LoraUserConfig.model_validate(data)
        except Exception as e:
            logger.warning(
                "Failed to parse %s: %s. Using default config.",
                self._paths.config_file,
                e,
            )
            return LoraUserConfig()

    def save_config(self, config: LoraUserConfig) -> None:
        """Saves user config to config.yaml."""
        raw_dict = config.model_dump()
        # Convert tuples to lists for cleaner YAML serialization
        clean_dict = self._jsonify(raw_dict)
        with open(self._paths.config_file, "w", encoding="utf-8") as f:
            yaml.dump(clean_dict, f, default_flow_style=False, sort_keys=False)

    def apply_ntfs_acls(self) -> bool:
        """Secures the ~/.lora directory using Windows icacls.exe.

        Restricts access strictly to the current Windows user and SYSTEM.
        """
        if sys.platform != "win32":
            logger.debug(
                "Non-Windows environment detected; skipping icacls configuration."
            )
            return True

        import getpass

        username = os.environ.get("USERNAME") or os.environ.get("USER")
        if not username:
            try:
                username = getpass.getuser()
            except Exception:
                username = None
        if not username:
            logger.warning(
                "USERNAME environment variable not found; skipping icacls."
            )
            return False

        try:
            target_str = str(self._paths.base_dir.resolve())
            cmd = [
                "icacls.exe",
                target_str,
                "/inheritance:r",
                "/grant:r",
                f"{username}:(OI)(CI)F",
                "NT AUTHORITY\\SYSTEM:(OI)(CI)F",
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                logger.info(
                    "Locked NTFS ACLs for %s to user %s", target_str, username
                )
                return True
            else:
                logger.warning(
                    "icacls exited with code %d: %s",
                    result.returncode,
                    result.stderr,
                )
                return False
        except Exception as e:
            logger.warning("Error running icacls on %s: %s", self._paths.base_dir, e)
            return False

    def _jsonify(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._jsonify(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._jsonify(v) for v in obj]
        return obj
