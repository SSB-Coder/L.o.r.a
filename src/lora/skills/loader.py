import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class SkillDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str = ""
    triggers: tuple[str, ...] = Field(default_factory=tuple)
    required_tools: tuple[str, ...] = Field(default_factory=tuple)
    system_prompt_addon: str | None = None
    default_depth: str = "deep"


class SkillLoader:
    """Discovers and parses custom skills from ~/.lora/skills/<skill_name>/skill.yaml."""

    def __init__(self, skills_dir: Path) -> None:
        self._skills_dir = skills_dir
        self._skills: dict[str, SkillDefinition] = {}

    @property
    def skills(self) -> dict[str, SkillDefinition]:
        return dict(self._skills)

    def load_skills(self) -> dict[str, SkillDefinition]:
        """Scans the skills directory and loads all valid skill.yaml definitions."""
        self._skills.clear()
        if not self._skills_dir.is_dir():
            return {}

        for item in self._skills_dir.iterdir():
            if item.is_dir():
                yaml_file = item / "skill.yaml"
                if yaml_file.is_file():
                    try:
                        with open(yaml_file, encoding="utf-8") as f:
                            raw = yaml.safe_load(f) or {}
                        skill = self._parse_skill_dict(raw, fallback_name=item.name)
                        self._skills[skill.name] = skill
                        logger.info("Loaded custom skill: %s", skill.name)
                    except Exception as e:
                        logger.warning("Failed to load skill from %s: %e", yaml_file, e)

        return dict(self._skills)

    def match_skill(self, prompt: str) -> SkillDefinition | None:
        """Finds the first skill whose trigger phrases are contained within the user prompt."""
        if not self._skills:
            self.load_skills()

        prompt_clean = prompt.lower().strip()
        for skill in self._skills.values():
            for trigger in skill.triggers:
                if trigger.lower() in prompt_clean:
                    return skill

        return None

    def _parse_skill_dict(self, raw: dict[str, Any], fallback_name: str) -> SkillDefinition:
        triggers_raw = raw.get("triggers", [])
        if isinstance(triggers_raw, str):
            triggers_tuple = (triggers_raw,)
        else:
            triggers_tuple = tuple(triggers_raw)

        tools_raw = raw.get("required_tools", [])
        if isinstance(tools_raw, str):
            tools_tuple = (tools_raw,)
        else:
            tools_tuple = tuple(tools_raw)

        return SkillDefinition(
            name=str(raw.get("name", fallback_name)),
            description=str(raw.get("description", "")),
            triggers=triggers_tuple,
            required_tools=tools_tuple,
            system_prompt_addon=raw.get("system_prompt_addon"),
            default_depth=str(raw.get("default_depth", "deep")),
        )
