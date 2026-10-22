"""SkillLoader supporting both Anthropic-style SKILL.md (YAML frontmatter + Markdown) and skill.yaml."""

import logging
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n(.*)$", re.DOTALL)

BUILTIN_SKILL_MDS: dict[str, str] = {
    "deep-research": """---
name: deep-research
description: Multi-source web & scholarly research with inline [S1] citations and reference integrity checks.
triggers:
  - deep research
  - literature review
  - systematic review
  - state of the art
  - research report
required_tools:
  - web_search
  - fetch_page
  - pdf_fetch
default_depth: deep
agent: research
---
# Deep Research Skill

When answering this request:
1. Synthesize evidence across multiple independent sources and cite every concrete claim inline using `[S1]`, `[S2]` markers.
2. Lead with a direct executive answer, then organize key mechanisms, empirical benchmarks/numbers, and trade-offs or limitations into scannable sections.
3. Explicitly note where sources disagree or where evidence is incomplete rather than guessing.
""",
    "web-navigator": """---
name: web-navigator
description: Smart web & desktop navigation, site-specific topic search, and external AI delegation.
triggers:
  - navigate to
  - open wikipedia on
  - ask chatgpt
  - ask claude
  - ask gemini
  - ask perplexity
required_tools:
  - open_url
  - open_app
  - ask_other_ai
  - fetch_page
default_depth: quick
agent: system_navigator
---
# Web Navigator Skill

When executing navigation or external AI delegation:
1. Resolve target web applications (`wikipedia`, `youtube`, `arxiv`, `github`, `perplexity`) and construct direct topic/search URLs when a page or query is specified.
2. When delegating to an external AI (`chatgpt`, `claude`, `gemini`, `perplexity`), scrub private local paths/tokens first and match the prompt complexity to the user's request.
""",
    "code-architect": """---
name: code-architect
description: Idiomatic software engineering, debugging, refactoring, and standalone code file generation.
triggers:
  - write python
  - write code
  - refactor this
  - debug this
  - implement a script
required_tools:
  - read_file
  - search_files
  - create_artifact
default_depth: standard
agent: workspace_code
---
# Code Architect Skill

When writing, debugging, or refactoring code:
1. Produce clean, modular, type-annotated code with explicit error handling and edge-case coverage.
2. If the user asks to create or save a script, module, or web page, generate the standalone file artifact so they can open or run it directly.
3. Keep explanations concise: focus on root causes, complexity trade-offs, and verification steps.
""",
    "artifact-studio": """---
name: artifact-studio
description: Creates structured documents, Markdown notes, CSV tables, JSON files, and single-file HTML tools.
triggers:
  - create a document
  - export as csv
  - build an html
  - create a report file
  - save as file
required_tools:
  - create_note
  - create_artifact
default_depth: standard
agent: artifact_studio
---
# Artifact Studio Skill

When creating a deliverable file for the user:
1. Produce complete, well-structured content without placeholders or truncation.
2. Save the deliverable to the user's LORA artifacts directory and attach an interactive file card in the response so the user can open or inspect it with one click.
""",
    "visual-creator": """---
name: visual-creator
description: Generates SVG diagrams, architecture flowcharts, data plots, and edited images displayed inline in chat.
triggers:
  - draw a diagram
  - create an image
  - generate an image
  - make a chart
  - plot a graph
  - generate svg
required_tools:
  - generate_visual
default_depth: standard
agent: artifact_studio
---
# Visual Creator Skill

When asked for an image, diagram, chart, or visual asset:
1. Generate a clean, high-contrast visual artifact (SVG diagram, matplotlib chart, or Pillow composition) with legible labels and balanced spacing.
2. Present the visual inline in the chat response alongside a brief summary of what is shown.
""",
    "truth-auditor": """---
name: truth-auditor
description: Evidence-based claim review with cited primary sources and explicit uncertainty.
triggers:
  - fact check
  - verify claims
  - audit this
  - is it true that
  - double check
required_tools:
  - web_search
  - fetch_page
default_depth: deep
agent: research
---
# Truth Auditor Skill

When auditing or verifying claims:
1. Look up primary evidence relevant to the requested claims and cite the passages used.
2. Explain what the sources say, where they disagree, and what remains uncertain.
3. Do not claim an independent verification pipeline ran or invent a truth score.
""",
}


def _find_builtin_skills_dir() -> Path | None:
    for base in (*Path(__file__).resolve().parents, Path.cwd()):
        cand = base / "config" / "skills"
        if cand.is_dir():
            return cand
    return None


class SkillDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str = ""
    triggers: tuple[str, ...] = Field(default_factory=tuple)
    required_tools: tuple[str, ...] = Field(default_factory=tuple)
    system_prompt_addon: str | None = None
    default_depth: str = "deep"
    agent: str = "auto"
    builtin: bool = False


class SkillLoader:
    """Discovers and parses custom skills (~/.lora/skills/*) and built-in skills (SKILL.md & skill.yaml)."""

    def __init__(self, skills_dir: Path | None = None, include_builtin: bool = False) -> None:
        self._skills_dir = Path(skills_dir) if skills_dir else None
        self._include_builtin = include_builtin
        self._skills: dict[str, SkillDefinition] = {}

    @property
    def skills(self) -> dict[str, SkillDefinition]:
        return dict(self._skills)

    def load_skills(self) -> dict[str, SkillDefinition]:
        """Scans configured skill directories and embedded built-ins for SKILL.md and skill.yaml definitions."""
        self._skills.clear()

        if self._include_builtin:
            for name, md_text in BUILTIN_SKILL_MDS.items():
                try:
                    self._skills[name] = self._parse_skill_md(md_text, fallback_name=name, is_builtin=True)
                except Exception as e:
                    logger.warning("Failed to parse embedded built-in skill %s: %s", name, e)
            builtin_dir = _find_builtin_skills_dir()
            if builtin_dir and builtin_dir.is_dir():
                self._scan_directory(builtin_dir, is_builtin=True)

        if self._skills_dir and self._skills_dir.is_dir():
            self._scan_directory(self._skills_dir, is_builtin=False)

        return dict(self._skills)

    def _scan_directory(self, directory: Path, is_builtin: bool = False) -> None:
        for item in sorted(directory.iterdir()):
            if not item.is_dir():
                continue
            md_file = item / "SKILL.md"
            yaml_file = item / "skill.yaml"
            if yaml_file.is_file():
                try:
                    with open(yaml_file, encoding="utf-8") as f:
                        raw = yaml.safe_load(f) or {}
                    skill = self._parse_skill_dict(raw, fallback_name=item.name, is_builtin=is_builtin)
                    self._skills[skill.name] = skill
                    logger.info("Loaded skill (yaml): %s", skill.name)
                except Exception as e:
                    logger.warning("Failed to load skill.yaml from %s: %s", yaml_file, e)
            elif md_file.is_file():
                try:
                    skill = self._parse_skill_md(md_file.read_text(encoding="utf-8"), fallback_name=item.name, is_builtin=is_builtin)
                    self._skills[skill.name] = skill
                    logger.info("Loaded skill (SKILL.md): %s", skill.name)
                except Exception as e:
                    logger.warning("Failed to load SKILL.md from %s: %s", md_file, e)

    def get_skill(self, name: str) -> SkillDefinition | None:
        if not self._skills:
            self.load_skills()
        clean = (name or "").strip().lower()
        if not clean or clean == "auto":
            return None
        if clean in self._skills:
            return self._skills[clean]
        for k, sk in self._skills.items():
            if k.lower() == clean or sk.name.lower() == clean or clean in k.lower():
                return sk
        return None

    def match_skill(self, prompt: str, explicit_skill: str | None = None) -> SkillDefinition | None:
        """Finds an explicitly selected skill or the first skill whose trigger phrases are contained in the prompt."""
        if not self._skills:
            self.load_skills()

        if explicit_skill and explicit_skill.lower() != "auto":
            chosen = self.get_skill(explicit_skill)
            if chosen is not None:
                return chosen

        prompt_clean = (prompt or "").lower().strip()
        ordered = sorted(self._skills.values(), key=lambda s: (s.builtin, s.name))
        for skill in ordered:
            for trigger in skill.triggers:
                if trigger.lower() in prompt_clean:
                    return skill

        return None

    def _parse_skill_md(self, content: str, fallback_name: str, is_builtin: bool = False) -> SkillDefinition:
        m = _FRONTMATTER_RE.match(content.strip())
        if not m:
            return SkillDefinition(
                name=fallback_name,
                description=fallback_name,
                system_prompt_addon=content.strip(),
                builtin=is_builtin,
            )
        front_raw, body = m.group(1), m.group(2).strip()
        meta = yaml.safe_load(front_raw) or {}
        if not isinstance(meta, dict):
            meta = {}
        meta.setdefault("system_prompt_addon", body)
        return self._parse_skill_dict(meta, fallback_name=fallback_name, is_builtin=is_builtin)

    def _parse_skill_dict(self, raw: dict[str, Any], fallback_name: str, is_builtin: bool = False) -> SkillDefinition:
        triggers_raw = raw.get("triggers", [])
        if isinstance(triggers_raw, str):
            triggers_tuple = (triggers_raw,)
        else:
            triggers_tuple = tuple(str(t) for t in triggers_raw if t)

        tools_raw = raw.get("required_tools", [])
        if isinstance(tools_raw, str):
            tools_tuple = (tools_raw,)
        else:
            tools_tuple = tuple(str(t) for t in tools_raw if t)

        return SkillDefinition(
            name=str(raw.get("name", fallback_name)),
            description=str(raw.get("description", "")),
            triggers=triggers_tuple,
            required_tools=tools_tuple,
            system_prompt_addon=raw.get("system_prompt_addon"),
            default_depth=str(raw.get("default_depth", "deep")),
            agent=str(raw.get("agent", "auto")),
            builtin=is_builtin,
        )
