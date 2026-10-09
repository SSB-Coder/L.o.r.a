from pathlib import Path

import yaml

from lora.skills.loader import SkillDefinition, SkillLoader
from lora.storage.export import ReportExporter
from lora.storage.memory import MemoryStore


def test_skills_loader_discovery_and_matching(tmp_path: Path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    # Create skill 1: Arxiv Monitor
    skill1_dir = skills_dir / "arxiv_monitor"
    skill1_dir.mkdir()
    skill1_yaml = skill1_dir / "skill.yaml"
    skill1_data = {
        "name": "arxiv_monitor",
        "description": "Monitors latest physics preprints on arXiv",
        "triggers": ["check arxiv preprints", "arxiv physics"],
        "required_tools": ["arxiv_search"],
        "system_prompt_addon": "Prioritize high-energy physics papers.",
        "default_depth": "deep",
    }
    with open(skill1_yaml, "w", encoding="utf-8") as f:
        yaml.dump(skill1_data, f)

    loader = SkillLoader(skills_dir=skills_dir)
    loaded_skills = loader.load_skills()

    assert len(loaded_skills) == 1
    assert "arxiv_monitor" in loaded_skills
    skill = loaded_skills["arxiv_monitor"]
    assert isinstance(skill, SkillDefinition)
    assert skill.default_depth == "deep"

    # Match skill from prompt
    matched = loader.match_skill("Can you check arxiv preprints for quantum gravity?")
    assert matched is not None
    assert matched.name == "arxiv_monitor"

    # Unmatched prompt
    no_match = loader.match_skill("What is the weather today?")
    assert no_match is None


def test_memory_store_read_and_append(tmp_path: Path):
    profile_file = tmp_path / "user_profile.md"
    profile_file.write_text("# Initial Profile\n- User likes brevity\n", encoding="utf-8")

    memory = MemoryStore(profile_path=profile_file)
    initial_text = memory.get_profile_markdown()
    assert "User likes brevity" in initial_text

    # Append note
    memory.append_note(category="Coding Preferences", note="Prefers strict type annotations")
    updated_text = memory.get_profile_markdown()
    assert "Coding Preferences" in updated_text
    assert "Prefers strict type annotations" in updated_text


def test_report_exporter_markdown_and_html(tmp_path: Path):
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()

    exporter = ReportExporter(outputs_dir=outputs_dir)

    # Export markdown
    md_path = exporter.export_markdown(
        title="Quantum Supremacy 2026",
        content="# Quantum Supremacy\n\nRecent benchmark results show...",
    )
    assert md_path.is_file()
    assert md_path.suffix == ".md"
    assert "Quantum Supremacy" in md_path.read_text(encoding="utf-8")

    # Export html
    html_path = exporter.export_html(
        title="Quantum Supremacy 2026",
        markdown_content="# Quantum Supremacy\n\nRecent benchmark results...",
    )
    assert html_path.is_file()
    assert html_path.suffix == ".html"
    assert "<html" in html_path.read_text(encoding="utf-8").lower()
