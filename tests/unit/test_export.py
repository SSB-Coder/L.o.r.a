"""Unit tests for ReportExporter and export_report tool/command wiring."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from lora.storage.export import ReportExporter
from lora.tools.registry import ToolRegistry
from lora.safety.policy import ActionRiskTier
from lora.agent.commands import parse_command


def test_report_exporter_markdown(tmp_path: Path):
    exporter = ReportExporter(outputs_dir=tmp_path)
    title = "Quantum Computing Survey"
    content = "## Abstract\nQuantum computing leverages superposition."
    md_file = exporter.export_markdown(title=title, content=content)

    assert md_file.exists()
    assert md_file.name == "quantum_computing_survey.md"
    text = md_file.read_text(encoding="utf-8")
    assert "# Quantum Computing Survey" in text
    assert "Exported by L.O.R.A" in text
    assert "Quantum computing leverages superposition." in text


def test_report_exporter_html(tmp_path: Path):
    exporter = ReportExporter(outputs_dir=tmp_path)
    title = "AI Verification Architecture"
    content = "# Introduction\nV1 through V8 verification checks."
    html_file = exporter.export_html(title=title, markdown_content=content)

    assert html_file.exists()
    assert html_file.name == "ai_verification_architecture.html"
    text = html_file.read_text(encoding="utf-8")
    assert "<title>AI Verification Architecture - L.O.R.A Report</title>" in text
    assert "<h1>Introduction</h1>" in text


def test_export_report_tool_in_registry(tmp_path: Path):
    reg = ToolRegistry(media_db=MagicMock())
    # Override outputs_dir to tmp_path for test isolation
    reg.report_exporter = ReportExporter(outputs_dir=tmp_path)

    tool = reg._tools.get("export_report")
    assert tool is not None
    assert tool.tier == ActionRiskTier.TIER2

    res = reg._handle_export_report(
        title="Federated Learning",
        content="Federated learning protects user privacy.",
        format="both",
    )
    assert res["status"] == "success"
    assert len(res["paths"]) == 2
    assert any(p.endswith(".md") for p in res["paths"])
    assert any(p.endswith(".html") for p in res["paths"])


def test_export_report_command_parsing():
    cmd1 = parse_command("export report called My Research Findings")
    assert cmd1 is not None
    assert cmd1.kind == "export_report"
    assert cmd1.target == "My Research Findings"
    assert cmd1.args["format"] == "markdown"

    cmd2 = parse_command("save the report as html titled Neural Networks")
    assert cmd2 is not None
    assert cmd2.kind == "export_report"
    assert cmd2.target == "Neural Networks"
    assert cmd2.args["format"] == "html"

    cmd3 = parse_command("export report as both")
    assert cmd3 is not None
    assert cmd3.kind == "export_report"
    assert cmd3.args["format"] == "both"
