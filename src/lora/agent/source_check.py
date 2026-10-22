"""Cheap reference integrity checks. These do not establish factual accuracy."""

import re
from collections.abc import Iterable
from typing import Any

from lora.agent.research import Source


def check_references(answer: str, sources: Iterable[Source]) -> dict[str, Any]:
    available = {source.id for source in sources}
    # Only inspect citation groups, not bare identifiers in prose or code.
    groups = re.findall(r"\[(S\d+(?:\s*[,;]\s*S\d+)*)\]", answer)
    cited = {identifier for group in groups for identifier in re.findall(r"S\d+", group)}
    unknown = sorted(cited - available)
    if unknown:
        status = "invalid_references"
        note = "Some citations do not match the supplied sources: " + ", ".join(unknown) + "."
    elif not available:
        status, note = "no_sources", "No source text was available."
    elif not cited:
        status, note = "missing_citations", "Sources were read, but the answer has no inline citations."
    else:
        status, note = "references_checked", "Citation references checked; claims were not independently fact-checked."
    return {"status": status, "note": note, "cited_sources": len(cited & available),
            "total_sources": len(available), "unknown_ids": unknown}


def source_metadata(answer: str, sources: Iterable[Source]) -> dict[str, Any]:
    return {"verdict": "unchecked", "source_check": check_references(answer, sources)}


def mobile_answer(result: dict[str, Any]) -> dict[str, Any]:
    """Share one presentation contract between completed jobs and WebSocket events."""
    metadata = result.get("verification") or {}
    return {
        "text": result.get("answer", ""),
        "kind": result.get("kind", "chat"),
        "source_check": metadata.get("source_check"),
        "citations": [{"id": source.get("id", f"S{i + 1}"), "title": source.get("title", ""),
                       "source_url": source.get("url", ""), "quote": source.get("quote", ""),
                       "tier": source.get("tier", 2)}
                      for i, source in enumerate(result.get("sources") or []) if isinstance(source, dict)],
    }
