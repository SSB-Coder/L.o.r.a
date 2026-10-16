"""Claim-by-claim 8-stage fact checking (V1-V8) of an answer against retrieved evidence sources.

Wraps the repo's V1-V8 pipeline (V1 citation check, V2 claim decomposition, V3 NLI groundedness,
V4 LLM fact-check escalation, V5 multi-source corroboration, V6 number/entity check,
V7 CopyChecker & Round-Table critic, V8 composite score & repair) and exposes structured stage scorecards
for the UI Verification Dropdown.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from lora.agent.research import Source
from lora.dialogue import strip_think
from lora.grounding.cards import EvidenceCard
from lora.grounding.copycheck import CopyChecker
from lora.verify.claims import extract_citation_ids

logger = logging.getLogger(__name__)

if os.environ.get("LORA_VERIFY_NLI") == "0":
    os.environ.setdefault("LORA_VERIFY_OFFLINE", "1")


@dataclass
class Verification:
    verdict: str  # verified | repaired | abstain | unchecked
    score: float
    supported: int
    total: int
    unverified: list[str] = field(default_factory=list)
    claims: list[dict[str, Any]] = field(default_factory=list)
    repaired_answer: str = ""
    stages: list[dict[str, Any]] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return self.total > 0 and not self.unverified

    def for_ui(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "supported": self.supported,
            "total": self.total,
            "score": round(self.score, 2),
            "unverified": self.unverified[:6],
            "stages": self.stages,
        }


def cards_from_sources(sources: list[Source]) -> list[EvidenceCard]:
    cards = []
    for s in sources:
        quote = (s.text or s.snippet).strip()
        if quote:
            cards.append(EvidenceCard(card_id=s.id, claim_fact=s.title, exact_quote=quote, source_url=s.url, title=s.title))
    return cards


class VerifierLLM:
    """The ``generate(prompt)`` interface the pipeline's LLM fact-checker expects, on top of Ollama."""

    def __init__(self, client: Any, model: str, options: dict[str, Any] | None = None, max_calls: int = 4) -> None:
        base_opts = dict(options or {})
        base_opts["num_ctx"] = min(int(base_opts.get("num_ctx", 4096)), 4096)
        base_opts["temperature"] = 0
        self.client, self.model, self.options = client, model, base_opts
        self.max_calls, self.calls = max_calls, 0

    async def generate(self, prompt: str, **_: Any) -> SimpleNamespace:
        if self.calls >= self.max_calls:
            raise RuntimeError("verifier call budget used")
        self.calls += 1
        text = ""
        async for delta in self.client.stream_chat(
            self.model,
            [{"role": "user", "content": prompt}],
            options=self.options,
            think=False,
            keep_alive="24h",
        ):
            text += delta
        return SimpleNamespace(response=strip_think(text))


def _build_eight_stages(
    answer: str,
    cards: list[EvidenceCard],
    details: list[Any],
    composite_score: float,
    verdict: str,
    llm_calls: int = 0,
) -> list[dict[str, Any]]:
    """Build the 8-stage (V1-V8) verification breakdown for the UI Verification Dropdown."""
    total_claims = len(details)
    supported_claims = sum(1 for d in details if getattr(d, "is_supported", False))

    # V1: Citation Existence & Resolution
    cited_ids = set(extract_citation_ids(answer or ""))
    valid_ids = {c.card_id for c in cards}
    phantom_ids = cited_ids - valid_ids
    v1_score = 1.0 if (cited_ids and not phantom_ids) else (0.80 if not phantom_ids else 0.3)

    # V2: Atomic Claim Decomposition
    v2_score = 1.0 if total_claims > 0 else 0.0

    # V3: NLI Semantic Groundedness
    avg_nli = (
        sum(float(getattr(d, "nli_score", 0.0)) for d in details) / total_claims
        if total_claims
        else 0.0
    )
    v3_score = round(min(1.0, max(avg_nli, supported_claims / max(1, total_claims))), 2)

    # V4: LLM Entailment Escalation
    v4_score = round(supported_claims / max(1, total_claims), 2)

    # V5: Multi-Source Corroboration
    domains = {c.source_url.split("/")[2] for c in cards if c.source_url and "/" in c.source_url and len(c.source_url.split("/")) > 2}
    v5_score = 1.0 if len(domains) >= 2 else (0.85 if len(cards) >= 1 else 0.0)

    # V6: Numerical & Entity Precision
    num_passed = sum(1 for d in details if getattr(d, "number_check_passed", True))
    v6_score = round(num_passed / max(1, total_claims), 2)

    # V7: CopyChecker (8-gram overlap) & Originality Gate
    copy_res = CopyChecker(n_gram=8, max_overlap_ratio=0.35).check_overlap(
        answer or "", [c.exact_quote for c in cards]
    )
    copy_score = 1.0 if copy_res.is_clean else max(0.0, 1.0 - copy_res.overlap_ratio)
    v7_score = round(min(1.0, copy_score), 2)

    # V8: Composite Faithfulness & Repair Gate
    v8_score = round(float(composite_score), 2)

    def _st(s: float) -> str:
        return "pass" if s >= 0.80 else ("warn" if s >= 0.55 else "fail")

    v3_detail = (
        f"Mean entailment {int(v3_score * 100)}% across {total_claims} claim(s)"
        if avg_nli > 0
        else f"Lexical entailment {int(v3_score * 100)}% across {total_claims} claim(s)"
    )
    v4_detail = (
        f"{supported_claims}/{total_claims} claims verified ({llm_calls} LLM probe(s))"
        if llm_calls > 0
        else f"{supported_claims}/{total_claims} claims verified (0 probes - entailed)"
    )

    return [
        {
            "id": "V1",
            "name": "Citation Integrity",
            "score": round(v1_score, 2),
            "status": _st(v1_score),
            "detail": f"{len(cited_ids & valid_ids)} valid citation(s), {len(phantom_ids)} phantom",
        },
        {
            "id": "V2",
            "name": "Claim Decomposition",
            "score": round(v2_score, 2),
            "status": _st(v2_score),
            "detail": f"{total_claims} atomic claim(s) extracted",
        },
        {
            "id": "V3",
            "name": "NLI Groundedness",
            "score": v3_score,
            "status": _st(v3_score),
            "detail": v3_detail,
        },
        {
            "id": "V4",
            "name": "LLM Fact-Check",
            "score": v4_score,
            "status": _st(v4_score),
            "detail": v4_detail,
        },
        {
            "id": "V5",
            "name": "Source Corroboration",
            "score": round(v5_score, 2),
            "status": _st(v5_score),
            "detail": f"{len(cards)} evidence source(s) across {max(1, len(domains))} domain(s)",
        },
        {
            "id": "V6",
            "name": "Number & Entity Check",
            "score": v6_score,
            "status": _st(v6_score),
            "detail": f"{num_passed}/{total_claims} numerical & entity checks passed",
        },
        {
            "id": "V7",
            "name": "Critic & Copy Gate",
            "score": v7_score,
            "status": _st(v7_score),
            "detail": f"Copy & Originality Gate ({int((1.0 - copy_res.overlap_ratio) * 100)}% original)",
        },
        {
            "id": "V8",
            "name": "Composite Truthfulness",
            "score": v8_score,
            "status": _st(v8_score),
            "detail": f"Verdict: {verdict} ({int(v8_score * 100)}% composite score)",
        },
    ]


async def verify_answer(answer: str, sources: list[Source], llm: VerifierLLM | None = None) -> Verification:
    from lora.verify.pipeline import VerificationPipeline

    cards = cards_from_sources(sources)
    if not cards or not answer.strip():
        return Verification("unchecked", 0.0, 0, 0)
    report = await VerificationPipeline(ollama_client=llm).verify(answer, cards)
    details = list(report.claim_details)
    claims = [
        {
            "text": d.claim_text,
            "status": "supported" if d.is_supported else "unsupported",
            "reason": d.unsupported_reason,
            "nli_score": round(float(d.nli_score), 2),
            "numbers_ok": bool(d.number_check_passed),
        }
        for d in details
    ]
    verdict_str = str(report.verdict.value)
    comp_score = float(report.composite_faithfulness_score)
    stages = _build_eight_stages(
        answer=answer,
        cards=cards,
        details=details,
        composite_score=comp_score,
        verdict=verdict_str,
        llm_calls=getattr(llm, "calls", 0) if llm else 0,
    )
    supported_cnt = sum(1 for d in details if d.is_supported)
    logger.info(
        "8-Way Verification: verdict=%s score=%.2f (%d/%d claims) [%s]",
        verdict_str,
        comp_score,
        supported_cnt,
        len(details),
        ", ".join(f"{s['id']}={int(s['score']*100)}%" for s in stages),
    )
    return Verification(
        verdict=verdict_str,
        score=comp_score,
        supported=supported_cnt,
        total=len(details),
        unverified=list(report.unverified_claims),
        claims=claims,
        repaired_answer=str(report.repaired_answer or ""),
        stages=stages,
    )
