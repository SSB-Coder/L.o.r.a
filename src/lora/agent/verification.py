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
from lora.verify.claims import ClaimDecomposer, extract_citation_ids

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
        # v1.5 Plan §3.3: Keep the model's exact num_ctx (8192) instead of shrinking to 4096,
        # otherwise Ollama unloads and reloads the model in VRAM on every verification step!
        base_opts["num_ctx"] = int(base_opts.get("num_ctx", 8192))
        base_opts["temperature"] = 0
        self.client, self.model, self.options = client, model, base_opts
        self.max_calls, self.calls = max_calls, 0
        self.succeeded = 0
        self.timed_out = 0

    async def generate(self, prompt: str, **_: Any) -> SimpleNamespace:
        if self.calls >= self.max_calls:
            raise RuntimeError("verifier call budget used")
        self.calls += 1
        text = ""
        try:
            async for delta in self.client.stream_chat(
                self.model,
                [{"role": "user", "content": prompt}],
                options=self.options,
                think=False,
                keep_alive="24h",
            ):
                text += delta
            self.succeeded += 1
            return SimpleNamespace(response=strip_think(text))
        except Exception:
            self.timed_out += 1
            raise


def _build_eight_stages(
    answer: str,
    cards: list[EvidenceCard],
    details: list[Any],
    composite_score: float,
    verdict: str,
    llm_calls: int = 0,
    llm_succeeded: int = 0,
    llm_timed_out: int = 0,
) -> list[dict[str, Any]]:
    """Build the 8-stage (V1-V8) verification breakdown for the UI Verification Dropdown."""
    total_claims = len(details)
    supported_claims = sum(1 for d in details if getattr(d, "is_supported", False))

    # V1: Citation Existence & Per-Claim Citation Coverage
    cited_ids = set(extract_citation_ids(answer or ""))
    valid_ids = {c.card_id for c in cards}
    phantom_ids = cited_ids - valid_ids
    decomposed = ClaimDecomposer().decompose(text=answer or "") if answer else []
    claims_with_cites = sum(1 for c in decomposed if c.cited_sources)
    cite_coverage = (claims_with_cites / max(1, len(decomposed))) if decomposed else (1.0 if cited_ids else 0.0)
    if phantom_ids:
        v1_score = max(0.20, round(0.60 * cite_coverage - 0.20 * len(phantom_ids), 2))
    elif cited_ids:
        v1_score = round(0.75 + 0.25 * cite_coverage, 2)
    else:
        v1_score = 0.50

    # V2: Atomic Claim Decomposition & Support Ratio
    v2_score = round(supported_claims / max(1, total_claims), 2) if total_claims > 0 else 0.0

    # V3: NLI Semantic Groundedness (true continuous mean NLI score — never inflated by binary support count)
    avg_nli = (
        sum(float(getattr(d, "nli_score", 0.0)) for d in details) / total_claims
        if total_claims
        else 0.0
    )
    v3_score = round(min(1.0, max(0.0, avg_nli)), 2)

    # V4: LLM Entailment Escalation & Spot-Check Audit
    if llm_timed_out > 0 and llm_succeeded == 0:
        v4_score = round(min(0.65, avg_nli * 0.75), 2)
        v4_detail = (
            f"LLM probe timed out ({llm_timed_out}/{max(1, llm_calls)}); "
            f"CPU NLI fallback ({supported_claims}/{total_claims} claims)"
        )
    elif llm_succeeded > 0:
        support_ratio = supported_claims / max(1, total_claims)
        v4_score = round(min(1.0, 0.65 * support_ratio + 0.35 * avg_nli), 2)
        timeout_note = f", {llm_timed_out} timed out" if llm_timed_out > 0 else ""
        v4_detail = f"{supported_claims}/{total_claims} claims verified ({llm_succeeded} LLM probe(s){timeout_note})"
    else:
        v4_score = round(min(0.88, avg_nli), 2)
        v4_detail = f"CPU NLI only (0 LLM probes; {supported_claims}/{total_claims} claims entailed)"

    # V5: Multi-Source Corroboration (measures actual cited source breadth + domain diversity)
    domains = {
        c.source_url.split("/")[2]
        for c in cards
        if c.source_url and "/" in c.source_url and len(c.source_url.split("/")) > 2
    }
    cited_valid_count = len(cited_ids & valid_ids)
    cite_source_ratio = cited_valid_count / max(1, len(cards))
    domain_ratio = min(1.0, len(domains) / max(1, min(3, len(cards)))) if cards else 0.0
    if len(cards) == 1:
        v5_score = round(0.85 * cite_source_ratio, 2)
    else:
        v5_score = round(0.55 * domain_ratio + 0.45 * cite_source_ratio, 2)

    # V6: Numerical & Entity Precision
    num_passed = sum(1 for d in details if getattr(d, "number_check_passed", True))
    v6_score = round(num_passed / max(1, total_claims), 2)

    # V7: CopyChecker (8-gram overlap) & Originality Gate (reports true originality %, not snapped to 100%)
    copy_res = CopyChecker(n_gram=8, max_overlap_ratio=0.35).check_overlap(
        answer or "", [c.exact_quote for c in cards]
    )
    orig_ratio = max(0.0, 1.0 - copy_res.overlap_ratio)
    v7_score = round(orig_ratio if copy_res.is_clean else min(0.54, orig_ratio), 2)

    # V8: Composite Faithfulness & Repair Gate (continuous blend of pipeline score and V1-V7 stage scores)
    stage_mean = (v1_score + v2_score + v3_score + v4_score + v5_score + v6_score + v7_score) / 7.0
    v8_score = round(0.55 * float(composite_score) + 0.45 * stage_mean, 2)
    if llm_timed_out > 0 and llm_succeeded == 0:
        v8_score = round(min(v8_score, 0.78), 2)

    def _st(s: float) -> str:
        return "pass" if s >= 0.80 else ("warn" if s >= 0.55 else "fail")

    v3_detail = (
        f"Mean entailment {int(round(v3_score * 100))}% across {total_claims} claim(s)"
        if avg_nli > 0
        else f"Lexical entailment {int(round(v3_score * 100))}% across {total_claims} claim(s)"
    )

    return [
        {
            "id": "V1",
            "name": "Citation Integrity",
            "score": round(v1_score, 2),
            "status": _st(v1_score),
            "detail": f"{cited_valid_count} valid source(s) cited ({claims_with_cites}/{max(1, len(decomposed))} claims), {len(phantom_ids)} phantom",
        },
        {
            "id": "V2",
            "name": "Claim Decomposition",
            "score": round(v2_score, 2),
            "status": _st(v2_score),
            "detail": f"{supported_claims}/{total_claims} atomic claim(s) supported",
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
            "detail": f"{cited_valid_count}/{len(cards)} source(s) cited across {max(1, len(domains))} domain(s)",
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
            "detail": f"Copy & Originality Gate ({int(round(orig_ratio * 100))}% original)",
        },
        {
            "id": "V8",
            "name": "Composite Truthfulness",
            "score": v8_score,
            "status": _st(v8_score),
            "detail": f"Verdict: {verdict} ({int(round(v8_score * 100))}% composite score)",
        },
    ]


async def verify_answer(answer: str, sources: list[Source], llm: VerifierLLM | None = None) -> Verification:
    from lora.verify.pipeline import VerificationPipeline

    cards = cards_from_sources(sources)
    if not cards or not answer.strip():
        return Verification("unchecked", 0.0, 0, 0)
    pipe = VerificationPipeline(ollama_client=llm)
    report = await pipe.verify(answer, cards)
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
    v4_stats = pipe.last_v4_stats
    llm_calls = max(v4_stats.get("attempted", 0), getattr(llm, "calls", 0) if llm else 0)
    llm_succeeded = v4_stats.get("succeeded", getattr(llm, "succeeded", 0) if llm else 0)
    llm_timed_out = v4_stats.get("timed_out", getattr(llm, "timed_out", 0) if llm else 0)
    stages = _build_eight_stages(
        answer=answer,
        cards=cards,
        details=details,
        composite_score=comp_score,
        verdict=verdict_str,
        llm_calls=llm_calls,
        llm_succeeded=llm_succeeded,
        llm_timed_out=llm_timed_out,
    )
    v8_stage_score = float(stages[-1]["score"]) if stages else comp_score
    supported_cnt = sum(1 for d in details if d.is_supported)
    logger.info(
        "8-Way Verification: verdict=%s score=%.2f (%d/%d claims) [%s]",
        verdict_str,
        v8_stage_score,
        supported_cnt,
        len(details),
        ", ".join(f"{s['id']}={int(round(s['score']*100))}%" for s in stages),
    )
    return Verification(
        verdict=verdict_str,
        score=v8_stage_score,
        supported=supported_cnt,
        total=len(details),
        unverified=list(report.unverified_claims),
        claims=claims,
        repaired_answer=str(report.repaired_answer or ""),
        stages=stages,
    )
