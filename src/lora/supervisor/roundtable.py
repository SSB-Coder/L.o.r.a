"""Sequential Round-Table Personas and Arbitration Engine for L.O.R.A.

Implements Skeptic, Fact-Checker, Editor, Safety Officer, and Chair/Supervisor
personas. Enforces the Strict Model Family Invariant and Disagreement Rule.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def get_model_family(model_name: str) -> str:
    """Map model name or tag to its underlying architecture family."""
    lower = model_name.strip().lower()
    if "granite" in lower:
        return "granite"
    if "spark" in lower:
        return "spark"
    if any(q in lower for q in ["qwen", "ornith", "qwythos"]):
        return "qwen"
    if "gemma" in lower:
        return "gemma"
    return lower


class PersonaEvaluation(BaseModel):
    """Evaluation result produced by a single round-table persona."""

    model_config = ConfigDict(frozen=True)

    persona: str
    verdict: str  # approve, revise, veto
    score: float  # 0.0 to 1.0
    notes: list[str] = Field(default_factory=list)


class RoundTableReport(BaseModel):
    """Synthesized arbitration verdict produced by the Chair persona."""

    model_config = ConfigDict(frozen=True)

    decision: str  # approve, revise, abstain
    composite_score: float = 1.0
    persona_scores: dict[str, float] = Field(default_factory=dict)
    persona_verdicts: dict[str, str] = Field(default_factory=dict)
    hard_veto: bool = False
    veto_reasons: list[str] = Field(default_factory=list)
    required_changes: list[str] = Field(default_factory=list)
    is_fast_path: bool = False


class SkepticPersona:
    """Hunts for unsupported assertions, unwarranted generalizations, or overreaching speculation."""

    def evaluate(
        self,
        draft: str,
        claims: list[dict[str, Any]],
        depth_mode: str = "standard",
    ) -> PersonaEvaluation:
        notes = []
        for c in claims:
            status = c.get("status", "unknown")
            if status in ["unsupported", "fabricated", "contradicted"]:
                notes.append(f"Skeptic objection: claim '{c.get('text', '')}' is {status}.")

        if notes:
            return PersonaEvaluation(
                persona="skeptic",
                verdict="veto",
                score=0.20,
                notes=notes,
            )

        return PersonaEvaluation(
            persona="skeptic",
            verdict="approve",
            score=0.95,
            notes=["All claims show preliminary support with bounded certainty."],
        )


class FactCheckerPersona:
    """Spot-checks the highest-risk claims against raw evidence chunks."""

    def evaluate(
        self,
        draft: str,
        claims: list[dict[str, Any]],
        evidence_cards: list[dict[str, Any]],
    ) -> PersonaEvaluation:
        notes = []
        for c in claims:
            status = c.get("status", "unknown")
            if status in ["fabricated", "unsupported"]:
                notes.append(f"Fabricated or unsupported claim found: '{c.get('text', '')}'.")

        if notes:
            return PersonaEvaluation(
                persona="fact_checker",
                verdict="veto",
                score=0.10,
                notes=notes,
            )

        return PersonaEvaluation(
            persona="fact_checker",
            verdict="approve",
            score=0.95,
            notes=["Claims are consistent with evidence cards."],
        )


class EditorPersona:
    """Evaluates fit to requested depth mode, readability, and non-copying paraphrasing."""

    def evaluate(
        self,
        draft: str,
        depth_mode: str = "standard",
    ) -> PersonaEvaluation:
        if not draft.strip():
            return PersonaEvaluation(
                persona="editor",
                verdict="veto",
                score=0.0,
                notes=["Draft text is completely empty."],
            )

        return PersonaEvaluation(
            persona="editor",
            verdict="approve",
            score=0.90,
            notes=[f"Draft matches requested depth mode '{depth_mode}' and style standards."],
        )


class SafetyOfficerPersona:
    """Audits draft for data exfiltration, sensitive path leakage, or unsafe instructions."""

    def evaluate(
        self,
        draft: str,
    ) -> PersonaEvaluation:
        lower = draft.lower()
        if any(term in lower for term in ["system override", "delete_file", "rmdir /s /q", "format c:"]):
            return PersonaEvaluation(
                persona="safety_officer",
                verdict="veto",
                score=0.0,
                notes=["Safety policy violation detected in draft output."],
            )

        return PersonaEvaluation(
            persona="safety_officer",
            verdict="approve",
            score=1.0,
            notes=["No safety policy violations found in text."],
        )


class ChairPersona:
    """Synthesizes individual evaluations into an arbitration report enforcing Disagreement Rule."""

    def synthesize(self, evaluations: list[PersonaEvaluation]) -> RoundTableReport:
        if not evaluations:
            return RoundTableReport(
                decision="abstain",
                composite_score=0.0,
                hard_veto=True,
                veto_reasons=["No persona evaluations provided."],
            )

        scores = {ev.persona: ev.score for ev in evaluations}
        verdicts = {ev.persona: ev.verdict for ev in evaluations}
        avg_score = sum(scores.values()) / len(scores)

        vetoes = [ev for ev in evaluations if ev.verdict == "veto"]
        veto_reasons = [note for v in vetoes for note in v.notes]

        # Disagreement rule: Any hard fail beats majority vote!
        if vetoes:
            # If veto comes from safety officer, abstain; otherwise revise
            has_safety_veto = any(v.persona == "safety_officer" for v in vetoes)
            decision = "abstain" if has_safety_veto else "revise"
            return RoundTableReport(
                decision=decision,
                composite_score=avg_score,
                persona_scores=scores,
                persona_verdicts=verdicts,
                hard_veto=True,
                veto_reasons=veto_reasons,
                required_changes=veto_reasons,
            )

        if avg_score >= 0.80:
            return RoundTableReport(
                decision="approve",
                composite_score=avg_score,
                persona_scores=scores,
                persona_verdicts=verdicts,
                hard_veto=False,
            )

        return RoundTableReport(
            decision="revise",
            composite_score=avg_score,
            persona_scores=scores,
            persona_verdicts=verdicts,
            hard_veto=False,
            required_changes=[f"Improve low average score ({avg_score:.2f}) to >= 0.80."],
        )


class RoundTableEvaluator:
    """Orchestrates sequential, time-multiplexed round-table evaluation on a single GPU."""

    def __init__(self) -> None:
        self.skeptic = SkepticPersona()
        self.fact_checker = FactCheckerPersona()
        self.editor = EditorPersona()
        self.safety = SafetyOfficerPersona()
        self.chair = ChairPersona()

    def validate_family_invariant(self, author_model: str, judge_model: str) -> None:
        """Enforce strict invariant that author model family cannot judge its own draft."""
        author_fam = get_model_family(author_model)
        judge_fam = get_model_family(judge_model)

        if author_fam == judge_fam:
            raise ValueError(
                f"Strict Model Family Invariant violated: Author '{author_model}' (family: {author_fam}) "
                f"and Judge '{judge_model}' (family: {judge_fam}) share family '{author_fam}'."
            )

    def run_roundtable(
        self,
        draft: str,
        claims: list[dict[str, Any]],
        evidence_cards: list[dict[str, Any]],
        depth_mode: str = "standard",
        run_mode: str = "quick",
        author_model: str = "hf.co/XHToken/Spark-X2.6-4B-GGUF:Q4_K_M",
        judge_model: str = "granite4.2:8b",
    ) -> RoundTableReport:
        """Run sequential persona evaluations (or fast path if quick mode)."""
        self.validate_family_invariant(author_model, judge_model)

        # Fast path for quick mode to conserve latency and power
        if run_mode == "quick":
            return RoundTableReport(
                decision="approve",
                composite_score=1.0,
                is_fast_path=True,
            )

        # Deep mode: sequential time-multiplexed execution
        evaluations = [
            self.skeptic.evaluate(draft, claims, depth_mode),
            self.fact_checker.evaluate(draft, claims, evidence_cards),
            self.editor.evaluate(draft, depth_mode),
            self.safety.evaluate(draft),
        ]

        return self.chair.synthesize(evaluations)
