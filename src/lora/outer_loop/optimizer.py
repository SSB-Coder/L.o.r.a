"""Trace-Driven Prompt Optimizer for L.O.R.A Outer Loop."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lora.outer_loop.genome import EditValidator, PromptGenome
from lora.outer_loop.scorer import ScoreBreakdown


class PromptMutationProposal(BaseModel):
    """Structured proposal representing a single targeted prompt edit."""

    model_config = ConfigDict(frozen=True)

    target_flaw: str
    action: str = "append_rule"
    rule_to_add: str
    rationale: str = ""


class TraceDiagnostic(BaseModel):
    """Diagnosis of observed synthesis and verification failures."""

    model_config = ConfigDict(frozen=True)

    target_flaws: tuple[str, ...] = Field(default_factory=tuple)
    priority_dimension: str = "faithfulness"
    feedback_summary: str = ""


class TraceAnalyzer:
    """Extracts concrete failure signals from Phase 7 and Phase 9 verification logs."""

    def analyze(
        self,
        verification_report: dict[str, Any],
        roundtable_report: dict[str, Any],
        score_breakdown: ScoreBreakdown,
    ) -> TraceDiagnostic:
        """Parse reports into actionable failure diagnoses."""
        flaws = []

        # 1. Unverified claims from Phase 7
        for claim in verification_report.get("unverified_critical_claims", []):
            flaws.append(f"Unverified claim: '{claim}'")

        # 2. Veto reasons from Phase 9 Round Table
        for reason in roundtable_report.get("veto_reasons", []):
            flaws.append(f"Round Table veto: {reason}")

        # 3. Identify weakest scoring dimension
        components = score_breakdown.component_scores or {}
        # Exclude efficiency from being the primary mutation target
        eval_components = {k: v for k, v in components.items() if k != "efficiency"}
        if eval_components:
            priority_dim = min(eval_components, key=eval_components.get)
        else:
            priority_dim = "faithfulness"

        if score_breakdown.feedback_text:
            flaws.append(score_breakdown.feedback_text)

        return TraceDiagnostic(
            target_flaws=flaws or ["General rubric refinement."],
            priority_dimension=priority_dim,
            feedback_summary=score_breakdown.feedback_text,
        )


class PromptOptimizerAgent:
    """Formulates surgical, trace-grounded mutations to improve prompt genomes."""

    def __init__(self, edit_validator: EditValidator | None = None) -> None:
        self.edit_validator = edit_validator or EditValidator()

    def propose_mutation(self, diagnostic: TraceDiagnostic) -> PromptMutationProposal:
        """Formulate a targeted mutation proposal addressing the highest priority flaw."""
        target_flaw = diagnostic.target_flaws[0] if diagnostic.target_flaws else "Improve synthesis precision."
        dim = diagnostic.priority_dimension

        if dim == "faithfulness":
            rule = "Strictly assert only facts directly corroborated by provided evidence cards."
            rationale = "Eliminates unsupported or speculative claims."
        elif dim == "citation_precision":
            rule = "Append inline bracketed citation tags [1] to every factual assertion."
            rationale = "Ensures citation traceability for all claims."
        elif dim == "coverage":
            rule = "Synthesize all key definitions and experimental outcomes mentioned in cards."
            rationale = "Increases coverage of source evidence."
        elif dim == "depth_fit":
            rule = "Match explanation length and technical density strictly to requested depth."
            rationale = "Improves depth mode alignment."
        else:
            rule = "Maintain clear, unambiguous structure with verified evidence backing."
            rationale = "General clarity and precision refinement."

        return PromptMutationProposal(
            target_flaw=target_flaw,
            action="append_rule",
            rule_to_add=rule,
            rationale=rationale,
        )

    def apply_mutation(
        self,
        base_genome: PromptGenome,
        proposal: PromptMutationProposal,
    ) -> PromptGenome:
        """Apply the proposed edit to create a candidate genome, enforcing edit constraints."""
        if not proposal.target_flaw.strip():
            raise ValueError("Target flaw cannot be empty. Every mutation must state its concrete target.")

        if not proposal.rule_to_add.strip():
            raise ValueError("Rule to add cannot be empty.")

        new_rules = tuple([*base_genome.refinement_rules, proposal.rule_to_add.strip()])
        candidate = base_genome.model_copy(
            update={
                "refinement_rules": new_rules,
                "version": base_genome.version + 1,
            }
        )

        # Validate with EditValidator
        is_valid, reason = self.edit_validator.validate_edit(base=base_genome, candidate=candidate)
        if not is_valid:
            raise ValueError(f"Mutation rejected by EditValidator: {reason}")

        return candidate
