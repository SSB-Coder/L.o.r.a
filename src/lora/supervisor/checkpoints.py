"""Supervisor Checkpoints for L.O.R.A.

Implements Plan Review, Action Review (deterministic ActionGate first + sanity check),
Evidence Review, Answer Review, and Anomaly Watcher (loop & injection detection).
"""

import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lora.safety.gate import ActionGate
from lora.safety.policy import PolicyVerdict
from lora.supervisor.state import AnomalyReport, SupervisorAction


class ReviewResult(BaseModel):
    """Immutable outcome from a supervisor checkpoint review."""

    model_config = ConfigDict(frozen=True)

    approved: bool
    verdict: str  # approved, hard_deny, reject, sanity_failed, revise, abstain, insufficient_evidence
    reason: str = ""
    required_changes: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class AnomalyWatcher:
    """Monitors workflow execution for loops, injection traps, and scope drift."""

    INJECTION_PATTERNS = [
        re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions", re.IGNORECASE),
        re.compile(r"system\s+override", re.IGNORECASE),
        re.compile(r"disregard\s+(?:all\s+)?prior\s+instructions", re.IGNORECASE),
        re.compile(r"developer\s+mode\s+enabled", re.IGNORECASE),
        re.compile(r"you\s+must\s+execute\s*:\s*(?:format|delete|rmdir)", re.IGNORECASE),
    ]

    SENSITIVE_PATHS = ["system32", "windows", "registry", "/etc/shadow", "sam"]

    def __init__(self, max_repeated_actions: int = 3) -> None:
        self.max_repeated_actions = max_repeated_actions

    def _normalize_val(self, val: Any) -> Any:
        if isinstance(val, str):
            return val.strip().lower()
        if isinstance(val, dict):
            return {k: self._normalize_val(v) for k, v in sorted(val.items())}
        if isinstance(val, list):
            return [self._normalize_val(i) for i in val]
        return val

    def check_repetition(
        self,
        current_action: SupervisorAction,
        action_history: list[SupervisorAction],
    ) -> AnomalyReport:
        """Detect if the identical action is being executed in a repetitive loop."""
        norm_tool = current_action.tool.strip().lower()
        norm_params = self._normalize_val(current_action.params)

        matches = 0
        for past in action_history:
            if past.tool.strip().lower() == norm_tool:
                if self._normalize_val(past.params) == norm_params:
                    matches += 1

        if matches + 1 >= self.max_repeated_actions:
            return AnomalyReport(
                detected=True,
                anomaly_type="loop_detected",
                message=f"Repeated action '{current_action.tool}' with identical parameters {matches + 1} times.",
                details={"tool": current_action.tool, "repetitions": matches + 1},
            )

        return AnomalyReport(detected=False)

    def check_injection(self, text: str) -> AnomalyReport:
        """Inspect retrieved text for prompt injection patterns attempting to hijack control."""
        for pattern in self.INJECTION_PATTERNS:
            if pattern.search(text):
                return AnomalyReport(
                    detected=True,
                    anomaly_type="prompt_injection_trap",
                    message="Potential prompt injection instruction detected in text.",
                    details={"pattern": pattern.pattern},
                )
        return AnomalyReport(detected=False)

    def check_scope_drift(
        self,
        user_request: str,
        proposed_action: SupervisorAction,
    ) -> AnomalyReport:
        """Detect if proposed action targets sensitive areas completely outside user request scope."""
        req_lower = user_request.lower()
        target_str = str(proposed_action.params.get("target", "") or proposed_action.params.get("path", "")).lower()

        # Check sensitive OS paths
        if any(sp in target_str for sp in self.SENSITIVE_PATHS):
            if not any(sp in req_lower for sp in self.SENSITIVE_PATHS):
                return AnomalyReport(
                    detected=True,
                    anomaly_type="scope_drift",
                    message=f"Action targeting sensitive path '{target_str}' unrelated to request '{user_request}'.",
                    details={"target": target_str},
                )

        return AnomalyReport(detected=False)


class PlanReviewer:
    """Verifies that generated execution plans are minimal, necessary, and safe."""

    FORBIDDEN_TOOLS = {
        "delete",
        "delete_file",
        "delete_dir",
        "rmdir",
        "remove_file",
        "destroy",
        "format",
        "unlink",
    }

    def review_plan(
        self,
        user_request: str,
        planned_actions: list[SupervisorAction],
    ) -> ReviewResult:
        """Review plan prior to execution."""
        if not planned_actions:
            return ReviewResult(
                approved=False,
                verdict="reject",
                reason="Plan contains no actions.",
            )

        for action in planned_actions:
            tool_name = action.tool.strip().lower()
            if tool_name in self.FORBIDDEN_TOOLS or any(fb in tool_name for fb in ["delete", "destroy", "format", "unlink"]):
                return ReviewResult(
                    approved=False,
                    verdict="hard_deny",
                    reason=f"Forbidden or destructive tool '{action.tool}' detected in plan.",
                    required_changes=["Remove all destructive tool references."],
                )

        return ReviewResult(
            approved=True,
            verdict="approved",
            reason="Plan verified safe and consistent.",
        )


class ActionReviewer:
    """Evaluates individual actions: deterministic ActionGate policy FIRST, then sanity check."""

    def __init__(
        self,
        action_gate: ActionGate,
        sanity_check_fn: Callable[[str, SupervisorAction], bool] | None = None,
    ) -> None:
        self.action_gate = action_gate
        self.sanity_check_fn = sanity_check_fn or self._default_sanity_check

    def _default_sanity_check(self, user_request: str, action: SupervisorAction) -> bool:
        """Deterministic sanity check verifying that Tier >= 1 actions directly serve the user's request."""
        req = user_request.lower()

        if action.tool == "open_app":
            app = str(action.params.get("app_name", "")).lower()
            # Common app aliases
            app_terms = [app]
            if app in ["calc", "calculator"]:
                app_terms.extend(["calc", "calculator"])
            elif app in ["notepad", "editor"]:
                app_terms.extend(["notepad", "editor"])
            elif app in ["browser", "chrome", "firefox"]:
                app_terms.extend(["browser", "chrome", "firefox", "internet", "web"])
            return any(term in req for term in app_terms)

        if action.tool == "open_path":
            target = str(action.params.get("target", "") or action.params.get("path", "")).lower()
            return any(word in req for word in target.replace("/", " ").replace("\\", " ").split() if len(word) > 2)

        return True

    def review_action(
        self,
        user_request: str,
        action: SupervisorAction,
        origin: str = "desktop",
        provenance: dict[str, Any] | None = None,
    ) -> ReviewResult:
        """Verify action against ActionGate policy first; if allowed and Tier >= 1, run sanity check."""
        # 1. Deterministic ActionGate policy evaluation (CANNOT BE OVERRIDDEN)
        gate_res = self.action_gate.verify_and_gate(
            tool=action.tool,
            origin=origin,
            tier=action.tier,
            params=action.params,
            provenance=provenance,
        )

        if not gate_res.allowed:
            return ReviewResult(
                approved=False,
                verdict="hard_deny" if gate_res.verdict == PolicyVerdict.HARD_DENY else "denied",
                reason=f"ActionGate deterministic policy denied: {gate_res.reason}",
                details={"rule_matched": gate_res.rule_matched},
            )

        # 2. Tier 0 actions (read-only research/queries) require no higher-tier sanity gating
        if action.tier == 0:
            return ReviewResult(
                approved=True,
                verdict="approved",
                reason="Tier 0 action approved directly.",
            )

        # 3. Tier >= 1 action sanity check
        sanity_ok = self.sanity_check_fn(user_request, action)
        if not sanity_ok:
            return ReviewResult(
                approved=False,
                verdict="sanity_failed",
                reason=f"Sanity check failed: action '{action.tool}' with params {action.params} does not align with request '{user_request}'.",
            )

        return ReviewResult(
            approved=True,
            verdict="approved",
            reason="ActionGate and Sanity Check passed.",
        )


class EvidenceReviewer:
    """Audits retrieved evidence cards for relevance, quantity, and domain diversity."""

    def review_evidence(
        self,
        query: str,
        cards: list[dict[str, Any]],
    ) -> ReviewResult:
        """Evaluate evidence sufficiency and diversity."""
        if not cards:
            return ReviewResult(
                approved=False,
                verdict="insufficient_evidence",
                reason="No evidence cards retrieved to support research claims.",
            )

        domains = {card.get("domain") for card in cards if card.get("domain")}
        return ReviewResult(
            approved=True,
            verdict="approved",
            reason=f"Evidence approved: {len(cards)} cards across {len(domains)} distinct domains.",
            details={"card_count": len(cards), "unique_domains": len(domains)},
        )


class AnswerReviewer:
    """Audits synthesized answers against Phase 7 verification reports and claim tables."""

    def __init__(self, composite_gate_threshold: float = 0.85) -> None:
        self.composite_gate_threshold = composite_gate_threshold

    def review_answer(
        self,
        answer: str,
        verification_report: dict[str, Any],
        claims: list[dict[str, Any]],
    ) -> ReviewResult:
        """Evaluate whether draft answer passes composite faithfulness threshold."""
        composite_score = float(verification_report.get("composite_faithfulness_score", 0.0))
        unverified = verification_report.get("unverified_critical_claims", [])

        if composite_score < self.composite_gate_threshold or unverified:
            req_changes = [f"Verify or remove claim: {c}" for c in unverified]
            return ReviewResult(
                approved=False,
                verdict="revise",
                reason=f"Composite score ({composite_score:.2f}) below threshold ({self.composite_gate_threshold:.2f}) or unverified critical claims present.",
                required_changes=req_changes,
                details={
                    "composite_score": composite_score,
                    "unverified_count": len(unverified),
                },
            )

        return ReviewResult(
            approved=True,
            verdict="approved",
            reason=f"Answer approved with composite faithfulness {composite_score:.2f}.",
            details={"composite_score": composite_score},
        )
