"""Action Gate & Provenance Tracker for L.O.R.A.

Enforces execution budgets, repetitive loop detection, taint tracking,
audit logging, and dry-run modes before tool calls reach the operating system.
"""

import logging
import threading
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from lora.core.logging import scrub_sensitive_data
from lora.safety.killswitch import KillSwitch
from lora.safety.policy import ActionRiskTier, PolicyEngine, PolicyResult, PolicyVerdict

logger = logging.getLogger("lora.safety.gate")


class DataProvenance(StrEnum):
    USER = "user"  # Local user direct spoken or typed input
    SYSTEM = "system"  # Verified system configuration or local code
    USER_REMOTE = "user_remote"  # Mobile LAN gateway input
    UNTRUSTED = "untrusted"  # Web pages, external PDFs, scraped text


def has_untrusted_taint(data: Any) -> bool:
    """Recursively search any mapping, list, or enum value for UNTRUSTED taint."""
    if data == DataProvenance.UNTRUSTED or data == "untrusted":
        return True
    if isinstance(data, Mapping):
        return any(has_untrusted_taint(v) for v in data.values())
    if isinstance(data, (list, tuple, set)):
        return any(has_untrusted_taint(item) for item in data)
    return False


class ActionGate:
    """Gatekeeper and taint-tracker wrapping PolicyEngine execution."""

    def __init__(
        self,
        policy_engine: PolicyEngine,
        kill_switch: KillSwitch | None = None,
        max_actions_per_task: int = 15,  # v1.5 Plan §9.1: tightened default action budget to 15
        max_repeated_actions: int = 3,
        dry_run: bool = False,
    ) -> None:
        self.policy_engine = policy_engine
        self.kill_switch = kill_switch or KillSwitch()
        self.max_actions_per_task = max_actions_per_task
        self.max_repeated_actions = max_repeated_actions
        self.dry_run = dry_run
        self.action_count = 0
        self.action_history: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def reset_task(self) -> None:
        """Reset per-task counters for a new task session."""
        with self._lock:
            self.action_count = 0
            self.action_history.clear()

    def _log_audit(self, tool: str, origin: str, tier: int, res: PolicyResult, params: dict[str, Any]) -> None:
        if tier >= 1 or not res.allowed:
            scrubbed = scrub_sensitive_data(params)
            logger.info(
                "AUDIT origin=%s tool=%s tier=%d verdict=%s allowed=%s params=%s",
                origin,
                tool,
                tier,
                res.verdict,
                res.allowed,
                scrubbed,
            )

    def verify_and_gate(
        self,
        tool: str,
        origin: str,
        tier: int | ActionRiskTier,
        params: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> PolicyResult:
        """Verify an action request against kill switch, budgets, taint, and policy engine."""
        params_dict = params or {}
        provenance_dict = provenance or {}
        tier_int = int(tier)
        tool_lower = tool.lower()

        # 1. Kill Switch verification
        if self.kill_switch.is_halted():
            res = PolicyResult(
                allowed=False,
                verdict=PolicyVerdict.HARD_DENY,
                reason=f"Emergency kill switch is active ({self.kill_switch.halt_reason})",
                rule_matched="killswitch_halted",
            )
            self._log_audit(tool, origin, tier_int, res, params_dict)
            return res

        with self._lock:
            # 2. Action Budget enforcement
            if self.action_count >= self.max_actions_per_task:
                res = PolicyResult(
                    allowed=False,
                    verdict=PolicyVerdict.HARD_DENY,
                    reason=f"Task action budget exceeded (limit: {self.max_actions_per_task})",
                    rule_matched="action_budget_exceeded",
                )
                self._log_audit(tool, origin, tier_int, res, params_dict)
                return res

            # 3. Repetitive action loop detection
            current_signature = {"tool": tool_lower, "params": params_dict}
            recent_matches = 0
            for prior in reversed(self.action_history):
                if prior.get("tool") == tool_lower and prior.get("params") == params_dict:
                    recent_matches += 1
                else:
                    break

            if recent_matches >= self.max_repeated_actions:
                res = PolicyResult(
                    allowed=False,
                    verdict=PolicyVerdict.ASK,
                    reason=(
                        f"Repetitive action loop detected: '{tool}' invoked "
                        f"{recent_matches} times with identical parameters"
                    ),
                    rule_matched="repetitive_loop_detected",
                )
                self._log_audit(tool, origin, tier_int, res, params_dict)
                return res

        # 4. Taint Tracking (Prompt Injection Defense)
        if has_untrusted_taint(provenance_dict):
            if tier_int >= ActionRiskTier.TIER1:
                res = PolicyResult(
                    allowed=False,
                    verdict=PolicyVerdict.HARD_DENY,
                    reason=f"Untrusted tainted parameter cannot be used in Tier {tier_int} action '{tool}'",
                    rule_matched="taint_tracking_tier_denied",
                )
                self._log_audit(tool, origin, tier_int, res, params_dict)
                return res

            if tool_lower in ["browser_type", "ask_other_ai", "open_app", "open_path"]:
                res = PolicyResult(
                    allowed=False,
                    verdict=PolicyVerdict.HARD_DENY,
                    reason=f"Untrusted tainted input cannot be typed or executed by tool '{tool}'",
                    rule_matched="taint_tracking_typing_denied",
                )
                self._log_audit(tool, origin, tier_int, res, params_dict)
                return res

        # 5. Path checks for filesystem tools
        path_arg = params_dict.get("path") or params_dict.get("file_path") or params_dict.get("target")
        if path_arg:
            path_mode = "write" if tool_lower in ["export_report", "write_file", "create_note"] else "read"
            path_result = self.policy_engine.check_path(path=path_arg, origin=origin, mode=path_mode)
            if not path_result.allowed:
                self._log_audit(tool, origin, tier_int, path_result, params_dict)
                return path_result

        # 6. Evaluate via PolicyEngine
        policy_result = self.policy_engine.check_action(
            tool=tool,
            origin=origin,
            tier=tier_int,
            params=params_dict,
        )

        if policy_result.allowed:
            with self._lock:
                self.action_count += 1
                self.action_history.append(current_signature)

        self._log_audit(tool, origin, tier_int, policy_result, params_dict)
        return policy_result
