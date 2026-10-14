"""Typed decision router for L.O.R.A using Convai Laya.

Provides fast, non-autoregressive, calibrated decision routing on CPU:
- Intent: research | control | chat
- Action Risk: tier0 | tier1 | tier2 | tier3
- Depth Mode: eli5 | simple | standard | detailed | expert
- Run Mode: quick | standard | deep
"""

from typing import Any

try:
    import laya
except ImportError:  # optional until the Laya router is actually used
    laya = None  # type: ignore[assignment]

from pydantic import BaseModel, Field


class LayaDecision(BaseModel):
    intent: str
    intent_confidence: float
    intent_probabilities: dict[str, float] = Field(default_factory=dict)
    action_risk: str = "tier0"
    risk_confidence: float = 1.0
    needs_confirmation: bool = False
    depth_mode: str = "standard"
    run_mode: str = "quick"
    raw_usage: dict[str, Any] = Field(default_factory=dict)


class LayaDecisionRouter:
    """Wrapper around laya.Router for non-autoregressive decision classification."""

    def __init__(
        self,
        confidence_threshold: float = 0.70,
        risky_threshold: float = 0.30,
        model_variant: str = "english",
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.risky_threshold = risky_threshold
        self.model_variant = model_variant
        if laya is not None:
            self.router = laya.Router(default=model_variant)
        else:
            self.router = None
        self._setup_questions()

    def _setup_questions(self) -> None:
        self.questions = {
            "intent": {
                "type": "choice",
                "instructions": "What is the primary intent of the user request?",
                "criteria": {
                    "research": "Search the web, read literature, summarize papers, or find factual answers",
                    "control": "Open an application, open a file or folder, or focus a desktop window",
                    "chat": "Casual greeting, conversation, joke, or simple question not requiring research",
                },
            },
            "action_risk": {
                "type": "choice",
                "instructions": "What is the risk tier of this computer action?",
                "criteria": {
                    "tier0": "Read-only search, inspection, or viewing open windows",
                    "tier1": "Opening an allowlisted application or opening a file",
                    "tier2": "Writing a new output file or submitting to an external AI site",
                    "tier3": "Destructive commands, deleting files, shell commands, or changing settings",
                },
            },
            "depth_mode": {
                "type": "choice",
                "instructions": "What explanation depth is requested?",
                "criteria": {
                    "eli5": "Explain like I'm 5, everyday analogies, very simple words",
                    "simple": "Plain language, short summary, one example",
                    "standard": "Balanced summary with key terms",
                    "detailed": "Mechanisms, evidence, competing views, and citations",
                    "expert": "Technical methodology, equations, and advanced nuances",
                },
            },
        }

    @staticmethod
    def _heuristic_predict(utterance: str) -> dict[str, Any]:
        low = utterance.strip().lower()
        if any(w in low for w in ("delete", "format disk", "rmdir", "destroy", "overwrite", "system32")):
            return {
                "answers": {
                    "intent": {"choice": "control", "answer_confidence": 0.92, "probabilities": {"control": 0.92}},
                    "action_risk": {"choice": "tier3", "answer_confidence": 0.95, "probabilities": {"tier3": 0.95}},
                    "depth_mode": {"choice": "standard"},
                },
                "usage": {},
            }
        if any(low.startswith(v) or f" {v} " in f" {low} " for v in ("open", "launch", "focus", "switch to", "start")):
            return {
                "answers": {
                    "intent": {"choice": "control", "answer_confidence": 0.95, "probabilities": {"control": 0.95, "research": 0.03, "chat": 0.02}},
                    "action_risk": {"choice": "tier1", "answer_confidence": 0.90, "probabilities": {"tier1": 0.90, "tier0": 0.10}},
                    "depth_mode": {"choice": "standard"},
                },
                "usage": {},
            }
        if any(w in low for w in ("hi ", "hello", "good morning", "good afternoon", "joke", "how are you", "thanks")):
            return {
                "answers": {
                    "intent": {"choice": "chat", "answer_confidence": 0.94, "probabilities": {"chat": 0.94, "research": 0.04, "control": 0.02}},
                    "action_risk": {"choice": "tier0", "answer_confidence": 0.99, "probabilities": {"tier0": 0.99}},
                    "depth_mode": {"choice": "standard"},
                },
                "usage": {},
            }
        depth = "expert" if "deep research" in low or "expert" in low else ("eli5" if "eli5" in low else "standard")
        return {
            "answers": {
                "intent": {"choice": "research", "answer_confidence": 0.95, "probabilities": {"research": 0.95, "chat": 0.03, "control": 0.02}},
                "action_risk": {"choice": "tier0", "answer_confidence": 0.99, "probabilities": {"tier0": 0.99}},
                "depth_mode": {"choice": depth},
            },
            "usage": {},
        }

    def predict(self, utterance: str) -> LayaDecision:
        """Classify a user utterance into structured decision parameters."""
        if self.router is not None:
            raw = self.router.predict(state=utterance, questions=self.questions)
            ans_intent = raw.get("answers", {}).get("intent", {})
            if ans_intent.get("answer_confidence", 0.0) < self.confidence_threshold and "probabilities" not in ans_intent:
                raw = self._heuristic_predict(utterance)
        else:
            raw = self._heuristic_predict(utterance)
        answers = raw.get("answers", {})

        # 1. Intent extraction
        intent_info = answers.get("intent", {})
        intent_choice = intent_info.get("choice", "research")
        intent_conf = intent_info.get("answer_confidence", 0.5)
        intent_probs = intent_info.get("probabilities", {})

        # 2. Risk extraction
        risk_info = answers.get("action_risk", {})
        risk_choice = risk_info.get("choice", "tier0")
        risk_conf = risk_info.get("answer_confidence", 1.0)
        tier3_prob = risk_info.get("probabilities", {}).get("tier3", 0.0)

        # Flag for confirmation if Tier >= 2 or Tier 3 probability is above safety threshold
        needs_confirm = (risk_choice in ["tier2", "tier3"]) or (tier3_prob >= self.risky_threshold)

        # 3. Depth extraction
        depth_info = answers.get("depth_mode", {})
        depth_choice = depth_info.get("choice", "standard")

        # 4. Infer run mode
        run_mode = "quick"
        lower_utt = utterance.lower()
        if "deep research" in lower_utt or "with refinement" in lower_utt or "outer loop" in lower_utt:
            run_mode = "deep"

        return LayaDecision(
            intent=intent_choice,
            intent_confidence=round(intent_conf, 4),
            intent_probabilities=intent_probs,
            action_risk=risk_choice,
            risk_confidence=round(risk_conf, 4),
            needs_confirmation=needs_confirm,
            depth_mode=depth_choice,
            run_mode=run_mode,
            raw_usage=raw.get("usage", {}),
        )
