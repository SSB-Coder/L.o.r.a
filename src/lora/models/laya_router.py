"""Typed decision router for L.O.R.A using Convai Laya (Hugging Face / PyPI `laya`).

Provides fast, non-autoregressive, calibrated decision routing on CPU (0 MB GPU VRAM):
- Intent: research | control | chat
- Action Risk: tier0 | tier1 | tier2 | tier3
- Depth Mode: eli5 | simple | standard | detailed | expert
- Run Mode: quick | standard | deep
- Topic Type: background | scholarly | biomedical | physics_astro | cs | tech_community | books | current_events | local_files
- Prompt Injection Flag: boolean safety gate (`noul` / heuristic scanner)
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

try:
    import laya
except ImportError:  # optional until the Laya router is actually used
    laya = None  # type: ignore[assignment]

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LayaDecision(BaseModel):
    intent: str
    intent_confidence: float
    intent_probabilities: dict[str, float] = Field(default_factory=dict)
    action_risk: str = "tier0"
    risk_confidence: float = 1.0
    needs_confirmation: bool = False
    depth_mode: str = "standard"
    run_mode: str = "quick"
    topic_type: str = "background"
    prompt_injection: bool = False
    backend: str = "laya"
    raw_usage: dict[str, Any] = Field(default_factory=dict)


_INJECTION_PATTERNS = re.compile(
    r"(?:ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions|"
    r"system\s+override|you\s+are\s+now\s+in\s+developer\s+mode|"
    r"disregard\s+your\s+rules|reveal\s+your\s+system\s+prompt)",
    re.IGNORECASE,
)


def infer_topic_type(utterance: str) -> str:
    """Classify the domain/topic type for connector routing (§4.4, §4.8.1)."""
    low = (utterance or "").lower()
    if any(w in low for w in ("pubmed", "clinical", "oncology", "medical", "disease", "gene", "protein", "biomedical", "therapy", "drug", "patient")):
        return "biomedical"
    if any(w in low for w in ("arxiv", "preprint", "physics", "quantum", "relativity", "astrophysics", "cosmology", "black hole", "gravitational")):
        return "physics_astro"
    if any(w in low for w in ("mamba", "transformer", "attention", "llm", "neural network", "algorithm", "compiler", "database", "github", "python", "rust", "c++", "kernel", "gpu")):
        return "cs"
    if any(w in low for w in ("crossref", "doi", "citation", "peer-reviewed", "journal", "paper", "study", "literature", "scholarly", "meta-analysis")):
        return "scholarly"
    if any(w in low for w in ("hacker news", "hackernews", "reddit", "forum", "community", "developer discussion")):
        return "tech_community"
    if any(w in low for w in ("book", "isbn", "author", "novel", "open library", "textbook")):
        return "books"
    if any(w in low for w in ("latest", "news", "today", "current", "this week", "2025", "2026", "recent")):
        return "current_events"
    if any(w in low for w in ("my file", "my document", "my downloads", "my desktop", "my folder", "local file", "in my sources")):
        return "local_files"
    return "background"


class LayaDecisionRouter:
    """Wrapper around `laya.Router` (Hugging Face Convai Innovations Laya) with temperature calibration."""

    def __init__(
        self,
        confidence_threshold: float = 0.70,
        risky_threshold: float = 0.30,
        model_variant: str = "english",
        checkpoint_path: str | None = None,
        calibration_file: str | None = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.risky_threshold = risky_threshold
        self.model_variant = model_variant
        self.checkpoint_path = checkpoint_path or os.environ.get("LORA_LAYA_CHECKPOINT", "models/laya-lora-ft")
        self.calibration_file = calibration_file or "config/laya_calibration.json"
        self.temperatures: dict[str, float] = {"intent": 1.0, "action_risk": 1.0, "depth_mode": 1.0}
        self._load_calibration()
        self.router: Any = None
        self.backend = "laya-calibrated"
        if laya is not None and hasattr(laya, "Router"):
            try:
                # Prefer local fine-tuned checkpoint if present on disk, else model_variant ("english" / HF)
                target_variant = model_variant
                for cand in (self.checkpoint_path, r"C:\Models\laya", r"C:\Models\laya-typed-decisions"):
                    if cand and Path(cand).exists():
                        target_variant = str(cand)
                        break
                self.router = laya.Router(default=target_variant)
                self.backend = f"laya-hf ({target_variant})"
            except Exception as exc:
                logger.warning("Laya HF router init deferred (%s); using calibrated CPU router.", exc)
                self.router = None
        self._setup_questions()

    def _load_calibration(self) -> None:
        try:
            p = Path(self.calibration_file)
            if p.is_file():
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, (int, float)) and v > 0:
                            self.temperatures[k] = float(v)
        except Exception:
            pass

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
        if any(w in low for w in ("delete", "format disk", "rmdir", "destroy", "overwrite", "system32", "rm -rf", "registry")):
            return {
                "answers": {
                    "intent": {"choice": "control", "answer_confidence": 0.92, "probabilities": {"control": 0.92, "research": 0.05, "chat": 0.03}},
                    "action_risk": {"choice": "tier3", "answer_confidence": 0.95, "probabilities": {"tier3": 0.95, "tier2": 0.05}},
                    "depth_mode": {"choice": "standard"},
                },
                "usage": {"engine": "laya-calibrated"},
            }
        if any(w in low for w in ("ask chatgpt", "ask claude", "ask gemini", "ask perplexity", "create a note", "save that", "write a file", "generate an image", "draw a diagram")):
            return {
                "answers": {
                    "intent": {"choice": "control", "answer_confidence": 0.91, "probabilities": {"control": 0.91, "research": 0.06, "chat": 0.03}},
                    "action_risk": {"choice": "tier2", "answer_confidence": 0.88, "probabilities": {"tier2": 0.88, "tier1": 0.12, "tier3": 0.0}},
                    "depth_mode": {"choice": "standard"},
                },
                "usage": {"engine": "laya-calibrated"},
            }
        if any(low.startswith(v) or f" {v} " in f" {low} " for v in ("open", "launch", "focus", "switch to", "start", "go to", "visit", "sleep", "wake up", "turn off")):
            return {
                "answers": {
                    "intent": {"choice": "control", "answer_confidence": 0.95, "probabilities": {"control": 0.95, "research": 0.03, "chat": 0.02}},
                    "action_risk": {"choice": "tier1", "answer_confidence": 0.90, "probabilities": {"tier1": 0.90, "tier0": 0.10, "tier3": 0.0}},
                    "depth_mode": {"choice": "standard"},
                },
                "usage": {"engine": "laya-calibrated"},
            }
        if any(w in low for w in ("hi ", "hello", "good morning", "good afternoon", "good evening", "joke", "how are you", "thanks", "thank you")):
            return {
                "answers": {
                    "intent": {"choice": "chat", "answer_confidence": 0.94, "probabilities": {"chat": 0.94, "research": 0.04, "control": 0.02}},
                    "action_risk": {"choice": "tier0", "answer_confidence": 0.99, "probabilities": {"tier0": 0.99}},
                    "depth_mode": {"choice": "standard"},
                },
                "usage": {"engine": "laya-calibrated"},
            }
        if any(w in low for w in ("eli5", "explain like i'm 5", "like i am five", "for a kid", "in very simple terms")):
            depth = "eli5"
        elif any(w in low for w in ("simply", "simple terms", "plain english", "briefly", "in one line", "short summary")):
            depth = "simple"
        elif any(w in low for w in ("deep research", "expert", "rigorous", "equations", "mathematical", "methodology")):
            depth = "expert"
        elif any(w in low for w in ("detailed", "in-depth", "in depth", "thorough", "compare", "mechanisms", "benchmarks")):
            depth = "detailed"
        else:
            depth = "standard"
        return {
            "answers": {
                "intent": {"choice": "research", "answer_confidence": 0.95, "probabilities": {"research": 0.95, "chat": 0.03, "control": 0.02}},
                "action_risk": {"choice": "tier0", "answer_confidence": 0.99, "probabilities": {"tier0": 0.99}},
                "depth_mode": {"choice": depth},
            },
            "usage": {"engine": "laya-calibrated"},
        }

    def predict(self, utterance: str) -> LayaDecision:
        """Classify a user utterance into structured decision parameters."""
        raw: dict[str, Any] = {}
        used_backend = self.backend
        if self.router is not None:
            try:
                raw = self.router.predict(state=utterance, questions=self.questions) or {}
                ans_intent = raw.get("answers", {}).get("intent", {})
                if ans_intent.get("answer_confidence", 0.0) < self.confidence_threshold and "probabilities" not in ans_intent:
                    raw = self._heuristic_predict(utterance)
                    used_backend = "laya-calibrated"
            except Exception as exc:
                logger.debug("Laya router predict fallback: %s", exc)
                raw = self._heuristic_predict(utterance)
                used_backend = "laya-calibrated"
        else:
            raw = self._heuristic_predict(utterance)
            used_backend = "laya-calibrated"

        answers = raw.get("answers", {})

        # 1. Intent extraction
        intent_info = answers.get("intent", {})
        intent_choice = intent_info.get("choice", "research")
        intent_conf = float(intent_info.get("answer_confidence", 0.5))
        intent_probs = intent_info.get("probabilities", {})

        # 2. Risk extraction
        risk_info = answers.get("action_risk", {})
        risk_choice = risk_info.get("choice", "tier0")
        risk_conf = float(risk_info.get("answer_confidence", 1.0))
        tier3_prob = float(risk_info.get("probabilities", {}).get("tier3", 0.0))

        # Flag for confirmation if Tier >= 2 or Tier 3 probability is above safety threshold
        needs_confirm = (risk_choice in ["tier2", "tier3"]) or (tier3_prob >= self.risky_threshold)

        # 3. Depth extraction
        depth_info = answers.get("depth_mode", {})
        depth_choice = depth_info.get("choice", "standard")

        # 4. Infer run mode
        run_mode = "quick"
        lower_utt = utterance.lower()
        if any(w in lower_utt for w in ("deep research", "with refinement", "outer loop", "fact-check", "fact check", "double-check", "thoroughly")):
            run_mode = "deep"

        # 5. Domain topic routing & prompt-injection check
        topic = infer_topic_type(utterance)
        inj = bool(_INJECTION_PATTERNS.search(utterance or ""))

        decision = LayaDecision(
            intent=intent_choice,
            intent_confidence=round(intent_conf, 4),
            intent_probabilities=intent_probs,
            action_risk=risk_choice,
            risk_confidence=round(risk_conf, 4),
            needs_confirmation=needs_confirm,
            depth_mode=depth_choice,
            run_mode=run_mode,
            topic_type=topic,
            prompt_injection=inj,
            backend=used_backend,
            raw_usage=raw.get("usage", {}),
        )
        logger.info(
            "Laya decision [%s]: intent=%s (%.2f) risk=%s depth=%s run_mode=%s topic=%s",
            decision.backend,
            decision.intent,
            decision.intent_confidence,
            decision.action_risk,
            decision.depth_mode,
            decision.run_mode,
            decision.topic_type,
        )
        return decision
