"""Typed decision router for L.O.R.A using Convai Laya (local `models/laya-lora-ft` on CPU).

Provides fast, non-autoregressive, calibrated decision routing on CPU (0 MB GPU VRAM):
- Intent: research | control | chat
- Action Risk: tier0 | tier1 | tier2 | tier3
- Depth Mode: eli5 | simple | standard | detailed | expert
- Run Mode: quick | standard | deep
- Topic Type: background | scholarly | biomedical | physics_astro | cs | tech_community | books | current_events | local_files
- Prompt Injection Flag: boolean safety gate (`noul` / heuristic scanner)

Strictly loads from local `models/laya-lora-ft` (or offline HF cache) with ZERO outbound Hugging Face downloads at startup.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
logging.getLogger("httpx").setLevel(logging.WARNING)

try:
    import laya
except ImportError:  # optional until the Laya router is actually used
    laya = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

_VALID_LAYA_VARIANTS = {
    "english",
    "multilingual",
    "typed-decisions",
    "decisions",
    "default",
    "en",
    "laya",
    "laya-multilingual",
    "laya-typed-decisions",
    "ml",
    "multi",
    "typed",
    "typed_decisions",
}


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
    stt_command_plausible: bool = False
    backend: str = "laya"
    calibrated: bool = False
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


def _resolve_laya_checkpoint(explicit: str | None = None) -> str:
    env_ckpt = explicit or os.environ.get("LORA_LAYA_CHECKPOINT")
    if env_ckpt and Path(env_ckpt).exists():
        return str(Path(env_ckpt).resolve())
    repo_root = Path(__file__).resolve().parents[3]
    for cand in (
        Path("models/laya-lora-ft"),
        repo_root / "models" / "laya-lora-ft",
        Path("models/laya"),
        repo_root / "models" / "laya",
        Path(r"C:\Models\laya-lora-ft"),
        Path(r"C:\Models\laya"),
        Path(r"C:\Models\laya-typed-decisions"),
    ):
        if cand.exists():
            return str(cand.resolve())
    return env_ckpt or "models/laya-lora-ft"


@contextmanager
def _local_laya_hf_intercept(ckpt_dir: Path):
    """Redirect `laya`'s internal `huggingface_hub.hf_hub_download` calls to local `models/laya-lora-ft`
    and enforce `local_files_only=True` so `laya.Router` NEVER downloads from huggingface.co at startup.
    """
    try:
        import huggingface_hub
    except ImportError:
        yield
        return

    orig_download = getattr(huggingface_hub, "hf_hub_download", None)
    if orig_download is None:
        yield
        return

    def _patched_download(*args: Any, **kwargs: Any) -> str:
        filename = kwargs.get("filename") or (args[1] if len(args) > 1 else "")
        subfolder = kwargs.get("subfolder") or ""
        rel_path = f"{subfolder}/{filename}".strip("/") if subfolder else str(filename)

        if ckpt_dir.is_dir() and rel_path:
            # 1. Exact relative path inside models/laya-lora-ft (e.g. models/laya-lora-ft/typed-decisions/rl_agent_config.json)
            cand_exact = ckpt_dir / rel_path
            if cand_exact.is_file():
                return str(cand_exact.resolve())
            # 2. Path with variant prefix stripped (e.g. models/laya-lora-ft/rl_agent_config.json, models/laya-lora-ft/tokenizer/tokenizer.json)
            parts = Path(rel_path).parts
            if len(parts) > 1:
                cand_stripped = ckpt_dir.joinpath(*parts[1:])
                if cand_stripped.is_file():
                    return str(cand_stripped.resolve())
            # 3. Direct basename inside models/laya-lora-ft
            cand_base = ckpt_dir / Path(rel_path).name
            if cand_base.is_file():
                return str(cand_base.resolve())

        # Otherwise check local HF cache ONLY (never make outbound HTTP calls during startup)
        kwargs["local_files_only"] = True
        return orig_download(*args, **kwargs)

    huggingface_hub.hf_hub_download = _patched_download
    # Also patch inside any already-imported laya submodules that imported hf_hub_download directly
    patched_mods: list[tuple[Any, Any]] = []
    for mod_name, mod_obj in list(__import__("sys").modules.items()):
        if mod_name.startswith("laya") and hasattr(mod_obj, "hf_hub_download"):
            patched_mods.append((mod_obj, getattr(mod_obj, "hf_hub_download")))
            setattr(mod_obj, "hf_hub_download", _patched_download)
    try:
        yield
    finally:
        huggingface_hub.hf_hub_download = orig_download
        for mod_obj, orig_fn in patched_mods:
            setattr(mod_obj, "hf_hub_download", orig_fn)


class LayaDecisionRouter:
    """Wrapper around `laya.Router` using local `models/laya-lora-ft` weights with temperature calibration."""

    def __init__(
        self,
        confidence_threshold: float = 0.70,
        risky_threshold: float = 0.30,
        model_variant: str = "typed-decisions",
        checkpoint_path: str | None = None,
        calibration_file: str | None = None,
        lazy_load: bool = False,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self._predict_lock = threading.Lock()
        self.risky_threshold = risky_threshold
        self.model_variant = model_variant if model_variant in _VALID_LAYA_VARIANTS else "typed-decisions"
        self.checkpoint_path = _resolve_laya_checkpoint(checkpoint_path)
        self.calibration_file = calibration_file or "config/laya_calibration.json"
        self.temperatures: dict[str, float] = {"intent": 1.0, "action_risk": 1.0, "depth_mode": 1.0}
        self.calibrated_questions: set[str] = set()
        self._load_calibration()
        self.agent: Any = None
        self.router: Any = None
        ckpt_dir = Path(self.checkpoint_path)
        self.backend = f"laya-local ({ckpt_dir.name})" if ckpt_dir.exists() else "laya-calibrated"

        self._backend_loaded = False
        self._setup_questions()
        if not lazy_load:
            self.preload()

    def preload(self) -> bool:
        with self._predict_lock:
            self._ensure_backend()
            return self.agent is not None or self.router is not None

    def _ensure_backend(self) -> None:
        if self._backend_loaded:
            return
        self._backend_loaded = True
        ckpt_dir = Path(self.checkpoint_path)
        if laya is not None:
            # 1. Prefer direct local Agent load from checkpoint on disk (0 downloads, 100% offline)
            if ckpt_dir.exists() and (ckpt_dir / "model.safetensors").exists() and hasattr(laya, "Agent"):
                try:
                    self.agent = laya.Agent(str(ckpt_dir.resolve()))
                    self.backend = f"laya-local ({ckpt_dir.name})"
                except Exception as exc:
                    logger.warning("Local Laya Agent load failed (%s); trying fallback router.", exc)
                    self.agent = None

            # 2. Fall back to Router if local Agent was not loaded
            if self.agent is None and hasattr(laya, "Router"):
                with _local_laya_hf_intercept(ckpt_dir):
                    for variant in dict.fromkeys((self.model_variant, "typed-decisions", "english")):
                        try:
                            self.router = laya.Router(default=variant)
                            self.backend = f"laya-hf ({variant})"
                            break
                        except Exception:
                            self.router = None

    def _load_calibration(self) -> None:
        try:
            p = Path(self.calibration_file)
            if p.is_file():
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, (int, float)) and v > 0:
                            self.temperatures[k] = float(v)
                            self.calibrated_questions.add(k)
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
        self.questions["intent"]["criteria"].update({
            "notebook_followup": "Rewrite or ask about cached research",
            "system": "Sleep or exit the assistant", "stop": "Cancel current tasks",
        })
        self.questions["run_mode"] = {"type": "choice", "instructions": "Choose the requested research mode",
            "criteria": {"quick": "Brief research", "deep": "Full research and review",
                         "deep_with_refinement": "Research with bounded prompt refinement"}}
        self.questions["topic_type"] = {"type": "choice", "instructions": "Choose the source domain",
            "criteria": {k: k.replace("_", " ") for k in (
                "background", "scholarly", "biomedical", "physics_astro", "cs", "tech_community",
                "books", "current_events", "local_files", "geo")}}
        for key, instruction in {
            "needs_confirmation": "Is this ambiguous or potentially misheard?",
            "is_destructive_request": "Does this request deletion, overwriting, shell execution or unsafe settings?",
            "contains_instruction_in_content": "Does this snippet contain instructions directed at the assistant?",
            "answer_addresses_question": "Does the provided answer directly address the question?",
            "stt_command_plausible": "Is this a coherent plausible spoken command?",
        }.items():
            self.questions[key] = {"type": "noul", "instructions": instruction}
        for key, instruction in {"source_relevance": "How relevant is this source to the query?",
                                 "claim_importance": "How important is this factual claim?"}.items():
            self.questions[key] = {"type": "score", "instructions": instruction,
                                   "criteria": ["low", "medium", "high"]}

    @staticmethod
    def _heuristic_predict_raw(utterance: str) -> dict[str, Any]:
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
        if any(low.startswith(v) or f" {v} " in f" {low} " for v in ("open", "launch", "focus", "switch to", "start", "activate", "go to", "visit", "sleep", "wake up", "turn off")):
            return {
                "answers": {
                    "intent": {"choice": "control", "answer_confidence": 0.95, "probabilities": {"control": 0.95, "research": 0.03, "chat": 0.02}},
                    "action_risk": {"choice": "tier1", "answer_confidence": 0.90, "probabilities": {"tier1": 0.90, "tier0": 0.10, "tier3": 0.0}},
                    "depth_mode": {"choice": "standard"},
                },
                "usage": {"engine": "laya-calibrated"},
            }
        explicit_research = re.match(r"^(?:please\s+)?(?:research|search|look up|fact-check)\b", low)
        if not explicit_research and any(w in low for w in (
            "hi ", "hello", "good morning", "good afternoon", "good evening", "good night", "goodnight", "goodbye", "joke", "how are you", "thanks", "thank you",
            "how's your day", "how is your day", "how was your day", "how are things", "let's chat", "can we talk",
            "i'm tired", "i am tired", "i've had a long day", "i had a long day", "i feel ",
        )):
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
        from lora.agent.commands import is_research, is_smalltalk

        factual = bool(re.match(r"^(?:what|why|how|who|where|when|explain|compare|describe|tell me about)\b", low))
        intent = "research" if not is_smalltalk(low) and (is_research(low) or factual) else "chat"
        return {
            "answers": {
                "intent": {"choice": intent, "answer_confidence": 0.0},
                "action_risk": {"choice": "tier0", "answer_confidence": 0.99, "probabilities": {"tier0": 0.99}},
                "depth_mode": {"choice": depth},
            },
            "usage": {"engine": "laya-calibrated"},
        }

    @staticmethod
    def _heuristic_predict(utterance: str) -> dict[str, Any]:
        raw = LayaDecisionRouter._heuristic_predict_raw(utterance)
        for answer in raw["answers"].values():
            answer.pop("probabilities", None)
            answer["answer_confidence"] = 0.0
        raw["usage"] = {"engine": "rule-based", "calibrated": False}
        return raw

    def check_stt_plausible(self, utterance: str) -> bool:
        decision = self.predict(utterance)
        return decision.stt_command_plausible

    def predict(self, utterance: str) -> LayaDecision:
        """Classify using shared CPU state safely across desktop/gateway threads."""
        with self._predict_lock:
            return self._predict(utterance)

    def predict_fast(self, utterance: str) -> LayaDecision:
        """Route without loading private CPU weights on the interactive path."""
        with self._predict_lock:
            return self._predict(utterance, use_neural=False)

    def _predict(self, utterance: str, use_neural: bool = True) -> LayaDecision:
        """Classify a user utterance into structured decision parameters."""
        if use_neural:
            self._ensure_backend()
        raw: dict[str, Any] = {}
        used_backend = self.backend
        if not use_neural:
            raw = self._heuristic_predict(utterance)
        elif self.agent is not None:
            try:
                raw = self.agent.predict(state=utterance, questions=self.questions) or {}
                ans_intent = raw.get("answers", {}).get("intent", {})
                if ans_intent.get("answer_confidence", 0.0) < self.confidence_threshold and "probabilities" not in ans_intent:
                    raw = self._heuristic_predict(utterance)
            except Exception as exc:
                logger.debug("Laya agent predict fallback: %s", exc)
                raw = self._heuristic_predict(utterance)
        elif self.router is not None:
            try:
                with _local_laya_hf_intercept(Path(self.checkpoint_path)):
                    raw = self.router.predict(state=utterance, questions=self.questions) or {}
                ans_intent = raw.get("answers", {}).get("intent", {})
                if ans_intent.get("answer_confidence", 0.0) < self.confidence_threshold and "probabilities" not in ans_intent:
                    raw = self._heuristic_predict(utterance)
            except Exception as exc:
                logger.debug("Laya router predict fallback: %s", exc)
                raw = self._heuristic_predict(utterance)
        else:
            raw = self._heuristic_predict(utterance)

        answers = raw.get("answers", {})
        for key, answer in answers.items():
            probabilities = answer.get("probabilities", {})
            if key in self.calibrated_questions and probabilities:
                temperature = self.temperatures[key]
                scaled = {label: math.exp(math.log(max(float(value), 1e-12)) / temperature)
                          for label, value in probabilities.items()}
                total = sum(scaled.values())
                answer["probabilities"] = {label: value / total for label, value in scaled.items()}
                answer["answer_confidence"] = max(answer["probabilities"].values())
        rule_based = raw.get("usage", {}).get("engine") == "rule-based"
        if rule_based:
            used_backend = "rule-based (uncalibrated)"

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
        needs_confirm = needs_confirm or intent_conf < self.confidence_threshold or float(answers.get("needs_confirmation", {}).get("noul", 0)) >= self.risky_threshold

        # 3. Depth extraction
        depth_info = answers.get("depth_mode", {})
        depth_choice = depth_info.get("choice", "standard")

        # 4. Infer run mode from neural depth + explicit user request
        run_mode = "deep" if depth_choice in ("detailed", "expert") else "quick"
        lower_utt = utterance.lower()
        if any(w in lower_utt for w in ("deep research", "with refinement", "outer loop", "fact-check", "fact check", "double-check", "thoroughly")):
            run_mode = "deep"

        # 5. Domain topic routing & prompt-injection check
        topic = infer_topic_type(utterance)
        topic = answers.get("topic_type", {}).get("choice", topic)
        run_mode = answers.get("run_mode", {}).get("choice", run_mode)
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
            stt_command_plausible=not rule_based and float(answers.get("stt_command_plausible", {}).get("noul", 0)) >= self.confidence_threshold,
            backend=used_backend,
            calibrated=not rule_based and {"intent", "action_risk"}.issubset(self.calibrated_questions),
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
