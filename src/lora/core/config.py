"""Type-safe configuration and policy management for LORA."""

from fnmatch import fnmatch
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class PolicyDecision(BaseModel):
    allowed: bool
    reason: str


class DenyRule(BaseModel):
    tool_match: str | None = None
    path_extensions: list[str] | None = None
    origin: str | None = None
    tiers: list[int] | None = None
    tools: list[str] | None = None
    reason: str


class AllowRule(BaseModel):
    origin: str
    tiers: list[int]
    tools: list[str] | None = None
    action: str = "allow"


class PolicyConfig(BaseModel):
    version: str = "1.2"
    deny_rules: list[DenyRule] = Field(default_factory=list)
    allow_rules: list[AllowRule] = Field(default_factory=list)
    typing_domain_allowlist: list[str] = Field(default_factory=list)

    def evaluate_action(self, tool: str, origin: str, tier: int) -> PolicyDecision:
        """
        Evaluate tool execution policy:
        1. Hard-deny rules are evaluated FIRST and are non-overridable.
        2. Allow/Ask rules are evaluated next with last-match-wins within non-denied space.
        """
        # Step 1: Check hard deny rules
        for rule in self.deny_rules:
            # Check tool match glob
            if rule.tool_match and fnmatch(tool.lower(), rule.tool_match.lower()):
                return PolicyDecision(allowed=False, reason=rule.reason)

            # Check origin restrictions
            if rule.origin and rule.origin.lower() == origin.lower():
                if rule.tiers and tier in rule.tiers:
                    return PolicyDecision(allowed=False, reason=rule.reason)
                if rule.tools and tool.lower() in [t.lower() for t in rule.tools]:
                    return PolicyDecision(allowed=False, reason=rule.reason)

        # Step 2: Check allow rules
        allowed = False
        reason = f"No matching allow rule for origin '{origin}', tier {tier}, tool '{tool}'"

        for rule in self.allow_rules:
            if rule.origin.lower() == origin.lower() and tier in rule.tiers:
                if rule.tools is None or tool.lower() in [t.lower() for t in rule.tools]:
                    allowed = True
                    reason = f"Allowed by rule for origin '{origin}' tier {tier}"

        return PolicyDecision(allowed=allowed, reason=reason)

    def evaluate_path(self, path: str | Path, origin: str) -> PolicyDecision:
        """Evaluate filesystem path policy (blocks executable / macro extensions)."""
        suffix = Path(path).suffix.lower()

        # Check deny rules for blocked extensions
        for rule in self.deny_rules:
            if rule.path_extensions and suffix in [ext.lower() for ext in rule.path_extensions]:
                return PolicyDecision(allowed=False, reason=rule.reason)

        return PolicyDecision(allowed=True, reason="Path extension permitted")


# Sub-models for lora.yaml
class SpeakerVerifierConfig(BaseModel):
    enabled: bool = True
    model: str = "models/voice/user_verifier.onnx"
    threshold: float = 0.70


class WakeConfig(BaseModel):
    engine: str = "openwakeword"
    model: str = "models/voice/hey_lora.onnx"
    threshold: float = 0.65
    consecutive: int = 2
    vad_gate: bool = True
    speaker_verifier: SpeakerVerifierConfig = Field(default_factory=SpeakerVerifierConfig)


class STTConfig(BaseModel):
    engine: str = "faster-whisper"
    model: str = "distil-small.en"
    device: str = "cpu"
    compute: str = "int8"
    beam: int = 1
    vad_filter: bool = True
    initial_prompt: str = ""


class TTSQualityConfig(BaseModel):
    engine: str = "kokoro-onnx"
    model_path: str = "models/kokoro/kokoro-v1.0.onnx"
    voices_path: str = "models/kokoro/voices-v1.0.bin"
    speaker: str = "af_sky"
    dtype: str = "float32"
    attn: str = "sdpa"


class TTSFastConfig(BaseModel):
    engine: str = "piper"
    voice: str = "en_US-lessac-medium"


class VoiceConfig(BaseModel):
    mode: str = "auto"
    wake: WakeConfig = Field(default_factory=WakeConfig)
    hotkey: str = "ctrl+alt+space"
    vad: str = "silero"
    stt: STTConfig = Field(default_factory=STTConfig)
    stt_fallback: list[str] = Field(default_factory=list)
    tts_quality: TTSQualityConfig = Field(default_factory=TTSQualityConfig)
    tts_fast: TTSFastConfig = Field(default_factory=TTSFastConfig)


class DecisionConfig(BaseModel):
    checkpoint: str = "models/laya-lora-ft"
    base_fallback: str = "convaiinnovations/laya"
    device: str = "cpu"
    tau: float = 0.70
    risky_threshold: float = 0.30
    calibration_file: str = "config/laya_calibration.json"


class GatewayAuthConfig(BaseModel):
    desktop_token_env: str = "LORA_DESKTOP_TOKEN"
    mobile_token_env: str = "LORA_MOBILE_TOKEN"
    rate_limit_failed_attempts: int = 5


class MobileGatewayConfig(BaseModel):
    max_tier: int = 0
    max_run_mode: str = "standard"
    allow_uploads: bool = False
    allow_local_files: bool = False


class GatewayConfig(BaseModel):
    desktop_bind: str = "127.0.0.1:8765"
    mobile_bind: str = "192.168.50.10:8765"
    auth: GatewayAuthConfig = Field(default_factory=GatewayAuthConfig)
    max_concurrent_llm_jobs: int = 1
    mobile: MobileGatewayConfig = Field(default_factory=MobileGatewayConfig)


class ModelRoleConfig(BaseModel):
    name: str
    family: str | None = None
    think: bool | str = False
    temperature: float | None = None
    num_ctx: int = 4096
    fallback: list[str] = Field(default_factory=list)


class OllamaEnvConfig(BaseModel):
    OLLAMA_MAX_LOADED_MODELS: int = 1
    base_url: str = "http://127.0.0.1:11434"


class ModelsConfig(BaseModel):
    planner: ModelRoleConfig
    controller: ModelRoleConfig
    writer: ModelRoleConfig
    reader: ModelRoleConfig
    verifier: ModelRoleConfig
    supervisor: ModelRoleConfig
    optimizer: ModelRoleConfig
    ollama_env: OllamaEnvConfig = Field(default_factory=OllamaEnvConfig)


class GroundingConfig(BaseModel):
    nli_fast: str = "vectara-hhem-2.1-open"
    nli_secondary: str = "minicheck"
    embedder: str = "models/embeddinggemma-2"
    vector_store: str = "faiss"
    reranker: str = "bge-reranker-base"


class ResearchConfig(BaseModel):
    max_sources: int = 12
    evidence_cards: int = 10
    min_entailment: float = 0.80
    copy_overlap_max_ngram: int = 8
    query_mode_min_similarity: float = 0.35


class OuterLoopConfig(BaseModel):
    max_iters: int = 5
    epsilon: str | float = "auto"
    wallclock_minutes: int = 20
    faithfulness_floor: float = 0.85


class SafetyConfig(BaseModel):
    max_actions_per_task: int = 40
    dry_run_over: int = 3
    killswitch: str = "ctrl+esc"
    allowed_roots: list[str] = Field(default_factory=list)
    safe_extensions: list[str] = Field(default_factory=list)
    blocked_extensions: list[str] = Field(default_factory=list)


class LoraConfig(BaseModel):
    models_dir: str = "models"
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    models: ModelsConfig
    grounding: GroundingConfig = Field(default_factory=GroundingConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    outerloop: OuterLoopConfig = Field(default_factory=OuterLoopConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)


class ConfigManager:
    """Manages loading and validation of LORA configuration and policy files."""

    def load_lora_config(self, path: Path | str) -> LoraConfig:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return LoraConfig.model_validate(data)

    def load_policy(self, path: Path | str) -> PolicyConfig:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return PolicyConfig.model_validate(data)


def load_config(path: Path | str) -> LoraConfig:
    """Convenience function to load master configuration."""
    return ConfigManager().load_lora_config(path)
