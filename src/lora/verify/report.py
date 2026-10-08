"""Verification report and confidence models."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from lora.verify.corroborate import CorroborationResult


class VerificationVerdict(StrEnum):
    """Overall outcome of the V1-V8 verification pipeline."""

    VERIFIED = "verified"
    REPAIRED = "repaired"
    ABSTAIN = "abstain"


class ConfidenceLevel(StrEnum):
    """Calibrated confidence badge assigned to verified research output."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ClaimVerificationDetail(BaseModel):
    """Detailed audit trace for a single atomic claim."""

    model_config = ConfigDict(frozen=True)

    claim_id: str
    claim_text: str
    nli_score: float
    is_supported: bool
    number_check_passed: bool = True
    corroboration: CorroborationResult | None = None
    unsupported_reason: str = ""


class VerificationReport(BaseModel):
    """The final verification audit report delivered by the Supervisor gate."""

    model_config = ConfigDict(frozen=True)

    verdict: VerificationVerdict
    composite_faithfulness_score: float
    confidence: ConfidenceLevel
    claim_details: list[ClaimVerificationDetail]
    repaired_answer: str
    unverified_claims: list[str] = Field(default_factory=list)
    requires_abstention: bool = False
