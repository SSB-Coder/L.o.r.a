"""Core event models and system states for LORA."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SystemState(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    ACTING = "acting"
    VERIFYING = "verifying"
    SPEAKING = "speaking"
    ERROR = "error"


class LoraEvent(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
