"""Pydantic schemas for LLM requests, responses, and structured decoding."""

from typing import Any

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class OllamaChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    format: str | dict[str, Any] | None = None
    stream: bool = False
    options: dict[str, Any] | None = None


class OllamaChatResponse(BaseModel):
    model: str = "unknown"
    message: ChatMessage
    done: bool = True
    total_duration: int | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None
