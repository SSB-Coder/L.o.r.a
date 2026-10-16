"""Pydantic schemas for LLM requests, responses, and structured decoding."""

from typing import Any

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str
    images: list[str] | None = None

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(*args, **kwargs)


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
