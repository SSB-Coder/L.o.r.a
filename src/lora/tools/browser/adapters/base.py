"""Base Chatbot DOM Adapter for L.O.R.A."""

from typing import Any, Protocol


class ChatbotAdapter(Protocol):
    """Protocol for per-site DOM interaction with external AI assistants."""

    name: str
    domain: str
    url: str

    async def submit_prompt(self, page: Any, prompt: str, timeout_seconds: float = 30.0) -> str:
        """Type prompt into chatbot input, submit, and extract streaming/final response."""
        ...
