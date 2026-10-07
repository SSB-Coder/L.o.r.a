"""Async HTTP client for local Ollama service with structured output validation."""

import json
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from lora.models.schemas import ChatMessage, OllamaChatResponse

T = TypeVar("T", bound=BaseModel)


class OllamaClientError(Exception):
    """Base exception for Ollama client errors."""
    pass


class OllamaClient:
    """Async client interfacing with the local Ollama daemon."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    async def is_online(self) -> bool:
        """Check if the Ollama daemon is reachable and responding."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(f"{self.base_url}/api/tags")
                return res.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
            return False

    async def list_models(self) -> list[dict[str, Any]]:
        """List currently downloaded models in Ollama."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            res = await client.get(f"{self.base_url}/api/tags")
            if res.status_code != 200:
                raise OllamaClientError(f"Failed to list models: HTTP {res.status_code}")
            data = res.json()
            return data.get("models", [])

    async def generate_chat(
        self,
        model: str,
        messages: list[ChatMessage | dict[str, str]],
        format: str | dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        """Send chat request and return assistant message content."""
        normalized_messages = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                normalized_messages.append(msg.model_dump())
            elif isinstance(msg, dict):
                normalized_messages.append(msg)

        payload: dict[str, Any] = {
            "model": model,
            "messages": normalized_messages,
            "stream": False,
        }
        if format:
            payload["format"] = format
        if options:
            payload["options"] = options

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                res = await client.post(f"{self.base_url}/api/chat", json=payload)
                if res.status_code != 200:
                    raise OllamaClientError(f"Ollama chat error: HTTP {res.status_code} - {res.text}")
                data = res.json()
                response = OllamaChatResponse.model_validate(data)
                return response.message.content
            except httpx.HTTPError as e:
                raise OllamaClientError(f"Network error communicating with Ollama: {e}") from e

    async def generate_structured(
        self,
        model: str,
        prompt: str,
        response_model: type[T],
        system_prompt: str | None = None,
        temperature: float = 0.0,
        num_ctx: int = 4096,
    ) -> T:
        """
        Generate structured output adhering to a Pydantic schema with automatic repair retry.
        """
        schema_json = json.dumps(response_model.model_json_schema())
        effective_system = (
            system_prompt or "You are a precise, truth-grounded assistant."
        ) + f"\nYou must respond ONLY with a valid JSON object matching this schema:\n{schema_json}"

        messages = [
            {"role": "system", "content": effective_system},
            {"role": "user", "content": prompt},
        ]

        options = {
            "temperature": temperature,
            "num_ctx": num_ctx,
        }

        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            raw_reply = ""
            try:
                raw_reply = await self.generate_chat(
                    model=model,
                    messages=messages,
                    format="json",
                    options=options,
                )

                # Clean any markdown block wrapping if present
                clean_text = raw_reply.strip()
                if clean_text.startswith("```json"):
                    clean_text = clean_text[7:]
                if clean_text.startswith("```"):
                    clean_text = clean_text[3:]
                if clean_text.endswith("```"):
                    clean_text = clean_text[:-3]
                clean_text = clean_text.strip()

                parsed_json = json.loads(clean_text)
                return response_model.model_validate(parsed_json)

            except (json.JSONDecodeError, ValidationError) as e:
                last_error = e
                if attempt < self.max_retries:
                    # Append error feedback for self-repair
                    messages.append({"role": "assistant", "content": raw_reply})
                    messages.append({
                        "role": "user",
                        "content": f"The previous output failed validation: {e}. Fix the error and return ONLY valid JSON."
                    })

        raise OllamaClientError(
            f"Failed to obtain valid structured output for {response_model.__name__} after {self.max_retries + 1} attempts: {last_error}"
        ) from last_error
