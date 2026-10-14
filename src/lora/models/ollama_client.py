"""Async HTTP client for local Ollama service with structured output validation."""

import json
import re
from collections.abc import AsyncIterator, Sequence
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from lora.models.schemas import ChatMessage, OllamaChatResponse

T = TypeVar("T", bound=BaseModel)


class OllamaClientError(Exception):
    """Base exception for Ollama client errors."""
    pass


def describe_exception(exc: BaseException) -> str:
    """Readable text for an exception. httpx timeouts stringify to '' otherwise."""
    text = str(exc).strip()
    name = type(exc).__name__
    if isinstance(exc, httpx.TimeoutException):
        return (
            f"{name}: Ollama did not answer in time. The model may still be loading "
            "(large models take a while on first use) or the GPU is busy."
        )
    return f"{name}: {text}" if text else name


def _normalize_model_name(name: str) -> tuple[str, str]:
    base = name.strip().lower().split("/")[-1]
    tag = ""
    if ":" in base:
        base, tag = base.split(":", 1)
    return re.sub(r"[^a-z0-9]", "", base), tag


def match_models(wanted: str, available: Sequence[str]) -> list[str]:
    """All installed tags that satisfy ``wanted`` (exact first, then fuzzy)."""
    if not wanted:
        return []
    exact = [a for a in available if a == wanted or a.lower() == wanted.lower()]
    if exact:
        return exact
    want_base, want_tag = _normalize_model_name(wanted)
    if not want_base:
        return []
    found: list[str] = []
    for cand in available:
        cand_base, cand_tag = _normalize_model_name(cand)
        close = cand_base == want_base or (
            min(len(cand_base), len(want_base)) >= 8
            and (cand_base.startswith(want_base) or want_base.startswith(cand_base))
        )
        if not close:
            continue
        if want_tag and cand_tag and want_tag != cand_tag and "latest" not in (want_tag, cand_tag):
            continue
        found.append(cand)
    return found


def match_model(wanted: str, available: Sequence[str]) -> str | None:
    """Map a configured model name onto an installed Ollama tag.

    Exact match first, then case-insensitive, then a namespace/punctuation-insensitive
    prefix match, so ``spark-x25-4b-q4`` finds ``SparkLLM/Spark-X2.5-4B:latest``.
    """
    if not wanted:
        return None
    if wanted in available:
        return wanted
    lowered = {a.lower(): a for a in available}
    if wanted.lower() in lowered:
        return lowered[wanted.lower()]

    want_base, want_tag = _normalize_model_name(wanted)
    if not want_base:
        return None
    for cand in available:
        cand_base, cand_tag = _normalize_model_name(cand)
        close = cand_base == want_base or (
            min(len(cand_base), len(want_base)) >= 8
            and (cand_base.startswith(want_base) or want_base.startswith(cand_base))
        )
        if not close:
            continue
        if want_tag and cand_tag and want_tag != cand_tag and "latest" not in (want_tag, cand_tag):
            continue
        return cand
    return None


class OllamaClient:
    """Async client interfacing with the local Ollama daemon."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 60.0,
        max_retries: int = 2,
        chat_timeout: float = 300.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        # Cold-loading a multi-GB model into 6 GB of VRAM can take well over a minute.
        self.chat_timeout = max(chat_timeout, timeout)

    def _chat_timeout(self) -> httpx.Timeout:
        return httpx.Timeout(self.chat_timeout, connect=5.0)

    async def warm_up(
        self,
        model: str,
        options: dict[str, Any] | None = None,
        keep_alive: str | int = "24h",
        timeout: float = 240.0,
    ) -> bool:
        """Load ``model`` into memory now (empty prompt) so the first real request is fast.

        Pass the same ``options`` (notably num_ctx) the chat uses, otherwise Ollama reloads.
        """
        payload: dict[str, Any] = {"model": model, "keep_alive": keep_alive, "stream": False}
        if options:
            payload["options"] = options
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=5.0)) as client:
                res = await client.post(f"{self.base_url}/api/generate", json=payload)
                return res.status_code == 200
        except httpx.HTTPError:
            return False

    async def installed_models(self) -> list[str]:
        """Names of installed models (empty list if none)."""
        return [m.get("name", "") for m in await self.list_models() if m.get("name")]

    async def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.2,
        model: str | None = None,
    ) -> str:
        """Convenience single-turn generation method used by PromptOptimizerAgent."""
        from lora.models.selection import plan_chat

        target_model = model
        options: dict[str, Any] = {"temperature": temperature, "num_ctx": 4096}
        if not target_model:
            plan = await plan_chat(self, role="optimizer")
            if plan is None:
                raise OllamaClientError("No Ollama models installed")
            target_model = plan.model
            options = {**plan.options, "temperature": temperature}
        msgs: list[dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return await self.generate_chat(model=target_model, messages=msgs, options=options, think=False)

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
        think: bool | str | None = None,
        keep_alive: str | int | None = None,
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
        if think is not None:
            payload["think"] = think
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        async with httpx.AsyncClient(timeout=self._chat_timeout()) as client:
            try:
                res = await client.post(f"{self.base_url}/api/chat", json=payload)
                if res.status_code == 400 and think is not None and "think" in res.text.lower():
                    # Model does not support the think flag: retry without it.
                    payload.pop("think", None)
                    res = await client.post(f"{self.base_url}/api/chat", json=payload)
                if res.status_code != 200:
                    raise OllamaClientError(f"Ollama chat error: HTTP {res.status_code} - {res.text}")
                data = res.json()
                response = OllamaChatResponse.model_validate(data)
                return response.message.content
            except httpx.HTTPError as e:
                raise OllamaClientError(
                    f"Network error communicating with Ollama: {describe_exception(e)}"
                ) from e

    async def stream_chat(
        self,
        model: str,
        messages: list[ChatMessage | dict[str, str]],
        options: dict[str, Any] | None = None,
        think: bool | str | None = None,
        keep_alive: str | int | None = None,
    ) -> AsyncIterator[str]:
        """Yield assistant text deltas as Ollama produces them (thinking tokens are skipped)."""
        normalized = [m.model_dump() if isinstance(m, ChatMessage) else m for m in messages]
        payload: dict[str, Any] = {"model": model, "messages": normalized, "stream": True}
        if options:
            payload["options"] = options
        if think is not None:
            payload["think"] = think
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        url = f"{self.base_url}/api/chat"
        async with httpx.AsyncClient(timeout=self._chat_timeout()) as client:
            try:
                for attempt in range(2):
                    async with client.stream("POST", url, json=payload) as res:
                        if res.status_code != 200:
                            body = (await res.aread()).decode("utf-8", "replace")
                            if attempt == 0 and "think" in payload and "think" in body.lower():
                                payload.pop("think")  # model does not support the think flag
                                continue
                            raise OllamaClientError(
                                f"Ollama chat error: HTTP {res.status_code} - {body.strip()}"
                            )
                        async for line in res.aiter_lines():
                            if not line.strip():
                                continue
                            try:
                                chunk = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if chunk.get("error"):
                                raise OllamaClientError(f"Ollama error: {chunk['error']}")
                            delta = (chunk.get("message") or {}).get("content") or ""
                            if delta:
                                yield delta
                            if chunk.get("done"):
                                return
                        return
            except httpx.HTTPError as e:
                raise OllamaClientError(
                    f"Network error communicating with Ollama: {describe_exception(e)}"
                ) from e

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
