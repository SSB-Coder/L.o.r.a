import json

import httpx
import pytest
import respx
from pydantic import BaseModel, Field

from lora.models.ollama_client import OllamaClient


class ResearchSummary(BaseModel):
    title: str = Field(description="Title of research topic")
    findings: list[str] = Field(description="Key bullet points")
    confidence: float = Field(description="Confidence between 0 and 1")

@pytest.mark.asyncio
@respx.mock
async def test_ollama_health_check_online():
    respx.get("http://127.0.0.1:11434/api/tags").respond(
        status_code=200,
        json={"models": [{"name": "granite4.2:8b", "size": 5300000000}]}
    )

    client = OllamaClient(base_url="http://127.0.0.1:11434")
    is_online = await client.is_online()
    assert is_online is True

@pytest.mark.asyncio
@respx.mock
async def test_ollama_health_check_offline():
    respx.get("http://127.0.0.1:11434/api/tags").mock(
        side_effect=httpx.ConnectError("Connection refused")
    )

    client = OllamaClient(base_url="http://127.0.0.1:11434")
    is_online = await client.is_online()
    assert is_online is False

@pytest.mark.asyncio
@respx.mock
async def test_structured_output_generation():
    expected_data = {
        "title": "Quantum Entanglement",
        "findings": ["Particles remain connected", "Non-local behavior"],
        "confidence": 0.98
    }

    respx.post("http://127.0.0.1:11434/api/chat").respond(
        status_code=200,
        json={
            "model": "granite4.2:8b",
            "message": {
                "role": "assistant",
                "content": json.dumps(expected_data)
            },
            "done": True
        }
    )

    client = OllamaClient(base_url="http://127.0.0.1:11434")
    result = await client.generate_structured(
        model="granite4.2:8b",
        prompt="Summarize quantum entanglement",
        response_model=ResearchSummary,
        temperature=0.0
    )

    assert isinstance(result, ResearchSummary)
    assert result.title == "Quantum Entanglement"
    assert len(result.findings) == 2
    assert result.confidence == 0.98

@pytest.mark.asyncio
@respx.mock
async def test_structured_output_retry_on_invalid_json():
    # First response is invalid JSON, second response is valid JSON
    expected_data = {
        "title": "Retry Test",
        "findings": ["Success on second attempt"],
        "confidence": 0.9
    }

    route = respx.post("http://127.0.0.1:11434/api/chat")
    route.side_effect = [
        httpx.Response(200, json={"message": {"role": "assistant", "content": "Not a valid JSON {"}}),
        httpx.Response(200, json={"message": {"role": "assistant", "content": json.dumps(expected_data)}})
    ]

    client = OllamaClient(base_url="http://127.0.0.1:11434", max_retries=1)
    result = await client.generate_structured(
        model="granite4.2:8b",
        prompt="Test retry",
        response_model=ResearchSummary
    )

    assert isinstance(result, ResearchSummary)
    assert result.title == "Retry Test"
