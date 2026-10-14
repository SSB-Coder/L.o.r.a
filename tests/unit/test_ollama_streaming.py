import json

import httpx
import pytest
import respx

from lora.models.ollama_client import (
    OllamaClient,
    OllamaClientError,
    describe_exception,
    match_model,
)

URL = "http://127.0.0.1:11434"


def _ndjson(*objs):
    return ("\n".join(json.dumps(o) for o in objs) + "\n").encode()


@pytest.mark.asyncio
@respx.mock
async def test_stream_chat_yields_deltas_and_skips_thinking():
    body = _ndjson(
        {"message": {"role": "assistant", "content": "", "thinking": "hmm"}, "done": False},
        {"message": {"role": "assistant", "content": "Hel"}, "done": False},
        {"message": {"role": "assistant", "content": "lo"}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True},
    )
    route = respx.post(f"{URL}/api/chat").respond(200, content=body)
    client = OllamaClient(base_url=URL)
    out = [d async for d in client.stream_chat("m", [{"role": "user", "content": "hi"}], think=False, keep_alive="30m")]
    assert "".join(out) == "Hello"
    sent = json.loads(route.calls[0].request.content)
    assert sent["stream"] is True and sent["think"] is False and sent["keep_alive"] == "30m"


@pytest.mark.asyncio
@respx.mock
async def test_stream_chat_retries_without_think_when_model_rejects_it():
    route = respx.post(f"{URL}/api/chat")
    route.side_effect = [
        httpx.Response(400, json={"error": "\"m\" does not support thinking"}),
        httpx.Response(200, content=_ndjson({"message": {"content": "ok"}, "done": True})),
    ]
    client = OllamaClient(base_url=URL)
    out = [d async for d in client.stream_chat("m", [], think=False)]
    assert out == ["ok"]
    assert "think" not in json.loads(route.calls[1].request.content)


@pytest.mark.asyncio
@respx.mock
async def test_stream_chat_surfaces_midstream_error():
    respx.post(f"{URL}/api/chat").respond(200, content=_ndjson({"error": "model failed to load"}))
    client = OllamaClient(base_url=URL)
    with pytest.raises(OllamaClientError, match="model failed to load"):
        [d async for d in client.stream_chat("m", [])]


@pytest.mark.asyncio
@respx.mock
async def test_timeout_message_is_not_blank():
    respx.post(f"{URL}/api/chat").mock(side_effect=httpx.ReadTimeout(""))
    client = OllamaClient(base_url=URL)
    with pytest.raises(OllamaClientError) as exc:
        await client.generate_chat("m", [{"role": "user", "content": "hi"}])
    assert "did not answer in time" in str(exc.value)


def test_describe_exception_has_type_name_even_when_message_empty():
    assert describe_exception(ValueError("")) == "ValueError"


@pytest.mark.parametrize(
    "wanted, available, expected",
    [
        ("granite4.2:8b", ["granite4.2:8b", "gemma4:e4b"], "granite4.2:8b"),
        ("GEMMA4:E4B", ["gemma4:e4b"], "gemma4:e4b"),
        ("spark-x25-4b-q4", ["SparkLLM/Spark-X2.5-4B:latest"], "SparkLLM/Spark-X2.5-4B:latest"),
        ("spark-x25-4b", ["granite4.2:8b", "SparkLLM/Spark-X2.5-4B:latest"], "SparkLLM/Spark-X2.5-4B:latest"),
        ("granite4.2:8b", ["granite4.2:micro"], None),
        ("nothing", ["gemma4:e4b"], None),
        ("", ["gemma4:e4b"], None),
    ],
)
def test_match_model(wanted, available, expected):
    assert match_model(wanted, available) == expected
