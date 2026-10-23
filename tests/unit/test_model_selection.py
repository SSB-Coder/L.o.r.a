import json
from types import SimpleNamespace

import pytest
import respx

from lora.models.ollama_client import OllamaClient
from lora.models.selection import (
    MAX_INTERACTIVE_CTX,
    PREFERRED_MODEL,
    QWYTHOS_MODEL,
    build_options,
    candidate_names,
    choose_model,
    load_role,
    plan_chat,
)

YAML = """
models:
  writer:
    name: spark-x25-4b-q4
    fallback: [ornith-1.5-9b-q4]
    temperature: 0.3
    num_ctx: 32768
    think: false
"""


def test_load_role_and_options(tmp_path):
    cfg = tmp_path / "lora.yaml"
    cfg.write_text(YAML, encoding="utf-8")
    role = load_role("writer", cfg)
    assert role["name"] == "spark-x25-4b-q4"
    options, think = build_options(role)
    assert options == {"temperature": 0.3, "num_ctx": MAX_INTERACTIVE_CTX}  # capped for a 6 GB GPU
    assert think is False


def test_missing_config_gives_sane_defaults(tmp_path):
    assert load_role("writer", tmp_path / "nope.yaml") == {}
    options, think = build_options({})
    assert options["num_ctx"] == 4096 and think is False


def test_spark_is_always_first_even_when_the_user_config_says_otherwise(monkeypatch):
    monkeypatch.delenv("LORA_MODEL", raising=False)
    names = candidate_names("ornith-1.5:9b", {"name": "role", "fallback": ["fb"]})
    assert names[:4] == [PREFERRED_MODEL, "role", "ornith-1.5:9b", "fb"]
    assert names.index("gemma4:e4b") < names.index("granite4.2:8b")
    monkeypatch.setenv("LORA_MODEL", "my-model")
    assert candidate_names(None, {})[0] == "my-model"

    # gemma4:26b (18 GB) must be excluded from auto-selection
    installed = ["gemma4:26b", "granite4.2:8b", "spark-x25-4b-8k:latest"]
    assert choose_model(["gemma4", "granite4.2:8b"], installed) == ("granite4.2:8b", True)
    assert QWYTHOS_MODEL in candidate_names(None, {}, role="complex")


def test_prefers_the_small_context_variant_of_spark():
    installed = ["ornith-1.5:9b", "spark-x25-4b-32k:latest", "spark-x25-4b-8k:latest"]
    assert choose_model(["spark-x25-4b"], installed) == ("spark-x25-4b-8k:latest", True)


def test_choose_model_falls_back_to_first_installed():
    assert choose_model(["a", "b"], ["z:1"]) == ("z:1", False)
    assert choose_model(["a"], []) == (None, False)
    assert choose_model(["spark-x25-4b-q4"], ["SparkLLM/Spark-X2.5-4B:latest"]) == (
        "SparkLLM/Spark-X2.5-4B:latest",
        True,
    )


@pytest.mark.asyncio
@respx.mock
async def test_plan_chat_end_to_end(tmp_path):
    cfg = tmp_path / "lora.yaml"
    cfg.write_text(YAML, encoding="utf-8")
    respx.get("http://127.0.0.1:11434/api/tags").respond(
        200, json={"models": [{"name": "SparkLLM/Spark-X2.5-4B:latest"}]}
    )
    plan = await plan_chat(OllamaClient(), user_default="spark-x25-4b-q4", config_path=cfg)
    assert plan is not None
    assert plan.model == "SparkLLM/Spark-X2.5-4B:latest" and plan.think is False
    assert plan.keep_alive == "24h"


@pytest.mark.asyncio
@respx.mock
async def test_plan_chat_returns_none_without_models():
    respx.get("http://127.0.0.1:11434/api/tags").respond(200, json={"models": []})
    assert await plan_chat(OllamaClient()) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["writer", "reader", "planner"])
@pytest.mark.parametrize("legacy_installed", [True, False])
@respx.mock
async def test_spark26_roles_use_installed_quantized_model_before_and_after_legacy_removal(
    role, legacy_installed, monkeypatch
):
    monkeypatch.delenv("LORA_MODEL", raising=False)
    legacy = "hf.co/XHToken/Spark-X2.5-4B-GGUF:Q4_K_M"
    installed = ["granite4.2:8b", "bonsai-2:9b", PREFERRED_MODEL]
    if legacy_installed:
        installed.insert(0, legacy)
    url = "http://127.0.0.1:11434"
    respx.get(f"{url}/api/tags").respond(
        200, json={"models": [{"name": name} for name in installed]}
    )
    warm = respx.post(f"{url}/api/generate").respond(200, json={"done": True})
    chat = respx.post(f"{url}/api/chat").respond(
        200, json={"model": PREFERRED_MODEL, "message": {"role": "assistant", "content": "Ready."}, "done": True}
    )
    client = OllamaClient()
    # Existing user config may still name 2.5; the new role/default must win.
    plan = await plan_chat(client, user_default=legacy, role=role)
    assert plan is not None and plan.model == PREFERRED_MODEL
    assert plan.options["num_ctx"] == 8192 and plan.think is False
    assert await client.warm_up(plan.model, options=plan.options, keep_alive=plan.keep_alive)
    answer = await client.generate_chat(
        plan.model, [{"role": "user", "content": "Hello"}],
        options=plan.options, think=plan.think, keep_alive=plan.keep_alive,
    )
    assert answer == "Ready."
    warm_body = json.loads(warm.calls[0].request.content)
    chat_body = json.loads(chat.calls[0].request.content)
    assert warm_body["model"] == chat_body["model"] == PREFERRED_MODEL
    assert warm_body["options"]["num_ctx"] == chat_body["options"]["num_ctx"] == 8192
    assert chat_body["think"] is False and chat_body["keep_alive"] == "24h"


def test_spark26_does_not_match_spark25_or_a_different_quantization():
    from lora.models.ollama_client import match_models

    legacy = "hf.co/XHToken/Spark-X2.5-4B-GGUF:Q4_K_M"
    other_quant = "hf.co/XHToken/Spark-X2.6-4B-GGUF:Q8_0"
    assert match_models(PREFERRED_MODEL, [legacy, other_quant]) == []
    assert choose_model([PREFERRED_MODEL], [legacy, PREFERRED_MODEL]) == (PREFERRED_MODEL, True)


@pytest.mark.asyncio
@respx.mock
async def test_diagnostics_recognize_namespaced_spark26(monkeypatch):
    from lora.agent.diagnostics import _check_engine

    monkeypatch.delenv("LORA_MODEL", raising=False)
    respx.get("http://127.0.0.1:11434/api/tags").respond(
        200, json={"models": [{"name": PREFERRED_MODEL}]}
    )
    check = await _check_engine(SimpleNamespace(
        ollama_url="http://127.0.0.1:11434", user_default_model="spark-x25-4b-q4",
    ))
    assert check.ok and check.summary == "ready."
    assert "preferred_present=True" in check.detail
