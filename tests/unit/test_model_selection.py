import pytest
import respx

from lora.models.ollama_client import OllamaClient
from lora.models.selection import (
    MAX_INTERACTIVE_CTX,
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
    assert names[:4] == ["spark-x25-4b", "role", "ornith-1.5:9b", "fb"]
    assert names.index("gemma4:e4b") < names.index("granite4.2:8b")
    monkeypatch.setenv("LORA_MODEL", "my-model")
    assert candidate_names(None, {})[0] == "my-model"


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
