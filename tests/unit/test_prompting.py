import pytest
from lora.agent.prompting import parse_plan, PLAN_SYSTEM


def test_parse_plan_valid_english():
    raw = '{"question": "What is redshift in astronomy?", "queries": ["redshift definition", "redshift astronomy"]}'
    fallback = "thorough and deep research about red shift"
    plan = parse_plan(raw, fallback)
    assert plan.engineered is True
    assert plan.question == "What is redshift in astronomy?"
    assert plan.queries == ["redshift definition", "redshift astronomy"]


def test_parse_plan_cjk_hallucination_guard():
    # If the model hallucinates Chinese from an English prompt, reject and fallback to English query
    raw = '{"question": "请对红移进行全面且深入的 research", "queries": ["红移 定义 原理", "红移 天文学 应用 研究"]}'
    fallback = "thorough and deep research about red shift"
    plan = parse_plan(raw, fallback)
    assert plan.engineered is False
    assert plan.question == fallback
    assert plan.queries == [fallback]


def test_parse_plan_preserves_legitimate_cjk_query():
    # If the user actually asked in Chinese, Chinese output should NOT be rejected
    raw = '{"question": "红移的物理原理是什么？", "queries": ["红移 物理原理", "多普勒红移"]}'
    fallback = "请解释红移"
    plan = parse_plan(raw, fallback)
    assert plan.engineered is True
    assert plan.question == "红移的物理原理是什么？"
    assert "多普勒红移" in plan.queries


def test_plan_system_prompt_has_language_constraint():
    assert "EXACT same language" in PLAN_SYSTEM
    assert "NEVER translate into Chinese" in PLAN_SYSTEM
