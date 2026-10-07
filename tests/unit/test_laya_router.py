"""Unit tests for LayaDecisionRouter business logic and schema parsing."""

from unittest.mock import MagicMock, patch

import pytest

from lora.models.laya_router import LayaDecision, LayaDecisionRouter


@pytest.fixture
def mock_laya_router():
    """Create a mock LayaDecisionRouter to test classification logic in isolation."""
    with patch("laya.Router") as mock_router_cls:
        mock_instance = MagicMock()
        mock_router_cls.return_value = mock_instance
        router = LayaDecisionRouter(confidence_threshold=0.70, risky_threshold=0.30)
        yield router, mock_instance


def test_laya_router_initialization(mock_laya_router):
    router, _ = mock_laya_router
    assert router is not None
    assert "intent" in router.questions
    assert "action_risk" in router.questions
    assert "depth_mode" in router.questions


def test_laya_control_intent(mock_laya_router):
    router, mock_instance = mock_laya_router
    mock_instance.predict.return_value = {
        "answers": {
            "intent": {
                "choice": "control",
                "answer_confidence": 0.95,
                "probabilities": {"control": 0.95, "research": 0.03, "chat": 0.02},
            },
            "action_risk": {
                "choice": "tier1",
                "answer_confidence": 0.90,
                "probabilities": {"tier0": 0.10, "tier1": 0.85, "tier2": 0.05, "tier3": 0.0},
            },
            "depth_mode": {"choice": "standard"},
        },
        "usage": {"cpu_time_ms": 12.5},
    }

    res = router.predict("Hey LORA, open VS Code and my Downloads folder")
    assert isinstance(res, LayaDecision)
    assert res.intent == "control"
    assert res.intent_confidence == 0.95
    assert res.action_risk == "tier1"
    assert res.needs_confirmation is False
    assert res.run_mode == "quick"


def test_laya_research_intent_deep_mode(mock_laya_router):
    router, mock_instance = mock_laya_router
    mock_instance.predict.return_value = {
        "answers": {
            "intent": {
                "choice": "research",
                "answer_confidence": 0.98,
                "probabilities": {"research": 0.98},
            },
            "action_risk": {"choice": "tier0", "answer_confidence": 0.99},
            "depth_mode": {"choice": "expert"},
        },
        "usage": {},
    }

    res = router.predict("Perform deep research on sparse attention mechanisms")
    assert isinstance(res, LayaDecision)
    assert res.intent == "research"
    assert res.depth_mode == "expert"
    assert res.run_mode == "deep"


def test_laya_risky_confirmation_flag_tier3(mock_laya_router):
    router, mock_instance = mock_laya_router
    mock_instance.predict.return_value = {
        "answers": {
            "intent": {"choice": "control", "answer_confidence": 0.90},
            "action_risk": {
                "choice": "tier3",
                "answer_confidence": 0.85,
                "probabilities": {"tier3": 0.85},
            },
            "depth_mode": {"choice": "standard"},
        },
        "usage": {},
    }

    res = router.predict("Delete the system partition and format disk C")
    assert isinstance(res, LayaDecision)
    assert res.action_risk == "tier3"
    assert res.needs_confirmation is True


def test_laya_risky_threshold_trigger(mock_laya_router):
    router, mock_instance = mock_laya_router
    # Even if choice is tier1, if tier3 probability is >= 0.30, needs_confirmation triggers
    mock_instance.predict.return_value = {
        "answers": {
            "intent": {"choice": "control", "answer_confidence": 0.80},
            "action_risk": {
                "choice": "tier1",
                "answer_confidence": 0.60,
                "probabilities": {"tier1": 0.65, "tier3": 0.35},
            },
            "depth_mode": {"choice": "standard"},
        },
        "usage": {},
    }

    res = router.predict("Overwriting configuration in system directory")
    assert isinstance(res, LayaDecision)
    assert res.needs_confirmation is True
