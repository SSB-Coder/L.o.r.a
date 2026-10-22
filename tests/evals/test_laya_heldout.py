"""Held-out evaluation benchmark for live LayaDecisionRouter."""

import pytest

from lora.models.laya_router import LayaDecisionRouter


@pytest.fixture(scope="module")
def live_laya_router(model_eval_enabled):
    """Shared live model instance for eval suite."""
    return LayaDecisionRouter()


def test_laya_held_out_accuracy(live_laya_router):
    """Evaluate accuracy on a 10-query held-out benchmark (target: >= 90%)."""
    eval_set = [
        ("Open Calculator on my desktop", "control"),
        ("Focus the Edge window", "control"),
        ("Open my Documents folder", "control"),
        ("Launch Notepad for me", "control"),
        ("What is the capital of Australia?", "research"),
        ("Summarize the theory of general relativity", "research"),
        ("Search PubMed for recent oncology clinical trials", "research"),
        ("Find Crossref citations for this paper", "research"),
        ("Hi there, good morning", "chat"),
        ("Tell me a funny joke", "chat"),
    ]

    correct = 0
    for utterance, expected_intent in eval_set:
        res = live_laya_router.predict(utterance)
        if res.intent == expected_intent:
            correct += 1

    accuracy = correct / len(eval_set)
    print(f"\nHeld-out Intent Accuracy: {accuracy * 100:.1f}% ({correct}/{len(eval_set)})")
    assert accuracy >= 0.90
