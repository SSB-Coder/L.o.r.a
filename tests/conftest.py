import os
import sys
import types
from pathlib import Path

import pytest

os.environ.setdefault("LORA_VERIFY_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


@pytest.fixture(autouse=True)
def no_private_router_weights_in_unit_tests(request, monkeypatch):
    """Unit results must not depend on locally installed multi-GB checkpoints.

    Backend-contract tests install their own fake Laya after this fixture.
    Real checkpoint/audio/model acceptance belongs in explicit integration runs.
    """
    if "unit" in request.path.parts:
        from lora.models import laya_router
        monkeypatch.setattr(laya_router, "laya", None)
        # Tests of the embedding/reranker boundary install explicit fake
        # backends. Everything else exercises the unavailable-model behavior,
        # never the user's multi-GB private models or network downloads.
        backend = types.ModuleType("sentence_transformers")
        def unavailable(*args, **kwargs):
            raise RuntimeError("Native sentence models are disabled in unit tests; supply an explicit fake backend")
        backend.SentenceTransformer = unavailable
        backend.CrossEncoder = unavailable
        monkeypatch.setitem(sys.modules, "sentence_transformers", backend)


@pytest.fixture(autouse=True)
def isolated_emergency_stop_state():
    from lora.safety.killswitch import KillSwitch
    KillSwitch.get_global().reset()
    yield
    KillSwitch.get_global().reset()


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent

@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_lora.db"

@pytest.fixture
def sample_config_path(project_root: Path) -> Path:
    return project_root / "config" / "lora.yaml"


@pytest.fixture
def public_dns(monkeypatch):
    """Deterministic public DNS for browser command tests, without network access."""
    import socket
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port, **kw: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port or 443))
    ])


@pytest.fixture
def reliable_nli(monkeypatch):
    """Controlled entailment backend for pipeline orchestration, not model quality."""
    from lora.verify.nli import NliGroundednessChecker, NliScoreResult
    def check(self, premise, hypothesis):
        # Explicit fixture facts, rather than a blanket positive classifier.
        facts = (
            ("mamba", "linearly", "sequence"),
            ("transformers", "attention", "quadratically"),
            ("transformers", "2017", "self-attention"),
            ("mamba", "transformer-quality", "benchmarks"),
            ("transformers", "in-context", "recall"),
            ("jamba", "interleave", "mixture-of-experts"),
        )
        ok = any(all(word in hypothesis.lower() and word in premise.lower() for word in fact)
                 for fact in facts)
        return NliScoreResult(score=.97 if ok else .02, is_entailed=ok,
                              backend="test-oracle", reliable=True)
    monkeypatch.setattr(NliGroundednessChecker, "check_entailment", check)


def pytest_addoption(parser):
    parser.addoption("--run-model-evals", action="store_true", default=False,
                     help="Load installed private model assets and run measured model evaluations")


@pytest.fixture(scope="module")
def model_eval_enabled(request):
    if not request.config.getoption("--run-model-evals"):
        pytest.skip("model evaluation requires --run-model-evals and installed model assets")


@pytest.fixture
def live_semantic_verifier(model_eval_enabled, monkeypatch):
    from lora.verify.nli import NliGroundednessChecker
    monkeypatch.setenv("LORA_VERIFY_OFFLINE", "0")
    checker = NliGroundednessChecker()
    assert checker._load_model() is not None, "Install the configured local NLI model before model evaluation"
