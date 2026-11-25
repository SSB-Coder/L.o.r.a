"""Supported-runtime boundaries and retained experimental compatibility."""

import importlib
import importlib.util
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_default_runtime_imports_without_optional_or_experimental_backends():
    script = """
import importlib.abc
import sys

class UnsupportedBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'faiss', 'laya', 'openwakeword', 'qwen_tts', 'torchvision'} or fullname.startswith('lora.experimental'):
            raise AssertionError('Default startup imported optional backend: ' + fullname)

sys.meta_path.insert(0, UnsupportedBackend())
import lora.agent.runtime
import lora.main
from lora.dialogue import ConversationHistory
from lora.grounding.vector_index import make_index
from lora.voice.tts.engine import TtsEngine
runtime = lora.agent.runtime.AgentRuntime('http://unused', None, ConversationHistory(), lambda: None)
assert make_index(3).d == 3
assert TtsEngine().worker.default_voice == 'af_sky'
assert not any(name.startswith('lora.experimental') for name in sys.modules)
print('supported runtime ready')
"""
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "LORA_VECTOR_BACKEND": "numpy"}
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'supported runtime ready' in result.stdout


@pytest.mark.parametrize(('old', 'new'), [
    ('lora.voice.controller', 'lora.experimental.legacy.voice_controller'),
    ('lora.core.controller', 'lora.experimental.legacy.controller'),
    ('lora.core.planner', 'lora.experimental.legacy.planner'),
    ('lora.grounding.pipeline', 'lora.experimental.legacy.research_pipeline'),
    ('lora.agent.verification', 'lora.experimental.agent.verification'),
    ('lora.agent.deep', 'lora.experimental.agent.deep'),
    ('lora.verify.nli', 'lora.experimental.verify.nli'),
    ('lora.supervisor.state', 'lora.experimental.supervisor.state'),
    ('lora.outer_loop.genome', 'lora.experimental.outer_loop.genome'),
])
def test_legacy_imports_share_the_canonical_experimental_module(old, new):
    assert importlib.import_module(old) is importlib.import_module(new)


def test_historical_worker_name_only_uses_cpu_kokoro(monkeypatch):
    from lora.voice.tts import worker

    class Kokoro:
        def get_voices(self):
            return ['af_sky']

        def create(self, text, **kwargs):
            assert kwargs['voice'] == 'af_sky'
            return np.ones(240, dtype=np.float32), 24000

    monkeypatch.setattr(worker, 'get_kokoro_model', lambda path: Kokoro())
    old = worker.Qwen3TtsWorkerProcess(tokenizer_dir='unused', spawn_process=True,
                                      cpu_only=False, default_voice='vivian')
    old.start()
    chunk = old.synthesize('Compatibility still speaks.')
    assert len(chunk.samples) == 240 and chunk.error is None
    assert old.cpu_only and not old.spawn_process
    assert old.diagnosis()['backend'] == 'kokoro'
    assert worker.tts_diagnosis()['device'] == 'cpu'
    old.stop()
    assert not hasattr(worker, 'get_qwen_model')
    assert not (ROOT / 'src/lora/voice/tts/server_main.py').exists()
    assert not (ROOT / 'src/lora/voice/tts/external.py').exists()


def test_optional_packages_are_not_mandatory_and_lock_matches_extras():
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']
    from packaging.requirements import Requirement

    requirements = [Requirement(value) for value in project['dependencies']]
    base = {requirement.name for requirement in requirements if requirement.marker is None}
    assert not base & {'faiss-cpu', 'laya', 'openwakeword', 'torchvision'}
    vision = next(requirement for requirement in requirements if requirement.name == 'torchvision')
    assert vision.marker.evaluate({'sys_platform': 'linux', 'platform_machine': 'x86_64'})
    assert vision.marker.evaluate({'sys_platform': 'win32', 'platform_machine': 'AMD64'})
    assert vision.marker.evaluate({'sys_platform': 'darwin', 'platform_machine': 'arm64'})
    assert not vision.marker.evaluate({'sys_platform': 'darwin', 'platform_machine': 'x86_64'})
    locked = tomllib.loads((ROOT / 'uv.lock').read_text(encoding='utf-8'))
    package = next(p for p in locked['package'] if p['name'] == 'lora')
    for extra, dependencies in project['optional-dependencies'].items():
        expected = {Requirement(requirement).name for requirement in dependencies}
        assert {d['name'] for d in package['optional-dependencies'][extra]} == expected


def test_shipped_master_settings_are_recognized_by_the_schema():
    from lora.core.config import load_config

    raw = yaml.safe_load((ROOT / 'config/lora.yaml').read_text(encoding='utf-8'))
    config = load_config(ROOT / 'config/lora.yaml')

    def walk(values, model):
        for key, value in values.items():
            assert key in type(model).model_fields, f'Ignored setting: {key}'
            child = getattr(model, key)
            if isinstance(value, dict) and hasattr(child, 'model_fields'):
                walk(value, child)

    walk(raw, config)
    assert config.models.writer.num_ctx == 8192
    assert config.voice.stt.device == 'cpu'


@pytest.mark.parametrize(('arguments', 'expected'), [
    ([], ['kokoro', 'embeddinggemma']),
    (['--all-models'], ['kokoro', 'embeddinggemma', 'laya', 'reranker']),
    (['--laya-only'], ['laya']),
    (['--reranker-only'], ['reranker']),
])
def test_model_setup_only_downloads_requested_assets(monkeypatch, tmp_path, arguments, expected):
    spec = importlib.util.spec_from_file_location('lora_model_setup', ROOT / 'scripts/download_models.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'MODELS_DIR', tmp_path)
    called = []
    for label, function in [('kokoro', 'setup_kokoro'), ('embeddinggemma', 'setup_embeddinggemma'),
                            ('laya', 'setup_laya'), ('reranker', 'setup_bge_reranker')]:
        monkeypatch.setattr(module, function, lambda force, name=label: called.append(name) or True)
    monkeypatch.setattr(sys, 'argv', ['download_models.py', *arguments])
    assert module.main() == 0
    assert called == expected


def test_new_seed_preferences_match_reference_checks_and_preserve_existing_profile(monkeypatch, tmp_path):
    from lora.config.manager import ConfigManager

    monkeypatch.setattr(Path, 'home', lambda: tmp_path / 'home')
    manager = ConfigManager(tmp_path / 'user')
    manager.ensure_ecosystem(apply_acls=False)
    profile = manager.paths.user_profile_file
    assert 'lightweight source-reference checks' in profile.read_text(encoding='utf-8')
    skill = yaml.safe_load((manager.paths.skills_dir / 'deep_research_brief/skill.yaml').read_text(encoding='utf-8'))
    assert skill['default_depth'] == 'deep'
    assert 'independent fact checking' in skill['system_prompt_addon']
    profile.write_text('My actual personal preferences.', encoding='utf-8')
    manager.ensure_ecosystem(apply_acls=False)
    assert profile.read_text(encoding='utf-8') == 'My actual personal preferences.'


def test_package_and_child_imports_share_model_classes():
    from lora.experimental.outer_loop.genome import PromptGenome as CanonicalGenome
    from lora.experimental.outer_loop.scorer import ScoreBreakdown as CanonicalScore
    from lora.outer_loop import PromptGenome, ScoreBreakdown
    from lora.outer_loop.genome import PromptGenome as ChildGenome
    from lora.outer_loop.scorer import ScoreBreakdown as ChildScore

    assert PromptGenome is ChildGenome is CanonicalGenome
    assert ScoreBreakdown is ChildScore is CanonicalScore
