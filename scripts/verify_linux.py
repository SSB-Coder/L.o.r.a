"""Verify this checkout's Fedora runtime; --models also exercises cached models.

No microphone recording or audio playback is performed. The optional language
model check calls only the local Ollama endpoint.
"""
import argparse
import importlib.metadata
import json
import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(models=False):
    if Path(sys.prefix).resolve() != (ROOT / '.venv').resolve():
        raise RuntimeError('Use Lora/.venv/bin/python, not another project environment')
    os.chdir(ROOT)
    import numpy as np
    import onnxruntime
    import sounddevice
    import torch
    from playwright.sync_api import sync_playwright
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWidgets import QApplication

    assert torch.ones(2).sum().item() == 2
    assert torch.version.cuda is None, 'Lora Linux profile expects CPU PyTorch'
    app = QApplication.instance() or QApplication([])
    view = QWebEngineView()
    view.close()
    app.processEvents()
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_content('<title>Lora Fedora verification</title>')
        assert page.title() == 'Lora Fedora verification'
        browser.close()
    report = {
        'python': sys.version, 'executable': sys.executable,
        'torch': torch.__version__, 'torch_cuda_build': torch.version.cuda,
        'numpy': np.__version__, 'onnxruntime': onnxruntime.__version__,
        'qt_webengine': 'passed', 'playwright_chromium': 'passed',
        'audio_devices': [d['name'] for d in sounddevice.query_devices()],
        'session': os.environ.get('XDG_SESSION_TYPE'),
        'wayland_limitations': 'Global hotkeys, window control and injected typing unavailable',
        'installed_packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
    }
    if models:
        from faster_whisper import WhisperModel
        from kokoro_onnx import Kokoro

        from lora.grounding.embed import EmbeddingEngine
        kokoro = Kokoro(str(ROOT / 'models/kokoro/kokoro-v1.0.onnx'), str(ROOT / 'models/kokoro/voices-v1.0.bin'))
        samples, rate = kokoro.create('Lora is ready on Fedora.', voice='af_sky')
        assert len(samples) > 0 and np.isfinite(samples).all()
        report['kokoro'] = {'status': 'passed', 'samples': len(samples), 'sample_rate': rate}
        whisper = WhisperModel('base.en', device='cpu', compute_type='int8', local_files_only=True)
        segments, _ = whisper.transcribe(np.zeros(16000, dtype=np.float32), language='en', vad_filter=False)
        list(segments)
        import soundfile as sf
        with tempfile.TemporaryDirectory(prefix='lora-speech-check-') as folder:
            audio_path = Path(folder) / 'speech.wav'
            sf.write(audio_path, samples, rate)
            segments, _ = whisper.transcribe(str(audio_path), language='en', beam_size=3)
            transcription = ' '.join(segment.text for segment in segments).strip()
        assert transcription, 'Synthesized speech was not transcribed'
        report['whisper_cpu_int8'] = {'status': 'passed', 'tts_roundtrip': transcription}
        embedder = EmbeddingEngine()
        vectors = embedder.encode(['A Linux computer.', 'An artificial life simulation.'])
        assert not embedder._use_fallback and vectors.shape[0] == 2 and np.isfinite(vectors).all()
        report['embeddings'] = {'status': 'passed', 'shape': list(vectors.shape), 'fallback': False}
        from lora.models.profiles import PREFERRED_MODEL
        request = urllib.request.Request('http://127.0.0.1:11434/api/generate',
            data=json.dumps({'model': PREFERRED_MODEL, 'prompt': 'Reply with the word ready.', 'stream': False,
                             'think': False, 'options': {'num_ctx': 8192, 'num_predict': 24}}).encode(),
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)
        assert result.get('response', '').strip(), result
        report['ollama'] = {'status': 'passed', 'model': PREFERRED_MODEL, 'response': result['response']}
        import asyncio

        from lora.agent.runtime import AgentRuntime
        from lora.dialogue import ConversationHistory
        from lora.tools.registry import ToolRegistry
        async def conversation():
            runtime = AgentRuntime('http://127.0.0.1:11434', PREFERRED_MODEL,
                                   ConversationHistory(), ToolRegistry)
            return await asyncio.wait_for(runtime.run_turn('hello', lambda _: None, lambda *_: None), timeout=180)
        turn = asyncio.run(conversation())
        assert turn.kind == 'chat' and turn.text.strip()
        report['lora_conversation'] = {'status': 'passed', 'response': turn.text}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'logs/fedora-verification.json')
    args = parser.parse_args()
    try:
        report = {'status': 'passed', **verify(args.models)}
    except Exception as error:
        import traceback
        report = {'status': 'failed', 'error': str(error), 'traceback': traceback.format_exc()}
    report['checked_at_unix'] = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'installed_packages'}, indent=2))
    sys.exit(0 if report['status'] == 'passed' else 1)
