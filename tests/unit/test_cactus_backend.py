"""Cactus output is evidence/proposals, never an alternate tool executor."""
import asyncio
import base64
import io
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from lora.agent.runtime import AgentRuntime
from lora.core.config import CactusConfig
from lora.dialogue import ConversationHistory
from lora.models.cactus_backend import (
    CactusCancelledError,
    CactusError,
    CactusWorker,
    _describe_mirage,
    tool_schemas,
    validate_calls,
)


def registry():
    result = MagicMock()
    result.app_launcher.applications = {'zed': {'name': 'Zed'}}
    result.list_tools.return_value = [
        {'name': 'open_app', 'description': 'Launch app', 'schema': {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']}},
        {'name': 'read_file', 'description': 'Read file', 'schema': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}},
        {'name': 'shell_exec', 'description': 'Never allowed', 'schema': {'type': 'object'}},
    ]
    return result


def runtime():
    reg = registry()
    rt = AgentRuntime('http://unused', None, ConversationHistory(), lambda: reg,
                      cactus_config=CactusConfig(needle_enabled=True, mirage_enabled=True))
    rt._tool_with_provenance = AsyncMock(return_value={'status': 'success', 'message': 'Opened Zed.'})
    rt._stream = AsyncMock(return_value='A test image.')
    return rt


def response(*calls, confidence=.95):
    return {'type': 'call', 'success': True, 'confidence': confidence,
            'function_calls': [{'name': name, 'arguments': args} for name, args in calls]}


def test_mirage_recovers_nonfinite_scores_without_changing_quoted_text():
    model = SimpleNamespace(describe=lambda *_, **__: json.loads(
        r'{"objects":[{"name":"cat","score":-nan},{"name":"ear","score":0.5}],'
        r'"text":[{"string":"quoted \"score\":-nan and Infinity","score":inf}]}'
    ))
    result = _describe_mirage(model, None)
    assert result['objects'] == [{'name': 'cat', 'score': None}, {'name': 'ear', 'score': .5}]
    assert result['text'] == [{'string': 'quoted "score":-nan and Infinity', 'score': None}]


def test_mirage_still_rejects_truncated_json():
    model = SimpleNamespace(describe=lambda *_, **__: json.loads('{"objects":['))
    with pytest.raises(json.JSONDecodeError):
        _describe_mirage(model, None)


@pytest.mark.parametrize('bad', [
    ('open_app', {'name': 'bash'}), ('open_app', {'name': 'zed', 'command': 'sh'}),
    ('shell_exec', {'command': 'sh'}), ('read_file', {'path': True}),
])
def test_invalid_proposal_batch_rejected_before_any_execution(bad):
    schemas = tool_schemas(registry())
    with pytest.raises(CactusError):
        validate_calls(response(('open_app', {'name': 'zed'}), bad), schemas)


def test_mobile_schema_does_not_expose_desktop_actions():
    assert tool_schemas(registry(), allow_actions=False) == []


def test_cancelled_request_never_starts_native_worker(tmp_path):
    engine = CactusWorker('needle', tmp_path, 1, 1)
    with pytest.raises(CactusCancelledError):
        engine.call({}, lambda: True)
    assert engine._process is None
    engine.close()


@pytest.mark.asyncio
async def test_duplicate_proposals_execute_only_once():
    rt = runtime()
    rt._cactus_call = AsyncMock(return_value=response(('open_app', {'name': 'zed'})))
    result = await rt._run_needle_tools('Open Zed', lambda _: None, lambda *_: None)
    assert result.meta['tool_backend'] == 'needle'
    rt._tool_with_provenance.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('confidence', [None, .1, float('nan'), 2])
async def test_uncertain_proposals_need_confirmation(confidence):
    rt = runtime()
    rt.confirm = AsyncMock(return_value=False)
    rt._cactus_call = AsyncMock(return_value=response(('open_app', {'name': 'zed'}), confidence=confidence))
    result = await rt._run_needle_tools('Open Zed', lambda _: None, lambda *_: None)
    rt.confirm.assert_awaited_once()
    rt._tool_with_provenance.assert_not_called()
    assert result.meta['status'] == 'not_confirmed'


@pytest.mark.asyncio
async def test_later_untrusted_output_cannot_drive_actions():
    rt = runtime()
    rt._cactus_call = AsyncMock(side_effect=[response(('read_file', {'path': '/tmp/example.txt'})), response(('open_app', {'name': 'zed'}))])
    result = await rt._run_needle_tools('Read the file', lambda _: None, lambda *_: None)
    rt._tool_with_provenance.assert_awaited_once()
    assert result.meta['tool_results'][-1]['result']['status'] == 'denied'


@pytest.mark.asyncio
async def test_model_failure_never_falls_through_to_another_executor():
    rt = runtime()
    rt._cactus_call = AsyncMock(side_effect=CactusError('native worker stopped'))
    result = await rt.run_turn('open zed', lambda _: None, lambda *_: None)
    assert result.kind == 'error'
    rt._tool_with_provenance.assert_not_called()


@pytest.mark.asyncio
async def test_mirage_observations_use_text_writer_and_never_web_or_raw_image_tokens():
    rt = runtime()
    image = Image.new('RGB', (20, 20), 'white')
    data = io.BytesIO()
    image.save(data, format='PNG')
    rt._cactus_call = AsyncMock(return_value={'scene': 'A white rectangle', 'reasoning': 'UNVERIFIED_TRACE', 'text': [], '_worker': {'backend': 'mirage'}})
    rt._research_fn = AsyncMock(side_effect=AssertionError('An image question must not become web research'))
    result = await rt.run_turn('What is in this image?', lambda _: None, lambda *_: None,
        attachments=[{'name': 'test.png', 'mime': 'image/png', 'data_url': 'data:image/png;base64,' + base64.b64encode(data.getvalue()).decode()}])
    assert result.meta['vision_backend'] == 'mirage'
    assert rt._stream.call_args.kwargs['role'] == 'writer'
    assert 'images' not in rt._stream.call_args.args[0][-1]
    assert 'A white rectangle' in rt._stream.call_args.args[0][-1]['content']
    assert 'UNVERIFIED_TRACE' not in rt._stream.call_args.args[0][-1]['content']
    rt._research_fn.assert_not_called()


@pytest.mark.asyncio
async def test_async_cancellation_signals_worker_abort(monkeypatch):
    rt = runtime()
    saw_abort = threading.Event()
    def call(request, abort):
        while not abort():
            saw_abort.wait(.01)
        saw_abort.set()
        raise CactusCancelledError('cancelled')
    monkeypatch.setattr('lora.agent.cactus_workflows.worker', lambda *_: SimpleNamespace(call=call))
    task = asyncio.create_task(rt._cactus_call('needle', {}))
    await asyncio.sleep(.03)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await asyncio.to_thread(saw_abort.wait, 1)


def test_emergency_config_switch_keeps_existing_backends(monkeypatch):
    rt = runtime()
    monkeypatch.setenv('LORA_DISABLE_CACTUS', '1')
    assert not rt._cactus_enabled('needle') and not rt._cactus_enabled('mirage')


@pytest.mark.asyncio
async def test_literal_app_proposal_has_independent_user_grounding_but_still_uses_gate():
    from lora.tools.control.apps import AppLauncher
    rt = runtime()
    rt._get_registry().app_launcher = AppLauncher()
    rt.confirm = AsyncMock(return_value=False)
    rt._cactus_call = AsyncMock(return_value=response(('open_app', {'name': 'zed'}), confidence=.2))
    result = await rt._run_needle_tools('Open Zed', lambda _: None, lambda *_: None)
    rt._tool_with_provenance.assert_awaited_once()
    rt.confirm.assert_not_called()
    assert result.meta['tool_backend'] == 'needle'


@pytest.mark.asyncio
async def test_cancel_between_batch_actions_stops_the_second_action():
    rt = runtime()
    rt._cactus_call = AsyncMock(return_value=response(('read_file', {'path': '/tmp/a'}), ('read_file', {'path': '/tmp/b'})))
    def stop_after_first(*args):
        rt._should_abort = lambda: 'cancelled'
        from lora.safety.killswitch import KillSwitch
        KillSwitch.get_global().trigger('cancelled')
        return {'status': 'success', 'text': 'example'}
    rt._tool_with_provenance = AsyncMock(side_effect=stop_after_first)
    with pytest.raises(Exception, match='cancelled'):
        await rt._run_needle_tools('Read the files', lambda _: None, lambda *_: None)
    rt._tool_with_provenance.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(('query', 'name', 'args', 'prompt'), [
    ('Open Zed', 'open_app', {'name': 'zed'}, 'Open zed'),
    ('Open Wikipedia', 'open_url', {'url': 'https://en.wikipedia.org'}, 'Open https://en.wikipedia.org'),
    ('Open wiki', 'open_url', {'url': 'https://en.wikipedia.org'}, 'Open https://en.wikipedia.org'),
])
async def test_catalogue_arguments_grounded_and_literal_action_finishes(query, name, args, prompt):
    from lora.tools.registry import ToolRegistry
    rt = runtime()
    reg = ToolRegistry()
    rt._get_registry = lambda: reg
    rt.confirm = AsyncMock(return_value=False)
    rt._cactus_call = AsyncMock(side_effect=[response((name, args), confidence=.2), CactusError('must not plan again')])
    try:
        result = await rt._run_needle_tools(query, lambda _: None, lambda *_: None)
        assert result.kind == 'agent'
        rt._cactus_call.assert_awaited_once()
        request = rt._cactus_call.call_args.args[1]
        assert request['text'] == prompt
        assert [tool['name'] for tool in request['tools']] == [name]
        rt._tool_with_provenance.assert_awaited_once()
        rt.confirm.assert_not_called()
    finally:
        await reg.close()
