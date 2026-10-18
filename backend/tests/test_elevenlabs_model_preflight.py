"""Model configuration reporting is local, allowlisted and not a provider probe."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest


def module(name):
    path = Path(__file__).resolve().parents[2] / 'scripts' / (name + '.py')
    spec = importlib.util.spec_from_file_location('isolated_' + name, path)
    value = importlib.util.module_from_spec(spec)
    with patch('dotenv.load_dotenv'):
        spec.loader.exec_module(value)
    return value


@pytest.mark.asyncio
@pytest.mark.parametrize('model,transport', [
    ('eleven_v4_turbo', 'dialogue_websocket'), ('eleven_flash_v2_5', 'http_stream'),
    ('eleven_multilingual_v2', 'http_stream'), ('test-secret-not-a-model', None),
])
async def test_elevenlabs_local_preflight_reports_only_supported_model(monkeypatch, capsys, model, transport):
    preflight = module('elevenlabs_preflight')
    monkeypatch.setenv('ELEVENLABS_TTS_MODEL', model)
    monkeypatch.setenv('ELEVENLABS_API_KEY', 'test-secret-key')
    await preflight.check(SimpleNamespace(check_account=False, check_stt=False))
    output = capsys.readouterr().out
    result = json.JSONDecoder().raw_decode(output)[0]
    assert result['network'] == 'not_requested' and result['tts_generated'] is False
    assert result['tts_transport'] == transport
    assert result['tts_model'] == (model if transport else None)
    assert result['tts_model_supported'] is bool(transport)
    assert 'test-secret' not in output


@pytest.mark.asyncio
@pytest.mark.parametrize('model', ['eleven_v4_turbo', 'eleven_flash_v2_5', 'test-secret-model'])
async def test_remote_health_speech_model_is_allowlisted(model):
    preflight = module('conversation_preflight')
    payload = {'version': '0.2.0-conversation', 'conversation': {
        'websocket': '/api/conversation/ws', 'input': 'pcm_s16le_24000_mono',
        'speech_provider': 'elevenlabs', 'speech_model': model}}
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request:
        httpx.Response(200, json=payload))) as client:
        result = await preflight.check_backend('http://127.0.0.1:8001', client=client)
    assert result['compatible']
    assert result['conversation']['speech_model'] == (None if model.startswith('test-secret') else model)
    assert 'test-secret' not in json.dumps(result)


@pytest.mark.asyncio
async def test_product_search_preflight_does_not_claim_catalog_probe_checks_wording(monkeypatch, capsys):
    preflight = module('conversation_preflight')
    monkeypatch.setenv('TEXT_PROVIDER', 'anthropic')
    monkeypatch.setattr(preflight.BrainSettings, 'from_env', lambda: SimpleNamespace(
        provider='deepseek', model='deepseek-flash', reasoning_key='', deepseek_key='',
        perceptron_key='', catalog_path='missing-catalog'))
    monkeypatch.setattr(preflight.ConversationConfig, 'from_env', lambda: SimpleNamespace(
        product_mode='product_search', transcription_provider='elevenlabs',
        speech_provider='elevenlabs', elevenlabs_tts_model='eleven_v4_turbo'))
    monkeypatch.setattr(preflight, 'ShoppingBrain', lambda *a, **kw: pytest.fail('No probe without a key'))
    await preflight.check(True)
    output = capsys.readouterr().out
    result = json.JSONDecoder().raw_decode(output)[0]
    assert result['reasoning_probe_scope'] == 'reviewed_catalog_only'
    assert result['reasoning_provider'] == 'deepseek'
    assert result['active_analysis_mode'] == 'product_search'
    assert result['product_search_text_provider'] == 'anthropic'
    assert result['live_reasoning'] == 'not_run_missing_key'
    assert result['cloud_requests_made'] == 0
    assert 'do not validate product-search wording' in output


@pytest.mark.asyncio
async def test_product_search_remote_gemini_health_remains_a_supported_configuration():
    preflight = module('conversation_preflight')
    payload = {'conversation': {'websocket': '/api/conversation/ws',
        'input': 'pcm_s16le_24000_mono', 'analysis_mode': 'product_search',
        'reasoning_provider': 'gemini'}}
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request:
        httpx.Response(200, json=payload))) as client:
        result = await preflight.check_backend('http://127.0.0.1:8001', client=client)
    assert result['compatible'] and result['conversation']['reasoning_provider'] == 'gemini'
