import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest


@pytest.fixture
def preflight():
    path = Path(__file__).resolve().parents[2] / 'scripts' / 'conversation_preflight.py'
    spec = importlib.util.spec_from_file_location('isolated_health_preflight', path)
    module = importlib.util.module_from_spec(spec)
    with patch('dotenv.load_dotenv'):
        spec.loader.exec_module(module)
    return module


def healthy():
    return {'version': '0.2.0-conversation', 'status': 'ok',
        'conversation': {'websocket': '/api/conversation/ws', 'input': 'pcm_s16le_24000_mono',
                         'reasoning_provider': 'anthropic', 'transcription_provider': 'elevenlabs', 'speech_provider': 'elevenlabs'},
        'providers': {'anthropic': True, 'elevenlabs': False}}


@pytest.mark.asyncio
async def test_health_is_exact_credential_free_get_and_only_returns_allowlisted_fields(preflight):
    calls = []
    payload = healthy()
    payload['private_field'] = 'private-remote-secret'
    payload['providers']['extra'] = 'private-remote-secret'
    payload['conversation']['private'] = 'private-remote-secret'
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=payload)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler),
            headers={'Authorization': 'Bearer private-client-secret', 'xi-api-key': 'private-client-secret'},
            cookies={'session': 'private-client-secret'}, auth=('private-user', 'private-pass')) as client:
        result = await preflight.check_backend('http://localhost:8000/', client=client)
        assert not client.is_closed
    assert len(calls) == 1 and calls[0].method == 'GET'
    assert str(calls[0].url) == 'http://localhost:8000/api/health'
    assert not any(name in calls[0].headers for name in ('authorization', 'cookie', 'xi-api-key'))
    assert result['compatible'] and result['reachable']
    assert result['version'] == '0.2.0-conversation'
    assert result['providers_configured'] == {'anthropic': True, 'elevenlabs': False}
    assert 'private' not in json.dumps(result)


@pytest.mark.asyncio
async def test_old_server_reachable_but_explicitly_incompatible(preflight):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200,
            json={'status': 'ok', 'version': '0.1.0', 'providers': {'perceptron': True}}))) as client:
        result = await preflight.check_backend('http://127.0.0.1:8000', client=client)
    assert result['reachable'] and not result['compatible'] and result['version'] == '0.1.0'
    assert result['status'] == 'incompatible'
    assert '/api/conversation/ws' in result['message'] and 'pcm_s16le_24000_mono' in result['message']


@pytest.mark.asyncio
@pytest.mark.parametrize('origin', ['http://user:private-secret@localhost', 'https://localhost/api',
    'https://localhost?token=private-secret', 'https://localhost#private-secret', 'file:///private-secret',
    'http://localhost:99999', 'http://localhost/\n', 'not-a-url', 'http://localhost?', 'http://localhost#'])
async def test_invalid_origin_never_connects_or_echoes_supplied_secret(preflight, origin):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: pytest.fail('must not connect'))) as client:
        result = await preflight.check_backend(origin, client=client)
    assert result['status'] == 'invalid_origin' and not result['reachable']
    assert 'private-secret' not in json.dumps(result)


@pytest.mark.asyncio
async def test_redirect_is_never_followed_or_echoed(preflight):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(302, headers={'location': 'https://example.test/?private-secret'})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
        result = await preflight.check_backend('http://localhost:8000', client=client)
    assert len(calls) == 1 and result['status'] == 'redirect_rejected'
    assert result['reachable'] and not result['compatible'] and 'private-secret' not in json.dumps(result)


@pytest.mark.asyncio
@pytest.mark.parametrize('response', [httpx.Response(200, content=b'private-secret' * 2000),
    httpx.Response(200, headers={'content-length': '20000'}, content=b'private-secret'),
    httpx.Response(200, content=b'not JSON private-secret'),
    httpx.Response(401, content=b'private-secret')])
async def test_bounded_or_invalid_response_never_prints_body(preflight, response):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as client:
        result = await preflight.check_backend('http://localhost:8000', client=client)
    assert result['reachable'] and not result['compatible']
    assert result['status'] in ('response_too_large', 'invalid_health_response', 'http_error')
    assert 'private-secret' not in json.dumps(result)


@pytest.mark.asyncio
async def test_stream_without_length_stops_at_16k_limit(preflight):
    class Stream(httpx.AsyncByteStream):
        reads = 0
        closed = False
        async def __aiter__(self):
            for _ in range(100):
                self.reads += 1
                yield bytes(4096)
        async def aclose(self): self.closed = True
    stream = Stream()
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream))) as client:
        result = await preflight.check_backend('http://localhost:8000', client=client)
    assert result['status'] == 'response_too_large' and stream.reads == 5 and stream.closed


@pytest.mark.asyncio
async def test_timeout_redaction_and_unknown_provider_filter(preflight):
    def timeout(_): raise httpx.ReadTimeout('private-secret remote payload')
    async with httpx.AsyncClient(transport=httpx.MockTransport(timeout)) as client:
        result = await preflight.check_backend('http://localhost:8000', client=client)
    assert result['status'] == 'unavailable' and 'private-secret' not in json.dumps(result)
    payload = healthy()
    payload['conversation']['speech_provider'] = 'private-secret'
    payload['conversation']['input'] = 'other-format-private-secret'
    payload['version'] = 'private-secret'
    payload['providers']['elevenlabs'] = 'private-secret'
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))) as client:
        result = await preflight.check_backend('http://localhost:8000', client=client)
    assert result['status'] == 'incompatible' and result['version'] is None
    assert result['conversation']['speech_provider'] is None
    assert 'private-secret' not in json.dumps(result)
