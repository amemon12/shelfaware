from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from app.main import create_app
from app.providers import Settings


@pytest.mark.asyncio
async def test_voice_session_prevents_parallel_legacy_mutations():
    app = create_app(Settings())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        session = (await client.post('/api/sessions', json={'mode': 'fixture'})).json()
        app.state.conversation.runtimes[session['id']] = SimpleNamespace(connection=object())
        base = '/api/sessions/' + session['id']
        requests = [('turn', {'text': 'hello', 'client_turn_id': str(uuid4()), 'recipient_id': 'self'}),
                    ('review', {'product_id': 'untrusted', 'confirmed': True}),
                    ('confirm', {'action_id': str(uuid4())})]
        for route, data in requests:
            response = await client.post(base + '/' + route, json=data)
            assert response.status_code == 409, response.text
        app.state.conversation.runtimes.clear()
        await app.state.service.providers.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('mode,expected_reasoning', [('reviewed_catalog','deepseek'),('product_search','gemini')])
async def test_health_exposes_selection_without_claiming_provider_success(monkeypatch, mode, expected_reasoning):
    monkeypatch.setenv('CONVERSATION_PRODUCT_MODE',mode)
    monkeypatch.setenv('CONVERSATION_REASONING_PROVIDER','deepseek')
    app = create_app(Settings())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        data = (await client.get('/api/health')).json()
        assert data['conversation']['reasoning_provider'] == expected_reasoning
        assert 'configured_not_provider_tested' in data['conversation']
        assert isinstance(data['providers']['deepseek'], bool)
        await app.state.service.providers.close()
