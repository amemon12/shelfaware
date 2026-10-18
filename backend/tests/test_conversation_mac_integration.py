"""Mac merge contracts: preserve existing API and accept its bounded still photos."""
import asyncio
import base64
import io
import random
from datetime import datetime, timezone

import httpx
import pytest
from PIL import Image

from app.conversation_brain import BrainSettings, ReviewedCatalog, ShoppingBrain
from app.conversation_transport import ConversationConfig
from app.main import create_app
from app.providers import Settings
from addons.pair_pipeline.perceptron_adapter import Frame, _validate_frame
from test_conversation_transport import cleanup, connected
from test_conversation_scene import ReadyTranscriber


def large_png():
    image = Image.frombytes('RGB', (1000, 800), random.Random(7).randbytes(1000 * 800 * 3))
    buffer = io.BytesIO()
    image.save(buffer, 'PNG')
    assert 2 * 1024 * 1024 < len(buffer.getvalue()) < 4 * 1024 * 1024
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_product_search_accepts_existing_large_still_without_widening_audio(monkeypatch):
    seen_timeouts = []
    timeout = asyncio.timeout
    def record_timeout(delay):
        seen_timeouts.append(delay)
        return timeout(delay)
    monkeypatch.setattr(asyncio, 'timeout', record_timeout)
    config = ConversationConfig(product_mode='product_search', transcription_provider='disabled', speech_provider='native')
    owner, session, browser, connection, task, _ = await connected(config=config,
        transcriber_factory=lambda *_: ReadyTranscriber())
    try:
        await browser.send('input.text', text='camera', client_epoch=3)
        request = await browser.until('camera.request')
        await browser.send('camera.response', request_id=request['request_id'], width=1000, height=800,
            image_data_url='data:image/png;base64,' + base64.b64encode(large_png()).decode())
        response = await browser.until('response.text')
        assert response['text'] == 'camera received'
        assert 10 in seen_timeouts and 50 in seen_timeouts
        await browser.send('playback.finished', context_id=response['context_id'], client_epoch=3)
        await browser.until('input.enabled', lambda event: event['enabled'])
        # The bound still applies while listening; muted audio is intentionally
        # discarded before decoding or validation under the half-duplex contract.
        await browser.send('audio.append', audio='x' * 4001, client_epoch=3)
        assert (await browser.until('error'))['code'] == 'invalid_message'
    finally:
        await cleanup(owner, task)


@pytest.mark.asyncio
async def test_product_search_rejects_over_four_mib_frame():
    config = ConversationConfig(product_mode='product_search', transcription_provider='disabled', speech_provider='native')
    owner, session, browser, connection, task, _ = await connected(config=config)
    try:
        await browser.send('input.text', text='camera', client_epoch=3)
        request = await browser.until('camera.request')
        raw = large_png()
        raw += bytes(4 * 1024 * 1024 + 1 - len(raw))
        await browser.send('camera.response', request_id=request['request_id'], width=1000, height=800,
            image_data_url='data:image/png;base64,' + base64.b64encode(raw).decode())
        assert (await browser.until('error'))['code'] == 'invalid_camera_image'
        assert (await browser.until('response.text'))['text'] == 'please try camera again'
    finally:
        await cleanup(owner, task)


def test_catalog_detector_keeps_its_original_two_mib_limit():
    frame = Frame('test', 'data:image/png;base64,' + base64.b64encode(large_png()).decode(),
                  1000, 800, datetime.now(timezone.utc).isoformat())
    with pytest.raises(ValueError, match='size limit'):
        _validate_frame(frame)
    _validate_frame(frame, max_image_bytes=4 * 1024 * 1024)


@pytest.mark.asyncio
async def test_health_preserves_product_provider_contract_and_adds_voice_capabilities(monkeypatch):
    monkeypatch.setenv('CONVERSATION_PRODUCT_MODE', 'product_search')
    monkeypatch.setenv('CONVERSATION_REASONING_PROVIDER', 'deepseek')
    monkeypatch.setenv('CONVERSATION_TRANSCRIPTION_PROVIDER', 'elevenlabs')
    monkeypatch.setenv('CONVERSATION_SPEECH_PROVIDER', 'elevenlabs')
    monkeypatch.setenv('ELEVENLABS_API_KEY', 'private-test-voice-key')
    monkeypatch.setenv('OPENAI_API_KEY', 'private-test-unused-key')
    app = create_app(Settings(text_provider='deepseek', deepseek_key='private-test-text-key', api_token='test-token'))
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            health = await client.get('/api/health')
            assert health.status_code == 200
            assert health.json()['providers'] == {'perceptron': False, 'gemini': False, 'deepseek': True, 'anthropic': False}
            voice = health.json()['conversation']
            assert voice['analysis_mode'] == 'product_search'
            assert voice['evidence_scope'] == 'unverified_catalog_search'
            assert voice['providers_configured']['elevenlabs'] is True
            assert voice['providers_configured']['openai_optional'] is False
            assert 'private-test' not in health.text
            assert (await client.get('/api/conversation/sessions/invalid')).status_code == 401
            assert (await client.get('/api/fixtures')).status_code == 401
    finally:
        await app.state.conversation.close()
        await app.state.service.providers.close()


@pytest.mark.asyncio
async def test_empty_catalog_does_not_block_basic_conversation_or_make_provider_requests():
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: pytest.fail('Greeting must stay local'))) as client:
        brain = ShoppingBrain({'consented': True}, settings=BrainSettings(), client=client,
            catalog=ReviewedCatalog.from_payload({'version': 1, 'products': []}))
        try:
            result = await brain.respond('Hello')
            assert result['status'] == 'ready' and result['reply']
            assert brain.snapshot()['catalog_ready'] is False
        finally:
            await brain.close()
