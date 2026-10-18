"""Local scene hints trigger fresh product capture only at a consented idle boundary."""
import asyncio
import base64
import io
import time

import pytest
from PIL import Image

from app.conversation_transport import ConversationConfig, ConversationCoordinator
from test_conversation_transport import Brain, Browser, Speech, cleanup, service_session


class ReadyTranscriber:
    ready = True

    async def start(self): pass
    async def close(self): pass
    def discard_turn(self): pass


class RecordingBrowser(Browser):
    def __init__(self):
        super().__init__()
        self.events = []

    async def send_json(self, event):
        self.events.append(event)
        await super().send_json(event)


class SceneBrain(Brain):
    def __init__(self, *args):
        super().__init__(*args)
        self.auto_calls = 0

    def automatic_question(self):
        return 'Compare these products using my latest conversation preference.'

    async def respond_automatically(self, frame_provider=None):
        self.auto_calls += 1
        return await super().respond('camera', frame_provider)


class QuietGuidanceBrain(SceneBrain):
    async def respond_automatically(self, frame_provider=None):
        result = await super().respond_automatically(frame_provider)
        return {**result, 'silent': True}

    async def respond(self, text, frame_provider=None):
        return {**await super().respond(text, frame_provider), 'silent': True}


async def scene_connection(*, enabled=True, product_mode='product_search', mode='live_glasses',
                           brain_factory=SceneBrain, speech_factory=None, speech_provider='disabled',
                           service_and_session=None):
    service, session = service_and_session or service_session(['Avoid sugar'])
    service.get(session.id).session.mode = mode
    owner = ConversationCoordinator(service,
        config=ConversationConfig(transcription_provider='disabled', speech_provider=speech_provider,
                                  product_mode=product_mode),
        brain_factory=brain_factory, transcriber_factory=lambda *_: ReadyTranscriber(),
        speech_factory=speech_factory)
    browser = RecordingBrowser()
    connection = await owner.attach(browser, session.id)
    task = asyncio.create_task(connection.run())
    await browser.send('session.start', session_id=session.id, voice_consented=True, client_epoch=2,
                       automatic_capture_enabled=enabled)
    ready = await browser.until('session.ready')
    await browser.until('state', lambda event: event['state'] == 'listening')
    return owner, session, browser, connection, task, ready


async def send_photo(browser, request, *, epoch=2):
    data = io.BytesIO()
    Image.new('RGB', (100, 80), 'white').save(data, 'PNG')
    await browser.send('camera.response', client_epoch=epoch, request_id=request['request_id'],
        image_data_url='data:image/png;base64,' + base64.b64encode(data.getvalue()).decode(),
        width=100, height=80)


@pytest.mark.asyncio
async def test_quiet_automatic_guidance_resumes_input_without_playback_but_manual_reply_cannot_be_silenced():
    owner, _, browser, connection, task, _ = await scene_connection(
        brain_factory=QuietGuidanceBrain, speech_provider='native')
    try:
        await browser.until('input.enabled', lambda event: event['enabled'])
        await browser.send('scene.stable', client_epoch=2)
        await browser.until('scene.accepted')
        request = await browser.until('camera.request')
        assert not connection.input_enabled
        await send_photo(browser, request)
        await browser.until('snapshot')
        await browser.until('input.enabled', lambda event: event['enabled'])
        assert connection.runtime.state == 'listening' and not connection.playback_pending
        assert not any(event['type'] in ('response.text', 'audio.chunk', 'audio.done') for event in browser.events)
        assert not any(row['role'] == 'assistant' for row in connection.runtime.recent)
        await browser.send('input.text', text='Hello', client_epoch=3)
        response = await browser.until('response.text')
        assert response['text'] == 'Hello'
        await browser.until('state', lambda event: event['state'] == 'listening')
        assert connection.playback_pending and not connection.input_enabled
        await browser.send('playback.finished', client_epoch=3, context_id=response['context_id'])
        await browser.until('input.enabled', lambda event: event['enabled'])
    finally:
        await cleanup(owner, task)


@pytest.mark.asyncio
async def test_one_scene_one_fresh_camera_no_fake_transcript_duplicate_busy_and_cooldown():
    owner, session, browser, connection, task, ready = await scene_connection()
    try:
        assert ready['automatic_capture_enabled'] is True
        generation = connection.runtime.generation
        await browser.send('scene.stable', client_epoch=2)
        accepted = await browser.until('scene.accepted')
        request = await browser.until('camera.request')
        assert request['trigger'] == 'scene.stable'
        await browser.send('scene.stable', client_epoch=2)
        assert (await browser.until('scene.ignored'))['reason'] == 'busy'
        assert connection.runtime.generation == generation
        await send_photo(browser, request)
        assert (await browser.until('response.text'))['text'] == 'camera received'
        await browser.until('state', lambda event: event['state'] == 'listening')
        await browser.send('scene.stable', client_epoch=2)
        assert (await browser.until('scene.ignored'))['reason'] == 'cooldown'
        assert sum(event['type'] == 'camera.request' for event in browser.events) == 1
        assert not any(event['type'] == 'transcript.final' for event in browser.events)
        recent = owner.snapshot(session.id)['recent']
        assert all(item['source'] == 'automatic_scene' for item in recent)
        assert recent[0]['turn_id'] == accepted['turn_id']
        assert recent[0]['text'] == connection.runtime.brain.automatic_question()
        connection.last_automatic_scene_at = time.monotonic() - 8.01
        await browser.send('scene.stable', client_epoch=2)
        await browser.until('scene.accepted')
        second = await browser.until('camera.request')
        assert second['request_id'] != request['request_id']
        assert connection.runtime.brain.auto_calls == 2
    finally:
        await cleanup(owner, task)


@pytest.mark.asyncio
@pytest.mark.parametrize('options,reason', [
    ({'enabled': False}, 'disabled'), ({'enabled': 'true'}, 'disabled'),
    ({'product_mode': 'reviewed_catalog'}, 'unsupported_mode'),
    ({'mode': 'live_phone'}, 'unsupported_mode'),
])
async def test_automatic_capture_requires_explicit_flag_live_glasses_and_product_search(options, reason):
    owner, _, browser, connection, task, ready = await scene_connection(**options)
    try:
        assert ready['automatic_capture_enabled'] is False
        await browser.send('scene.stable', client_epoch=2)
        assert (await browser.until('scene.ignored'))['reason'] == reason
        assert connection.response_task is None and connection.last_automatic_scene_at is None
        assert not connection.runtime.brain.cancelled
    finally:
        await cleanup(owner, task)


@pytest.mark.asyncio
@pytest.mark.parametrize('epoch', [None, True, 1, 3])
async def test_scene_epoch_must_exactly_match_and_cannot_advance_context(epoch):
    owner, _, browser, connection, task, _ = await scene_connection()
    try:
        for kind in ['scene.stable', 'scene.configure']:
            await browser.send(kind, client_epoch=epoch, enabled=False)
            assert (await browser.until('scene.ignored'))['reason'] == 'stale_epoch'
        assert connection.client_epoch == 2 and connection.automatic_capture_enabled
        assert connection.response_task is None
    finally:
        await cleanup(owner, task)


@pytest.mark.asyncio
@pytest.mark.parametrize('busy', ['input', 'transcript', 'vad', 'text_only'])
async def test_scene_cannot_interrupt_pending_speech_transcription_or_unavailable_microphone(busy):
    owner, _, browser, connection, task, _ = await scene_connection()
    try:
        if busy == 'input': connection.input_turn = 'user-turn'
        elif busy == 'transcript': connection.transcript_deadline_task = asyncio.create_task(asyncio.Event().wait())
        elif busy == 'vad': connection.detector.active = True
        else: connection.runtime.state = 'text_only'
        await browser.send('scene.stable', client_epoch=2)
        assert (await browser.until('scene.ignored'))['reason'] == 'busy'
        assert not connection.runtime.brain.cancelled and connection.response_task is None
        assert connection.last_automatic_scene_at is None
    finally:
        await cleanup(owner, task)


@pytest.mark.asyncio
@pytest.mark.parametrize('speech_provider', ['disabled', 'native'])
async def test_scene_waits_for_actual_playback_completion_without_cancelling_audio(speech_provider):
    speaker = Speech(count=2)
    owner, _, browser, connection, task, _ = await scene_connection(speech_provider=speech_provider,
        speech_factory=(lambda: speaker) if speech_provider == 'disabled' else None)
    try:
        await browser.send('input.text', text='hello', client_epoch=3)
        answer = await browser.until('response.text')
        if speech_provider == 'disabled': await browser.until('audio.done')
        await browser.until('state', lambda event: event['state'] == 'listening')
        assert connection.playback_pending
        await browser.send('scene.stable', client_epoch=3)
        assert (await browser.until('scene.ignored'))['reason'] == 'busy'
        assert not any(event['type'] == 'response.cancelled' for event in browser.events)
        await browser.send('playback.finished', context_id=answer['context_id'], client_epoch=2)
        await browser.send('scene.stable', client_epoch=3)
        assert (await browser.until('scene.ignored'))['reason'] == 'busy'
        await browser.send('playback.finished', context_id=answer['context_id'], client_epoch=3)
        await browser.until('state', lambda event: event['state'] == 'listening')
        await browser.send('scene.stable', client_epoch=3)
        await browser.until('scene.accepted')
        assert (await browser.until('camera.request'))['trigger'] == 'scene.stable'
    finally:
        await cleanup(owner, task)


@pytest.mark.asyncio
@pytest.mark.parametrize('control', ['disable', 'interrupt', 'context'])
async def test_disable_stop_or_context_cancel_pending_automatic_capture_and_ignore_late_photo(control):
    owner, _, browser, connection, task, _ = await scene_connection()
    try:
        await browser.send('scene.stable', client_epoch=2)
        await browser.until('scene.accepted')
        request = await browser.until('camera.request')
        old_generation = connection.runtime.generation
        if control == 'disable':
            await browser.send('scene.configure', enabled=False, client_epoch=2)
            assert (await browser.until('scene.configured'))['enabled'] is False
        elif control == 'interrupt':
            await browser.send('interrupt', client_epoch=3)
            await browser.until('state', lambda event: event['state'] == 'listening')
        else:
            await browser.send('context.update', recipient_id='household', client_epoch=3)
            await browser.until('snapshot', lambda event: event['snapshot']['recipient_id'] == 'household')
        assert connection.runtime.generation > old_generation and not connection.camera_pending
        await send_photo(browser, request)
        await browser.send('ping')
        await browser.until('pong')
        assert not any(event['type'] == 'response.text' for event in browser.events)
        assert connection.response_task is None
    finally:
        await cleanup(owner, task)


@pytest.mark.asyncio
async def test_disable_does_not_cancel_manual_turn_and_toggle_cannot_bypass_cooldown():
    owner, _, browser, connection, task, _ = await scene_connection()
    try:
        await browser.send('input.text', text='camera', client_epoch=3)
        request = await browser.until('camera.request')
        assert 'trigger' not in request
        generation = connection.runtime.generation
        connection.last_automatic_scene_at = time.monotonic()
        await browser.send('scene.configure', enabled=False, client_epoch=3)
        await browser.until('scene.configured')
        assert connection.runtime.generation == generation and connection.camera_pending
        await send_photo(browser, request, epoch=3)
        await browser.until('response.text')
        await browser.until('state', lambda event: event['state'] == 'listening')
        await browser.send('scene.configure', enabled=True, client_epoch=3)
        assert (await browser.until('scene.configured'))['enabled'] is True
        await browser.send('scene.stable', client_epoch=3)
        assert (await browser.until('scene.ignored'))['reason'] == 'cooldown'
        assert owner.snapshot(connection.session_id)['recent'][0]['source'] == 'text'
    finally:
        await cleanup(owner, task)


@pytest.mark.asyncio
async def test_scene_carries_no_image_and_never_uses_supplied_media():
    owner, _, browser, connection, task, _ = await scene_connection()
    try:
        await browser.send('scene.stable', client_epoch=2, image_data_url='do not inspect this')
        assert (await browser.until('scene.ignored'))['reason'] == 'unexpected_media'
        assert connection.response_task is None
    finally:
        await cleanup(owner, task)


@pytest.mark.asyncio
async def test_automatic_scene_uses_real_product_adapter_and_exactly_one_capture():
    from app.conversation_product_search import ProductSearchBrain
    from test_conversation_product_search import setup
    service, guide, sid, _ = setup()
    owner, _, browser, connection, task, _ = await scene_connection(
        service_and_session=(service, service.get(sid).session),
        brain_factory=lambda profile, recipient: ProductSearchBrain(service, profile, recipient))
    try:
        await browser.send('scene.stable', client_epoch=2)
        await browser.until('scene.accepted')
        await send_photo(browser, await browser.until('camera.request'))
        assert (await browser.until('response.text'))['text']
        snapshot = (await browser.until('snapshot'))['snapshot']
        assert len(snapshot['products']) == 2
        assert guide.calls == ['identify', 'search', 'search', 'respond']
        await browser.send('input.text', text='Why that one?', client_epoch=3)
        await browser.until('response.text')
        assert guide.calls == ['identify', 'search', 'search', 'respond', 'respond']
        assert sum(event['type'] == 'camera.request' for event in browser.events) == 1
    finally:
        await cleanup(owner, task)
