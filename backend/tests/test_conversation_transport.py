import asyncio
import base64
import io
import json
import struct
import threading
from types import SimpleNamespace
from dataclasses import replace

import httpx
import pytest
from fastapi import FastAPI, WebSocketDisconnect
from fastapi.testclient import TestClient
from PIL import Image

from app.conversation_transport import (ConversationConfig, ConversationCoordinator,
    create_conversation_router, _speech_chunks)
from app.local_transcription import LocalTranscription, LocalWhisperEngine
from app.openai_transcription import OpenAITranscription, Transcript, TranscriptionError, session_configuration
from app.speech_detection import SpeechDetector, vad_pcm16_16khz
from app.service import ShoppingService
from app.models import CreateSession, Profile, CartItem


class Browser:
    def __init__(self):
        self.incoming, self.outgoing = asyncio.Queue(), asyncio.Queue()
        self.close_code = None

    async def receive_text(self):
        value = await self.incoming.get()
        if value is None:
            raise WebSocketDisconnect()
        return json.dumps(value)

    async def send_json(self, event):
        await self.outgoing.put(event)

    async def close(self, code=1000):
        self.close_code = code
        await self.incoming.put(None)

    async def send(self, kind, **fields):
        await self.incoming.put({'type': kind, **fields})

    async def until(self, kind, predicate=lambda event: True):
        async with asyncio.timeout(4):
            while True:
                event = await self.outgoing.get()
                if event['type'] == 'error' and kind != 'error':
                    raise AssertionError(event)
                if event['type'] == kind and predicate(event):
                    return event


class Brain:
    def __init__(self, profile, recipient_id):
        self.recipient_id, self.cart, self.pending = recipient_id, [], None
        self.cancelled = []

    def set_recipient(self, recipient_id):
        self.recipient_id = recipient_id

    def snapshot(self):
        return {'recipient_id': self.recipient_id, 'cart': self.cart, 'pending_action': self.pending}

    def cancel(self, clear_pending=True):
        self.cancelled.append(clear_pending)
        if clear_pending:
            self.pending = None

    async def respond(self, text, frame_provider=None):
        if text == 'camera':
            frame = await frame_provider()
            text = 'camera received' if frame else 'please try camera again'
        elif text == 'propose':
            self.pending = {'kind': 'test'}
        elif text == 'yes':
            text = 'confirmed' if self.pending else 'nothing to confirm'
        return {'reply': text, 'snapshot': self.snapshot()}

    async def close(self):
        pass


class Speech:
    def __init__(self, count=3):
        self.count, self.parts, self.closed = count, [], 0

    async def stream(self, request, cancel):
        from addons.pair_pipeline.voice_adapter import PCMChunk
        self.parts.append(request.text)
        try:
            for index in range(self.count):
                await asyncio.sleep(0)
                yield PCMChunk(request.context_id, index, b'\x00\x00' * 2400)
        finally:
            self.closed += 1


class Upstream:
    def __init__(self):
        self.incoming, self.sent = asyncio.Queue(), asyncio.Queue()
        self.closed = False

    async def send(self, raw):
        event = json.loads(raw)
        await self.sent.put(event)
        if event['type'] == 'session.update':
            await self.incoming.put({'type': 'session.updated'})

    def __aiter__(self):
        return self

    async def __anext__(self):
        event = await self.incoming.get()
        if event is None:
            raise StopAsyncIteration
        return json.dumps(event)

    async def close(self):
        self.closed = True
        await self.incoming.put(None)

    async def event(self, kind, **fields):
        await self.incoming.put({'type': kind, **fields})

    async def until(self, kind):
        async with asyncio.timeout(3):
            while True:
                event = await self.sent.get()
                if event['type'] == kind:
                    return event


def service_session(goals=None):
    service = ShoppingService(None)
    session = service.create(CreateSession(mode='live_glasses', profile=Profile(consented=True, goals=goals or [])))
    return service, session


async def connected(*, brain_factory=Brain, config=None, **options):
    service, session = service_session(['Prefer less sodium'])
    config = config or ConversationConfig(transcription_provider='disabled')
    owner = ConversationCoordinator(service, config=config, brain_factory=brain_factory, **options)
    browser = Browser()
    connection = await owner.attach(browser, session.id)
    task = asyncio.create_task(connection.run())
    await browser.send('session.start', session_id=session.id, voice_consented=True, client_epoch=2, recipient_id='self')
    ready = await browser.until('session.ready')
    return owner, session, browser, connection, task, ready


async def cleanup(owner, task):
    await owner.close()
    await asyncio.wait_for(task, 2)


def test_vad_onset_preroll_silence_and_max_duration():
    detector = SpeechDetector(classify=lambda audio: audio != bytes(640))
    silence, voice = bytes(960), struct.pack('<480h', *([1000] * 480))
    assert len(vad_pcm16_16khz(voice)) == 640
    assert all(not detector.feed(silence) for _ in range(20))
    assert all(not detector.feed(voice) for _ in range(4))
    start = detector.feed(voice)
    assert start[0].kind == 'start' and len(start[0].pcm) == 15 * 960
    events = [event for _ in range(33) for event in detector.feed(silence)]
    assert events[-1].kind == 'end' and events[-1].reason == 'silence'
    detector = SpeechDetector(classify=lambda _: True, max_speech_ms=200)
    events = [event for _ in range(10) for event in detector.feed(voice)]
    assert events[-1].reason == 'max_duration'
    detector.reset()
    assert len(detector.pending) == 0 and not detector.active


@pytest.mark.asyncio
async def test_text_confirmations_epoch_and_default_no_openai_calls():
    owner, session, browser, connection, task, ready = await connected(config=ConversationConfig(
        openai_api_key='present-but-must-not-be-used', transcription_provider='disabled', speech_provider='native'))
    try:
        assert ready['client_epoch'] == 2 and not ready['audio_output_available']
        assert ready['speech_provider'] == 'native'
        for epoch, text in [(3, 'propose'), (4, 'yes')]:
            await browser.send('input.text', text=text, client_epoch=epoch)
            answer = await browser.until('response.text')
            assert answer['client_epoch'] == epoch
        assert answer['text'] == 'confirmed'
        await browser.send('input.text', text='stale', client_epoch=3)
        await browser.send('ping')
        await browser.until('pong')
        assert all(item['text'] != 'stale' for item in owner.snapshot(session.id)['recent'])
        await browser.send('context.update', recipient_id='household', client_epoch=5)
        await browser.until('snapshot', lambda e: e['snapshot']['recipient_id'] == 'household')
        assert owner.service.get(session.id).session.recipient_id == 'household'
        await browser.send('input.text', text='yes', client_epoch=6)
        assert (await browser.until('response.text'))['text'] == 'nothing to confirm'
    finally:
        await cleanup(owner, task)


@pytest.mark.asyncio
async def test_tts_barge_in_flushes_old_epoch_and_reconnect_reuses_brain():
    speaker = Speech(count=100)
    owner, session, browser, connection, task, _ = await connected(speech_factory=lambda: speaker)
    try:
        await browser.send('input.text', text='first', client_epoch=3)
        first = await browser.until('audio.chunk')
        await browser.send('interrupt', client_epoch=4)
        await browser.until('response.cancelled')
        await browser.send('input.text', text='second', client_epoch=5)
        second = await browser.until('response.text')
        assert second['client_epoch'] == 5 and second['context_id'] != first['context_id']
        for _ in range(2):
            assert (await browser.until('audio.chunk'))['client_epoch'] == 5
        old_brain = connection.runtime.brain
        replacement = Browser()
        newer = await owner.attach(replacement, session.id)
        assert browser.close_code == 4001 and newer.runtime.brain is old_brain
        await asyncio.wait_for(task, 2)
        assert speaker.closed >= 2
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_upstream_out_of_order_completion_is_bound_to_committed_item():
    ws, events, errors = Upstream(), [], []
    async def connect(_): return ws
    async def on_event(event): events.append(event)
    async def on_error(error): errors.append(error)
    transcriber = OpenAITranscription('fake', on_event, on_error, connector=connect)
    await transcriber.start()
    try:
        assert session_configuration()['session']['audio']['input']['turn_detection'] is None
        for turn in ('first', 'second'):
            transcriber.begin_turn(turn)
            transcriber.append(bytes(4800))
            transcriber.commit(turn)
            await ws.until('input_audio_buffer.commit')
        await ws.event('conversation.item.input_audio_transcription.completed', item_id='item-2', transcript='second answer')
        await ws.event('input_audio_buffer.committed', item_id='item-1')
        await ws.event('input_audio_buffer.committed', item_id='item-2')
        await ws.event('conversation.item.input_audio_transcription.completed', item_id='item-1', transcript='first answer')
        async with asyncio.timeout(2):
            while len(events) != 2:
                await asyncio.sleep(.001)
        assert [(e.turn_id, e.text) for e in events] == [('second', 'second answer'), ('first', 'first answer')]
        assert not errors
        await ws.event('error', error={'secret': 'must not appear'})
        async with asyncio.timeout(2):
            while not ws.closed:
                await asyncio.sleep(.001)
        assert len(errors) == 1 and 'secret' not in errors[0]
    finally:
        await transcriber.close()


@pytest.mark.asyncio
async def test_continuous_vad_to_upstream_to_reply_and_stale_final_drop():
    ws = Upstream()
    async def connect(_): return ws
    factory = lambda callback, error: OpenAITranscription('fake', callback, error, connector=connect)
    owner, session, browser, connection, task, ready = await connected(transcriber_factory=factory,
        vad_factory=lambda: SpeechDetector(classify=lambda pcm: pcm != bytes(640)))
    try:
        assert ready['audio_input_available']
        voiced = base64.b64encode(struct.pack('<480h', *([1000] * 480))).decode()
        silent = base64.b64encode(bytes(960)).decode()
        for _ in range(5): await browser.send('audio.append', audio=voiced)
        start = await browser.until('speech.started')
        for _ in range(33): await browser.send('audio.append', audio=silent)
        await ws.until('input_audio_buffer.commit')
        await ws.event('input_audio_buffer.committed', item_id='one')
        await ws.event('conversation.item.input_audio_transcription.delta', item_id='one', delta='hello')
        await ws.event('conversation.item.input_audio_transcription.completed', item_id='one', transcript='hello')
        assert (await browser.until('transcript.partial'))['turn_id'] == start['turn_id']
        assert (await browser.until('response.text'))['text'] == 'hello'
        await browser.send('interrupt', client_epoch=3)
        await browser.until('response.cancelled')
        await connection.on_transcript(Transcript('final', start['turn_id'], 'stale', 'one'))
        assert all(item['text'] != 'stale' for item in owner.snapshot(session.id)['recent'])
    finally:
        await cleanup(owner, task)


@pytest.mark.asyncio
async def test_camera_request_matches_current_id_and_decodes_actual_pixels():
    owner, session, browser, connection, task, _ = await connected()
    try:
        await browser.send('input.text', text='camera', client_epoch=3)
        request = await browser.until('camera.request')
        await browser.send('camera.response', request_id='old', error='old request')
        buffer = io.BytesIO()
        Image.new('RGB', (100, 80), 'white').save(buffer, 'PNG')
        await browser.send('camera.response', request_id=request['request_id'], image_data_url='data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode(), width=100, height=80,
                           captured_at='2000-01-01T00:00:00Z')
        assert (await browser.until('response.text'))['text'] == 'camera received'
        await browser.send('input.text', text='camera', client_epoch=4)
        request = await browser.until('camera.request')
        await browser.send('camera.response', request_id=request['request_id'], error='no camera')
        assert (await browser.until('response.text'))['text'] == 'please try camera again'
    finally:
        await cleanup(owner, task)


@pytest.mark.asyncio
@pytest.mark.parametrize('camera_available', [True, False])
async def test_photo_receipt_reports_processing_before_slow_analysis_without_reopening_input(camera_available):
    received, release = asyncio.Event(), asyncio.Event()

    class SlowVisionBrain(Brain):
        async def respond(self, text, frame_provider=None):
            frame = await frame_provider()
            received.set()
            await release.wait()
            return {'reply': 'checked' if frame else 'camera unavailable', 'snapshot': self.snapshot()}

    owner, session, browser, connection, task, _ = await connected(brain_factory=SlowVisionBrain)
    try:
        await browser.send('input.text', text='look at these', client_epoch=3)
        request = await browser.until('camera.request')
        # A stale request cannot satisfy capture or emit a false receipt.
        await browser.send('camera.response', request_id='old', error='old request')
        if camera_available:
            buffer = io.BytesIO()
            Image.new('RGB', (100, 80), 'white').save(buffer, 'PNG')
            await browser.send('camera.response', request_id=request['request_id'],
                image_data_url='data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode(),
                width=100, height=80)
        else:
            await browser.send('camera.response', request_id=request['request_id'], error='capture_unavailable')
        await asyncio.wait_for(received.wait(), 2)
        assert not connection.input_enabled
        queued = []
        while not browser.outgoing.empty():
            queued.append(browser.outgoing.get_nowait())
        progress = [e for e in queued if e['type'] == 'turn.progress']
        if camera_available:
            assert connection.runtime.state == 'thinking'
            assert len(progress) == 1
            assert progress[0]['request_id'] == request['request_id']
            assert progress[0]['stage'] == 'photo_received'
            assert progress[0]['client_epoch'] == 3
        else:
            assert not progress
        assert not any(e['type'] == 'response.text' for e in queued)
        release.set()
        assert (await browser.until('response.text'))['text'] == ('checked' if camera_available else 'camera unavailable')
    finally:
        release.set()
        await cleanup(owner, task)


@pytest.mark.asyncio
async def test_cloud_speech_splits_long_replies_without_losing_words():
    speaker = Speech(count=1)
    text = 'A factual sentence with portion and source. ' * 25
    chunks = [chunk async for chunk in _speech_chunks(speaker, 'context', text, 'quiet', asyncio.Event())]
    assert len(chunks) > 1 and all(len(part) <= 600 for part in speaker.parts)
    assert ' '.join(speaker.parts) == text.strip()
    assert [chunk.sequence for chunk in chunks] == list(range(len(chunks)))


def test_auth_and_profile_consent_before_websocket_accept():
    service, session = service_session()
    owner = ConversationCoordinator(service, config=ConversationConfig(api_token='test-secret', transcription_provider='disabled'), brain_factory=Brain)
    app = FastAPI()
    app.include_router(create_conversation_router(owner))
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as denied:
            with client.websocket_connect(f'/api/conversation/ws?session_id={session.id}'):
                pass
        assert denied.value.code == 4401
        service.get(session.id).session.profile.consented = False
        with pytest.raises(WebSocketDisconnect) as unconsented:
            with client.websocket_connect(f'/api/conversation/ws?session_id={session.id}', headers={'Authorization': 'Bearer test-secret'}):
                pass
        assert unconsented.value.code == 4403


class LocalEngine:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def process(self, pcm):
        if pcm is None: return ''
        self.started.set()
        await self.release.wait()
        return 'recognized locally'


@pytest.mark.asyncio
async def test_local_transcription_buffer_cap_cancellation_and_exact_turn():
    engine, events, errors = LocalEngine(), [], []
    async def event(value): events.append(value)
    async def error(value): errors.append(value)
    transcriber = LocalTranscription(event, error, engine=engine)
    await transcriber.start()
    try:
        transcriber.begin_turn('old')
        transcriber.append(bytes(4800))
        transcriber.commit('old')
        await asyncio.wait_for(engine.started.wait(), 1)
        transcriber.discard_turn()
        engine.release.set()
        await asyncio.sleep(.01)
        assert events == []
        transcriber.begin_turn('new')
        transcriber.append(bytes(4800))
        transcriber.commit('new')
        async with asyncio.timeout(1):
            while not events: await asyncio.sleep(.001)
        assert events[0].turn_id == 'new' and events[0].text == 'recognized locally'
        transcriber.begin_turn('too-long')
        with pytest.raises(TranscriptionError): transcriber.append(bytes(720002))
        assert len(transcriber.pcm) == 0 and not errors
    finally:
        await transcriber.close()


@pytest.mark.asyncio
async def test_real_brain_identity_portion_goal_and_cart_confirmations_through_transport():
    from app.conversation_brain import ShoppingBrain, ReviewedCatalog, BrainSettings
    from test_conversation_brain import catalog_payload, response, png
    requests = []
    def provider(request):
        requests.append(request)
        return response({'candidate_ids': ['a', 'b']})
    client = httpx.AsyncClient(transport=httpx.MockTransport(provider))
    def make_brain(profile, recipient):
        return ShoppingBrain(profile, recipient, settings=BrainSettings(deepseek_key='fake-test-key'),
                             client=client, catalog=ReviewedCatalog.from_payload(catalog_payload()))
    owner, session, browser, connection, task, _ = await connected(brain_factory=make_brain)
    async def ask(text, epoch):
        await browser.send('input.text', text=text, client_epoch=epoch)
        answer = await browser.until('response.text')
        snapshot = (await browser.until('snapshot'))['snapshot']
        return answer, snapshot
    try:
        await browser.until('snapshot')
        await browser.send('input.text', text='compare these', client_epoch=3)
        camera = await browser.until('camera.request')
        await browser.send('camera.response', request_id=camera['request_id'], image_data_url='data:image/png;base64,' + base64.b64encode(png()).decode(), width=100, height=80)
        await browser.until('response.text')
        snapshot = (await browser.until('snapshot'))['snapshot']
        assert snapshot['pending_action']['kind'] == 'identity'
        _, snapshot = await ask('yes', 4)
        assert snapshot['pair'] == ['a', 'b']
        _, snapshot = await ask('both whole bottles', 5)
        assert snapshot['pending_action']['kind'] == 'goals'
        _, snapshot = await ask('yes', 6)
        assert snapshot['comparison']['candidate_product_id'] == 'b'
        _, snapshot = await ask('add it', 7)
        assert snapshot['pending_action']['kind'] == 'cart' and not snapshot['cart']
        _, snapshot = await ask('yes', 8)
        assert len(snapshot['cart']) == 1
        assert owner.service.get(session.id).session.cart[0].product_name == 'Synthetic Drink B'
        assert len(requests) == 1
        _, snapshot = await ask('this is for my family', 9)
        assert snapshot['recipient_id'] == 'household' and not snapshot['portions']
        assert owner.service.get(session.id).session.recipient_id == 'household'
    finally:
        await cleanup(owner, task)
        await client.aclose()


@pytest.mark.asyncio
async def test_local_cpu_worker_resamples_and_bounds_pending_work():
    pytest.importorskip('numpy', reason='Optional local transcription PCM conversion requires NumPy')
    entered, released = threading.Event(), threading.Event()
    calls = []
    class Model:
        def transcribe(self, audio, **options):
            calls.append((len(audio), options))
            entered.set()
            released.wait(3)
            return iter([SimpleNamespace(text=' local words ')]), None
    engine = LocalWhisperEngine(model_factory=Model)
    await engine.process(None)
    first = asyncio.create_task(engine.process(bytes(48000)))
    try:
        async with asyncio.timeout(2):
            while not entered.is_set(): await asyncio.sleep(.001)
        second = asyncio.create_task(engine.process(bytes(24000)))
        await asyncio.sleep(.01)
        with pytest.raises(TranscriptionError, match='busy'):
            await engine.process(bytes(960))
        first.cancel()
        with pytest.raises(asyncio.CancelledError): await first
        released.set()
        assert await asyncio.wait_for(second, 2) == 'local words'
        assert [size for size, _ in calls] == [16000, 8000]
        assert all(options['condition_on_previous_text'] is False for _, options in calls)
    finally:
        released.set()


@pytest.mark.asyncio
async def test_local_microphone_failure_keeps_text_and_reports_text_only():
    class Failing:
        async def process(self, pcm): raise TranscriptionError('private dependency details')
    service, session = service_session()
    owner = ConversationCoordinator(service, config=ConversationConfig(transcription_provider='local', speech_provider='native'), brain_factory=Brain,
        transcriber_factory=lambda callback, error: LocalTranscription(callback, error, engine=Failing()))
    browser = Browser()
    connection = await owner.attach(browser, session.id)
    task = asyncio.create_task(connection.run())
    try:
        await browser.send('session.start', session_id=session.id, voice_consented=True)
        assert (await browser.until('error'))['code'] == 'transcription_unavailable'
        ready = await browser.until('session.ready')
        assert ready['transcription_provider'] == 'local' and not ready['audio_input_available']
        assert (await browser.until('state'))['state'] == 'text_only'
        await browser.send('input.text', text='still works', client_epoch=1)
        assert (await browser.until('response.text'))['text'] == 'still works'
    finally:
        await cleanup(owner, task)


@pytest.mark.asyncio
async def test_playback_telemetry_requires_current_context_and_epoch():
    owner, session, browser, connection, task, _ = await connected()
    try:
        await browser.send('input.text', text='hello', client_epoch=3)
        answer = await browser.until('response.text')
        await browser.send('playback.started', context_id=answer['context_id'], client_epoch=3)
        assert (await browser.until('state', lambda event: event['state'] == 'speaking'))['client_epoch'] == 3
        await browser.send('playback.finished', context_id='wrong', client_epoch=3)
        await browser.send('ping')
        await browser.until('pong')
        assert owner.snapshot(session.id)['state'] == 'speaking'
        await browser.send('playback.finished', context_id=answer['context_id'], client_epoch=3)
        assert (await browser.until('state'))['state'] == 'text_only'
    finally:
        await cleanup(owner, task)


@pytest.mark.asyncio
async def test_unreviewed_voice_consent_never_starts_transcriber():
    service, session = service_session()
    owner = ConversationCoordinator(service, brain_factory=Brain,
        transcriber_factory=lambda *_: pytest.fail('transcriber started without voice consent'))
    browser = Browser()
    connection = await owner.attach(browser, session.id)
    task = asyncio.create_task(connection.run())
    try:
        await browser.send('session.start', session_id=session.id, voice_consented=False)
        assert (await browser.until('error'))['code'] == 'voice_consent_required'
        assert not connection.started
    finally:
        await cleanup(owner, task)


@pytest.mark.asyncio
async def test_invalid_decodable_image_size_fails_without_hanging_turn():
    owner, session, browser, connection, task, _ = await connected()
    try:
        await browser.send('input.text', text='camera', client_epoch=3)
        request = await browser.until('camera.request')
        buffer = io.BytesIO()
        Image.new('RGB', (100, 80)).save(buffer, 'PNG')
        await browser.send('camera.response', request_id=request['request_id'],
            image_data_url='data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode(), width=101, height=80)
        assert (await browser.until('error'))['code'] == 'invalid_camera_image'
        assert (await browser.until('response.text'))['text'] == 'please try camera again'
    finally:
        await cleanup(owner, task)


@pytest.mark.asyncio
async def test_stale_audio_epoch_never_restarts_vad_after_typed_interrupt():
    ws = Upstream()
    async def connector(_): return ws
    owner, session, browser, connection, task, _ = await connected(
        transcriber_factory=lambda callback, error: OpenAITranscription('fake', callback, error, connector=connector),
        vad_factory=lambda: SpeechDetector(classify=lambda _: True))
    try:
        await browser.send('input.text', text='current', client_epoch=3)
        await browser.until('response.text')
        for _ in range(5):
            await browser.send('audio.append', audio=base64.b64encode(bytes(960)).decode(), client_epoch=2)
        await browser.send('ping')
        await browser.until('pong')
        assert connection.input_turn is None and not connection.detector.active
    finally:
        await cleanup(owner, task)
