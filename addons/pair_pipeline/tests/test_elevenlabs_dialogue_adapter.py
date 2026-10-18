"""Offline v4 Turbo protocol/cancellation tests. No provider requests."""
import asyncio
import base64
import json
from types import SimpleNamespace

import pytest

from addons.pair_pipeline import elevenlabs_dialogue_adapter as dialogue
from addons.pair_pipeline.elevenlabs_voice_adapter import (
    ElevenLabsSpeechAdapter, create_elevenlabs_speech_adapter,
)
from addons.pair_pipeline.voice_adapter import MAX_AUDIO_BYTES, OUTPUT_CHUNK_BYTES, SpeechRequest, VoiceError


def event(**values): return json.dumps(values)
def audio(data, **values): return event(audio=base64.b64encode(data).decode(), **values)


class Socket:
    def __init__(self, messages=(), *, wait_on=None):
        self.messages = list(messages)
        self.sent = []
        self.closed = False
        self.wait_on = wait_on
        self.blocked = asyncio.Event()

    async def send(self, raw):
        self.sent.append(json.loads(raw))
        if self.wait_on == 'send':
            self.blocked.set()
            await asyncio.Event().wait()

    async def recv(self):
        if self.messages:
            return self.messages.pop(0)
        if self.wait_on == 'recv':
            self.blocked.set()
            await asyncio.Event().wait()
        raise RuntimeError('private provider body and test-secret')

    async def close(self): self.closed = True


def setup(messages=(), *, wait_on=None):
    socket = Socket(messages, wait_on=wait_on)
    calls = []
    async def connect(url, key):
        calls.append((url, key))
        return socket
    adapter = dialogue.ElevenLabsDialogueSpeechAdapter('test-secret', connector=connect)
    return adapter, socket, calls


async def collect(adapter, cancel=None):
    return [chunk async for chunk in adapter.stream(
        SpeechRequest('session:turn:epoch2', 'For 30g, this has 12g of carbohydrate.'), cancel=cancel)]


@pytest.mark.asyncio
async def test_exact_single_voice_protocol_pcm_context_and_final_flush():
    pcm = b'\x01\x00' * 5000
    adapter, socket, calls = setup([audio(pcm[:1]), audio(pcm[1:301]),
        audio(pcm[301:]), event(is_final_audio_for_turn=True), event(is_final=True)])
    chunks = await collect(adapter)
    assert calls == [(dialogue.ENDPOINT+'?model_id=eleven_v4_turbo&output_format=pcm_24000', 'test-secret')]
    assert socket.sent == [
        {'voices': [adapter.voice_id]},
        {'inputs': [{'text': 'For thirty grams, this has twelve grams of carbohydrate.',
                     'voice_id': adapter.voice_id, 'new_turn': False}]},
        {'close_socket': True},
    ]
    assert 'test-secret' not in json.dumps(socket.sent) and 'test-secret' not in calls[0][0]
    assert b''.join(c.pcm for c in chunks) == pcm
    assert [c.sequence for c in chunks] == list(range(len(chunks)))
    assert all(c.context_id == 'session:turn:epoch2' and c.sample_rate == 24000
        and c.channels == 1 and c.encoding == 'pcm_s16le' for c in chunks)
    assert all(len(c.pcm) % 2 == 0 and len(c.pcm) <= OUTPUT_CHUNK_BYTES for c in chunks)
    assert socket.closed


@pytest.mark.asyncio
async def test_final_can_carry_last_pcm_and_short_pcm_is_preserved():
    adapter, socket, _ = setup([audio(b'\x01\x00', is_final=True)])
    assert b''.join(c.pcm for c in await collect(adapter)) == b'\x01\x00'
    assert socket.closed


@pytest.mark.asyncio
@pytest.mark.parametrize('messages', [
    [event(is_final=True)],
    [audio(b'\x01'), event(is_final=True)],
    [audio(b'\x00\x00'), event(is_final_audio_for_turn=True)],
    ['not-json'], ['[]'], [b'not-text'],
    [event(audio=1)], [event(audio='%%%%')], [event(audio='')],
    [event(is_final='true')], [event(is_final_audio_for_turn=1)],
    [event(error='test-secret private provider body')],
    [audio(b'\x00\x00', voice_id='different-voice')],
    [audio(b'\x00\x00', sample_rate=16000)],
    [audio(b'\x00\x00', sample_rate=True)],
    [audio(b'\x00\x00', channels=2)],
    [audio(b'\x00\x00', encoding='mp3')],
    [audio(b'RIFF1234WAVE'), event(is_final=True)],
    [audio(b'ID3xxx'), event(is_final=True)],
    [event(unknown='test-secret private provider body')],
])
async def test_invalid_or_incomplete_response_fails_without_leaking_provider_data(messages):
    adapter, socket, _ = setup(messages)
    with pytest.raises(VoiceError) as error:
        await collect(adapter)
    assert 'test-secret' not in str(error.value) and 'private provider body' not in str(error.value)
    assert socket.closed


@pytest.mark.asyncio
async def test_event_count_and_size_and_audio_duration_are_bounded(monkeypatch):
    adapter, socket, _ = setup(['x' * (dialogue.MAX_EVENT_BYTES + 1)])
    with pytest.raises(VoiceError): await collect(adapter)
    assert socket.closed
    adapter, socket, _ = setup([audio(b'\x00\x00' * (MAX_AUDIO_BYTES // 2 + 1))])
    with pytest.raises(VoiceError, match='30-second'): await collect(adapter)
    assert socket.closed
    monkeypatch.setattr(dialogue, 'MAX_EVENTS', 3)
    adapter, socket, _ = setup([event(is_final_audio_for_turn=True)] * 4)
    with pytest.raises(VoiceError, match='complete audio'): await collect(adapter)
    assert socket.closed and len(socket.messages) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('method', ['event', 'task'])
@pytest.mark.parametrize('stage', ['connect', 'send', 'recv'])
async def test_cancellation_closes_socket_and_blocked_operation(method, stage):
    adapter, socket, _ = setup(wait_on=stage)
    cancelled = asyncio.Event()
    connect_started = asyncio.Event()
    connect_stopped = asyncio.Event()
    if stage == 'connect':
        async def connect(url, key):
            connect_started.set()
            try: await asyncio.Event().wait()
            finally: connect_stopped.set()
        adapter._connector = connect
    task = asyncio.create_task(collect(adapter, cancelled))
    await asyncio.wait_for((connect_started if stage == 'connect' else socket.blocked).wait(), 1)
    cancelled.set() if method == 'event' else task.cancel()
    with pytest.raises(asyncio.CancelledError): await asyncio.wait_for(task, 1)
    assert connect_stopped.is_set() if stage == 'connect' else socket.closed


@pytest.mark.asyncio
async def test_connection_cancel_race_closes_new_socket():
    cancel = asyncio.Event()
    socket = Socket()
    async def connect(url, key):
        cancel.set()
        return socket
    adapter = dialogue.ElevenLabsDialogueSpeechAdapter('test-secret', connector=connect)
    with pytest.raises(asyncio.CancelledError): await collect(adapter, cancel)
    assert socket.closed and socket.sent == []


@pytest.mark.asyncio
async def test_precancel_and_consumer_close_do_not_leave_provider_work():
    adapter, socket, calls = setup([audio(b'\x00\x00' * 10000)], wait_on='recv')
    cancel = asyncio.Event(); cancel.set()
    with pytest.raises(asyncio.CancelledError): await collect(adapter, cancel)
    assert not calls
    stream = adapter.stream(SpeechRequest('short', 'Approved reply.'))
    await anext(stream)
    await stream.aclose()
    assert socket.closed


@pytest.mark.asyncio
async def test_timeout_and_http_rejection_are_redacted():
    for failure in [TimeoutError('test-secret'),
                    SimpleProviderError(401), SimpleProviderError(302)]:
        async def connect(url, key): raise failure
        adapter = dialogue.ElevenLabsDialogueSpeechAdapter('test-secret', connector=connect)
        with pytest.raises(VoiceError) as error: await collect(adapter)
        assert 'test-secret' not in str(error.value)
        if isinstance(failure, SimpleProviderError): assert str(failure.response.status_code) in str(error.value)


class SimpleProviderError(Exception):
    def __init__(self, status):
        self.response = SimpleNamespace(status_code=status)
        super().__init__('test-secret private provider body')


def test_redirect_is_never_followed_and_explicit_factory_preserves_default():
    sentinel = SimpleProviderError(302)
    assert dialogue._NoRedirectConnect.process_redirect(None, sentinel) is sentinel
    assert isinstance(create_elevenlabs_speech_adapter('key'), ElevenLabsSpeechAdapter)
    assert isinstance(create_elevenlabs_speech_adapter('key', model='eleven_v4_turbo'), dialogue.ElevenLabsDialogueSpeechAdapter)
    with pytest.raises(VoiceError): create_elevenlabs_speech_adapter('key', model='unknown')


@pytest.mark.parametrize('key,voice,model', [
    ('', 'voice', dialogue.MODEL), ('secret\nvalue', 'voice', dialogue.MODEL),
    ('key', '../redirect', dialogue.MODEL), ('key', 'x'*101, dialogue.MODEL),
    ('key', 'voice', 'eleven_flash_v2_5'), ('key', 'voice', 'eleven_v4'),
])
def test_configuration_fails_closed_without_secret_values(key, voice, model):
    with pytest.raises(VoiceError) as error:
        dialogue.ElevenLabsDialogueSpeechAdapter(key, voice_id=voice, model=model)
    assert 'secret' not in str(error.value)


async def test_prepare_sends_only_voice_registration_and_stream_consumes_it_once():
    adapter, socket, calls = setup([audio(b'\x01\x00', is_final=True)])
    assert await adapter.prepare()
    assert await adapter.prepare()  # Idempotent while unused.
    assert socket.sent == [{'voices': [adapter.voice_id]}]
    assert len(calls) == 1 and not socket.closed
    chunks = await collect(adapter)
    assert len(calls) == 1 and socket.closed and chunks
    assert adapter._prepared_socket is None and adapter._expiry_task is None
    await adapter.aclose()


async def test_prepare_failure_falls_back_once_to_normal_startup_without_text_or_secret_leak():
    socket = Socket([audio(b'\x01\x00', is_final=True)])
    calls = []
    async def connect(url, key):
        calls.append(url)
        if len(calls) == 1:
            raise RuntimeError('test-secret private failure')
        return socket
    adapter = dialogue.ElevenLabsDialogueSpeechAdapter('test-secret', connector=connect)
    assert await adapter.prepare() is False
    assert adapter._prepared_socket is None
    assert await collect(adapter)
    assert len(calls) == 2 and socket.closed
    assert len([m for m in socket.sent if 'inputs' in m]) == 1


@pytest.mark.parametrize('mode', ['close', 'expiry', 'expired_before_timer'])
async def test_unused_prepared_socket_is_closed_without_synthesis(mode, monkeypatch):
    adapter, socket, _ = setup()
    if mode == 'expiry':
        monkeypatch.setattr(dialogue, 'PREPARED_IDLE_SECONDS', .01)
    assert await adapter.prepare()
    expiry = adapter._expiry_task
    if mode == 'expiry':
        await asyncio.wait_for(expiry, 1)
    elif mode == 'expired_before_timer':
        adapter._prepared_at -= 20
        assert await adapter._take_prepared() is None
    else:
        await adapter.aclose()
    assert socket.closed and socket.sent == [{'voices': [adapter.voice_id]}]
    assert adapter._prepared_socket is None and adapter._expiry_task is None and expiry.done()
    await adapter.aclose()


@pytest.mark.parametrize('method', ['event', 'task'])
@pytest.mark.parametrize('stage', ['connect', 'send'])
async def test_prepare_cancellation_closes_late_or_partially_registered_socket(method, stage):
    adapter, socket, _ = setup(wait_on=stage)
    cancel = asyncio.Event()
    started, stopped = asyncio.Event(), asyncio.Event()
    if stage == 'connect':
        async def connect(url, key):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        adapter._connector = connect
    task = asyncio.create_task(adapter.prepare(cancel=cancel))
    await asyncio.wait_for((started if stage == 'connect' else socket.blocked).wait(), 1)
    cancel.set() if method == 'event' else task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    await adapter.aclose()
    assert stopped.is_set() if stage == 'connect' else socket.closed
    assert adapter._prepared_socket is None and adapter._expiry_task is None
    assert not any('inputs' in m for m in socket.sent)


async def test_prepare_connection_cancel_race_and_registration_failure_close_sockets():
    cancel = asyncio.Event()
    socket = Socket()
    async def connect(url, key):
        cancel.set()
        return socket
    adapter = dialogue.ElevenLabsDialogueSpeechAdapter('test-secret', connector=connect)
    with pytest.raises(asyncio.CancelledError):
        await adapter.prepare(cancel=cancel)
    assert socket.closed and socket.sent == []
    second = Socket()
    async def reject(raw):
        raise RuntimeError('private provider failure')
    second.send = reject
    async def second_connect(url, key): return second
    adapter = dialogue.ElevenLabsDialogueSpeechAdapter('test-secret', connector=second_connect)
    assert await adapter.prepare() is False
    assert second.closed and adapter._prepared_socket is None


async def test_prepared_socket_is_not_reused_for_later_speech_parts():
    sockets = [Socket([audio(b'\x01\x00', is_final=True)]) for _ in range(2)]
    calls = []
    async def connect(url, key):
        socket = sockets[len(calls)]
        calls.append(url)
        return socket
    adapter = dialogue.ElevenLabsDialogueSpeechAdapter('test-secret', connector=connect)
    assert await adapter.prepare()
    assert await collect(adapter)
    assert await collect(adapter)
    assert len(calls) == 2 and all(socket.closed for socket in sockets)
    assert all(len([m for m in socket.sent if 'inputs' in m]) == 1 for socket in sockets)
    await adapter.aclose()


@pytest.mark.parametrize('status,category', [(401, 'authorization'), (403, 'authorization'),
    (429, 'rate_limited'), (402, 'balance'), (503, 'http_error')])
async def test_http_failure_has_only_safe_diagnostic_metadata(status, category):
    async def connect(url, key): raise SimpleProviderError(status)
    adapter = dialogue.ElevenLabsDialogueSpeechAdapter('test-secret', connector=connect)
    with pytest.raises(VoiceError) as caught:
        await collect(adapter)
    assert caught.value.diagnostic('elevenlabs') == {'provider': 'elevenlabs', 'stage': 'connect',
        'category': category, 'http_status': status}
    assert 'test-secret' not in json.dumps(caught.value.diagnostic('elevenlabs'))


@pytest.mark.parametrize('code,category', [('quota_exceeded', 'rate_limited'),
    ('invalid_api_key', 'authorization'), ('voice_not_found', 'configuration'),
    ('private provider body test-secret', 'invalid_response')])
async def test_provider_error_code_is_allowlisted_without_saving_provider_text(code, category):
    adapter, socket, _ = setup([event(error_code=code, error='private provider body test-secret')])
    with pytest.raises(VoiceError) as caught:
        await collect(adapter)
    diagnostic = caught.value.diagnostic('elevenlabs')
    assert diagnostic['category'] == category and diagnostic['stage'] == 'synthesis'
    assert diagnostic['http_status'] is None and socket.closed
    assert 'test-secret' not in str(caught.value) + json.dumps(diagnostic)


@pytest.mark.parametrize('has_audio', [False, True])
async def test_clean_websocket_close_before_audio_is_empty_and_mid_audio_is_network_failure(has_audio):
    from websockets.exceptions import ConnectionClosedOK
    from websockets.frames import Close
    adapter, socket, _ = setup([audio(b'\x01\x00' * 10)] if has_audio else [])
    original = socket.recv
    async def recv():
        if socket.messages:
            return await original()
        raise ConnectionClosedOK(Close(1000, 'private test-secret'), Close(1000, ''), True)
    socket.recv = recv
    with pytest.raises(VoiceError) as caught:
        await collect(adapter)
    assert caught.value.category == ('network' if has_audio else 'empty_response')
    assert socket.closed and 'test-secret' not in str(caught.value)


def test_voice_diagnostic_revalidates_mutated_attributes_and_excludes_message():
    error = VoiceError('private provider body test-secret', category='authorization', http_status=401)
    error.category, error.stage, error.http_status = 'private body', 'test-secret', True
    assert error.diagnostic('secret-provider') == {'provider': 'unknown', 'stage': 'synthesis',
        'category': 'unknown', 'http_status': None}
