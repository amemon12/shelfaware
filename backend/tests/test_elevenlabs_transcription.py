import asyncio
import base64
import json
from urllib.parse import parse_qs, urlparse

import pytest

from app.elevenlabs_transcription import ElevenLabsTranscription, connection_url
from app.openai_transcription import TranscriptionError
from app.conversation_transport import ConversationConfig


class Socket:
    def __init__(self, start=None):
        self.incoming, self.sent = asyncio.Queue(), asyncio.Queue()
        self.incoming.put_nowait(start or {'message_type': 'session_started', 'config': {'audio_format': 'pcm_24000'}})
        self.closed = False

    async def recv(self):
        event = await self.incoming.get()
        if event is None:
            raise RuntimeError('connection closed')
        return event if isinstance(event, str) else json.dumps(event)

    async def send(self, raw):
        await self.sent.put(json.loads(raw))

    def __aiter__(self): return self

    async def __anext__(self):
        return await self.recv()

    async def close(self):
        self.closed = True
        self.incoming.put_nowait(None)

    async def event(self, kind, **fields):
        await self.incoming.put({'message_type': kind, **fields})

    async def until_commit(self):
        messages = []
        async with asyncio.timeout(2):
            while True:
                event = await self.sent.get()
                messages.append(event)
                if event.get('commit'):
                    return messages


async def make(*, start=None, **options):
    sockets, events, errors, connections = [], [], [], []
    async def connect(url, key):
        connections.append((url, key))
        socket = Socket(start)
        sockets.append(socket)
        return socket
    async def event(value): events.append(value)
    async def error(value): errors.append(value)
    adapter = ElevenLabsTranscription('test-private-key', event, error, connector=connect, **options)
    await adapter.start()
    return adapter, sockets, events, errors, connections


async def until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(.001)


@pytest.mark.asyncio
async def test_manual_commit_24khz_and_short_yes_padding_exact_turn_mapping():
    adapter, sockets, events, errors, connections = await make()
    try:
        query = parse_qs(urlparse(connections[0][0]).query)
        assert query['model_id'] == ['scribe_v2_realtime']
        assert query['audio_format'] == ['pcm_24000'] and query['commit_strategy'] == ['manual']
        assert 'test-private-key' not in connections[0][0]
        adapter.begin_turn('yes-turn')
        audio = b'\x01\x02' * 2400
        adapter.append(audio)
        adapter.commit('yes-turn')
        messages = await sockets[0].until_commit()
        assert all(event['sample_rate'] == 24000 and event['message_type'] == 'input_audio_chunk' for event in messages)
        assert messages[-1]['audio_base_64'] == ''
        uploaded = b''.join(base64.b64decode(event['audio_base_64']) for event in messages)
        assert len(uploaded) == 96000 and uploaded[:len(audio)] == audio
        assert uploaded[len(audio):] == bytes(96000 - len(audio))
        await sockets[0].event('partial_transcript', text='ye')
        await sockets[0].event('committed_transcript', text='yes')
        await until(lambda: len(events) == 2)
        assert [(event.kind, event.turn_id, event.text) for event in events] == [('partial', 'yes-turn', 'ye'), ('final', 'yes-turn', 'yes')]
        await until(lambda: sockets[0].closed)
        assert not errors and adapter.ready
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_cancel_and_new_socket_drop_late_old_transcript():
    adapter, sockets, events, errors, _ = await make()
    try:
        adapter.begin_turn('old')
        adapter.append(bytes(4800))
        adapter.commit('old')
        await sockets[0].until_commit()
        adapter.discard_turn()
        adapter.begin_turn('new')
        adapter.append(bytes(4800))
        adapter.commit('new')
        await until(lambda: len(sockets) == 2)
        await sockets[1].until_commit()
        await sockets[0].event('committed_transcript', text='old answer')
        await sockets[1].event('committed_transcript', text='new answer')
        await until(lambda: len(events) == 1)
        assert events[0].turn_id == 'new' and events[0].text == 'new answer'
        assert sockets[0].closed and not errors
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_completed_then_new_turn_has_new_provider_connection():
    adapter, sockets, events, errors, _ = await make()
    try:
        for index in range(2):
            adapter.begin_turn(str(index))
            adapter.append(bytes(4800))
            adapter.commit(str(index))
            await until(lambda: len(sockets) > index)
            await sockets[index].until_commit()
            await sockets[index].event('committed_transcript', text=f'answer {index}')
            await until(lambda: len(events) > index)
        assert [event.turn_id for event in events] == ['0', '1']
        assert len(sockets) == 2 and not errors
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_idle_warm_socket_expires_then_next_turn_reconnects():
    adapter, sockets, events, errors, _ = await make(warm_idle_s=.01)
    try:
        await until(lambda: sockets[0].closed)
        assert adapter.ready  # next utterance creates its own socket
        adapter.begin_turn('after-idle')
        adapter.append(bytes(4800))
        adapter.commit('after-idle')
        await until(lambda: len(sockets) == 2)
        await sockets[1].until_commit()
        await sockets[1].event('committed_transcript', text='hello again')
        await until(lambda: events)
        assert events[0].turn_id == 'after-idle' and not errors
    finally:
        await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('kind,fields', [('quota_exceeded', {'error': 'private-key-and-text'}),
    ('committed_transcript', {'text': 'early'}), ('partial_transcript', {'text': 'x' * 4001})])
async def test_provider_error_malformed_or_premature_final_never_becomes_command(kind, fields):
    adapter, sockets, events, errors, _ = await make()
    try:
        adapter.begin_turn('turn')
        adapter.append(bytes(4800))
        await sockets[0].event(kind, **fields)
        await until(lambda: errors)
        assert not events and not adapter.ready
        assert 'private-key' not in errors[0]
        await until(lambda: sockets[0].closed)
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_auth_failure_is_sanitized_and_never_ready():
    sockets = []
    async def connect(url, key):
        socket = Socket({'message_type': 'auth_error', 'error': key})
        sockets.append(socket)
        return socket
    async def ignore(_): pass
    adapter = ElevenLabsTranscription('private-secret', ignore, ignore, connector=connect)
    with pytest.raises(TranscriptionError) as error:
        await adapter.start()
    assert 'private-secret' not in str(error.value) and not adapter.ready and sockets[0].closed


@pytest.mark.asyncio
async def test_audio_bounds_cancel_without_uploading_unbounded_data():
    adapter, sockets, events, errors, _ = await make()
    try:
        adapter.begin_turn('oversize')
        with pytest.raises(TranscriptionError): adapter.append(bytes(720002))
        assert adapter.turn is None
        adapter.begin_turn('bounded-queue')
        with pytest.raises(TranscriptionError): adapter.append(bytes(720000))
        assert adapter.turn is None and not events
    finally:
        await adapter.close()
    assert all(socket.closed for socket in sockets)


def test_default_provider_selection_is_elevenlabs_and_secrets_are_not_repr():
    config = ConversationConfig(elevenlabs_api_key='private-key', openai_api_key='private-other-key')
    assert config.transcription_provider == config.speech_provider == 'elevenlabs'
    assert config.elevenlabs_stt_model == 'scribe_v2_realtime'
    assert 'private-key' not in repr(config) and 'private-other-key' not in repr(config)
    with pytest.raises(ValueError): connection_url('unknown-model')


@pytest.mark.asyncio
async def test_elevenlabs_through_real_transport_vad_partial_final_and_reply():
    import struct
    from app.speech_detection import SpeechDetector
    from test_conversation_transport import connected, cleanup
    sockets = []
    async def connector(url, key):
        socket = Socket()
        sockets.append(socket)
        return socket
    owner, session, browser, connection, task, ready = await connected(
        config=ConversationConfig(elevenlabs_api_key='fake-key', speech_provider='native'),
        transcriber_factory=lambda callback, error: ElevenLabsTranscription('fake-key', callback, error, connector=connector),
        vad_factory=lambda: SpeechDetector(classify=lambda pcm: pcm != bytes(640)))
    try:
        assert ready['transcription_provider'] == 'elevenlabs' and ready['audio_input_available']
        voiced = base64.b64encode(struct.pack('<480h', *([1000] * 480))).decode()
        silent = base64.b64encode(bytes(960)).decode()
        for _ in range(5): await browser.send('audio.append', audio=voiced)
        start = await browser.until('speech.started')
        for _ in range(33): await browser.send('audio.append', audio=silent)
        await sockets[0].until_commit()
        await sockets[0].event('partial_transcript', text='hel')
        await sockets[0].event('committed_transcript', text='hello')
        assert (await browser.until('transcript.partial'))['turn_id'] == start['turn_id']
        assert (await browser.until('response.text'))['text'] == 'hello'
    finally:
        await cleanup(owner, task)


@pytest.mark.asyncio
async def test_elevenlabs_stt_to_actual_tts_adapter_emits_compatible_pcm():
    import httpx
    import struct
    from app.speech_detection import SpeechDetector
    from addons.pair_pipeline.elevenlabs_voice_adapter import ElevenLabsSpeechAdapter
    from test_conversation_transport import connected, cleanup
    sockets, requests = [], []
    async def connector(url, key):
        socket = Socket()
        sockets.append(socket)
        return socket
    def tts(request):
        requests.append(request)
        return httpx.Response(200, content=b'\x00\x00' * 4800, headers={'content-type': 'audio/pcm'})
    client = httpx.AsyncClient(transport=httpx.MockTransport(tts))
    owner, session, browser, connection, task, ready = await connected(
        config=ConversationConfig(elevenlabs_api_key='fake-key'),
        transcriber_factory=lambda callback, error: ElevenLabsTranscription('fake-key', callback, error, connector=connector),
        speech_factory=lambda: ElevenLabsSpeechAdapter('fake-key', client=client),
        vad_factory=lambda: SpeechDetector(classify=lambda pcm: pcm != bytes(640)))
    try:
        assert ready['speech_provider'] == 'elevenlabs' and ready['audio_output_available']
        voice = base64.b64encode(struct.pack('<480h', *([1000] * 480))).decode()
        for _ in range(5): await browser.send('audio.append', audio=voice)
        await browser.until('speech.started')
        for _ in range(33): await browser.send('audio.append', audio=base64.b64encode(bytes(960)).decode())
        await sockets[0].until_commit()
        await sockets[0].event('committed_transcript', text='8.91g')
        reply = await browser.until('response.text')
        chunk = await browser.until('audio.chunk')
        assert chunk['context_id'] == reply['context_id'] and chunk['sample_rate'] == 24000
        audio = bytearray(base64.b64decode(chunk['audio']))
        async with asyncio.timeout(2):
            while True:
                event = await browser.outgoing.get()
                if event['type'] == 'audio.done':
                    break
                if event['type'] == 'audio.chunk':
                    assert event['context_id'] == reply['context_id']
                    audio.extend(base64.b64decode(event['audio']))
        assert audio == bytes(9600)
        payload = json.loads(requests[0].content)
        assert payload['model_id'] == 'eleven_flash_v2_5'
        assert payload['text'] == 'eight point nine one grams'
        assert requests[0].headers['xi-api-key'] == 'fake-key'
        assert requests[0].url.params['output_format'] == 'pcm_24000'
    finally:
        await cleanup(owner, task)
        await client.aclose()
