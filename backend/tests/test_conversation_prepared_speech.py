"""Prepared speech overlaps reasoning, never validation or turn identity."""
import asyncio

import pytest

from app.conversation_transport import ConversationConfig
from addons.pair_pipeline.elevenlabs_dialogue_adapter import ElevenLabsDialogueSpeechAdapter
from addons.pair_pipeline.voice_adapter import VoiceError
from addons.pair_pipeline.tests.test_elevenlabs_dialogue_adapter import Socket, audio
from test_conversation_transport import Brain, connected, cleanup


class HeldBrain(Brain):
    def __init__(self, profile, recipient_id):
        super().__init__(profile, recipient_id)
        self.started, self.release = asyncio.Event(), asyncio.Event()
        self.result = 'approved answer'

    async def respond(self, text, frame_provider=None):
        self.started.set()
        await self.release.wait()
        if self.result == 'error':
            raise ValueError('invalid answer')
        return {'reply': self.result, 'silent': self.result == 'silent', 'snapshot': self.snapshot()}

    async def respond_automatically(self, frame_provider=None):
        return await self.respond('automatic', frame_provider)


class ObservedSocket(Socket):
    def __init__(self):
        super().__init__([audio(b'\x01\x00' * 100, is_final=True)])
        self.registered, self.closed_event = asyncio.Event(), asyncio.Event()

    async def send(self, raw):
        await super().send(raw)
        if 'voices' in self.sent[-1]:
            self.registered.set()

    async def close(self):
        await super().close()
        self.closed_event.set()


async def setup():
    sockets, speakers = [], []
    def factory():
        socket = ObservedSocket()
        sockets.append(socket)
        async def connect(url, key): return socket
        speaker = ElevenLabsDialogueSpeechAdapter('test-key', connector=connect)
        speakers.append(speaker)
        return speaker
    values = await connected(brain_factory=HeldBrain, speech_factory=factory,
        config=ConversationConfig(transcription_provider='disabled', product_mode='product_search'))
    return (*values, sockets, speakers)


async def test_preparation_overlaps_reasoning_but_audio_waits_for_validated_current_result():
    owner, _, browser, connection, task, _, sockets, speakers = await setup()
    try:
        await browser.send('input.text', text='question', client_epoch=3)
        brain = connection.runtime.brain
        await asyncio.wait_for(brain.started.wait(), 1)
        await asyncio.wait_for(sockets[0].registered.wait(), 1)
        assert sockets[0].sent == [{'voices': [speakers[0].voice_id]}]
        assert not connection.input_enabled and connection.context_id is None
        brain.release.set()
        result = await browser.until('response.text')
        chunk = await browser.until('audio.chunk')
        await browser.until('audio.done')
        assert result['text'] == 'approved answer'
        assert chunk['client_epoch'] == 3 and chunk['context_id'] == result['context_id']
        assert len(sockets) == 1 and sockets[0].closed
        assert len([m for m in sockets[0].sent if 'inputs' in m]) == 1
        assert connection.playback_pending and not connection.input_enabled
        await browser.send('playback.finished', context_id=result['context_id'], client_epoch=3)
        await browser.send('ping')
        await browser.until('pong')
        assert not connection.playback_pending
    finally:
        await cleanup(owner, task)
    assert speakers[0]._expiry_task is None


@pytest.mark.parametrize('ending', ['interrupt', 'disconnect', 'error', 'silent'])
async def test_abandoned_or_silent_reply_closes_preparation_without_sending_any_text(ending):
    owner, _, browser, connection, task, _, sockets, speakers = await setup()
    try:
        brain = connection.runtime.brain
        if ending == 'silent':
            connection._respond('automatic', 'auto-turn', source='automatic_scene')
        else:
            await browser.send('input.text', text='question', client_epoch=3)
        await asyncio.wait_for(brain.started.wait(), 1)
        await asyncio.wait_for(sockets[0].registered.wait(), 1)
        if ending == 'interrupt':
            await browser.send('interrupt', client_epoch=4)
        elif ending == 'disconnect':
            await browser.close()
        else:
            brain.result = ending
            brain.release.set()
            if ending == 'error':
                assert (await browser.until('error'))['code'] == 'turn_failed'
        await asyncio.wait_for(sockets[0].closed_event.wait(), 1)
        assert sockets[0].sent == [{'voices': [speakers[0].voice_id]}]
        assert connection.context_id is None
        assert not connection.playback_pending
    finally:
        await cleanup(owner, task)
    assert speakers[0]._expiry_task is None


async def test_interrupt_during_prepare_connection_cancels_pending_handshake():
    started, stopped = asyncio.Event(), asyncio.Event()
    async def connect(url, key):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
    speaker = ElevenLabsDialogueSpeechAdapter('test-key', connector=connect)
    owner, _, browser, connection, task, _ = await connected(brain_factory=HeldBrain,
        speech_factory=lambda: speaker)
    try:
        await browser.send('input.text', text='question', client_epoch=3)
        await asyncio.wait_for(started.wait(), 1)
        await browser.send('interrupt', client_epoch=4)
        await asyncio.wait_for(stopped.wait(), 1)
        assert connection.context_id is None and speaker._prepared_socket is None
    finally:
        await cleanup(owner, task)


class FailedSpeech:
    def __init__(self, category, status=None):
        self.category, self.status = category, status

    async def stream(self, request, cancel):
        raise VoiceError('private provider body or test-key', category=self.category,
                         stage='synthesis', http_status=self.status)
        yield  # Async iterator, intentionally never yields audio.


@pytest.mark.parametrize('category,status,hint', [('authorization', 401, 'access was rejected'),
    ('rate_limited', 429, 'service limit'), ('network', None, 'connection failed'),
    ('empty_response', None, 'returned no audio')])
async def test_safe_speech_failure_is_actionable_persisted_and_cleared_by_new_turn(category, status, hint):
    speaker = FailedSpeech(category, status)
    owner, session, browser, connection, task, _ = await connected(speech_factory=lambda: speaker)
    try:
        await browser.send('input.text', text='hello', client_epoch=3)
        await browser.until('response.text')
        error = await browser.until('error')
        assert error['code'] == 'speech_unavailable' and hint in error['message']
        diagnostic = owner.snapshot(session.id)['speech_failure']
        assert diagnostic == {'provider': 'elevenlabs', 'stage': 'synthesis',
            'category': category, 'http_status': status}
        assert 'private' not in str(error) + str(diagnostic) and 'test-key' not in str(error) + str(diagnostic)
        held = HeldBrain({}, 'self')
        connection.runtime.brain = held
        await browser.send('input.text', text='next', client_epoch=4)
        await asyncio.wait_for(held.started.wait(), 1)
        assert owner.snapshot(session.id)['speech_failure'] is None
    finally:
        await cleanup(owner, task)


async def test_speech_failure_survives_disconnect_but_context_update_clears_it():
    owner, session, browser, connection, task, _ = await connected(
        speech_factory=lambda: FailedSpeech('empty_response'))
    try:
        await browser.send('input.text', text='hello', client_epoch=3)
        await browser.until('error')
        await browser.close()
        await asyncio.wait_for(task, 1)
        assert owner.snapshot(session.id)['speech_failure']['category'] == 'empty_response'
        await owner.interrupt_session(session.id, 'context', recipient_id='household')
        assert owner.snapshot(session.id)['speech_failure'] is None
    finally:
        await owner.close()


async def test_cancelled_synthesis_cannot_publish_a_late_failure_diagnostic():
    entered, stopped = asyncio.Event(), asyncio.Event()
    class LateFailure:
        async def stream(self, request, cancel):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise VoiceError('private late failure', category='authorization', http_status=401)
            finally:
                stopped.set()
            yield
    owner, session, browser, connection, task, _ = await connected(speech_factory=LateFailure)
    try:
        await browser.send('input.text', text='hello', client_epoch=3)
        await asyncio.wait_for(entered.wait(), 1)
        await browser.send('interrupt', client_epoch=4)
        await asyncio.wait_for(stopped.wait(), 1)
        await browser.send('ping')
        await browser.until('pong')
        assert owner.snapshot(session.id)['speech_failure'] is None
    finally:
        await cleanup(owner, task)


async def test_configuration_failure_preserves_valid_text_and_gives_safe_hint():
    def factory():
        raise ValueError('private key or configuration value')
    owner, session, browser, connection, task, _ = await connected(speech_factory=factory)
    try:
        await browser.send('input.text', text='hello', client_epoch=3)
        assert (await browser.until('response.text'))['text'] == 'hello'
        error = await browser.until('error')
        assert 'not configured correctly' in error['message'] and 'private' not in error['message']
        assert owner.snapshot(session.id)['speech_failure'] == {'provider': 'elevenlabs',
            'stage': 'configuration', 'category': 'configuration', 'http_status': None}
    finally:
        await cleanup(owner, task)
