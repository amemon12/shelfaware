import asyncio
import base64
import struct

import pytest

from app.openai_transcription import Transcript
from app.speech_detection import SpeechDetector
from test_conversation_scene import scene_connection, send_photo
from test_conversation_transport import Brain, Speech, cleanup


class CountingTranscriber:
    ready = True

    def __init__(self):
        self.appended = 0
        self.begun = []
        self.committed = []

    def begin_turn(self, turn): self.begun.append(turn)
    def append(self, pcm): self.appended += 1
    def commit(self, turn): self.committed.append(turn)
    def discard_turn(self): pass
    async def close(self): pass


class CountingVAD:
    def __init__(self):
        self.detector = SpeechDetector(classify=lambda pcm: pcm != bytes(640))
        self.calls = 0

    @property
    def active(self): return self.detector.active

    def feed(self, pcm):
        self.calls += 1
        return self.detector.feed(pcm)

    def reset(self): self.detector.reset()


class WaitingBrain(Brain):
    def __init__(self, *args):
        super().__init__(*args)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def respond(self, text, frame_provider=None):
        self.entered.set()
        await self.release.wait()
        return await super().respond(text, frame_provider)


VOICE = base64.b64encode(struct.pack('<480h', *([1000] * 480))).decode()
SILENCE = base64.b64encode(bytes(960)).decode()


async def noise(browser, epoch=2):
    for _ in range(20):
        await browser.send('audio.append', audio=VOICE, client_epoch=epoch)
    await browser.send('ping')
    await browser.until('pong')


@pytest.mark.asyncio
async def test_commit_mutes_through_transcription_thinking_camera_tts_and_actual_playback():
    owner, _, browser, connection, task, ready = await scene_connection(
        brain_factory=WaitingBrain, speech_factory=lambda: Speech(count=5))
    connection.transcriber = transcriber = CountingTranscriber()
    connection.detector = detector = CountingVAD()
    try:
        assert ready['listening_enabled'] is True
        await browser.until('input.enabled', lambda event: event['enabled'])
        for _ in range(5): await browser.send('audio.append', audio=VOICE, client_epoch=2)
        turn = (await browser.until('speech.started'))['turn_id']
        for _ in range(33): await browser.send('audio.append', audio=SILENCE, client_epoch=2)
        await browser.until('input.enabled', lambda event: not event['enabled'])
        await browser.until('state', lambda event: event['state'] == 'transcribing')
        before = (detector.calls, transcriber.appended, len(connection.runtime.brain.cancelled), connection.runtime.generation)
        assert transcriber.committed == [turn]
        await noise(browser)
        assert before == (detector.calls, transcriber.appended, len(connection.runtime.brain.cancelled), connection.runtime.generation)
        await connection.on_transcript(Transcript('final', turn, 'camera', 'item'))
        await connection.runtime.brain.entered.wait()
        await noise(browser)
        assert not connection.input_enabled and detector.calls == before[0]
        connection.runtime.brain.release.set()
        request = await browser.until('camera.request')
        await noise(browser)
        assert not connection.input_enabled and detector.calls == before[0]
        await send_photo(browser, request)
        response = await browser.until('response.text')
        await browser.until('audio.chunk')
        await noise(browser)
        await browser.until('audio.done')
        await browser.until('state', lambda event: event['state'] == 'listening')
        assert not connection.input_enabled and connection.playback_pending
        await noise(browser)
        assert before == (detector.calls, transcriber.appended, len(connection.runtime.brain.cancelled), connection.runtime.generation)
        assert len(transcriber.begun) == 1
        assert not any(event['type'] == 'response.cancelled' for event in browser.events)
        await browser.send('playback.finished', client_epoch=1, context_id=response['context_id'])
        await browser.send('playback.finished', client_epoch=2, context_id='wrong')
        await noise(browser)
        assert not connection.input_enabled
        await browser.send('playback.finished', client_epoch=2, context_id=response['context_id'])
        await browser.until('input.enabled', lambda event: event['enabled'])
        assert connection.input_enabled
        for _ in range(5): await browser.send('audio.append', audio=VOICE, client_epoch=2)
        assert (await browser.until('speech.started'))['turn_id'] != turn
        assert len(transcriber.begun) == 2
    finally:
        await cleanup(owner, task)


@pytest.mark.asyncio
@pytest.mark.parametrize('automatic', [False, True])
async def test_typed_or_automatic_capture_mutes_before_work_and_explicit_interrupt_resumes(automatic):
    owner, _, browser, connection, task, _ = await scene_connection()
    connection.transcriber = transcriber = CountingTranscriber()
    connection.detector = detector = CountingVAD()
    try:
        await browser.until('input.enabled', lambda event: event['enabled'])
        if automatic:
            await browser.send('scene.stable', client_epoch=2)
        else:
            await browser.send('input.text', text='camera', client_epoch=2)
        await browser.until('input.enabled', lambda event: not event['enabled'])
        await browser.until('camera.request')
        generation = connection.runtime.generation
        await noise(browser)
        assert not connection.input_enabled and detector.calls == 0 and transcriber.appended == 0
        assert connection.runtime.generation == generation and connection.camera_pending
        await browser.send('interrupt', client_epoch=3)
        enabled = await browser.until('input.enabled', lambda event: event['enabled'])
        assert enabled['client_epoch'] == 3 and connection.input_enabled
        assert not connection.camera_pending
    finally:
        await cleanup(owner, task)


class BrokenSpeech:
    async def stream(self, request, cancel):
        raise RuntimeError('No audio produced')
        yield  # Make this an async generator, matching the provider interface.


class BrokenBrain(Brain):
    async def respond(self, text, frame_provider=None):
        raise RuntimeError('Response failed')


class EndingBrain(Brain):
    async def respond(self, text, frame_provider=None):
        return {'reply': 'Goodbye.', 'snapshot': self.snapshot(), 'end_conversation': True}


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['no_output', 'zero_audio', 'brain_error'])
async def test_text_only_or_failure_releases_microphone_after_current_turn(failure):
    owner, _, browser, connection, task, _ = await scene_connection(
        brain_factory=BrokenBrain if failure == 'brain_error' else Brain,
        speech_factory=(lambda: BrokenSpeech()) if failure == 'zero_audio' else None)
    try:
        await browser.until('input.enabled', lambda event: event['enabled'])
        await browser.send('input.text', text='hello', client_epoch=3)
        await browser.until('input.enabled', lambda event: not event['enabled'])
        if failure == 'no_output':
            response = await browser.until('response.text')
            assert (await browser.until('audio.done'))['context_id'] == response['context_id']
        else:
            code = 'speech_unavailable' if failure == 'zero_audio' else 'turn_failed'
            assert (await browser.until('error'))['code'] == code
        enabled = await browser.until('input.enabled', lambda event: event['enabled'])
        assert enabled['client_epoch'] == 3
        assert connection.input_enabled and not connection.playback_pending
    finally:
        await cleanup(owner, task)


@pytest.mark.asyncio
async def test_empty_final_transcript_resumes_without_creating_a_response():
    owner, _, browser, connection, task, _ = await scene_connection()
    try:
        await browser.until('input.enabled', lambda event: event['enabled'])
        connection.input_turn = 'empty-turn'
        connection.input_generation = connection.runtime.generation
        await connection._set_input_enabled(False)
        await browser.until('input.enabled', lambda event: not event['enabled'])
        await connection.on_transcript(Transcript('final', 'empty-turn', '', 'item'))
        assert (await browser.until('error'))['code'] == 'empty_transcript'
        await browser.until('input.enabled', lambda event: event['enabled'])
        assert connection.input_enabled and connection.response_task is None
    finally:
        await cleanup(owner, task)


@pytest.mark.asyncio
async def test_missing_playback_ack_cancels_context_before_resuming_and_old_timeout_is_ignored():
    owner, _, browser, connection, task, _ = await scene_connection(speech_provider='native')
    try:
        await browser.until('input.enabled', lambda event: event['enabled'])
        await browser.send('input.text', text='hello', client_epoch=3)
        response = await browser.until('response.text')
        await browser.until('state', lambda event: event['state'] == 'listening')
        assert not connection.input_enabled and connection.playback_deadline_task
        generation = connection.runtime.generation
        await connection._playback_deadline('old-context', generation, 0)
        assert not connection.input_enabled
        await connection._playback_deadline(response['context_id'], generation - 1, 0)
        assert not connection.input_enabled
        await connection._playback_deadline(response['context_id'], generation, 0)
        assert (await browser.until('error'))['code'] == 'playback_timeout'
        assert (await browser.until('response.cancelled'))['context_id'] == response['context_id']
        await browser.until('input.enabled', lambda event: event['enabled'])
        assert connection.input_enabled and connection.playback_deadline_task is None
        kinds = [event['type'] for event in browser.events]
        assert kinds.index('response.cancelled') < len(kinds) - 1
    finally:
        await cleanup(owner, task)


@pytest.mark.asyncio
@pytest.mark.parametrize('provider_ready', [True, False])
async def test_transcript_deadline_retries_ready_provider_without_disabling_microphone(monkeypatch, provider_ready):
    owner, _, browser, connection, task, _ = await scene_connection()
    connection.transcriber = transcriber = CountingTranscriber()
    connection.detector = detector = CountingVAD()
    try:
        await browser.until('input.enabled', lambda event: event['enabled'])
        connection.input_turn = 'timed-out-question'
        connection.input_generation = connection.runtime.generation
        await connection._set_input_enabled(False)
        await browser.until('input.enabled', lambda event: not event['enabled'])
        transcriber.ready = provider_ready
        async def immediate_deadline(delay):
            assert delay == 12
        with monkeypatch.context() as patch:
            patch.setattr(asyncio, 'sleep', immediate_deadline)
            await connection._transcript_deadline('timed-out-question', connection.runtime.generation)
        error = await browser.until('error')
        if provider_ready:
            assert error['code'] == 'transcription_retry'
            await browser.until('input.enabled', lambda event: event['enabled'])
            assert connection.input_enabled
            for _ in range(5): await browser.send('audio.append', audio=VOICE, client_epoch=2)
            await browser.until('speech.started')
            assert len(transcriber.begun) == 1 and detector.calls == 5
        else:
            assert error['code'] == 'transcription_unavailable'
            assert not connection.input_enabled
        assert connection.input_turn != 'timed-out-question'
    finally:
        await cleanup(owner, task)


@pytest.mark.asyncio
@pytest.mark.parametrize('output', ['native', 'none', 'failure', 'timeout', 'transcription_retry'])
async def test_spoken_end_holds_input_closed_after_ack_failure_or_timeout_until_manual_override(output):
    owner, _, browser, connection, task, _ = await scene_connection(brain_factory=EndingBrain,
        speech_provider='native' if output in ('native', 'timeout', 'transcription_retry') else 'disabled',
        speech_factory=(lambda: BrokenSpeech()) if output == 'failure' else None)
    try:
        await browser.until('input.enabled', lambda event: event['enabled'])
        await browser.send('input.text', text='goodbye', client_epoch=3)
        response = await browser.until('response.text')
        assert response['end_conversation'] is True
        assert connection.ending_context_id == response['context_id']
        if output == 'failure':
            assert (await browser.until('error'))['code'] == 'speech_unavailable'
        elif output == 'none':
            assert (await browser.until('audio.done'))['context_id'] == response['context_id']
        await browser.until('state', lambda event: event['state'] == 'ending')
        if output == 'timeout':
            await connection._playback_deadline(response['context_id'], connection.runtime.generation, 0)
            assert (await browser.until('error'))['code'] == 'playback_timeout'
            await browser.until('response.cancelled')
        elif output == 'transcription_retry':
            await connection.on_transcription_error('A stale transcript could not finish.')
            assert (await browser.until('error'))['code'] == 'transcription_retry'
        else:
            await browser.send('playback.finished', context_id=response['context_id'], client_epoch=3)
        await browser.send('ping')
        await browser.until('pong')
        assert not connection.input_enabled
        await browser.send('scene.stable', client_epoch=3)
        assert (await browser.until('scene.ignored'))['reason'] == 'busy'
        # A deliberate control is allowed to cancel the pending end request.
        await browser.send('interrupt', client_epoch=4)
        await browser.until('input.enabled', lambda event: event['enabled'])
        assert connection.input_enabled and connection.ending_context_id is None
    finally:
        await cleanup(owner, task)
