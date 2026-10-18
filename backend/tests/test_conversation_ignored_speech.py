"""Unrelated speech releases half-duplex input without a spoken placeholder."""
import asyncio

import pytest

from app.openai_transcription import Transcript
from addons.pair_pipeline.elevenlabs_dialogue_adapter import ElevenLabsDialogueSpeechAdapter
from test_conversation_prepared_speech import HeldBrain, ObservedSocket
from test_conversation_scene import scene_connection
from test_conversation_transport import cleanup


class IgnoreBrain(HeldBrain):
    async def respond(self, text, frame_provider=None):
        self.started.set()
        await self.release.wait()
        return {'status': 'ignored', 'silent': True, 'reply': 'No product question detected.',
                'snapshot': self.snapshot()}


@pytest.mark.parametrize('source', ['voice', 'text'])
async def test_ignored_turn_closes_prepared_speech_and_resumes_input_without_playback(source):
    socket = ObservedSocket()
    async def connect(url, key): return socket
    speaker = ElevenLabsDialogueSpeechAdapter('test-key', connector=connect)
    owner, session, browser, connection, task, _ = await scene_connection(
        brain_factory=IgnoreBrain, speech_factory=lambda: speaker)
    try:
        await browser.until('input.enabled', lambda event: event['enabled'])
        if source == 'text':
            await browser.send('input.text', text='unrelated fragment', client_epoch=2)
        else:
            await connection._set_input_enabled(False)
            connection.input_turn = 'voice-turn'
            connection.input_generation = connection.runtime.generation
            await connection.on_transcript(Transcript('final', 'voice-turn', 'unrelated fragment', 'item'))
        await asyncio.wait_for(connection.runtime.brain.started.wait(), 1)
        await asyncio.wait_for(socket.registered.wait(), 1)
        assert not connection.input_enabled
        connection.runtime.brain.release.set()
        await browser.until('input.enabled', lambda event: event['enabled'])
        assert socket.closed and socket.sent == [{'voices': [speaker.voice_id]}]
        assert connection.context_id is None and not connection.playback_pending
        assert connection.response_task is None and connection.input_enabled
        assert not any(event['type'] in ('response.text', 'audio.chunk', 'audio.done', 'camera.request')
                       for event in browser.events)
        assert not any(row['role'] == 'assistant' for row in connection.runtime.recent)
        summary = owner.snapshot(session.id)['turn_summaries'][-1]
        assert summary['source'] == source and summary['response_status'] == 'ignored'
        assert summary['status'] == 'silent' and summary['playback_state'] == 'not_requested'
        assert summary['turn_to_input_reenabled_ms'] is not None
        assert summary['turn_to_text_ms'] is None and summary['pcm_bytes_queued'] == 0
    finally:
        await cleanup(owner, task)
