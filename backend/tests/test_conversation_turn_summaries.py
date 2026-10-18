"""Bounded diagnostics distinguish phase latency and client playback reports."""
import asyncio
import json

import pytest

from app.conversation_transport import _phase_timings
from addons.pair_pipeline.voice_adapter import VoiceError
from test_conversation_transport import Brain, Browser, Speech, cleanup
from test_conversation_scene import scene_connection


PRIVATE = 'PRIVATE_TEXT_IMAGE_KEY_OR_SOURCE_URL'


def events(selection=120, guidance=0):
    return [
        {'type': 'product.vision_timing', 'stage': 'perception',
         'message': f'Perceptron selection: {selection} ms (parallel crop times overlap).'},
        *[{'type': 'product.vision_timing', 'stage': 'perception',
           'message': f'Perceptron crop_identity: {ms} ms (parallel crop times overlap).'} for ms in (210, 230)],
        {'type': 'product.detected', 'stage': 'perception',
         'message': f'Product 1: {PRIVATE}. Perceptron inspected 2 separate crop(s) in 355 ms.'},
        *[{'type': 'product.search', 'stage': 'search',
           'message': f'{PRIVATE}: catalog search finished in {ms} ms; 3 cited sources. Exact package formulation remains unverified.'}
          for ms in (400, 401)],
        {'type': 'product.guidance', 'stage': 'response',
         'message': f'DeepSeek returned a bounded response in {guidance} ms. Catalog estimates are separate from reviewed label facts.'},
        {'type': 'unrelated', 'stage': 'secret', 'message': PRIVATE, 'image': 'data:image/' + PRIVATE},
    ]


def summaries(owner, session):
    return owner.snapshot(session.id)['turn_summaries']


async def sync(browser):
    await browser.send('ping')
    await browser.until('pong')


class TimedBrain(Brain):
    async def respond(self, text, frame_provider=None):
        if text == 'camera':
            frame = await frame_provider()
            status = 'ready' if frame else 'needs_image'
        else:
            status = 'ready'
        return {'reply': PRIVATE, 'status': status, 'snapshot': self.snapshot(),
                'events': events(), 'end_conversation': text == 'goodbye'}

    async def respond_automatically(self, frame_provider=None):
        return {**await self.respond('automatic', frame_provider), 'silent': True}


def test_extracts_only_allowlisted_numbers_without_summing_parallel_phases():
    value = _phase_timings(events())
    assert value == {'vision_calls_ms': [
        {'stage': 'selection', 'milliseconds': 120},
        {'stage': 'crop_identity', 'milliseconds': 210},
        {'stage': 'crop_identity', 'milliseconds': 230}],
        'perception_group_ms': 355, 'catalog_search_group_ms': 401, 'guidance_phase_ms': 0}
    assert PRIVATE not in json.dumps(value)
    assert 'model' not in json.dumps(value).lower()  # Guidance can be deterministic.
    value = _phase_timings(events() * 100)
    assert len(value['vision_calls_ms']) == 4
    assert value['catalog_search_group_ms'] == 401


@pytest.mark.parametrize('bad', [None, 'private', {}, [None], [
    {'type': 'product.vision_timing', 'stage': 'perception',
     'message': 'Perceptron selection: 999999 ms (parallel crop times overlap).'},
    {'type': 'product.guidance', 'stage': 'response', 'message': PRIVATE * 2000},
    {'type': 'product.search', 'stage': 'search', 'message': 401},
    {'type': 'product.vision_timing', 'stage': 'perception',
     'message': 'Perceptron secret_stage: 5 ms (parallel crop times overlap).'},
    {'type': 'product.vision_timing', 'stage': 'wrong',
     'message': 'Perceptron selection: 5 ms (parallel crop times overlap).'},
]])
def test_malformed_or_unbounded_event_metadata_is_not_retained(bad):
    assert _phase_timings(bad) == _phase_timings([])


async def test_matching_ack_retains_numeric_report_across_disconnect_without_content():
    owner, session, browser, connection, task, _ = await scene_connection(
        brain_factory=TimedBrain, speech_factory=lambda: Speech(2))
    try:
        await browser.until('input.enabled', lambda event: event['enabled'])
        await browser.send('input.text', text=PRIVATE, client_epoch=3)
        response = await browser.until('response.text')
        await browser.until('audio.done')
        before = summaries(owner, session)[0]
        assert before['phase_timings'] == _phase_timings(events())
        assert before['playback_state'] == 'pending'
        assert before['pcm_bytes_queued'] == 9600
        assert before['text_to_playback_ack_ms'] is None
        assert not connection.input_enabled
        for epoch, context in ((2, response['context_id']), (3, 'wrong-context')):
            await browser.send('playback.finished', client_epoch=epoch, context_id=context)
        await sync(browser)
        assert summaries(owner, session)[0] == before
        await browser.send('playback.started', client_epoch=3, context_id=response['context_id'])
        await sync(browser)
        assert summaries(owner, session)[0]['playback_state'] == 'client_started'
        await browser.send('playback.finished', client_epoch=3, context_id=response['context_id'])
        await browser.until('input.enabled', lambda event: event['enabled'])
        after = summaries(owner, session)[0]
        assert after['playback_state'] == 'client_finished'
        assert 0 <= after['turn_to_text_ms'] <= 300_000
        assert 0 <= after['text_to_first_audio_queued_ms'] <= after['text_to_audio_done_queued_ms']
        assert after['text_to_playback_ack_ms'] >= after['text_to_first_audio_queued_ms']
        assert after['turn_to_input_reenabled_ms'] is not None
        assert PRIVATE not in json.dumps(after) and 'data:image' not in json.dumps(after)
        # Snapshots cannot mutate retained state, and duplicate/out-of-order
        # client telemetry cannot regress a completed diagnostic.
        after['phase_timings']['vision_calls_ms'].clear()
        assert summaries(owner, session)[0]['phase_timings']['vision_calls_ms']
        await browser.send('playback.started', client_epoch=3, context_id=response['context_id'])
        await sync(browser)
        assert summaries(owner, session)[0]['playback_state'] == 'client_finished'
        await browser.close()
        await asyncio.wait_for(task, 1)
        assert summaries(owner, session)[0]['playback_state'] == 'client_finished'
    finally:
        await owner.close()


async def test_only_last_five_turns_survive_and_local_response_does_not_claim_model_time():
    owner, session, browser, connection, task, _ = await scene_connection(brain_factory=TimedBrain)
    try:
        await browser.until('input.enabled', lambda event: event['enabled'])
        ids = []
        for epoch in range(3, 10):
            await browser.send('input.text', text=PRIVATE, client_epoch=epoch)
            await browser.until('response.text')
            await browser.until('input.enabled', lambda event: event['enabled'])
            ids.append(summaries(owner, session)[-1]['turn_id'])
        rows = summaries(owner, session)
        assert [row['turn_id'] for row in rows] == ids[-5:]
        assert all(row['playback_state'] == 'not_requested' for row in rows)
        assert all(row['text_to_first_audio_queued_ms'] is None for row in rows)
        assert all(row['phase_timings']['guidance_phase_ms'] == 0 for row in rows)
        assert PRIVATE not in json.dumps(rows)
    finally:
        await cleanup(owner, task)


async def test_disconnect_without_ack_and_old_connection_cannot_report_completion():
    owner, session, browser, connection, task, _ = await scene_connection(
        brain_factory=TimedBrain, speech_factory=lambda: Speech(1))
    try:
        await browser.send('input.text', text=PRIVATE, client_epoch=3)
        response = await browser.until('response.text')
        await browser.until('audio.done')
        replacement = await owner.attach(Browser(), session.id)
        await asyncio.wait_for(task, 1)
        row = summaries(owner, session)[0]
        assert row['playback_state'] == 'disconnected_unconfirmed'
        assert row['text_to_playback_ack_ms'] is None
        assert row['response_status'] == 'ready'
        connection._playback_summary('playback.finished',
            {'context_id': response['context_id'], 'client_epoch': 3})
        replacement._playback_summary('playback.finished',
            {'context_id': response['context_id'], 'client_epoch': 3})
        assert summaries(owner, session)[0] == row
    finally:
        await owner.close()


async def test_cancelled_old_brain_cannot_overwrite_replacement_turn_timings():
    entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    class LateBrain(TimedBrain):
        async def respond(self, text, frame_provider=None):
            if text == 'old':
                entered.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    await release.wait()
                finished.set()
                return {**await super().respond(text), 'events': events(selection=999)}
            return await super().respond(text)
    owner, session, browser, connection, task, _ = await scene_connection(brain_factory=LateBrain)
    try:
        await browser.send('input.text', text='old', client_epoch=3)
        await asyncio.wait_for(entered.wait(), 1)
        await browser.send('input.text', text='new', client_epoch=4)
        await browser.until('response.text')
        await browser.until('input.enabled', lambda event: event['enabled'])
        before = summaries(owner, session)
        assert before[0]['status'] == 'cancelled'
        assert before[0]['phase_timings'] == _phase_timings([])
        release.set()
        await asyncio.wait_for(finished.wait(), 1)
        await asyncio.sleep(0)
        assert summaries(owner, session) == before
    finally:
        release.set()
        await cleanup(owner, task)


async def test_playback_timeout_retains_specific_outcome_instead_of_generic_interrupt():
    owner, session, browser, connection, task, _ = await scene_connection(speech_provider='native')
    try:
        await browser.send('input.text', text=PRIVATE, client_epoch=3)
        response = await browser.until('response.text')
        await browser.until('state', lambda event: event['state'] == 'listening')
        await connection._playback_deadline(response['context_id'], connection.runtime.generation, 0)
        assert (await browser.until('error'))['code'] == 'playback_timeout'
        row = summaries(owner, session)[0]
        assert row['status'] == row['playback_state'] == 'playback_timeout'
        assert row['text_to_playback_ack_ms'] is None
        assert row['response_status'] == 'ready'
    finally:
        await cleanup(owner, task)


async def test_safe_speech_failure_survives_new_turn_and_contains_no_exception_text():
    class BrokenSpeech:
        async def stream(self, request, cancel):
            raise VoiceError(PRIVATE, category='rate_limited', stage='synthesis', http_status=429)
            yield
    owner, session, browser, connection, task, _ = await scene_connection(
        brain_factory=TimedBrain, speech_factory=BrokenSpeech, speech_provider='elevenlabs')
    try:
        await browser.send('input.text', text=PRIVATE, client_epoch=3)
        assert (await browser.until('error'))['code'] == 'speech_unavailable'
        first = summaries(owner, session)[0]
        assert first['playback_state'] == 'speech_failed'
        assert first['speech_failure'] == {'provider': 'elevenlabs', 'stage': 'synthesis',
                                         'category': 'rate_limited', 'http_status': 429}
        await browser.send('input.text', text='next', client_epoch=4)
        await browser.until('error')
        assert summaries(owner, session)[0] == first
        assert PRIVATE not in json.dumps(summaries(owner, session))
    finally:
        await cleanup(owner, task)


async def test_camera_failure_and_silent_scene_are_distinct_without_retaining_frame():
    owner, session, browser, connection, task, _ = await scene_connection(brain_factory=TimedBrain)
    try:
        await browser.send('input.text', text='camera', client_epoch=3)
        request = await browser.until('camera.request')
        await browser.send('camera.response', request_id=request['request_id'], client_epoch=3, error=PRIVATE)
        await browser.until('response.text')
        await browser.until('input.enabled', lambda event: event['enabled'])
        row = summaries(owner, session)[0]
        assert row['camera_request_id'] == request['request_id'] and row['camera_wait_ms'] is not None
        assert row['status'] == row['response_status'] == 'needs_image'
        await browser.send('scene.stable', client_epoch=3)
        await browser.until('scene.accepted')
        await browser.until('input.enabled', lambda event: event['enabled'])
        silent = summaries(owner, session)[-1]
        assert silent['source'] == 'automatic_scene' and silent['status'] == 'silent'
        assert silent['context_id'] is None and silent['turn_to_text_ms'] is None
        assert silent['pcm_bytes_queued'] == 0 and silent['playback_state'] == 'not_requested'
        assert PRIVATE not in json.dumps(summaries(owner, session))
    finally:
        await cleanup(owner, task)


@pytest.mark.parametrize('ending', [False, True])
async def test_native_ack_and_ending_do_not_invent_cloud_pcm_or_listening_resume(ending):
    owner, session, browser, connection, task, _ = await scene_connection(
        brain_factory=TimedBrain, speech_provider='native')
    try:
        await browser.until('input.enabled', lambda event: event['enabled'])
        await browser.send('input.text', text='goodbye' if ending else 'normal', client_epoch=3)
        response = await browser.until('response.text')
        await browser.send('playback.finished', client_epoch=3, context_id=response['context_id'])
        await sync(browser)
        row = summaries(owner, session)[0]
        assert row['audio_output'] == 'native' and row['pcm_bytes_queued'] == 0
        assert row['playback_state'] == 'client_finished'
        assert row['ending_response'] == ending
        assert (row['turn_to_input_reenabled_ms'] is None) == ending
    finally:
        await cleanup(owner, task)


async def test_zero_chunk_cloud_ack_is_not_recorded_as_audio_played():
    owner, session, browser, connection, task, _ = await scene_connection(speech_factory=lambda: Speech(0))
    try:
        await browser.send('input.text', text=PRIVATE, client_epoch=3)
        response = await browser.until('response.text')
        await browser.until('audio.done')
        await browser.send('playback.finished', client_epoch=3, context_id=response['context_id'])
        await sync(browser)
        row = summaries(owner, session)[0]
        assert row['playback_state'] == 'client_finished_no_audio'
        assert row['pcm_bytes_queued'] == 0 and row['text_to_first_audio_queued_ms'] is None
    finally:
        await cleanup(owner, task)


async def test_simultaneous_sessions_keep_phase_records_and_snapshots_separate():
    entered, release = asyncio.Event(), asyncio.Event()
    class HeldBrain(TimedBrain):
        async def respond(self, text, frame_provider=None):
            entered.set()
            await release.wait()
            return {**await super().respond(text), 'events': events(selection=777)}
    first = await scene_connection(brain_factory=HeldBrain)
    second = await scene_connection(brain_factory=TimedBrain)
    owner_a, session_a, browser_a, _, task_a, _ = first
    owner_b, session_b, browser_b, _, task_b, _ = second
    try:
        await browser_a.send('input.text', text=PRIVATE, client_epoch=3)
        await asyncio.wait_for(entered.wait(), 1)
        await browser_b.send('input.text', text=PRIVATE, client_epoch=3)
        await browser_b.until('response.text')
        before_b = summaries(owner_b, session_b)
        assert summaries(owner_a, session_a)[0]['status'] == 'running'
        release.set()
        await browser_a.until('response.text')
        assert summaries(owner_a, session_a)[0]['phase_timings']['vision_calls_ms'][0]['milliseconds'] == 777
        assert summaries(owner_b, session_b) == before_b
        assert before_b[0]['phase_timings']['vision_calls_ms'][0]['milliseconds'] == 120
        assert before_b[0]['turn_id'] != summaries(owner_a, session_a)[0]['turn_id']
    finally:
        release.set()
        await cleanup(owner_a, task_a)
        await cleanup(owner_b, task_b)


async def test_failed_recognition_keeps_returned_call_timing_without_claiming_group_completion():
    class FailedRecognition(TimedBrain):
        async def respond(self, text, frame_provider=None):
            return {**await super().respond(text), 'status': 'needs_image', 'events': events()[:1]}
    owner, session, browser, _, task, _ = await scene_connection(brain_factory=FailedRecognition)
    try:
        await browser.send('input.text', text=PRIVATE, client_epoch=3)
        await browser.until('response.text')
        row = summaries(owner, session)[0]
        assert row['response_status'] == 'needs_image'
        assert row['phase_timings']['vision_calls_ms'] == [{'stage': 'selection', 'milliseconds': 120}]
        assert row['phase_timings']['perception_group_ms'] is None
        assert row['phase_timings']['catalog_search_group_ms'] is None
        assert row['phase_timings']['guidance_phase_ms'] is None
    finally:
        await cleanup(owner, task)
