"""Cancellable continuous-audio shopping conversation, separate from REST turns."""
from __future__ import annotations

import asyncio
import base64
import binascii
import copy
import json
import io
import os
import re
import secrets
import time
import sys
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from .openai_transcription import OpenAITranscription, Transcript, TranscriptionError
from .speech_detection import SpeechDetector

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@dataclass(frozen=True)
class ConversationConfig:
    api_token: str = field(default='', repr=False)
    openai_api_key: str = field(default='', repr=False)
    elevenlabs_api_key: str = field(default='', repr=False)
    elevenlabs_stt_model: str = 'scribe_v2_realtime'
    elevenlabs_tts_model: str = 'eleven_flash_v2_5'
    elevenlabs_voice_id: str = 'JBFqnCBsd6RMkjVDRZzb'
    transcription_model: str = 'gpt-live-transcribe'
    transcription_provider: str = 'elevenlabs'
    speech_provider: str = 'elevenlabs'
    local_model: str = 'tiny.en'
    voice_style: str = 'quiet'
    heartbeat_timeout_s: float = 45
    startup_timeout_s: float = 8
    outgoing_queue_size: int = 64
    product_mode: str = 'reviewed_catalog'

    @classmethod
    def from_env(cls):
        return cls(api_token=os.getenv('SHELF_API_TOKEN', '').strip(),
                   openai_api_key=os.getenv('OPENAI_API_KEY', '').strip(),
                   elevenlabs_api_key=os.getenv('ELEVENLABS_API_KEY', '').strip(),
                   elevenlabs_stt_model=os.getenv('ELEVENLABS_STT_MODEL', 'scribe_v2_realtime').strip(),
                   elevenlabs_tts_model=os.getenv('ELEVENLABS_TTS_MODEL', 'eleven_flash_v2_5').strip(),
                   elevenlabs_voice_id=os.getenv('ELEVENLABS_VOICE_ID', 'JBFqnCBsd6RMkjVDRZzb').strip(),
                   transcription_provider=os.getenv('CONVERSATION_TRANSCRIPTION_PROVIDER', 'elevenlabs').strip(),
                   speech_provider=os.getenv('CONVERSATION_SPEECH_PROVIDER', 'elevenlabs').strip(),
                   local_model=os.getenv('CONVERSATION_LOCAL_MODEL', 'tiny.en').strip(),
                   product_mode=os.getenv('CONVERSATION_PRODUCT_MODE', 'reviewed_catalog').strip())


def _brain(profile, recipient_id):
    from .conversation_brain import ShoppingBrain
    return ShoppingBrain(profile=profile, recipient_id=recipient_id)


def _speech(key):
    from addons.pair_pipeline.openai_voice_adapter import OpenAISpeechAdapter
    return OpenAISpeechAdapter(key)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _phase_timings(events) -> dict:
    """Extract numbers only; parallel search/crop entries are never summed.

    These are existing service phase timings, not measurements of model calls.
    In particular guidance may be a locally computed, zero-provider response.
    """
    phases = {'vision_calls_ms': [], 'perception_group_ms': None,
              'catalog_search_group_ms': None, 'guidance_phase_ms': None}
    if not isinstance(events, list):
        return phases
    for event in events[:50]:
        if not isinstance(event, dict):
            continue
        message = event.get('message')
        if not isinstance(message, str) or len(message) > 2048:
            continue
        kind, stage = event.get('type'), event.get('stage')
        if kind == 'product.vision_timing' and stage == 'perception':
            match = re.fullmatch(r'Perceptron (selection|selection_retry|crop_identity): (\d{1,6}) ms '
                                 r'\(parallel crop times overlap\)\.', message)
            if match and int(match[2]) <= 300_000 and len(phases['vision_calls_ms']) < 4:
                phases['vision_calls_ms'].append({'stage': match[1], 'milliseconds': int(match[2])})
            continue
        key, match = None, None
        if kind == 'product.detected' and stage == 'perception':
            key = 'perception_group_ms'
            match = re.search(r'\. Perceptron inspected [12] separate crop\(s\) in (\d{1,6}) ms\.$', message)
        elif kind == 'product.search' and stage == 'search':
            key = 'catalog_search_group_ms'
            match = re.search(r': catalog search finished in (\d{1,6}) ms; \d{1,3} cited sources\. '
                              r'Exact package formulation remains unverified\.$', message)
        elif kind == 'product.guidance' and stage == 'response':
            key = 'guidance_phase_ms'
            match = re.fullmatch(r'(?:DeepSeek|Gemini|Anthropic Sonnet|Configured provider) returned a bounded '
                                 r'response in (\d{1,6}) ms\. Catalog estimates are separate from reviewed label facts\.', message)
        if match and int(match[1]) <= 300_000:
            phases[key] = max(phases[key] or 0, int(match[1]))
    return phases


@dataclass
class _TurnSummary:
    """Five small content-free records survive disconnect in the runtime only."""
    turn_id: str
    generation: int
    client_epoch: int
    source: str
    started: float = field(default_factory=time.monotonic)
    text_at: float | None = None
    values: dict = field(default_factory=dict)

    def __post_init__(self):
        self.values = {'turn_id': self.turn_id, 'generation': self.generation,
            'client_epoch': self.client_epoch, 'source': self.source, 'accepted_at': _now(),
            'status': 'running', 'response_status': None, 'context_id': None, 'camera_request_id': None,
            'camera_wait_ms': None, 'phase_timings': _phase_timings([]),
            'turn_to_text_ms': None, 'text_to_first_audio_queued_ms': None,
            'text_to_audio_done_queued_ms': None, 'text_to_playback_ack_ms': None,
            'turn_to_input_reenabled_ms': None, 'pcm_bytes_queued': 0,
            'audio_output': 'none', 'playback_state': 'not_requested', 'ending_response': False,
            'speech_failure': None}

    def elapsed(self, since: float | None = None) -> int:
        return min(300_000, max(0, round((time.monotonic() - (self.started if since is None else since)) * 1000)))

    def snapshot(self) -> dict:
        return copy.deepcopy(self.values)


@dataclass
class _Runtime:
    brain: Any
    generation: int = 0
    connection: Any = None
    state: str = 'disconnected'
    recent: deque = field(default_factory=lambda: deque(maxlen=20))
    updated_at: str = field(default_factory=_now)
    capture_metadata: dict | None = None
    speech_failure: dict | None = None
    turn_summaries: deque = field(default_factory=lambda: deque(maxlen=5))


class ConversationCoordinator:
    def __init__(self, service, *, config: ConversationConfig | None = None, brain_factory=None,
                 transcriber_factory=None, speech_factory=None, vad_factory=None):
        self.service = service
        self.config = config or ConversationConfig.from_env()
        if self.config.product_mode not in ('reviewed_catalog', 'product_search'):
            raise ValueError('Unknown conversation product mode')
        self.brain_factory = brain_factory or _brain
        self.transcriber_factory = transcriber_factory
        if self.config.transcription_provider not in ('elevenlabs', 'local', 'openai', 'disabled') or self.config.speech_provider not in ('elevenlabs', 'native', 'openai', 'disabled'):
            raise ValueError('Unknown conversation speech provider')
        self.speech_available = bool((self.config.speech_provider == 'openai' and self.config.openai_api_key)
            or (self.config.speech_provider == 'elevenlabs' and self.config.elevenlabs_api_key and self.config.elevenlabs_voice_id) or speech_factory)
        self.speech_factory = speech_factory or self._speaker
        self.vad_factory = vad_factory or SpeechDetector
        self.runtimes: dict[str, _Runtime] = {}
        self.lock = asyncio.Lock()

    def _speaker(self):
        if self.config.speech_provider == 'elevenlabs':
            from addons.pair_pipeline.elevenlabs_voice_adapter import create_elevenlabs_speech_adapter
            return create_elevenlabs_speech_adapter(self.config.elevenlabs_api_key, voice_id=self.config.elevenlabs_voice_id,
                                          model=self.config.elevenlabs_tts_model)
        return _speech(self.config.openai_api_key)

    def _session(self, session_id: str):
        try:
            UUID(session_id)
        except (ValueError, TypeError, AttributeError):
            raise HTTPException(404, 'Shopping session not found.') from None
        session = self.service.get(session_id).session
        if not session.profile.consented:
            raise HTTPException(403, 'Review and consent to the shopping profile before starting voice.')
        return session

    def snapshot(self, session_id: str) -> dict:
        self._session(session_id)
        runtime = self.runtimes.get(session_id)
        if runtime is None:
            return {'session_id': session_id, 'connected': False, 'state': 'not_started', 'recent': [],
                    'snapshot': None, 'turn_summaries': []}
        return {'session_id': session_id, 'connected': runtime.connection is not None,
                'state': runtime.state, 'updated_at': runtime.updated_at,
                'recent': list(runtime.recent), 'snapshot': runtime.brain.snapshot(),
                'capture': runtime.capture_metadata,
                'speech_failure': dict(runtime.speech_failure) if runtime.speech_failure else None,
                'turn_summaries': [summary.snapshot() for summary in runtime.turn_summaries]}

    async def attach(self, websocket: WebSocket, session_id: str):
        session = self._session(session_id)
        async with self.lock:
            # Expired/deleted sessions must not retain voice memory indefinitely.
            for key in list(self.runtimes):
                if key not in self.service.records:
                    stale = self.runtimes.pop(key)
                    if stale.connection:
                        await stale.connection.shutdown(4001)
                    await stale.brain.close()
            runtime = self.runtimes.get(session_id)
            if runtime is None:
                if len(self.runtimes) >= 64:
                    raise HTTPException(429, 'Voice session capacity reached.')
                runtime = _Runtime(self.brain_factory(session.profile.model_dump(), session.recipient_id))
                if hasattr(runtime.brain, 'session_id'):
                    runtime.brain.session_id = session_id
                if hasattr(runtime.brain, 'cart'):
                    runtime.brain.cart = [{'id': item.id, 'product_id': item.product_id,
                        'name': item.product_name, 'recipient_id': item.recipient_id,
                        'quantity': item.quantity, 'confirmed_at': item.confirmed_at} for item in session.cart]
                self.runtimes[session_id] = runtime
            if runtime.connection:
                await runtime.connection.shutdown(4001)
            connection = ConversationConnection(self, websocket, session_id, runtime)
            runtime.connection = connection
            return connection

    async def interrupt_session(self, session_id: str, reason: str, delete: bool = False,
                                recipient_id: str | None = None):
        runtime = self.runtimes.get(session_id)
        if runtime is None:
            return
        if runtime.connection:
            if delete:
                await runtime.connection.shutdown(1000)
            else:
                await runtime.connection.interrupt()
        else:
            runtime.generation += 1
            runtime.brain.cancel()
        if recipient_id is not None:
            runtime.speech_failure = None
            runtime.brain.set_recipient(recipient_id)
            if runtime.connection:
                await runtime.connection.emit({'type': 'snapshot', 'snapshot': runtime.brain.snapshot()})
        if delete:
            self.runtimes.pop(session_id, None)
            await runtime.brain.close()

    async def close(self):
        for session_id in list(self.runtimes):
            await self.interrupt_session(session_id, 'shutdown', delete=True)


class ConversationConnection:
    def __init__(self, owner: ConversationCoordinator, websocket: WebSocket, session_id: str, runtime: _Runtime):
        self.owner, self.ws, self.session_id, self.runtime = owner, websocket, session_id, runtime
        self.config = owner.config
        self.queue: asyncio.Queue[tuple[int | None, dict]] = asyncio.Queue(maxsize=self.config.outgoing_queue_size)
        self.writer_task: asyncio.Task | None = None
        self.response_task: asyncio.Task | None = None
        self.transcript_deadline_task: asyncio.Task | None = None
        self.playback_deadline_task: asyncio.Task | None = None
        self.cancel_event = asyncio.Event()
        self.context_id: str | None = None
        self.context_client_epoch = 0
        self.client_epoch = 0
        self.input_turn: str | None = None
        self.input_generation: int | None = None
        self.camera_pending: dict[str, tuple[int, asyncio.Future]] = {}
        self.detector = owner.vad_factory()
        self.transcriber = None
        self.started = False
        self.closed = False
        self.audio_error_sent = False
        self.audio_window: deque[tuple[float, int]] = deque()
        self.automatic_capture_enabled = False
        self.last_automatic_scene_at: float | None = None
        self.response_source: str | None = None
        self.context_source: str | None = None
        self.playback_pending = False
        self.input_enabled = False
        self.ending_context_id: str | None = None
        self.turn_summary: _TurnSummary | None = None

    def _current_summary(self, summary: _TurnSummary | None) -> bool:
        return bool(summary is not None and self.turn_summary is summary and not self.closed
                    and self.runtime.connection is self and summary.generation == self.runtime.generation)

    def _end_summary(self, outcome: str):
        summary = self.turn_summary
        if not self._current_summary(summary):
            return
        values = summary.values
        if values['playback_state'] in ('pending', 'client_started'):
            values['playback_state'] = outcome
            values['status'] = 'cancelled' if outcome == 'interrupted' else outcome
        elif values['status'] == 'running':
            values['status'] = 'cancelled' if outcome == 'interrupted' else outcome

    def _playback_summary(self, kind: str, event: dict):
        summary = self.turn_summary
        if (not self._current_summary(summary) or summary.client_epoch != self.client_epoch
                or event.get('client_epoch', self.client_epoch) != summary.client_epoch
                or event.get('context_id') != summary.values['context_id']
                or summary.values['playback_state'] not in ('pending', 'client_started')):
            return
        if kind == 'playback.started':
            summary.values['playback_state'] = 'client_started'
        else:
            state = 'client_finished' if kind == 'playback.finished' else 'client_interrupted'
            if state == 'client_finished' and summary.values['audio_output'] == 'cloud' and not summary.values['pcm_bytes_queued']:
                state = 'client_finished_no_audio'
            summary.values['playback_state'] = state
            summary.values['text_to_playback_ack_ms'] = summary.elapsed(summary.text_at)

    async def emit(self, payload: dict, generation: int | None = None):
        if self.closed or (generation is not None and generation != self.runtime.generation):
            return
        payload = {'client_epoch': self.client_epoch, **payload}
        try:
            async with asyncio.timeout(2):
                await self.queue.put((generation, payload))
        except TimeoutError:
            await self.shutdown(1013)

    async def state(self, state: str, generation: int | None = None):
        if generation is not None and generation != self.runtime.generation:
            return
        if state == 'listening' and not (self.transcriber and self.transcriber.ready):
            state = 'text_only'
        self.runtime.state, self.runtime.updated_at = state, _now()
        await self.emit({'type': 'state', 'state': state}, generation)

    async def error(self, code: str, message: str):
        await self.emit({'type': 'error', 'code': code, 'message': message})

    async def _set_input_enabled(self, enabled: bool, *, force: bool = False):
        if self.closed:
            return
        enabled = bool(enabled and self.started and self.transcriber and self.transcriber.ready)
        changed = enabled != self.input_enabled
        self.input_enabled = enabled
        summary = self.turn_summary
        if enabled and self._current_summary(summary) and summary.values['turn_to_input_reenabled_ms'] is None:
            summary.values['turn_to_input_reenabled_ms'] = summary.elapsed()
        if changed or force:
            await self.emit({'type': 'input.enabled', 'enabled': enabled}, self.runtime.generation)

    async def _resume_input_if_idle(self, *, force: bool = False):
        if (self.runtime.state == 'listening' and not self.playback_pending and self.ending_context_id is None
                and not (self.response_task and not self.response_task.done())
                and self.input_turn is None and not self.camera_pending
                and not (self.transcript_deadline_task and not self.transcript_deadline_task.done())):
            await self._set_input_enabled(True, force=force)

    async def _writer(self):
        try:
            while not self.closed:
                generation, payload = await self.queue.get()
                if generation is not None and generation != self.runtime.generation:
                    continue
                async with asyncio.timeout(3):
                    await self.ws.send_json(payload)
        except asyncio.CancelledError:
            raise
        except Exception:
            await self.shutdown(1011)

    async def run(self):
        self.writer_task = asyncio.create_task(self._writer())
        try:
            while not self.closed:
                async with asyncio.timeout(self.config.heartbeat_timeout_s if self.started else 10):
                    raw = await self.ws.receive_text()
                message_limit = (6 if self.config.product_mode == 'product_search' else 3) * 1024 * 1024
                if len(raw) > message_limit:
                    await self.error('message_too_large', 'Message exceeds the conversation limit.')
                    break
                try:
                    event = json.loads(raw)
                    if not isinstance(event, dict):
                        raise ValueError
                    if len(raw) > 3 * 1024 * 1024 and event.get('type') != 'camera.response':
                        await self.error('message_too_large', 'Message exceeds the conversation limit.')
                        break
                    await self.handle(event)
                except (ValueError, TypeError, binascii.Error):
                    await self.error('invalid_message', 'Check the event fields and audio format.')
                except HTTPException:
                    await self.shutdown(4403)
        except (WebSocketDisconnect, RuntimeError):
            pass
        except TimeoutError:
            await self.error('heartbeat_timeout', 'Conversation timed out. Reconnect to continue.')
        finally:
            await self.shutdown(1000)

    def _epoch(self, event: dict) -> bool:
        value = event.get('client_epoch', self.client_epoch)
        if type(value) is not int or not 0 <= value <= 2 ** 53:
            raise ValueError
        if value < self.client_epoch:
            return False
        self.client_epoch = value
        return True

    async def handle(self, event: dict):
        kind = event.get('type')
        if kind == 'ping':
            await self.emit({'type': 'pong'})
            return
        if kind == 'session.start':
            if self.started or event.get('session_id') != self.session_id or event.get('voice_consented') is not True:
                await self.error('voice_consent_required', 'Confirm microphone transcription consent before starting voice.')
                return
            if not self._epoch(event):
                return
            recipient = event.get('recipient_id', 'self')
            if recipient not in ('self', 'household'):
                raise ValueError
            self.runtime.brain.set_recipient(recipient)
            self._sync_recipient(recipient)
            self.started = True
            self.automatic_capture_enabled = (event.get('automatic_capture_enabled') is True
                and self.config.product_mode == 'product_search'
                and self.owner._session(self.session_id).mode == 'live_glasses')
            if self.owner.transcriber_factory:
                self.transcriber = self.owner.transcriber_factory(self.on_transcript, self.on_transcription_error)
            elif self.config.transcription_provider == 'elevenlabs' and self.config.elevenlabs_api_key:
                from .elevenlabs_transcription import ElevenLabsTranscription
                try:
                    self.transcriber = ElevenLabsTranscription(self.config.elevenlabs_api_key,
                        self.on_transcript, self.on_transcription_error, model=self.config.elevenlabs_stt_model,
                        startup_timeout_s=self.config.startup_timeout_s)
                except ValueError:
                    await self.error('transcription_unavailable', 'The configured ElevenLabs transcription model is unsupported. Check the backend configuration.')
            elif self.config.transcription_provider == 'local':
                from .local_transcription import LocalTranscription
                self.transcriber = LocalTranscription(self.on_transcript, self.on_transcription_error, model=self.config.local_model)
            elif self.config.transcription_provider == 'openai' and self.config.openai_api_key:
                self.transcriber = OpenAITranscription(self.config.openai_api_key, self.on_transcript,
                    self.on_transcription_error, model=self.config.transcription_model,
                    startup_timeout_s=self.config.startup_timeout_s)
            if self.transcriber:
                try:
                    await self.transcriber.start()
                except TranscriptionError:
                    await self.error('transcription_unavailable', 'Microphone transcription could not start. Check the configured speech provider or use text.')
                    self.transcriber = None
            elif self.config.transcription_provider != 'disabled' and not (self.config.transcription_provider == 'elevenlabs' and self.config.elevenlabs_api_key):
                await self.error('transcription_unavailable', 'The configured transcription provider key is missing on the backend. Use text until it is configured.')
            await self.emit({'type': 'session.ready', 'session_id': self.session_id,
                'client_epoch': self.client_epoch, 'audio_input_available': bool(self.transcriber and self.transcriber.ready),
                'listening_enabled': bool(self.transcriber and self.transcriber.ready),
                'audio_output_available': self.owner.speech_available, 'sample_rate': 24000,
                'automatic_capture_enabled': self.automatic_capture_enabled,
                'transcription_provider': self.config.transcription_provider,
                'speech_provider': self.config.speech_provider})
            await self.emit({'type': 'snapshot', 'snapshot': self.runtime.brain.snapshot()})
            await self.state('listening')
            await self._resume_input_if_idle(force=True)
            return
        if not self.started:
            await self.error('session_not_started', 'Send session.start with explicit voice consent first.')
            return
        self.owner._session(self.session_id)
        if kind in ('scene.configure', 'scene.stable'):
            await self._scene_event(event)
            return
        if kind in ('audio.append', 'camera.response') and event.get('client_epoch', self.client_epoch) != self.client_epoch:
            return
        if kind == 'session.stop':
            await self.shutdown(1000)
        elif kind == 'interrupt':
            if self._epoch(event):
                await self.interrupt()
        elif kind == 'context.update':
            if not self._epoch(event):
                return
            recipient = event.get('recipient_id')
            if recipient not in ('self', 'household'):
                raise ValueError
            await self.interrupt()
            self.runtime.speech_failure = None
            self.runtime.brain.set_recipient(recipient)
            self._sync_recipient(recipient)
            await self.emit({'type': 'snapshot', 'snapshot': self.runtime.brain.snapshot()})
        elif kind in ('playback.started', 'playback.finished', 'playback.interrupted'):
            if event.get('client_epoch', self.client_epoch) != self.client_epoch or event.get('context_id') != self.context_id:
                return
            # Client playback telemetry is not a measurement of what the wearer heard.
            self._playback_summary(kind, event)
            self.playback_pending = kind == 'playback.started'
            if self.playback_pending:
                await self._set_input_enabled(False)
            else:
                self._cancel_playback_deadline()
            await self.state('speaking' if kind == 'playback.started' else
                             'ending' if self.ending_context_id == self.context_id else 'listening')
            if not self.playback_pending:
                await self._resume_input_if_idle()
        elif kind == 'input.text':
            if not self._epoch(event):
                return
            text = event.get('text')
            if not isinstance(text, str) or not text.strip() or len(text) > 2000:
                raise ValueError
            await self.interrupt(clear_pending=False, resume_input=False)
            turn = str(uuid4())
            await self.emit({'type': 'transcript.final', 'text': text.strip(), 'turn_id': turn}, self.runtime.generation)
            self._respond(text.strip(), turn, source='text')
        elif kind == 'audio.append':
            # Half duplex: audio captured after the question is committed must
            # never reach VAD/STT or cancel a response, even from an old client.
            if not self.input_enabled:
                return
            audio = event.get('audio')
            if not isinstance(audio, str) or not 1 <= len(audio) <= 4000:
                raise ValueError
            pcm = base64.b64decode(audio, validate=True)
            if len(pcm) not in (960, 1920, 2880):
                raise ValueError
            stamp = time.monotonic()
            while self.audio_window and self.audio_window[0][0] < stamp - 2:
                self.audio_window.popleft()
            self.audio_window.append((stamp, len(pcm)))
            if sum(size for _, size in self.audio_window) > 192000:
                await self.error('audio_rate', 'Audio arrived faster than realtime. Reconnect the microphone.')
                await self.shutdown(1008)
                return
            if not self.transcriber or not self.transcriber.ready:
                if not self.audio_error_sent:
                    self.audio_error_sent = True
                    await self.error('transcription_unavailable', 'Microphone transcription is unavailable. Use text or reconnect.')
                return
            try:
                for action in self.detector.feed(pcm):
                    if action.kind == 'start':
                        await self.interrupt(reset_detector=False, clear_pending=False)
                        self.input_turn = str(uuid4())
                        self.input_generation = self.runtime.generation
                        await self.emit({'type': 'speech.started', 'turn_id': self.input_turn}, self.runtime.generation)
                        self.transcriber.begin_turn(self.input_turn)
                        self.transcriber.append(action.pcm)
                        await self.state('listening')
                    elif action.kind == 'audio':
                        self.transcriber.append(action.pcm)
                    elif action.kind == 'end' and self.input_turn:
                        await self._set_input_enabled(False)
                        self.transcriber.commit(self.input_turn)
                        self.transcript_deadline_task = asyncio.create_task(self._transcript_deadline(self.input_turn, self.runtime.generation))
                        await self.state('transcribing')
                        if action.reason == 'max_duration':
                            await self.error('speech_limit', 'That reached the 15-second turn limit. Keep the next question short.')
            except TranscriptionError:
                await self.on_transcription_error('Audio transcription stopped. Reconnect, or use text input.')
        elif kind == 'camera.response':
            request_id = event.get('request_id')
            pending = self.camera_pending.get(request_id) if isinstance(request_id, str) else None
            if not pending or pending[0] != self.runtime.generation:
                return
            future = pending[1]
            if event.get('error') is not None:
                if not future.done():
                    future.set_result(None)
                return
            from addons.pair_pipeline.perceptron_adapter import Frame, _validate_frame
            # Device clocks may differ. Receipt time is server UTC; freshness also
            # depends on the current request ID and the phone's monotonic gate.
            frame = Frame(request_id, event.get('image_data_url'), event.get('width'), event.get('height'), _now())
            image_limit = (4 if self.config.product_mode == 'product_search' else 2) * 1024 * 1024
            try:
                _validate_frame(frame, max_image_bytes=image_limit)
            except (ValueError, TypeError):
                if not future.done():
                    future.set_result(None)
                await self.error('invalid_camera_image', 'That camera frame is invalid. Please try again.')
                return
            from PIL import Image, ImageOps
            encoded = base64.b64decode(frame.image_data_url.split(',', 1)[1], validate=True)
            try:
                with Image.open(io.BytesIO(encoded)) as source:
                    if source.format not in ('JPEG', 'PNG') or source.width * source.height > 8_000_000:
                        raise ValueError
                    if max(source.size) > 4096:
                        raise ValueError
                    upright = ImageOps.exif_transpose(source)
                    upright.load()
                    if upright.size != (frame.width, frame.height):
                        raise ValueError
                    if source.getexif().get(274, 1) != 1:
                        buffer = io.BytesIO()
                        upright.convert('RGB').save(buffer, format='JPEG', quality=85)
                        frame = Frame(request_id, 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode(),
                                      frame.width, frame.height, frame.received_at)
                        _validate_frame(frame, max_image_bytes=image_limit)
            except Exception:
                if not future.done():
                    future.set_result(None)
                await self.error('invalid_camera_image', 'That camera frame could not be decoded. Please try again.')
                return
            if not future.done():
                device_timestamp = event.get('captured_at')
                if not isinstance(device_timestamp, str) or len(device_timestamp) > 60:
                    device_timestamp = None
                self.runtime.capture_metadata = {'request_id': request_id, 'received_at': frame.received_at,
                    'device_captured_at': device_timestamp, 'width': frame.width, 'height': frame.height}
                future.set_result(frame)
        else:
            await self.error('unknown_event', 'Unsupported conversation event.')

    def _scene_busy(self) -> bool:
        return (self.runtime.state != 'listening'
            or bool(self.response_task and not self.response_task.done())
            or self.input_turn is not None
            or bool(self.transcript_deadline_task and not self.transcript_deadline_task.done())
            or bool(self.camera_pending) or self.playback_pending
            or bool(getattr(self.detector, 'active', False)))

    async def _scene_event(self, event: dict):
        """A local stability hint may request one fresh photo, never interrupt a turn."""
        # Scene telemetry cannot advance the conversation epoch or carry an image.
        epoch = event.get('client_epoch')
        if type(epoch) is not int or epoch != self.client_epoch:
            await self.emit({'type': 'scene.ignored', 'reason': 'stale_epoch'})
            return
        if any(key in event for key in ('image_data_url', 'image', 'audio')):
            await self.emit({'type': 'scene.ignored', 'reason': 'unexpected_media'})
            return
        if (self.config.product_mode != 'product_search'
                or self.owner._session(self.session_id).mode != 'live_glasses'):
            await self.emit({'type': 'scene.ignored', 'reason': 'unsupported_mode'})
            return
        if event['type'] == 'scene.configure':
            enabled = event.get('enabled')
            if type(enabled) is not bool:
                raise ValueError
            self.automatic_capture_enabled = enabled
            if not enabled and ((self.response_source == 'automatic_scene'
                    and self.response_task and not self.response_task.done())
                    or (self.context_source == 'automatic_scene' and self.playback_pending)):
                await self.interrupt()
            await self.emit({'type': 'scene.configured', 'enabled': enabled})
            return
        if not self.automatic_capture_enabled:
            await self.emit({'type': 'scene.ignored', 'reason': 'disabled'})
            return
        if not self.input_enabled or self._scene_busy():
            await self.emit({'type': 'scene.ignored', 'reason': 'busy'})
            return
        stamp = time.monotonic()
        if self.last_automatic_scene_at is not None and stamp - self.last_automatic_scene_at < 8:
            await self.emit({'type': 'scene.ignored', 'reason': 'cooldown'})
            return
        generation = self.runtime.generation
        turn = str(uuid4())
        self.last_automatic_scene_at = stamp
        await self._set_input_enabled(False)
        await self.emit({'type': 'scene.accepted', 'turn_id': turn}, generation)
        # A microphone callback may interrupt while the outgoing queue is full.
        if self.closed or generation != self.runtime.generation or epoch != self.client_epoch or self._scene_busy():
            return
        question = (self.runtime.brain.automatic_question()
            if hasattr(self.runtime.brain, 'automatic_question') else 'Compare these products for my shopping goal.')
        self._respond(question, turn, source='automatic_scene')

    async def interrupt(self, *, reset_detector: bool = True, clear_pending: bool = True,
                        resume_input: bool = True, preserve_ending: bool = False):
        self._end_summary('interrupted')
        self.runtime.generation += 1
        self.runtime.brain.cancel(clear_pending=clear_pending)
        self.cancel_event.set()
        task, self.response_task = self.response_task, None
        self.response_source = None
        self.playback_pending = False
        self.context_source = None
        if not preserve_ending:
            self.ending_context_id = None
        self._cancel_playback_deadline()
        if not resume_input:
            self.input_enabled = False
        if task and task is not asyncio.current_task():
            task.cancel()
        self.input_turn = None
        self.input_generation = None
        if self.transcript_deadline_task and self.transcript_deadline_task is not asyncio.current_task():
            self.transcript_deadline_task.cancel()
        self.transcript_deadline_task = None
        if reset_detector:
            self.detector.reset()
        if self.transcriber and self.transcriber.ready:
            try:
                self.transcriber.discard_turn()
            except TranscriptionError:
                pass
        for _, future in self.camera_pending.values():
            if not future.done():
                future.cancel()
        self.camera_pending.clear()
        # Remove queued obsolete audio before enqueuing cancellation control.
        retained = []
        while not self.queue.empty():
            try:
                generation, payload = self.queue.get_nowait()
                if generation is None:
                    retained.append((generation, payload))
            except asyncio.QueueEmpty:
                break
        for item in retained:
            self.queue.put_nowait(item)
        if self.context_id:
            await self.emit({'type': 'response.cancelled', 'context_id': self.context_id,
                             'client_epoch': self.context_client_epoch})
            self.context_id = None
        if task and task is not asyncio.current_task():
            # Cancellation cleanup must not block receiving new microphone frames.
            task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
        if not resume_input:
            await self._set_input_enabled(False, force=True)
        await self.state('listening')
        if resume_input:
            await self._resume_input_if_idle(force=True)

    async def on_transcription_error(self, message: str):
        # A failed recognition is not a rejection of the last proposed action.
        await self.interrupt(clear_pending=False, resume_input=False,
                             preserve_ending=self.ending_context_id is not None)
        if self.transcriber and self.transcriber.ready:
            # A timed-out/local turn can fail while its transcription connection
            # remains usable. Do not tell the phone to disable microphone support.
            await self.error('transcription_retry', message)
            await self._resume_input_if_idle(force=True)
        else:
            await self.error('transcription_unavailable', message)

    async def on_transcript(self, event: Transcript):
        if self.closed or event.turn_id != self.input_turn or self.input_generation != self.runtime.generation:
            return
        await self.emit({'type': f'transcript.{event.kind}', 'text': event.text, 'turn_id': event.turn_id}, self.runtime.generation)
        if event.kind == 'final':
            self.input_turn = None
            if self.transcript_deadline_task:
                self.transcript_deadline_task.cancel()
                self.transcript_deadline_task = None
            if event.text:
                self._respond(event.text, event.turn_id)
            else:
                await self.error('empty_transcript', 'I did not catch that. Please try again.')
                await self.state('listening')
                await self._resume_input_if_idle()

    async def _transcript_deadline(self, turn_id, generation):
        await asyncio.sleep(27 if self.config.transcription_provider == 'local' else 12)
        if self.input_turn == turn_id and self.runtime.generation == generation:
            await self.on_transcription_error('Speech recognition timed out. Please repeat, or use text.')

    def _cancel_playback_deadline(self):
        task, self.playback_deadline_task = self.playback_deadline_task, None
        if task and task is not asyncio.current_task():
            task.cancel()

    def _watch_playback(self, context_id: str, generation: int, timeout: float):
        self._cancel_playback_deadline()
        if self.playback_pending and self.context_id == context_id and generation == self.runtime.generation:
            self.playback_deadline_task = asyncio.create_task(self._playback_deadline(context_id, generation, timeout))

    async def _playback_deadline(self, context_id: str, generation: int, timeout: float):
        await asyncio.sleep(timeout)
        if (not self.closed and self.playback_pending and self.context_id == context_id
                and self.runtime.generation == generation):
            # The phone must stop this context before input can reopen. A lost
            # completion callback must not leave a healthy connection muted forever.
            self._end_summary('playback_timeout')
            await self.error('playback_timeout', 'Playback completion was not confirmed. The reply was stopped; you can ask again.')
            await self.interrupt(resume_input=self.ending_context_id != context_id,
                                 preserve_ending=self.ending_context_id == context_id)

    def _respond(self, text: str, turn_id: str, *, source: str = 'voice'):
        self.runtime.speech_failure = None
        self.runtime.recent.append({'role': 'user', 'text': text, 'turn_id': turn_id, 'at': _now(), 'source': source})
        self.cancel_event = asyncio.Event()
        self.response_source = source
        generation, client_epoch = self.runtime.generation, self.client_epoch
        summary = _TurnSummary(turn_id, generation, client_epoch,
                               source if source in ('voice', 'text', 'automatic_scene') else 'unknown')
        self.turn_summary = summary
        self.runtime.turn_summaries.append(summary)
        self.response_task = asyncio.create_task(self._answer(text, turn_id, generation, client_epoch,
                                                              self.cancel_event, summary, source=source))

    def _sync_recipient(self, recipient: str):
        session = self.owner._session(self.session_id)
        if session.recipient_id != recipient:
            self.runtime.speech_failure = None
            from .models import ContextRequest
            self.owner.service.context(self.session_id, ContextRequest(recipient_id=recipient))

    async def _camera(self, generation: int, *, source: str = 'voice', summary: _TurnSummary | None = None):
        request_id = str(uuid4())
        started = time.monotonic()
        if self._current_summary(summary):
            summary.values['camera_request_id'] = request_id
        future = asyncio.get_running_loop().create_future()
        self.camera_pending[request_id] = (generation, future)
        try:
            await self.state('capturing', generation)
            await self.emit({'type': 'camera.request', 'request_id': request_id,
                **({'trigger': 'scene.stable'} if source == 'automatic_scene' else {})}, generation)
            async with asyncio.timeout(10):
                frame = await future
            if frame is not None and generation == self.runtime.generation and not self.closed:
                # Capture has finished; do not leave the phone saying that it
                # is taking a photo throughout recognition/search/wording.
                await self.state('thinking', generation)
                await self.emit({'type': 'turn.progress', 'stage': 'photo_received',
                                 'request_id': request_id}, generation)
            return frame
        except TimeoutError:
            await self.error('camera_timeout', 'No fresh camera view arrived. Hold the package steady and try again.')
            return None
        finally:
            if self._current_summary(summary):
                summary.values['camera_wait_ms'] = summary.elapsed(started)
            self.camera_pending.pop(request_id, None)

    async def _answer(self, text: str, turn_id: str, generation: int, client_epoch: int, cancel: asyncio.Event,
                      summary: _TurnSummary,
                      *, source: str = 'voice'):
        context_id = str(uuid4())
        speaker = None
        speech_factory_error = None
        preparation = None
        try:
            await self.state('thinking', generation)
            if self.owner.speech_available:
                try:
                    speaker = self.owner.speech_factory()
                    if callable(getattr(speaker, 'prepare', None)):
                        # Only register the configured voice while reasoning.
                        # No reply text reaches TTS before brain validation.
                        preparation = asyncio.create_task(speaker.prepare(cancel=cancel))
                except Exception as exc:
                    speech_factory_error = exc
            async with asyncio.timeout(50 if self.config.product_mode == 'product_search' else 30):
                frame_provider = lambda: self._camera(generation, source=source, summary=summary)
                if source == 'automatic_scene' and hasattr(self.runtime.brain, 'respond_automatically'):
                    result = await self.runtime.brain.respond_automatically(frame_provider=frame_provider)
                else:
                    result = await self.runtime.brain.respond(text, frame_provider=frame_provider)
            if self.closed or generation != self.runtime.generation:
                return
            reply = result.get('reply')
            if not isinstance(reply, str) or not reply.strip() or len(reply) > 4000:
                raise ValueError('Invalid brain reply')
            if self._current_summary(summary):
                status = result.get('status', 'ready')
                summary.values['status'] = status if status in ('ready', 'needs_image', 'needs_portion', 'needs_review',
                    'needs_confirmation', 'needs_clarification', 'needs_consent', 'unknown', 'cancelled', 'ignored') else 'unknown'
                summary.values['response_status'] = summary.values['status']
                summary.values['phase_timings'] = _phase_timings(result.get('events'))
            self._sync_recipient(self.runtime.brain.snapshot()['recipient_id'])
            if result.get('silent') is True and (source == 'automatic_scene' or result.get('status') == 'ignored'):
                if self._current_summary(summary):
                    summary.values['status'] = 'silent'
                # No playback context is created for unrelated speech or a
                # repeated scene prompt. Finally closes any prepared TTS socket
                # and releases input only after this response task finishes.
                await self.emit({'type': 'snapshot', 'snapshot': result.get('snapshot', self.runtime.brain.snapshot())}, generation)
                await self.state('listening', generation)
                return
            if hasattr(self.runtime.brain, 'cart'):
                from .models import CartItem
                session = self.owner._session(self.session_id)
                combined = {item.id: item for item in session.cart}
                for item in self.runtime.brain.cart:
                    mapped = CartItem(id=item['id'], product_id=item['product_id'], product_name=item['name'],
                        recipient_id=item['recipient_id'], quantity=item.get('quantity', 1),
                        confirmed_at=item.get('confirmed_at', _now()))
                    combined[mapped.id] = mapped
                session.cart = list(combined.values())[-100:]
            self.context_id, self.context_client_epoch = context_id, client_epoch
            if result.get('end_conversation') is True:
                self.ending_context_id = context_id
            self.context_source = source
            self.playback_pending = self.owner.speech_available or self.config.speech_provider == 'native'
            if self._current_summary(summary):
                summary.values['context_id'] = context_id
                summary.values['ending_response'] = self.ending_context_id == context_id
                summary.values['audio_output'] = ('cloud' if self.owner.speech_available else
                    'native' if self.config.speech_provider == 'native' else 'none')
                summary.values['playback_state'] = 'pending' if self.playback_pending else 'not_requested'
                summary.text_at = time.monotonic()
                summary.values['turn_to_text_ms'] = summary.elapsed()
            if not self.owner.speech_available and self.config.speech_provider == 'native':
                # Native speech length is unknown to the server; allow a slow
                # 100 words/minute plus startup margin before cancelling it.
                self._watch_playback(context_id, generation, max(20, len(reply.split()) * .6 + 10))
            self.runtime.recent.append({'role': 'assistant', 'text': reply, 'turn_id': turn_id, 'at': _now(), 'source': source})
            await self.emit({'type': 'response.text', 'text': reply, 'context_id': context_id,
                             'client_epoch': client_epoch,
                             **({'end_conversation': True} if self.ending_context_id == context_id else {})}, generation)
            await self.emit({'type': 'snapshot', 'snapshot': result.get('snapshot', self.runtime.brain.snapshot())}, generation)
            if self.owner.speech_available:
                try:
                    if speech_factory_error is not None:
                        from addons.pair_pipeline.voice_adapter import VoiceError
                        raise VoiceError('Speech configuration unavailable', category='configuration', stage='configuration')
                    if preparation is not None:
                        # Preparation is optional; a failed handshake can use
                        # the adapter's normal startup, never another provider.
                        await asyncio.gather(preparation, return_exceptions=True)
                    # Preserve every word; split long cloud speech at natural boundaries.
                    stream = _speech_chunks(speaker, context_id, reply, self.config.voice_style, cancel)
                    await self.state('speaking', generation)
                    first_audio = None
                    sent_bytes = 0
                    try:
                        async for chunk in stream:
                            if cancel.is_set() or generation != self.runtime.generation:
                                return
                            if first_audio is None:
                                first_audio = time.monotonic()
                            delay = first_audio + sent_bytes / 48000 - .2 - time.monotonic()
                            if delay > 0:
                                await asyncio.sleep(delay)
                            if cancel.is_set() or generation != self.runtime.generation:
                                return
                            await self.emit({'type': 'audio.chunk', 'context_id': context_id,
                                'client_epoch': client_epoch, 'sequence': chunk.sequence,
                                'audio': base64.b64encode(chunk.pcm).decode('ascii'), 'sample_rate': 24000}, generation)
                            if self._current_summary(summary):
                                if summary.values['text_to_first_audio_queued_ms'] is None:
                                    summary.values['text_to_first_audio_queued_ms'] = summary.elapsed(summary.text_at)
                                summary.values['pcm_bytes_queued'] += len(chunk.pcm)
                            sent_bytes += len(chunk.pcm)
                    finally:
                        await stream.aclose()
                    await self.emit({'type': 'audio.done', 'context_id': context_id, 'client_epoch': client_epoch}, generation)
                    if self._current_summary(summary):
                        summary.values['text_to_audio_done_queued_ms'] = summary.elapsed(summary.text_at)
                    if generation == self.runtime.generation:
                        self.runtime.speech_failure = None
                    self._watch_playback(context_id, generation, 15)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if self.closed or generation != self.runtime.generation:
                        return
                    self.playback_pending = False
                    diagnostic, hint = _speech_failure(exc, self.config.speech_provider)
                    self.runtime.speech_failure = diagnostic
                    if self._current_summary(summary):
                        summary.values['speech_failure'] = dict(diagnostic)
                        summary.values['playback_state'] = 'speech_failed'
                    await self.emit({'type': 'response.cancelled', 'context_id': context_id, 'client_epoch': client_epoch}, generation)
                    await self.error('speech_unavailable', hint)
            elif self.config.speech_provider != 'native':
                # Release the client's response context even when this reply
                # has no audio, before the authoritative input-enable event.
                await self.emit({'type': 'audio.done', 'context_id': context_id, 'client_epoch': client_epoch}, generation)
            await self.state('ending' if self.ending_context_id == context_id else 'listening', generation)
        except asyncio.CancelledError:
            raise
        except Exception:
            if not self.closed and generation == self.runtime.generation:
                if self._current_summary(summary):
                    summary.values['status'] = 'failed'
                self.playback_pending = False
                if self.ending_context_id == context_id:
                    await self.emit({'type': 'response.cancelled', 'context_id': context_id,
                                     'client_epoch': client_epoch}, generation)
                await self.error('turn_failed', 'That turn could not finish. Please try again.')
                await self.state('ending' if self.ending_context_id == context_id else 'listening', generation)
        finally:
            if preparation is not None:
                if not preparation.done():
                    preparation.cancel()
                await asyncio.gather(preparation, return_exceptions=True)
            if speaker is not None and callable(getattr(speaker, 'aclose', None)):
                try:
                    async with asyncio.timeout(3):
                        await speaker.aclose()
                except Exception:
                    pass
            if not self.closed and generation == self.runtime.generation and self.response_task is asyncio.current_task():
                self.response_task = None
                await self._resume_input_if_idle()

    async def shutdown(self, code: int = 1000):
        if self.closed:
            return
        self._end_summary('disconnected_unconfirmed')
        self.closed = True
        self.input_enabled = False
        self.runtime.generation += 1
        self.runtime.brain.cancel()
        self.cancel_event.set()
        current = asyncio.current_task()
        tasks = [t for t in (self.response_task, self.writer_task, self.transcript_deadline_task,
                            self.playback_deadline_task) if t and t is not current]
        for task in tasks:
            task.cancel()
        for _, future in self.camera_pending.values():
            if not future.done():
                future.cancel()
        self.camera_pending.clear()
        if self.transcriber:
            await self.transcriber.close()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self.runtime.connection is self:
            self.runtime.connection = None
            self.runtime.state, self.runtime.updated_at = 'disconnected', _now()
        try:
            await self.ws.close(code=code)
        except Exception:
            pass


def create_conversation_router(coordinator: ConversationCoordinator) -> APIRouter:
    router = APIRouter()

    @router.websocket('/api/conversation/ws')
    async def conversation(websocket: WebSocket, session_id: str):
        token = coordinator.config.api_token
        if token and not secrets.compare_digest(websocket.headers.get('authorization', ''), f'Bearer {token}'):
            await websocket.close(code=4401)
            return
        try:
            coordinator._session(session_id)
        except HTTPException:
            await websocket.close(code=4403)
            return
        await websocket.accept()
        try:
            connection = await coordinator.attach(websocket, session_id)
        except HTTPException:
            await websocket.close(code=4429)
            return
        await connection.run()

    return router


async def _speech_chunks(speaker, context_id, text, style, cancel):
    from addons.pair_pipeline.voice_adapter import SpeechRequest, PCMChunk
    sequence = 0
    remaining = text.strip()
    while remaining and not cancel.is_set():
        end = min(600, len(remaining))
        if end < len(remaining):
            boundary = max(remaining.rfind('. ', 0, end), remaining.rfind('? ', 0, end), remaining.rfind('! ', 0, end))
            end = boundary + 1 if boundary >= 100 else max(remaining.rfind(' ', 0, end), 1)
        part, remaining = remaining[:end].strip(), remaining[end:].strip()
        stream = speaker.stream(SpeechRequest(context_id, part, style), cancel=cancel)
        try:
            async for chunk in stream:
                yield PCMChunk(context_id, sequence, chunk.pcm, 24000)
                sequence += 1
        finally:
            await stream.aclose()


def _speech_failure(error, provider):
    from addons.pair_pipeline.voice_adapter import VoiceError
    diagnostic = (error.diagnostic(provider) if isinstance(error, VoiceError) else
                  VoiceError('', category='timeout' if isinstance(error, TimeoutError) else
                             'network' if isinstance(error, OSError) else 'unknown').diagnostic(provider))
    hints = {
        'configuration': 'Speech is not configured correctly. Check the server voice settings.',
        'authorization': 'Speech access was rejected. Check the server key and voice access.',
        'rate_limited': 'Speech service limit reached. Check account usage or try again shortly.',
        'balance': 'Speech account needs attention. Check billing or available credit.',
        'network': 'The speech connection failed. Please try again.',
        'timeout': 'The speech service timed out. Please try again.',
        'empty_response': 'The speech service returned no audio. Please try again.',
    }
    return diagnostic, hints.get(diagnostic['category'], 'Audio reply is unavailable. The full response is shown as text.')
