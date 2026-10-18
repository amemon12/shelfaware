"""Bounded transcription-only WebSocket. No provider text or audio is logged.

https://developers.openai.com/api/docs/guides/realtime-transcription
Final ordering is resolved through committed item IDs, never arrival order.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

ENDPOINT = 'wss://api.openai.com/v1/realtime?intent=transcription'
_logger = logging.getLogger('shelf_aware.private_realtime_transport')
_logger.disabled = True
_logger.propagate = False


class TranscriptionError(Exception):
    pass


@dataclass(frozen=True)
class Transcript:
    kind: str
    turn_id: str
    text: str
    item_id: str


def session_configuration(model: str = 'gpt-live-transcribe') -> dict:
    return {'type': 'session.update', 'session': {'type': 'transcription', 'audio': {'input': {
        'format': {'type': 'audio/pcm', 'rate': 24000},
        'transcription': {'model': model, 'delay': 'low', 'languages': ['en']},
        'turn_detection': None,
    }}}}


async def _connect(key: str):
    from websockets.asyncio.client import connect
    return await connect(ENDPOINT, additional_headers={'Authorization': f'Bearer {key}'},
                         open_timeout=8, close_timeout=2, ping_interval=20, ping_timeout=20,
                         max_size=128 * 1024, max_queue=16, logger=_logger)


class OpenAITranscription:
    def __init__(self, api_key: str, on_event: Callable[[Transcript], Awaitable[None]],
                 on_error: Callable[[str], Awaitable[None]], *, model: str = 'gpt-live-transcribe',
                 connector=None, startup_timeout_s: float = 8):
        self.key = api_key
        self.on_event, self.on_error = on_event, on_error
        self.model, self.connector = model, connector or _connect
        self.startup_timeout_s = startup_timeout_s
        self.ws = None
        self.ready = False
        self.closed = False
        self.active_turn: str | None = None
        self.commits: deque[str] = deque()
        self.item_turns: OrderedDict[str, str] = OrderedDict()
        self.pending_events: dict[str, list[dict]] = {}
        self.texts: dict[str, str] = {}
        self.finished: set[str] = set()
        self.queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=128)
        self.tasks: list[asyncio.Task] = []
        self.ready_event = asyncio.Event()
        self.failed = False

    async def start(self) -> None:
        if not self.key:
            raise TranscriptionError('Transcription is not configured.')
        try:
            async with asyncio.timeout(self.startup_timeout_s):
                self.ws = await self.connector(self.key)
                await self.ws.send(json.dumps(session_configuration(self.model)))
                self.tasks = [asyncio.create_task(self._writer()), asyncio.create_task(self._reader())]
                await self.ready_event.wait()
                if not self.ready:
                    raise TranscriptionError('Transcription could not start.')
        except asyncio.CancelledError:
            await self.close()
            raise
        except Exception:
            await self.close()
            raise TranscriptionError('Transcription could not connect. Text input remains available.') from None

    def _enqueue(self, event: dict) -> None:
        if not self.ready or self.closed:
            raise TranscriptionError('Transcription is unavailable. Reconnect to resume microphone input.')
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            raise TranscriptionError('Audio upload is behind. Stop and reconnect.') from None

    def begin_turn(self, turn_id: str) -> None:
        if self.active_turn is not None:
            self.discard_turn()
        self.active_turn = turn_id

    def append(self, pcm: bytes) -> None:
        if self.active_turn is None:
            return
        self._enqueue({'type': 'input_audio_buffer.append', 'audio': base64.b64encode(pcm).decode('ascii')})

    def commit(self, turn_id: str) -> None:
        if self.active_turn != turn_id:
            return
        if len(self.commits) >= 16:
            raise TranscriptionError('Too many unacknowledged audio turns. Reconnect.')
        self.commits.append(turn_id)
        try:
            self._enqueue({'type': 'input_audio_buffer.commit', 'event_id': f'commit_{turn_id}'})
        except TranscriptionError:
            self.commits.pop()
            raise
        self.active_turn = None

    def discard_turn(self) -> None:
        if self.active_turn is not None:
            self.active_turn = None
            self._enqueue({'type': 'input_audio_buffer.clear'})

    async def _fail(self) -> None:
        self.ready = False
        self.ready_event.set()
        if not self.closed and not self.failed:
            self.failed = True
            await self.on_error('Transcription connection ended. Reconnect, or use text input.')
            await self.close()

    async def _writer(self):
        try:
            while not self.closed:
                event = await self.queue.get()
                async with asyncio.timeout(3):
                    await self.ws.send(json.dumps(event))
        except asyncio.CancelledError:
            raise
        except Exception:
            await self._fail()

    async def _reader(self):
        try:
            async for raw in self.ws:
                if not isinstance(raw, str) or len(raw) > 128 * 1024:
                    raise TranscriptionError('Invalid upstream event')
                event = json.loads(raw)
                if not isinstance(event, dict):
                    raise TranscriptionError('Invalid upstream event')
                kind = event.get('type')
                if kind == 'session.updated':
                    self.ready = True
                    self.ready_event.set()
                elif kind == 'error':
                    raise TranscriptionError('Upstream rejected the transcription session')
                elif kind == 'input_audio_buffer.committed':
                    item_id = event.get('item_id')
                    if not isinstance(item_id, str) or len(item_id) > 200:
                        raise TranscriptionError('Invalid committed item')
                    if item_id in self.item_turns:
                        continue
                    if not self.commits:
                        raise TranscriptionError('Unmatched audio commit')
                    self.item_turns[item_id] = self.commits.popleft()
                    while len(self.item_turns) > 64:
                        old, _ = self.item_turns.popitem(last=False)
                        self.texts.pop(old, None)
                        self.finished.discard(old)
                    for buffered in self.pending_events.pop(item_id, []):
                        await self._transcript(buffered)
                elif kind in ('conversation.item.input_audio_transcription.delta', 'conversation.item.input_audio_transcription.completed'):
                    await self._transcript(event)
                elif kind == 'conversation.item.input_audio_transcription.failed':
                    raise TranscriptionError('Transcription failed')
            if not self.closed:
                await self._fail()
        except asyncio.CancelledError:
            raise
        except Exception:
            await self._fail()

    async def _transcript(self, event: dict) -> None:
        item_id = event.get('item_id')
        if not isinstance(item_id, str) or len(item_id) > 200:
            raise TranscriptionError('Invalid transcript item')
        value = event.get('delta') if event['type'].endswith('.delta') else event.get('transcript')
        if not isinstance(value, str) or len(value) > 4000:
            raise TranscriptionError('Transcript exceeded bounds')
        if item_id not in self.item_turns:
            # Never guess which utterance an unbound delta belongs to. In some
            # upstream schedules partial captions wait until the commit ack.
            if len(self.pending_events) >= 16 and item_id not in self.pending_events:
                raise TranscriptionError('Too many unbound transcript items')
            pending = self.pending_events.setdefault(item_id, [])
            if len(pending) >= 64 or sum(len(str(e.get('delta', e.get('transcript', '')))) for e in pending) + len(value) > 8000:
                raise TranscriptionError('Unbound transcript exceeded bounds')
            pending.append(event)
            return
        if item_id in self.finished:
            return
        turn_id = self.item_turns[item_id]
        if event['type'].endswith('.delta'):
            text = self.texts.get(item_id, '') + value
            if len(text) > 4000:
                raise TranscriptionError('Transcript exceeded bounds')
            self.texts[item_id] = text
            await self.on_event(Transcript('partial', turn_id, text, item_id))
        else:
            self.finished.add(item_id)
            self.texts.pop(item_id, None)
            await self.on_event(Transcript('final', turn_id, value.strip(), item_id))

    async def close(self) -> None:
        self.closed = True
        self.ready = False
        current = asyncio.current_task()
        tasks = [task for task in self.tasks if task is not current]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self.ws is not None:
            try:
                async with asyncio.timeout(2):
                    await self.ws.close()
            except Exception:
                pass
        self.pending_events.clear()
        self.texts.clear()
        self.commits.clear()
