"""ElevenLabs Scribe streaming transcription with isolated, cancellable turns.

The provider's transcript messages have no client turn/item ID. Each utterance
therefore owns one socket; late finals can never bind to the next utterance.
An initial warm socket expires before the provider's 15-second idle limit.
No audio, transcript, key, or provider error body is logged.

Protocol: https://elevenlabs.io/docs/api-reference/speech-to-text/v-1-speech-to-text-realtime
Commit: https://github.com/elevenlabs/elevenlabs-python/blob/main/src/elevenlabs/realtime/connection.py
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from dataclasses import dataclass, field
from urllib.parse import urlencode

from .openai_transcription import Transcript, TranscriptionError

ENDPOINT = 'wss://api.elevenlabs.io/v1/speech-to-text/realtime'
MODEL = 'scribe_v2_realtime'
_logger = logging.getLogger('shelf_aware.private_elevenlabs_transport')
_logger.disabled = True
_logger.propagate = False


def connection_url(model=MODEL):
    if model != MODEL:
        raise ValueError('Unsupported ElevenLabs realtime transcription model')
    return ENDPOINT + '?' + urlencode({'model_id': model, 'audio_format': 'pcm_24000',
        'language_code': 'en', 'commit_strategy': 'manual', 'include_timestamps': 'false'})


async def _connect(url, key):
    from websockets.asyncio.client import connect
    return await connect(url, additional_headers={'xi-api-key': key}, open_timeout=8,
        close_timeout=2, ping_interval=20, ping_timeout=20,
        max_size=64 * 1024, max_queue=16, logger=_logger)


def audio_message(pcm: bytes, commit=False):
    return {'message_type': 'input_audio_chunk', 'audio_base_64': base64.b64encode(pcm).decode('ascii'),
            'sample_rate': 24000, 'commit': commit}


def _event(raw):
    if not isinstance(raw, str) or len(raw) > 64 * 1024:
        raise TranscriptionError('Invalid transcription event')
    event = json.loads(raw)
    if not isinstance(event, dict):
        raise TranscriptionError('Invalid transcription event')
    return event


@dataclass
class _Turn:
    turn_id: str
    generation: int
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=128))
    buffer: bytearray = field(default_factory=bytearray)
    byte_count: int = 0
    committed: bool = False
    commit_sent: bool = False
    socket: object | None = None
    task: asyncio.Task | None = None


class ElevenLabsTranscription:
    def __init__(self, api_key, on_event, on_error, *, model=MODEL, connector=None,
                 startup_timeout_s=8, warm_idle_s=10):
        self.key, self.on_event, self.on_error = api_key, on_event, on_error
        self.url = connection_url(model)
        self.connector = connector or _connect
        self.startup_timeout_s, self.warm_idle_s = startup_timeout_s, warm_idle_s
        self.ready = False
        self.closed = False
        self.generation = 0
        self.active_turn = None
        self.turn: _Turn | None = None
        self.warm_socket = None
        self.warm_timer = None
        self.tasks: set[asyncio.Task] = set()

    async def _open(self):
        socket = None
        try:
            async with asyncio.timeout(self.startup_timeout_s):
                socket = await self.connector(self.url, self.key)
                event = _event(await socket.recv())
                if event.get('message_type') != 'session_started':
                    raise TranscriptionError('Transcription session was rejected')
                config = event.get('config', {})
                if not isinstance(config, dict) or config.get('audio_format', 'pcm_24000') != 'pcm_24000':
                    raise TranscriptionError('Unexpected transcription audio format')
                return socket
        except BaseException:
            if socket is not None:
                await self._close_socket(socket)
            raise

    async def start(self):
        if not self.key:
            raise TranscriptionError('ElevenLabs transcription is not configured.')
        try:
            self.warm_socket = await self._open()
            self.ready = True
            self.warm_timer = asyncio.create_task(self._expire_warm())
        except asyncio.CancelledError:
            await self.close()
            raise
        except Exception:
            await self.close()
            raise TranscriptionError('ElevenLabs transcription could not start. Check the backend key and account access.') from None

    async def _expire_warm(self):
        await asyncio.sleep(self.warm_idle_s)
        socket, self.warm_socket = self.warm_socket, None
        if socket is not None:
            await self._close_socket(socket)

    def begin_turn(self, turn_id):
        if not self.ready or self.closed:
            raise TranscriptionError('ElevenLabs transcription is unavailable. Reconnect.')
        self.discard_turn()
        turn = _Turn(turn_id, self.generation)
        self.turn, self.active_turn = turn, turn_id
        if self.warm_timer:
            self.warm_timer.cancel()
            self.warm_timer = None
        turn.socket, self.warm_socket = self.warm_socket, None
        turn.task = asyncio.create_task(self._run(turn))
        self.tasks.add(turn.task)
        turn.task.add_done_callback(self.tasks.discard)

    def _put(self, turn, pcm):
        try:
            turn.queue.put_nowait(pcm)
        except asyncio.QueueFull:
            self.discard_turn()
            raise TranscriptionError('Speech upload is behind. Reconnect and repeat the question.') from None

    def append(self, pcm):
        turn = self.turn
        if turn is None or turn.committed:
            return
        if not isinstance(pcm, bytes) or len(pcm) % 2 or turn.byte_count + len(pcm) > 720000:
            self.discard_turn()
            raise TranscriptionError('Speech exceeded the 15-second audio limit.')
        turn.byte_count += len(pcm)
        turn.buffer.extend(pcm)
        # Provider recommends 0.1–1 second chunks; phone/VAD stays at 20 ms.
        while len(turn.buffer) >= 4800:
            self._put(turn, bytes(turn.buffer[:4800]))
            del turn.buffer[:4800]

    def commit(self, turn_id):
        turn = self.turn
        if turn is None or turn.turn_id != turn_id or turn.committed:
            return
        # Scribe begins processing after ~2 seconds. Pad short confirmations
        # with silence so a spoken "yes" is not stranded below that boundary.
        if turn.byte_count < 96000:
            self.append(bytes(96000 - turn.byte_count))
        if turn.buffer:
            self._put(turn, bytes(turn.buffer))
            turn.buffer.clear()
        turn.committed = True
        self.active_turn = None
        self._put(turn, None)

    def _current(self, turn):
        return not self.closed and self.turn is turn and turn.generation == self.generation

    async def _read(self, turn):
        async for raw in turn.socket:
            event = _event(raw)
            kind = event.get('message_type')
            if kind in ('partial_transcript', 'committed_transcript'):
                text = event.get('text')
                if not isinstance(text, str) or len(text) > 4000:
                    raise TranscriptionError('Invalid transcription text')
                if kind == 'committed_transcript' and not turn.commit_sent:
                    raise TranscriptionError('Unexpected transcript segment')
                if self._current(turn):
                    await self.on_event(Transcript('partial' if kind == 'partial_transcript' else 'final',
                        turn.turn_id, text.strip(), turn.turn_id))
                if kind == 'committed_transcript':
                    return
            elif kind == 'session_started':
                continue
            elif kind == 'warning':
                # No unsupported zero-retention request is made. Never forward
                # arbitrary provider prose as an application instruction.
                continue
            else:
                raise TranscriptionError('Transcription provider rejected or ended the turn')
        raise TranscriptionError('Transcription connection ended before a final result')

    async def _run(self, turn):
        reader = None
        try:
            if turn.socket is None:
                turn.socket = await self._open()
            if not self._current(turn):
                return
            reader = asyncio.create_task(self._read(turn))
            while True:
                queued = asyncio.create_task(turn.queue.get())
                try:
                    done, _ = await asyncio.wait((reader, queued), return_when=asyncio.FIRST_COMPLETED)
                    if reader in done:
                        await reader  # propagate provider error before more audio uploads
                        raise TranscriptionError('Transcript arrived before manual commit')
                    chunk = queued.result()
                finally:
                    if not queued.done():
                        queued.cancel()
                    await asyncio.gather(queued, return_exceptions=True)
                if chunk is None:
                    turn.commit_sent = True
                async with asyncio.timeout(3):
                    await turn.socket.send(json.dumps(audio_message(chunk or b'', commit=chunk is None)))
                if chunk is None:
                    break
            async with asyncio.timeout(12):
                await reader
        except asyncio.CancelledError:
            raise
        except Exception:
            if self._current(turn):
                self.ready = False
                await self.on_error('ElevenLabs transcription stopped. Check account access and reconnect, or use text.')
        finally:
            if reader and not reader.done():
                reader.cancel()
            if reader:
                await asyncio.gather(reader, return_exceptions=True)
            if turn.socket is not None:
                await self._close_socket(turn.socket)
                turn.socket = None
            turn.buffer.clear()
            while not turn.queue.empty():
                turn.queue.get_nowait()
            if self.turn is turn:
                self.turn = None
                self.active_turn = None

    def discard_turn(self):
        self.generation += 1
        turn, self.turn = self.turn, None
        self.active_turn = None
        if turn:
            turn.buffer.clear()
            while not turn.queue.empty():
                turn.queue.get_nowait()
            if turn.task and turn.task is not asyncio.current_task():
                turn.task.cancel()
            # A task cancelled before its first scheduling never enters its
            # finally block. Close its already-assigned warm socket here too.
            socket, turn.socket = turn.socket, None
            if socket is not None:
                closing = asyncio.create_task(self._close_socket(socket))
                self.tasks.add(closing)
                closing.add_done_callback(self.tasks.discard)

    @staticmethod
    async def _close_socket(socket):
        try:
            async with asyncio.timeout(2):
                await socket.close()
        except Exception:
            pass

    async def close(self):
        self.closed, self.ready = True, False
        self.discard_turn()
        if self.warm_timer:
            self.warm_timer.cancel()
            await asyncio.gather(self.warm_timer, return_exceptions=True)
            self.warm_timer = None
        if self.warm_socket is not None:
            await self._close_socket(self.warm_socket)
            self.warm_socket = None
        tasks = [task for task in self.tasks if task is not asyncio.current_task()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
