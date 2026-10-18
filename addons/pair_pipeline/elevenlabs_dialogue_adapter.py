"""Explicit Eleven v4 Turbo Text-to-Dialogue WebSocket, isolated per reply.

Protocol checked 2026-10-18:
https://elevenlabs.io/docs/eleven-api/guides/how-to/websockets/realtime-tdd
https://elevenlabs.io/docs/api-reference/text-to-dialogue/ttd-websocket

This is a separate API from Flash HTTP TTS. One socket registers one configured
voice, sends an already-approved reply, flushes, and requires is_final. There is
no provider/model fallback, retained dialogue, logging, or audio storage here.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import time
from collections.abc import AsyncIterator
from urllib.parse import urlencode

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from .elevenlabs_voice_adapter import (
    VOICE, _await_or_cancel, _check_prefix, speech_text, validate_configuration,
)
from .voice_adapter import MAX_AUDIO_BYTES, OUTPUT_CHUNK_BYTES, PCMChunk, SpeechRequest, VoiceError

MODEL = 'eleven_v4_turbo'
ENDPOINT = 'wss://api.elevenlabs.io/v1/text-to-dialogue/stream-input'
MAX_EVENT_BYTES = 2 * 1024 * 1024  # At most the complete bounded PCM response, base64 encoded.
MAX_EVENTS = 1024
PREPARED_IDLE_SECONDS = 15  # Below the provider's 20-second idle timeout.
_logger = logging.getLogger('shelf_aware.private_elevenlabs_dialogue')
_logger.disabled = True
_logger.propagate = False


class _NoRedirectConnect(connect):
    def process_redirect(self, exc):
        # Never forward the credential header to a redirected endpoint.
        return exc


async def _connect(url, key):
    return await _NoRedirectConnect(url, additional_headers={'xi-api-key': key},
        open_timeout=5, close_timeout=2, ping_interval=20, ping_timeout=20,
        max_size=MAX_EVENT_BYTES, max_queue=4, logger=_logger)


async def _close(socket):
    try:
        async with asyncio.timeout(2):
            await socket.close()
    except Exception:
        pass


async def _bounded(operation, seconds):
    async with asyncio.timeout(seconds):
        return await operation()


def _http_category(status):
    return ('authorization' if status in (401, 403) else 'rate_limited' if status == 429 else
            'balance' if status == 402 else 'http_error')


def _rejection(event):
    detail = event.get('error')
    code = event.get('error_code')
    if not isinstance(code, str) and isinstance(detail, dict):
        code = detail.get('code', detail.get('type', detail.get('status')))
    status = event.get('status_code')
    if type(status) is not int or not 100 <= status <= 599:
        status = None
    # Interpret only documented-style machine codes; never forward provider
    # message/body text. Unknown codes remain a generic invalid response.
    codes = {'invalid_api_key': 'authorization', 'unauthorized': 'authorization',
        'voice_access_denied': 'authorization', 'rate_limit_exceeded': 'rate_limited',
        'too_many_requests': 'rate_limited', 'concurrent_request_limit_exceeded': 'rate_limited',
        'quota_exceeded': 'rate_limited', 'insufficient_credits': 'balance',
        'voice_not_found': 'configuration', 'model_not_found': 'configuration'}
    category = _http_category(status) if status is not None else codes.get(code, 'invalid_response') if isinstance(code, str) else 'invalid_response'
    return VoiceError('Speech provider rejected the dialogue request.', category=category, http_status=status)


def _event(raw, voice_id):
    if not isinstance(raw, str) or len(raw) > MAX_EVENT_BYTES or len(raw.encode('utf-8')) > MAX_EVENT_BYTES:
        raise VoiceError('Speech provider returned an invalid dialogue event.')
    try:
        event = json.loads(raw)
    except (ValueError, UnicodeError):
        raise VoiceError('Speech provider returned malformed dialogue data.') from None
    if not isinstance(event, dict):
        raise VoiceError('Speech provider returned an invalid dialogue event.')
    if 'error' in event or 'error_code' in event or event.get('message_type') == 'error':
        raise _rejection(event)
    if event.get('voice_id', voice_id) != voice_id:
        raise VoiceError('Speech provider returned a different voice.')
    for key in ('is_final', 'is_final_audio_for_turn'):
        if key in event and type(event[key]) is not bool:
            raise VoiceError('Speech provider returned invalid completion data.')
    # The API's selected output_format defines PCM16 little endian. If optional
    # format metadata is supplied, it must agree rather than being ignored.
    for key, expected in (('sample_rate', 24000), ('channels', 1), ('encoding', 'pcm_s16le')):
        if key in event and (type(event[key]) is not type(expected) or event[key] != expected):
            raise VoiceError('Speech provider returned an unexpected audio format.')
    encoded = event.get('audio')
    if encoded is not None:
        if not isinstance(encoded, str) or not encoded or len(encoded) > MAX_EVENT_BYTES:
            raise VoiceError('Speech provider returned invalid audio data.')
        try:
            audio = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            raise VoiceError('Speech provider returned invalid audio data.') from None
        if not audio:
            raise VoiceError('Speech provider returned empty audio data.')
    else:
        audio = b''
    if not audio and event.get('is_final') is not True and event.get('is_final_audio_for_turn') is not True:
        raise VoiceError('Speech provider returned an unsupported dialogue event.')
    return audio, event.get('is_final') is True


class ElevenLabsDialogueSpeechAdapter:
    def __init__(self, api_key: str, *, voice_id: str = VOICE, model: str = MODEL, connector=None):
        self._api_key, self.voice_id = validate_configuration(api_key, voice_id)
        if model != MODEL:
            raise VoiceError('Unsupported ElevenLabs dialogue speech model.', category='configuration', stage='configuration')
        self.model = model
        self._connector = connector or _connect
        self._prepare_lock = asyncio.Lock()
        self._prepared_socket = None
        self._prepared_at = 0.0
        self._expiry_task = None
        self._closed = False

    def _url(self):
        return ENDPOINT + '?' + urlencode({'model_id': self.model, 'output_format': 'pcm_24000'})

    async def _open_registered(self, cancel):
        socket = await _await_or_cancel(self._connector(self._url(), self._api_key), cancel, _close)
        try:
            await _await_or_cancel(_bounded(lambda: socket.send(json.dumps({'voices': [self.voice_id]})), 3), cancel)
            return socket
        except BaseException:
            await _close(socket)
            raise

    async def prepare(self, *, cancel: asyncio.Event | None = None) -> bool:
        """Warm one reply's handshake without text or audio generation.

        This is opportunistic: failure leaves normal stream startup available.
        A socket is never reused after it has synthesized any text.
        """
        async with self._prepare_lock:
            if self._closed or (cancel is not None and cancel.is_set()):
                return False
            if self._prepared_socket is not None:
                return True
            try:
                async with asyncio.timeout(5):
                    socket = await self._open_registered(cancel)
            except asyncio.CancelledError:
                raise
            except Exception:
                return False
            self._prepared_socket = socket
            self._prepared_at = time.monotonic()
            self._expiry_task = asyncio.create_task(self._expire_prepared(socket))
            return True

    async def _expire_prepared(self, socket):
        await asyncio.sleep(PREPARED_IDLE_SECONDS)
        async with self._prepare_lock:
            if self._prepared_socket is socket:
                self._prepared_socket = None
                await _close(socket)
            if self._expiry_task is asyncio.current_task():
                self._expiry_task = None

    async def _take_prepared(self):
        async with self._prepare_lock:
            socket, self._prepared_socket = self._prepared_socket, None
            expiry, self._expiry_task = self._expiry_task, None
            try:
                if expiry is not None:
                    expiry.cancel()
                    await asyncio.gather(expiry, return_exceptions=True)
                if socket is not None and time.monotonic() - self._prepared_at >= PREPARED_IDLE_SECONDS:
                    await _close(socket)
                    socket = None
                return socket
            except BaseException:
                if socket is not None:
                    await _close(socket)
                raise

    async def aclose(self):
        # The caller cancels/awaits prepare before closing. The lock also makes
        # direct concurrent callers safe without abandoning a late handshake.
        async with self._prepare_lock:
            self._closed = True
        socket = await self._take_prepared()
        if socket is not None:
            await _close(socket)

    async def stream(self, request: SpeechRequest, *, cancel: asyncio.Event | None = None) -> AsyncIterator[PCMChunk]:
        if cancel is not None and cancel.is_set():
            raise asyncio.CancelledError
        text = speech_text(request.text)
        socket = None
        total = sequence = 0
        pending = bytearray()
        checked_prefix = False
        final = False
        stage = 'connect'
        try:
            async with asyncio.timeout(30):
                if self._closed:
                    raise VoiceError('Speech adapter is closed.')
                socket = await self._take_prepared()
                if socket is None:
                    socket = await self._open_registered(cancel)
                stage = 'synthesis'
                # Credentials stay in the handshake header, never the URL or text.
                for payload in ({'inputs': [{'text': text, 'voice_id': self.voice_id, 'new_turn': False}]},
                        {'close_socket': True}):
                    await _await_or_cancel(_bounded(lambda: socket.send(json.dumps(payload)), 3), cancel)
                for _ in range(MAX_EVENTS):
                    raw = await _await_or_cancel(_bounded(socket.recv, 8), cancel)
                    block, final = _event(raw, self.voice_id)
                    total += len(block)
                    if total > MAX_AUDIO_BYTES:
                        raise VoiceError('Speech exceeded the 30-second output limit.')
                    pending.extend(block)
                    if not checked_prefix and (len(pending) >= 12 or final):
                        _check_prefix(bytes(pending[:12]))
                        checked_prefix = True
                    if checked_prefix:
                        usable = len(pending) - len(pending) % 2
                        for offset in range(0, usable, OUTPUT_CHUNK_BYTES):
                            if cancel is not None and cancel.is_set():
                                raise asyncio.CancelledError
                            yield PCMChunk(request.context_id, sequence,
                                bytes(pending[offset:min(offset + OUTPUT_CHUNK_BYTES, usable)]))
                            sequence += 1
                        del pending[:usable]
                    if final:
                        break
                if final and not total:
                    raise VoiceError('Speech provider returned no audio.', category='empty_response')
                if not final or pending:
                    raise VoiceError('Speech ended before a complete audio response was received.')
        except VoiceError:
            raise
        except Exception as exc:
            status = getattr(getattr(exc, 'response', None), 'status_code', None)
            if type(status) is int and 100 <= status <= 599:
                raise VoiceError(f'Speech provider returned HTTP {status}.', category=_http_category(status),
                                 stage=stage, http_status=status) from None
            code = getattr(getattr(exc, 'rcvd', None), 'code', None)
            category = ('empty_response' if isinstance(exc, ConnectionClosed) and code == 1000 and not total else
                        'timeout' if isinstance(exc, TimeoutError) else
                        'network' if isinstance(exc, (ConnectionClosed, OSError)) else 'invalid_response')
            raise VoiceError('Speech dialogue connection failed or timed out.', category=category, stage=stage) from None
        finally:
            if socket is not None:
                await _close(socket)
