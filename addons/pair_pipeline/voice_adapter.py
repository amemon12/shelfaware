"""Optional Gemini TTS adapter; no imports from or mutation of the running app.

REST/SSE contract verified 2026-10-18 against:
https://ai.google.dev/gemini-api/docs/speech-generation
https://ai.google.dev/gemini-api/docs/streaming

The caller must supply an already-approved reply, authorize its session/context,
and consume/close the iterator in the same task. Cancellation never rewrites text.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal

import httpx

MODEL = "gemini-3.8-flash-lite-tts"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/interactions"
SAMPLE_RATE = 24_000
MAX_EVENT_BYTES = 256 * 1024
MAX_AUDIO_BYTES = SAMPLE_RATE * 2 * 30  # 30 seconds, mono PCM16
OUTPUT_CHUNK_BYTES = 4_800  # 100 ms; playback may split further
_STYLES = {
    "quiet": "Warm, quiet, conversational delivery. Clear numbers, no theatrical emphasis.",
    "neutral": "Warm, natural, concise conversational delivery. Clear numbers.",
    "whisper": "Gentle whispering, while keeping all words and numbers intelligible.",
}


class VoiceError(Exception):
    """Safe error: never contains provider response bodies, text, or credentials."""

    def __init__(self, message, *, category='unknown', stage='synthesis', http_status=None):
        super().__init__(message)
        self.category = category if category in ('configuration', 'authorization', 'rate_limited',
            'balance', 'http_error', 'network', 'timeout', 'empty_response', 'invalid_response') else 'unknown'
        self.stage = stage if stage in ('configuration', 'connect', 'synthesis', 'playback') else 'synthesis'
        self.http_status = http_status if type(http_status) is int and 100 <= http_status <= 599 else None

    def diagnostic(self, provider='unknown'):
        # Revalidate even caller-supplied attributes; never persist error text.
        bounded = VoiceError('', category=self.category, stage=self.stage, http_status=self.http_status)
        return {'provider': provider if provider in ('elevenlabs', 'openai', 'native') else 'unknown',
                'stage': bounded.stage, 'category': bounded.category, 'http_status': bounded.http_status}


@dataclass(frozen=True)
class SpeechRequest:
    context_id: str
    text: str = field(repr=False)
    style: Literal["quiet", "neutral", "whisper"] = "quiet"

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,120}", self.context_id):
            raise VoiceError("Invalid speech context identifier.")
        if not isinstance(self.text, str) or not 1 <= len(self.text.strip()) <= 600:
            raise VoiceError("Speech requires a validated short reply of at most 600 characters.")
        if self.style not in _STYLES:
            raise VoiceError("Unsupported speech style.")


@dataclass(frozen=True)
class PCMChunk:
    context_id: str
    sequence: int
    pcm: bytes = field(repr=False)
    sample_rate: int = SAMPLE_RATE
    channels: int = 1
    sample_width_bytes: int = 2
    encoding: str = "pcm_s16le"


def request_payload(request: SpeechRequest) -> dict:
    """Keep the exact reply text separate from optional delivery instructions."""
    return {
        "model": MODEL,
        "input": [{"type": "user_input", "content": [{
            "type": "text", "text": request.text,
            "annotations": [{"type": "speech_metadata", "style": _STYLES[request.style]}],
        }]}],
        "response_format": {"type": "audio", "mime_type": "audio/l16", "sample_rate": SAMPLE_RATE},
        "generation_config": {"speech_config": [{"voice": "Kore"}]},
        "stream": True,
        "store": False,
    }


async def _sse_events(response: httpx.Response) -> AsyncIterator[dict]:
    """Bound memory before decoding even a malformed SSE line/event."""
    pending = bytearray()
    data_lines: list[bytes] = []
    event_size = 0
    async for block in response.aiter_bytes(chunk_size=4096):
        pending.extend(block)
        while b"\n" in pending:
            raw, _, rest = pending.partition(b"\n")
            pending = bytearray(rest)
            line = bytes(raw).rstrip(b"\r")
            if len(line) > MAX_EVENT_BYTES:
                raise VoiceError("Speech event exceeded the size limit.")
            if not line:
                if data_lines:
                    try:
                        event = json.loads(b"\n".join(data_lines))
                    except (ValueError, UnicodeError):
                        raise VoiceError("Speech provider returned malformed event data.") from None
                    if not isinstance(event, dict):
                        raise VoiceError("Speech provider returned an invalid event.")
                    yield event
                data_lines = []
                event_size = 0
            elif line.startswith(b"data:"):
                value = line[5:].lstrip(b" ")
                event_size += len(value)
                if event_size > MAX_EVENT_BYTES:
                    raise VoiceError("Speech event exceeded the size limit.")
                data_lines.append(value)
        if len(pending) > MAX_EVENT_BYTES:
            raise VoiceError("Speech event exceeded the size limit.")
    # Never interpret an abruptly truncated final event as a completed response.
    if pending.strip() or data_lines:
        raise VoiceError("Speech stream ended with an incomplete event.")


async def _next_event(events: AsyncIterator[dict], cancel: asyncio.Event | None) -> dict:
    if cancel is None:
        return await anext(events)
    if cancel.is_set():
        raise asyncio.CancelledError
    reading = asyncio.create_task(anext(events))
    cancelling = asyncio.create_task(cancel.wait())
    try:
        await asyncio.wait((reading, cancelling), return_when=asyncio.FIRST_COMPLETED)
        if cancel.is_set():
            raise asyncio.CancelledError
        return reading.result()
    finally:
        for task in (reading, cancelling):
            if not task.done():
                task.cancel()
        await asyncio.gather(reading, cancelling, return_exceptions=True)


class GeminiSpeechAdapter:
    def __init__(self, api_key: str, client: httpx.AsyncClient | None = None):
        if not api_key or not api_key.strip():
            raise VoiceError("Gemini speech is not configured.")
        self._api_key = api_key.strip()
        self._client = client

    async def stream(self, request: SpeechRequest, *, cancel: asyncio.Event | None = None) -> AsyncIterator[PCMChunk]:
        """Backpressured PCM; cancel the task/event or aclose() on playback stop.

        This reports generation, never actual playback. The iPhone owns
        play.started/play.finished and must reject chunks from stale contexts.
        """
        if cancel is not None and cancel.is_set():
            raise asyncio.CancelledError
        owned = self._client is None
        client = self._client or httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=5.0))
        total = 0
        sequence = 0
        completed = False
        carry = b""
        try:
            async with asyncio.timeout(30):
                async with client.stream(
                    "POST", ENDPOINT, json=request_payload(request),
                    headers={"x-goog-api-key": self._api_key, "Accept": "text/event-stream"},
                    follow_redirects=False,
                    timeout=httpx.Timeout(8.0, connect=5.0),
                ) as response:
                    if response.status_code != 200:
                        raise VoiceError(f"Speech provider returned HTTP {response.status_code}.")
                    if "text/event-stream" not in response.headers.get("content-type", "").lower():
                        raise VoiceError("Speech provider did not return an event stream.")
                    events = _sse_events(response)
                    try:
                        while True:
                            try:
                                event = await _next_event(events, cancel)
                            except StopAsyncIteration:
                                break
                            kind = event.get("event_type")
                            if kind in {"error", "interaction.failed"}:
                                raise VoiceError("Speech provider could not complete synthesis.")
                            if kind == "interaction.completed":
                                completed = True
                                break
                            if kind != "step.delta":
                                continue
                            delta = event.get("delta")
                            if not isinstance(delta, dict) or delta.get("type") != "audio":
                                continue
                            if delta.get("mime_type", "audio/l16") not in {"audio/l16", "audio/l16;rate=24000"}:
                                raise VoiceError("Speech provider returned an unexpected audio format.")
                            try:
                                data = base64.b64decode(delta["data"], validate=True)
                            except (KeyError, ValueError, TypeError, binascii.Error):
                                raise VoiceError("Speech provider returned invalid audio bytes.") from None
                            total += len(data)
                            if total > MAX_AUDIO_BYTES:
                                raise VoiceError("Speech exceeded the 30-second output limit.")
                            if total == len(data) and data.startswith(b"RIFF"):
                                raise VoiceError("Speech returned a WAV container instead of raw PCM.")
                            data = carry + data
                            usable = len(data) - len(data) % 2
                            carry = data[usable:]
                            for offset in range(0, usable, OUTPUT_CHUNK_BYTES):
                                if cancel is not None and cancel.is_set():
                                    raise asyncio.CancelledError
                                yield PCMChunk(request.context_id, sequence, data[offset:min(offset + OUTPUT_CHUNK_BYTES, usable)])
                                sequence += 1
                    finally:
                        await events.aclose()
            if not completed or not total or carry:
                raise VoiceError("Speech ended before a complete audio response was received.")
        except (httpx.HTTPError, TimeoutError):
            raise VoiceError("Speech connection failed or timed out.") from None
        finally:
            if owned:
                await client.aclose()
