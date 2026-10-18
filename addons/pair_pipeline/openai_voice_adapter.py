"""Optional OpenAI speech bridge using the existing 24 kHz PCM contract.

Verified docs 2026-10-18:
https://developers.openai.com/api/docs/guides/text-to-speech
https://developers.openai.com/api/reference/resources/audio/subresources/speech/methods/create
https://developers.openai.com/api/docs/deprecations

This TTS model is deprecated, with announced shutdown on 2027-01-06.
It remains a small migration bridge for the hackathon, not a long-term API choice.
No endpoint, player, microphone or running application is changed by this module.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TypeVar

import httpx

from .voice_adapter import MAX_AUDIO_BYTES, OUTPUT_CHUNK_BYTES, PCMChunk, SpeechRequest, VoiceError

MODEL = "gpt-4o-mini-tts-2025-12-15"
ENDPOINT = "https://api.openai.com/v1/audio/speech"
VOICE = "cedar"
_STYLES = {
    "quiet": "Warm, quiet, natural conversational delivery. Clear words and numbers, without theatrical emphasis.",
    "neutral": "Warm, natural conversational delivery. Clear words and numbers.",
    "whisper": "Gentle whispering while keeping every word and number intelligible.",
}
_T = TypeVar("_T")


def request_payload(request: SpeechRequest) -> dict:
    """Send the approved text unchanged; delivery instructions contain no user data."""
    return {
        "model": MODEL,
        "input": request.text,
        "voice": VOICE,
        "instructions": _STYLES[request.style] + " Read the supplied text as written; do not add or omit words.",
        "response_format": "pcm",
        "stream_format": "audio",
    }


async def _await_or_cancel(
    operation: Awaitable[_T], cancel: asyncio.Event | None,
    cleanup: Callable[[_T], Awaitable[None]] | None = None,
) -> _T:
    """Cancel blocked headers/body; close a response delivered in the same race."""
    pending = asyncio.ensure_future(operation)
    cancellation = asyncio.create_task(cancel.wait()) if cancel is not None else None
    returned = False
    try:
        if cancel is not None and cancel.is_set():
            raise asyncio.CancelledError
        if cancellation is None:
            value = await pending
        else:
            await asyncio.wait((pending, cancellation), return_when=asyncio.FIRST_COMPLETED)
            if cancel.is_set():
                raise asyncio.CancelledError
            value = pending.result()
        returned = True
        return value
    finally:
        if not pending.done():
            pending.cancel()
        if cancellation is not None and not cancellation.done():
            cancellation.cancel()
        results = await asyncio.gather(pending, *([cancellation] if cancellation else []), return_exceptions=True)
        if not returned and cleanup is not None and not isinstance(results[0], BaseException):
            await cleanup(results[0])


def _check_format(response: httpx.Response) -> None:
    media_type, *parameters = response.headers.get("content-type", "").lower().split(";")
    # The PCM choice, not the generic MIME name, defines little-endian encoding.
    if media_type.strip() not in {"audio/pcm", "audio/x-pcm", "application/octet-stream"}:
        raise VoiceError("Speech provider did not return raw PCM.")
    for parameter in parameters:
        name, _, value = parameter.strip().partition("=")
        if (name in {"rate", "samplerate", "sample_rate"} and value != "24000") or (name == "channels" and value != "1"):
            raise VoiceError("Speech provider returned an unexpected audio format.")
    content_length = response.headers.get("content-length")
    if content_length is not None:
        try:
            size = int(content_length)
        except ValueError:
            raise VoiceError("Speech provider returned an invalid content length.") from None
        if size < 0 or size > MAX_AUDIO_BYTES:
            raise VoiceError("Speech exceeded the 30-second output limit.")


def _check_prefix(prefix: bytes) -> None:
    if prefix.startswith((b"RIFF", b"RF64", b"OggS", b"fLaC", b"ID3")) or prefix[4:8] == b"ftyp":
        raise VoiceError("Speech returned a container instead of raw PCM.")


class OpenAISpeechAdapter:
    def __init__(self, api_key: str, client: httpx.AsyncClient | None = None):
        if not api_key or not api_key.strip():
            raise VoiceError("OpenAI speech is not configured.")
        self._api_key = api_key.strip()
        self._client = client

    async def stream(self, request: SpeechRequest, *, cancel: asyncio.Event | None = None) -> AsyncIterator[PCMChunk]:
        """Backpressured PCM; consume and aclose in one task on stop/disconnect.

        No retries. The playback owner must reject stale context IDs, bound its
        queue and report actual playback; HTTP completion does not mean heard.
        """
        if cancel is not None and cancel.is_set():
            raise asyncio.CancelledError
        owned = self._client is None
        client = self._client or httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=5.0))
        response: httpx.Response | None = None
        total = 0
        sequence = 0
        pending = bytearray()
        checked_prefix = False
        try:
            async with asyncio.timeout(30):
                http_request = client.build_request(
                    "POST", ENDPOINT, json=request_payload(request),
                    headers={"Authorization": f"Bearer {self._api_key}", "Accept": "audio/pcm, application/octet-stream"},
                    timeout=httpx.Timeout(8.0, connect=5.0),
                )
                response = await _await_or_cancel(
                    client.send(http_request, stream=True, follow_redirects=False),
                    cancel, lambda value: value.aclose(),
                )
                if response.status_code != 200:
                    raise VoiceError(f"Speech provider returned HTTP {response.status_code}.")
                _check_format(response)
                parts = response.aiter_bytes(chunk_size=4096)
                try:
                    while True:
                        try:
                            block = await _await_or_cancel(anext(parts), cancel)
                        except StopAsyncIteration:
                            break
                        total += len(block)
                        if total > MAX_AUDIO_BYTES:
                            raise VoiceError("Speech exceeded the 30-second output limit.")
                        pending.extend(block)
                        if not checked_prefix:
                            if len(pending) < 12:
                                continue
                            _check_prefix(bytes(pending[:12]))
                            checked_prefix = True
                        usable = len(pending) - len(pending) % 2
                        for offset in range(0, usable, OUTPUT_CHUNK_BYTES):
                            if cancel is not None and cancel.is_set():
                                raise asyncio.CancelledError
                            yield PCMChunk(request.context_id, sequence, bytes(pending[offset:min(offset + OUTPUT_CHUNK_BYTES, usable)]))
                            sequence += 1
                        del pending[:usable]
                finally:
                    await parts.aclose()
                if not total or len(pending) % 2:
                    raise VoiceError("Speech ended before a complete audio response was received.")
                if pending:  # Valid response shorter than the prefix inspection window.
                    _check_prefix(bytes(pending))
                    if cancel is not None and cancel.is_set():
                        raise asyncio.CancelledError
                    yield PCMChunk(request.context_id, sequence, bytes(pending))
        except (httpx.HTTPError, TimeoutError):
            raise VoiceError("Speech connection failed or timed out.") from None
        finally:
            if response is not None:
                await response.aclose()
            if owned:
                await client.aclose()
