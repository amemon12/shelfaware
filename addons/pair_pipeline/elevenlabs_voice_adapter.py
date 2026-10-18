"""ElevenLabs HTTP speech stream; same 24 kHz PCM contract as the other adapters.

Verified 2026-10-18:
https://elevenlabs.io/docs/api-reference/text-to-speech/stream
https://elevenlabs.io/docs/overview/models

Flash prioritizes latency. Numbers/units are expanded deterministically, without
an LLM rewriting approved facts. This endpoint does not promise whisper control;
quiet/whisper styles retain the configured voice and device playback volume.
No retries, provider fallback, logging, raw-audio storage or zero-retention claim.
"""
from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TypeVar

import httpx

from .voice_adapter import MAX_AUDIO_BYTES, OUTPUT_CHUNK_BYTES, PCMChunk, SpeechRequest, VoiceError

MODEL = "eleven_flash_v2_5"
VOICE = "JBFqnCBsd6RMkjVDRZzb"  # Official example; account availability needs verification.
ENDPOINT = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
_T = TypeVar("_T")
_ONES = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
         "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen")
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


def _integer_words(value: int) -> str:
    if value < 20:
        return _ONES[value]
    if value < 100:
        return _TENS[value // 10] + (" " + _ONES[value % 10] if value % 10 else "")
    if value < 1000:
        return _ONES[value // 100] + " hundred" + (" " + _integer_words(value % 100) if value % 100 else "")
    return _integer_words(value // 1000) + " thousand" + (" " + _integer_words(value % 1000) if value % 1000 else "")


def _number_words(value: str) -> str:
    sign = "minus " if value.startswith("-") else "plus " if value.startswith("+") else ""
    value = value.lstrip("+-")
    whole, point, fraction = value.partition(".")
    # Do not turn an identifier with leading zeros into a different identifier.
    integer = " ".join(_ONES[int(n)] for n in whole) if len(whole) > 1 and whole.startswith("0") else _integer_words(int(whole or "0"))
    return sign + integer + (" point " + " ".join(_ONES[int(n)] for n in fraction) if point else "")


def speech_text(text: str) -> str:
    """Exact English quantity wording; no float conversion, rounding or advice.

    We expand quantity units and standalone decimal tokens. Other words, product
    names, punctuation and numeric content are preserved. This is English-only.
    """
    number = r"[+-]?(?:[0-9]{1,6}(?:\.[0-9]{1,16})?|\.[0-9]{1,16})"
    units = {"g": "grams", "mg": "milligrams", "ml": "milliliters"}
    text = re.sub(r"(?<![\w.,])(" + number + r")\s*(mg|mL|ml|g)\b",
        lambda m: _number_words(m[1]) + " " + units[m[2].lower()], text)
    text = re.sub(r"(?<![\w.,])([+-]?(?:[0-9]{1,6}\.[0-9]{1,16}|\.[0-9]{1,16}))(?![\w.]|[0-9])",
        lambda m: _number_words(m[1]), text)
    if len(text) > 4000:
        raise VoiceError("Expanded speech exceeded the text limit.")
    return text


def request_payload(request: SpeechRequest, model: str = MODEL) -> dict:
    # Do not force enterprise-only Flash normalization or zero-retention flags.
    # Speaker boost is disabled to avoid its documented extra processing cost.
    return {"text": speech_text(request.text), "model_id": model,
            "voice_settings": {"use_speaker_boost": False}}


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


def validate_configuration(api_key: str, voice_id: str) -> tuple[str, str]:
    if not isinstance(api_key, str) or not api_key.strip():
        raise VoiceError("ElevenLabs speech is not configured.")
    if len(api_key.strip()) > 4096 or any(not 33 <= ord(c) <= 126 for c in api_key.strip()):
        raise VoiceError("ElevenLabs speech credential has an invalid format.")
    if not isinstance(voice_id, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,100}", voice_id) is None:
        raise VoiceError("Invalid ElevenLabs voice identifier.")
    return api_key.strip(), voice_id


def create_elevenlabs_speech_adapter(api_key: str, *, voice_id: str = VOICE, model: str = MODEL):
    """Explicit model dispatch; no fallback between HTTP TTS and dialogue WS."""
    if model == "eleven_v4_turbo":
        from .elevenlabs_dialogue_adapter import ElevenLabsDialogueSpeechAdapter
        return ElevenLabsDialogueSpeechAdapter(api_key, voice_id=voice_id, model=model)
    return ElevenLabsSpeechAdapter(api_key, voice_id=voice_id, model=model)


class ElevenLabsSpeechAdapter:
    def __init__(self, api_key: str, client: httpx.AsyncClient | None = None, *, voice_id: str = VOICE, model: str = MODEL):
        key, voice_id = validate_configuration(api_key, voice_id)
        if model not in {"eleven_flash_v2_5", "eleven_multilingual_v2"}:
            raise VoiceError("Unsupported ElevenLabs streaming speech model.")
        self._api_key = key
        self.voice_id, self.model = voice_id, model
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
                    "POST", ENDPOINT.format(voice_id=self.voice_id), params={"output_format": "pcm_24000"}, json=request_payload(request, self.model),
                    headers={"xi-api-key": self._api_key, "Accept": "audio/pcm, application/octet-stream"},
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
