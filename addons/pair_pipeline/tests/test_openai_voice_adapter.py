"""Offline transport tests; no API key, paid request or real audio is used."""
import asyncio
import json

import httpx
import pytest

from addons.pair_pipeline.openai_voice_adapter import ENDPOINT, MODEL, VOICE, OpenAISpeechAdapter
from addons.pair_pipeline.voice_adapter import MAX_AUDIO_BYTES, OUTPUT_CHUNK_BYTES, SpeechRequest, VoiceError


class Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks=(), wait=None):
        self.chunks = chunks
        self.wait = wait
        self.closed = False
        self.reading = asyncio.Event()

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        self.reading.set()
        if self.wait:
            await self.wait.wait()

    async def aclose(self):
        self.closed = True


async def collect(adapter, cancel=None):
    return [c async for c in adapter.stream(SpeechRequest("session:turn:g2", "For 30 g, this has 12 g of carbohydrate."), cancel=cancel)]


@pytest.mark.asyncio
async def test_exact_payload_context_pcm_and_borrowed_client():
    seen = []
    pcm = b"\x01\x00" * 5_000
    source = Chunks([pcm[:1], pcm[1:301], pcm[301:]])
    def handler(request):
        seen.append(request)
        return httpx.Response(200, headers={"content-type": "audio/pcm"}, stream=source)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await collect(OpenAISpeechAdapter("secret-value", client))
        assert not client.is_closed
    payload = json.loads(seen[0].content)
    assert str(seen[0].url) == ENDPOINT
    assert seen[0].headers["authorization"] == "Bearer secret-value"
    assert payload["model"] == MODEL and payload["voice"] == VOICE
    assert payload["input"] == "For 30 g, this has 12 g of carbohydrate."
    assert payload["response_format"] == "pcm" and payload["stream_format"] == "audio"
    assert payload["input"] not in payload["instructions"]
    assert b"".join(c.pcm for c in result) == pcm
    assert [c.sequence for c in result] == list(range(len(result)))
    assert all(c.context_id == "session:turn:g2" and c.encoding == "pcm_s16le" and c.sample_rate == 24_000 and c.channels == 1 for c in result)
    assert all(len(c.pcm) % 2 == 0 and len(c.pcm) <= OUTPUT_CHUNK_BYTES for c in result)
    assert source.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [302, 401, 429])
async def test_errors_redacted_redirects_not_followed(status):
    seen = []
    def handler(request):
        seen.append(request)
        return httpx.Response(status, headers={"location": "https://example.invalid/steal"}, content=b"secret-value private-reply")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
        with pytest.raises(VoiceError, match=f"HTTP {status}") as error:
            await collect(OpenAISpeechAdapter("secret-value", client))
    assert len(seen) == 1
    assert "secret-value" not in str(error.value) and "private-reply" not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("mime,payload", [
    ("audio/wav", b"\x00\x00"),
    ("text/event-stream", b"data: hello"),
    ("audio/pcm;rate=16000", b"\x00\x00"),
    ("audio/pcm;channels=2", b"\x00\x00"),
    ("audio/pcm", b"RIFF1234WAVE"),
    ("audio/pcm", b"ID3abc"),
    ("audio/pcm", b""),
    ("audio/pcm", b"\x00"),
], ids=["wav-mime", "sse", "wrong-rate", "stereo", "wav-container", "mp3-container", "empty", "odd-sample"])
async def test_invalid_audio_rejected(mime, payload):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, headers={"content-type": mime}, content=payload))) as client:
        with pytest.raises(VoiceError):
            await collect(OpenAISpeechAdapter("key", client))


@pytest.mark.asyncio
async def test_small_valid_pcm_and_output_size_limit():
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, headers={"content-type": "application/octet-stream"}, content=b"\x01\x00"))) as client:
        assert b"".join(c.pcm for c in await collect(OpenAISpeechAdapter("key", client))) == b"\x01\x00"
    source = Chunks([b"\x00\x00" * (MAX_AUDIO_BYTES // 2 + 1)])
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, headers={"content-type": "audio/pcm"}, stream=source))) as client:
        with pytest.raises(VoiceError, match="30-second"):
            await collect(OpenAISpeechAdapter("key", client))
    assert source.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["event", "task"])
async def test_cancel_blocked_body_closes_response(method):
    source = Chunks(wait=asyncio.Event())
    cancel = asyncio.Event()
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, headers={"content-type": "audio/pcm"}, stream=source))) as client:
        task = asyncio.create_task(collect(OpenAISpeechAdapter("key", client), cancel))
        await asyncio.wait_for(source.reading.wait(), 1)
        cancel.set() if method == "event" else task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
    assert source.closed


@pytest.mark.asyncio
async def test_cancel_while_waiting_for_headers():
    started = asyncio.Event()
    stopped = asyncio.Event()
    cancel = asyncio.Event()
    async def handler(request):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        task = asyncio.create_task(collect(OpenAISpeechAdapter("key", client), cancel))
        await asyncio.wait_for(started.wait(), 1)
        cancel.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
    assert stopped.is_set()


@pytest.mark.asyncio
async def test_cancel_header_response_race_closes_response():
    cancel = asyncio.Event()
    source = Chunks([b"\x00\x00"])
    async def handler(request):
        cancel.set()
        return httpx.Response(200, headers={"content-type": "audio/pcm"}, stream=source)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(asyncio.CancelledError):
            await collect(OpenAISpeechAdapter("key", client), cancel)
    assert source.closed


@pytest.mark.asyncio
async def test_aclose_after_first_chunk_closes_response():
    source = Chunks([b"\x00\x00" * 5000], wait=asyncio.Event())
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, headers={"content-type": "audio/pcm"}, stream=source))) as client:
        stream = OpenAISpeechAdapter("key", client).stream(SpeechRequest("x", "Reply."))
        await anext(stream)
        await stream.aclose()
        assert not client.is_closed
    assert source.closed


@pytest.mark.asyncio
async def test_transport_failure_is_safe():
    def handler(request):
        raise httpx.ReadError("secret-value private-reply", request=request)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(VoiceError) as error:
            await collect(OpenAISpeechAdapter("key", client))
    assert str(error.value) == "Speech connection failed or timed out."
