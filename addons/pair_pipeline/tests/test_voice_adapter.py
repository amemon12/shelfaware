"""No network/provider calls: contract, cancellation, framing and error tests."""
import asyncio
import base64
import json

import httpx
import pytest

from addons.pair_pipeline.voice_adapter import (
    ENDPOINT, MAX_EVENT_BYTES, MODEL, GeminiSpeechAdapter, SpeechRequest, VoiceError,
)


def sse(event):
    return b"data: " + json.dumps(event).encode() + b"\r\n\r\n"


def audio(data):
    return sse({"event_type": "step.delta", "delta": {"type": "audio", "data": base64.b64encode(data).decode()}})


DONE = sse({"event_type": "interaction.completed"})


class Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks, wait=None):
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


@pytest.mark.asyncio
async def test_exact_reply_payload_split_pcm_context_and_completion():
    seen = []
    source = audio(b"\x01\x00\x02") + audio(b"\x00\x03\x00") + DONE
    stream = Chunks([source[:17], source[17:41], source[41:]])
    def handler(request):
        seen.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        request = SpeechRequest("session1:reply1:g2", "For 30 g, this has 12 g of carbohydrate.")
        chunks = [part async for part in GeminiSpeechAdapter("secret-value", client).stream(request)]
    body = json.loads(seen[0].content)
    assert str(seen[0].url) == ENDPOINT
    assert body["model"] == MODEL
    assert body["input"][0]["content"][0]["text"] == request.text
    assert body["input"][0]["content"][0]["annotations"][0]["type"] == "speech_metadata"
    assert body["response_format"] == {"type": "audio", "mime_type": "audio/l16", "sample_rate": 24000}
    assert body["stream"] is True and body["store"] is False
    assert b"".join(c.pcm for c in chunks) == b"\x01\x00\x02\x00\x03\x00"
    assert [c.sequence for c in chunks] == list(range(len(chunks)))
    assert all(c.context_id == request.context_id and c.sample_rate == 24000 and c.channels == 1 for c in chunks)
    assert "secret-value" not in repr(chunks) and request.text not in repr(request)
    assert stream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("status,body", [(401, b"secret-value personal text"), (429, b"quota")])
async def test_http_error_redacts_response(status, body):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(status, content=body))) as client:
        with pytest.raises(VoiceError, match=f"HTTP {status}") as error:
            _ = [c async for c in GeminiSpeechAdapter("secret-value", client).stream(SpeechRequest("x", "Reply."))]
    assert "secret-value" not in str(error.value) and "personal text" not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [
    audio(b"\x00\x00"),  # Missing completion must not be reported as success.
    audio(b"\x00") + DONE,  # Incomplete PCM16 sample.
    sse({"event_type": "error", "error": {"message": "secret-value"}}),
    sse({"event_type": "step.delta", "delta": {"type": "audio", "data": "@@@"}}),
    audio(b"RIFFabcd") + DONE,
    b"data: " + b"x" * (MAX_EVENT_BYTES + 1),
], ids=["missing-completion", "odd-pcm-sample", "provider-error", "invalid-base64", "unexpected-wav", "oversize-event"])
async def test_bad_or_truncated_audio_is_rejected(payload):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=payload))) as client:
        with pytest.raises(VoiceError) as error:
            _ = [c async for c in GeminiSpeechAdapter("secret-value", client).stream(SpeechRequest("x", "Reply."))]
    assert "secret-value" not in str(error.value)


@pytest.mark.asyncio
async def test_cancel_event_interrupts_blocked_stream_and_closes_http():
    source = Chunks([], wait=asyncio.Event())
    cancel = asyncio.Event()
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=source))) as client:
        async def consume():
            return [c async for c in GeminiSpeechAdapter("key", client).stream(SpeechRequest("x", "Reply."), cancel=cancel)]
        task = asyncio.create_task(consume())
        await asyncio.wait_for(source.reading.wait(), 1)
        cancel.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
    assert source.closed


@pytest.mark.asyncio
async def test_task_cancellation_closes_http():
    source = Chunks([], wait=asyncio.Event())
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=source))) as client:
        async def consume():
            return [c async for c in GeminiSpeechAdapter("key", client).stream(SpeechRequest("x", "Reply."))]
        task = asyncio.create_task(consume())
        await asyncio.wait_for(source.reading.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert source.closed
