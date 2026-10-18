"""Offline transport tests; no API key, paid request or real audio is used."""
import asyncio
import json

import httpx
import pytest

from addons.pair_pipeline.elevenlabs_voice_adapter import ENDPOINT, MODEL, VOICE, ElevenLabsSpeechAdapter, speech_text
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
        result = await collect(ElevenLabsSpeechAdapter("secret-value", client))
        assert not client.is_closed
    payload = json.loads(seen[0].content)
    assert str(seen[0].url) == ENDPOINT.format(voice_id=VOICE) + "?output_format=pcm_24000"
    assert seen[0].headers["xi-api-key"] == "secret-value"
    assert "authorization" not in seen[0].headers
    assert payload["model_id"] == MODEL
    assert payload["text"] == "For thirty grams, this has twelve grams of carbohydrate."
    assert payload["voice_settings"] == {"use_speaker_boost": False}
    assert "apply_text_normalization" not in payload and "enable_logging" not in seen[0].url.params
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
            await collect(ElevenLabsSpeechAdapter("secret-value", client))
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
            await collect(ElevenLabsSpeechAdapter("key", client))


@pytest.mark.asyncio
async def test_small_valid_pcm_and_output_size_limit():
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, headers={"content-type": "application/octet-stream"}, content=b"\x01\x00"))) as client:
        assert b"".join(c.pcm for c in await collect(ElevenLabsSpeechAdapter("key", client))) == b"\x01\x00"
    source = Chunks([b"\x00\x00" * (MAX_AUDIO_BYTES // 2 + 1)])
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, headers={"content-type": "audio/pcm"}, stream=source))) as client:
        with pytest.raises(VoiceError, match="30-second"):
            await collect(ElevenLabsSpeechAdapter("key", client))
    assert source.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["event", "task"])
async def test_cancel_blocked_body_closes_response(method):
    source = Chunks(wait=asyncio.Event())
    cancel = asyncio.Event()
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, headers={"content-type": "audio/pcm"}, stream=source))) as client:
        task = asyncio.create_task(collect(ElevenLabsSpeechAdapter("key", client), cancel))
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
        task = asyncio.create_task(collect(ElevenLabsSpeechAdapter("key", client), cancel))
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
            await collect(ElevenLabsSpeechAdapter("key", client), cancel)
    assert source.closed


@pytest.mark.asyncio
async def test_aclose_after_first_chunk_closes_response():
    source = Chunks([b"\x00\x00" * 5000], wait=asyncio.Event())
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, headers={"content-type": "audio/pcm"}, stream=source))) as client:
        stream = ElevenLabsSpeechAdapter("key", client).stream(SpeechRequest("x", "Reply."))
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
            await collect(ElevenLabsSpeechAdapter("key", client))
    assert str(error.value) == "Speech connection failed or timed out."


@pytest.mark.parametrize('voice', ['../steal', 'https://attacker.invalid', 'voice?query=secret', 'voice#frag', 'voice/name', '', 'x' * 101])
def test_voice_path_cannot_change_endpoint(voice):
    with pytest.raises(VoiceError, match='voice identifier'):
        ElevenLabsSpeechAdapter('secret-value', voice_id=voice)


@pytest.mark.parametrize('key', [None, '', 'secret\nvalue', 'secret\rvalue', 'private-\u2603', 'x' * 4097])
def test_invalid_credential_error_is_sanitized(key):
    with pytest.raises(VoiceError) as error:
        ElevenLabsSpeechAdapter(key)
    assert 'secret' not in str(error.value) and 'private' not in str(error.value)


@pytest.mark.asyncio
async def test_configured_voice_and_quality_model_are_explicit_no_fallback():
    seen = []
    def handler(request):
        seen.append(request)
        return httpx.Response(200, headers={'content-type': 'application/octet-stream'}, content=b'\x01\x00')
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await collect(ElevenLabsSpeechAdapter('test-only', client, voice_id='ActualAccountVoice123', model='eleven_multilingual_v2'))
    assert len(seen) == 1
    assert seen[0].url.path == '/v1/text-to-speech/ActualAccountVoice123/stream'
    assert json.loads(seen[0].content)['model_id'] == 'eleven_multilingual_v2'
    with pytest.raises(VoiceError, match='Unsupported'):
        ElevenLabsSpeechAdapter('test-only', model='eleven_v4_turbo')  # Different API, not silently interchangeable.


def test_exact_decimal_units_no_rounding_or_ingredient_rewriting():
    assert speech_text('For 330mL: 8.91 versus 6.6 grams, sodium 12mg.') == (
        'For three hundred thirty milliliters: eight point nine one versus six point six grams, sodium twelve milligrams.')
    assert speech_text('At .5g and -1.05 g, Model2 contains 001g.') == (
        'At zero point five grams and minus one point zero five grams, Model2 contains zero zero one grams.')
    assert speech_text('No winner: unknown fiber. Product B12g. 1,000g label.') == (
        'No winner: unknown fiber. Product B12g. 1,000g label.')
    assert speech_text('0.0000000000000001 grams') == 'zero point zero zero zero zero zero zero zero zero zero zero zero zero zero zero zero one grams'


@pytest.mark.asyncio
async def test_precancelled_request_sends_nothing_and_does_not_claim_whisper():
    cancel = asyncio.Event()
    cancel.set()
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: pytest.fail('cancelled request was sent'))) as client:
        with pytest.raises(asyncio.CancelledError):
            await collect(ElevenLabsSpeechAdapter('test-only', client), cancel)
    from addons.pair_pipeline.elevenlabs_voice_adapter import request_payload
    text = 'Compare the portions.'
    assert request_payload(SpeechRequest('x', text, style='whisper')) == request_payload(SpeechRequest('x', text, style='neutral'))
