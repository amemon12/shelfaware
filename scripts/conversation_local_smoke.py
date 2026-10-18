"""Recorded CPU integration check; no hardware, network socket, or provider API.

From the project root:
  python scripts/conversation_local_smoke.py --wav .runtime/voice-preview-quiet.wav
Uses an explicitly supplied neutral recording, cached Whisper, real VAD, and the
real shopping brain with an empty catalog and disabled cloud credentials.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path
import sys
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
sys.path.insert(0, str(ROOT))

import httpx
from fastapi import WebSocketDisconnect

from app.conversation_brain import BrainSettings, ReviewedCatalog, ShoppingBrain
from app.conversation_transport import ConversationConfig, ConversationCoordinator
from app.models import CreateSession, Profile
from app.providers import Providers, Settings
from app.service import ShoppingService
from app.speech_detection import SpeechDetector


class MemorySocket:
    """The production transport's WebSocket interface, entirely in process."""
    def __init__(self):
        self.incoming = asyncio.Queue(maxsize=8)
        self.ready = asyncio.Event()
        self.events = []
        self.origin = time.monotonic()
        self.audio_available = False
        self.closed = False

    async def receive_text(self):
        value = await self.incoming.get()
        if value is None:
            raise WebSocketDisconnect()
        return json.dumps(value)

    async def send_json(self, event):
        if len(self.events) >= 128:
            raise RuntimeError('Smoke test event limit exceeded')
        record = {'at_ms': round((time.monotonic() - self.origin) * 1000, 1), **event}
        self.events.append(record)
        if event['type'] == 'session.ready':
            self.audio_available = event.get('audio_input_available') is True
            self.ready.set()

    async def close(self, code=1000):
        self.closed = True
        if not self.incoming.full():
            self.incoming.put_nowait(None)

    async def send(self, kind, **fields):
        await self.incoming.put({'type': kind, 'client_epoch': 0, **fields})


class TimedDetector(SpeechDetector):
    def __init__(self):
        super().__init__()
        self.last_voiced_at = None
        self.last_endpoint_at = None
        real_vad = self.classify
        def observed_vad(pcm):
            result = real_vad(pcm)
            if result:
                self.last_voiced_at = time.monotonic()
            return result
        self.classify = observed_vad

    def feed(self, pcm):
        events = super().feed(pcm)
        if any(event.kind == 'end' for event in events):
            self.last_endpoint_at = time.monotonic()
        return events


def read_wav(path):
    with wave.open(str(path), 'rb') as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getcomptype()) != (1, 2, 24000, 'NONE'):
            raise ValueError('Use an uncompressed WAV with 24 kHz, mono, signed 16-bit PCM.')
        if not 1 <= source.getnframes() <= 15 * 24000:
            raise ValueError('The recording must contain more than zero and at most 15 seconds.')
        return source.readframes(source.getnframes())


async def check(path, model):
    pcm = read_wav(path)
    calls = 0
    def refuse_network(request):
        nonlocal calls
        calls += 1
        raise RuntimeError('Provider API calls are disabled in this smoke test')
    blocked_transport = httpx.MockTransport(refuse_network)
    client = httpx.AsyncClient(transport=blocked_transport)
    providers = Providers(Settings(), transport=blocked_transport)
    service = ShoppingService(providers)
    session = service.create(CreateSession(mode='live_glasses', profile=Profile(consented=True, goals=[])))
    empty_catalog = ReviewedCatalog.from_payload({'version': 1, 'products': []})
    def make_brain(profile, recipient):
        return ShoppingBrain(profile, recipient, settings=BrainSettings(deepseek_key='', perceptron_key=''),
                             client=client, catalog=empty_catalog)
    detector = TimedDetector()
    owner = ConversationCoordinator(service, brain_factory=make_brain, vad_factory=lambda: detector,
        config=ConversationConfig(openai_api_key='', transcription_provider='local',
                                  speech_provider='native', local_model=model))
    socket = MemorySocket()
    connection = await owner.attach(socket, session.id)
    task = asyncio.create_task(connection.run())
    try:
        await socket.send('session.start', session_id=session.id, recipient_id='self', voice_consented=True)
        await asyncio.wait_for(socket.ready.wait(), 25)
        if not socket.audio_available:
            raise RuntimeError('Local recognizer unavailable. From backend, run python -m app.local_transcription --download first.')
        replay_start = time.monotonic()
        audio = pcm + bytes((-len(pcm)) % 960) + bytes(48000)  # one second of endpointing silence
        for index in range(0, len(audio), 960):
            await socket.send('audio.append', audio=base64.b64encode(audio[index:index + 960]).decode('ascii'))
            delay = replay_start + (index + 960) / 48000 - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
        async with asyncio.timeout(30):
            while True:
                completed = [event for event in socket.events if event['type'] == 'response.text']
                if (completed and not detector.active and connection.input_turn is None
                        and (connection.response_task is None or connection.response_task.done())):
                    break
                errors = [event for event in socket.events if event['type'] == 'error']
                if errors:
                    raise RuntimeError(errors[-1]['message'])
                await asyncio.sleep(.02)
        await asyncio.sleep(.02)  # allow the single writer to emit the final queued state
        transcripts = [event for event in socket.events if event['type'] == 'transcript.final']
        replies = [event for event in socket.events if event['type'] == 'response.text']
        starts = [event for event in socket.events if event['type'] == 'speech.started']
        if not starts or not transcripts or not replies:
            raise RuntimeError('The recording did not complete speech detection, transcription, and a reply.')
        if calls:
            raise RuntimeError('The offline smoke attempted a provider API call; its transport blocked the request.')
        last_voiced_ms = (detector.last_voiced_at - socket.origin) * 1000
        endpoint_ms = (detector.last_endpoint_at - socket.origin) * 1000
        last_transcript = transcripts[-1]['at_ms']
        last_reply = replies[-1]['at_ms']
        timeline = []
        for event in socket.events:
            if event['type'] in ('session.ready', 'speech.started', 'transcript.final', 'response.text', 'error', 'state'):
                timeline.append({key: event[key] for key in ('at_ms', 'type', 'state', 'text', 'code') if key in event})
        return {'ok': True, 'kind': 'recorded_cpu_integration',
            'limitations': ['Recorded neutral audio; not a live microphone or glasses test.',
                            'In-memory WebSocket interface; not a network transport test.',
                            'Empty catalog and disabled provider keys; reply may request setup or clarification.',
                            'Native iOS speech output is not exercised.'],
            'recording_seconds': round(len(pcm) / 48000, 3), 'local_model': model,
            'provider_api_calls': calls, 'speech_turns': len(starts),
            'timing_note': 'One recorded CPU run. A zero rounded interval is below observer resolution, not proof of zero processing cost.',
            'transcripts': [event['text'] for event in transcripts],
            'replies': [event['text'] for event in replies],
            'timing_ms': {'last_vad_voiced_to_final_transcript': round(last_transcript - last_voiced_ms, 1),
                          'last_vad_endpoint_to_final_transcript': round(last_transcript - endpoint_ms, 1),
                          'last_final_transcript_to_reply': round(last_reply - last_transcript, 1)},
            'events': timeline}
    finally:
        await owner.close()
        await asyncio.wait_for(task, 2)
        await providers.close()
        await client.aclose()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--wav', type=Path, required=True, help='Explicit neutral/synthetic 24 kHz mono PCM16 recording')
    parser.add_argument('--model', choices=('tiny.en', 'base.en', 'small.en'), default='tiny.en')
    args = parser.parse_args()
    try:
        result = asyncio.run(check(args.wav, args.model))
    except (ValueError, OSError, RuntimeError, TimeoutError, wave.Error) as error:
        print(json.dumps({'ok': False, 'kind': 'recorded_cpu_integration', 'error': str(error) or 'Timed out waiting for local speech.'}))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
