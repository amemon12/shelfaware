"""Explicit, single-request voice audition. Run as a module with --live to spend API credits."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import time
import wave

from dotenv import load_dotenv

from .voice_adapter import GeminiSpeechAdapter, MODEL, SpeechRequest, VoiceError

TEXT = 'I can compare these two. Would you drink the whole bottle, or just half?'


async def audition(root: Path, style: str, provider: str = 'elevenlabs') -> int:
    load_dotenv(root / '.env', override=False)
    key_name = {'openai': 'OPENAI_API_KEY', 'gemini': 'GEMINI_API_KEY', 'elevenlabs': 'ELEVENLABS_API_KEY'}[provider]
    key = os.environ.get(key_name, '').strip()
    if not key:
        print(f'{key_name} is not configured. No request made.')
        return 1
    audio = bytearray()
    first = None
    started = time.monotonic()
    try:
        model = MODEL
        if provider == 'elevenlabs':
            from .elevenlabs_voice_adapter import create_elevenlabs_speech_adapter
            model = os.getenv('ELEVENLABS_TTS_MODEL', 'eleven_flash_v2_5')
            adapter = create_elevenlabs_speech_adapter(key, voice_id=os.getenv('ELEVENLABS_VOICE_ID', 'JBFqnCBsd6RMkjVDRZzb'), model=model)
        elif provider == 'openai':
            from .openai_voice_adapter import OpenAISpeechAdapter, MODEL as OPENAI_MODEL
            adapter = OpenAISpeechAdapter(key)
            model = OPENAI_MODEL
        else:
            adapter = GeminiSpeechAdapter(key)
        async for chunk in adapter.stream(SpeechRequest('voice-audition', TEXT, style)):
            if first is None:
                first = time.monotonic() - started
            audio.extend(chunk.pcm)
    except VoiceError as error:
        print(str(error))
        return 1
    elapsed = time.monotonic() - started
    name = f'voice-preview-{style}' if provider == 'gemini' else f'voice-preview-{provider}-{style}'
    target = root / '.runtime' / f'{name}.wav'
    target.parent.mkdir(exist_ok=True)
    with wave.open(str(target), 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(audio)
    result = {'provider': provider, 'model': model, 'style': style, 'text': TEXT,
              'sample_count': 1, 'first_pcm_chunk_seconds': round(first, 3),
              'complete_response_seconds': round(elapsed, 3),
              'audio_duration_seconds': round(len(audio)/48_000, 3),
              'measures': 'Server PCM delivery only; not first audible playback or glasses latency.',
              'path': str(target)}
    target.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Make exactly one paid provider request.')
    parser.add_argument('--provider', choices=('gemini', 'openai', 'elevenlabs'), default='elevenlabs')
    parser.add_argument('--style', choices=('quiet', 'neutral', 'whisper'), default='quiet')
    arguments = parser.parse_args()
    if not arguments.live:
        print('No request made. Add --live for a single nonmedical voice sample using the backend key.')
        return 0
    return asyncio.run(audition(Path(__file__).resolve().parents[2], arguments.style, arguments.provider))


if __name__ == '__main__':
    raise SystemExit(main())
