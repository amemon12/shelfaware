"""Explicit ElevenLabs account/STT checks. No network calls without flags."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'backend'))
from dotenv import load_dotenv
load_dotenv(ROOT / '.env', override=False)


async def check(args):
    key = os.getenv('ELEVENLABS_API_KEY', '').strip()
    voice = os.getenv('ELEVENLABS_VOICE_ID', 'JBFqnCBsd6RMkjVDRZzb').strip()
    model = os.getenv('ELEVENLABS_STT_MODEL', 'scribe_v2_realtime').strip()
    tts_model = os.getenv('ELEVENLABS_TTS_MODEL', 'eleven_flash_v2_5').strip()
    supported_tts = tts_model in ('eleven_flash_v2_5', 'eleven_multilingual_v2', 'eleven_v4_turbo')
    result = {'key_configured': bool(key), 'voice_id': voice,
              'tts_model': tts_model if supported_tts else None,
              'tts_model_supported': supported_tts,
              'tts_transport': ('dialogue_websocket' if tts_model == 'eleven_v4_turbo'
                                else 'http_stream' if supported_tts else None),
              'stt_model': model, 'microphone_tested': False, 'tts_generated': False}
    if not args.check_account and not args.check_stt:
        result['network'] = 'not_requested'
    elif not key:
        result['network'] = 'not_run_missing_key'
    else:
        if args.check_account:
            import httpx
            if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', voice):
                result['voice'] = 'invalid_voice_id'
            else:
                try:
                    async with httpx.AsyncClient(timeout=8, follow_redirects=False) as client:
                        response = await client.get('https://api.elevenlabs.io/v1/voices/' + voice,
                                                    headers={'xi-api-key': key})
                        result['voice_status'] = response.status_code
                        if response.is_success:
                            data = response.json()
                            result['voice_available'] = data.get('voice_id') == voice
                            result['voice_name'] = data.get('name')
                        else:
                            result['voice'] = 'check_voice_access_and_key_permissions_in_ElevenLabs'
                except Exception:
                    result['voice'] = 'network_or_response_error'
        if args.check_stt:
            from app.elevenlabs_transcription import ElevenLabsTranscription
            async def event(_):
                pass
            async def error(_):
                result['stt_provider_error'] = True
            transcriber = ElevenLabsTranscription(key, event, error, model=model)
            started = time.perf_counter()
            try:
                async with asyncio.timeout(12):
                    await transcriber.start()
                result['stt_session_ready'] = bool(transcriber.ready)
                result['stt_connect_seconds'] = round(time.perf_counter() - started, 3)
            except Exception:
                result['stt_session_ready'] = False
                result['stt_message'] = 'Check speech-to-text permission, account quota, model and network.'
            finally:
                await transcriber.close()
    print(json.dumps(result, indent=2))
    print('Connection success does not prove transcription, synthesis, PCM playback, or glasses operation.')
    print('For one short paid TTS sample: python -m addons.pair_pipeline.audition_voice --provider elevenlabs --live')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-account', action='store_true', help='Read the configured voice using the private key.')
    parser.add_argument('--check-stt', action='store_true', help='Open and close one realtime STT session; provider usage rules apply. No microphone/audio is sent.')
    asyncio.run(check(parser.parse_args()))
