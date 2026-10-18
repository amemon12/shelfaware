"""Check conversation setup without exposing credentials. No cloud call by default."""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import re
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'backend'))
from dotenv import load_dotenv
load_dotenv(ROOT / '.env', override=False)
from app.conversation_brain import BrainSettings, Intent, ReviewedCatalog, ShoppingBrain, VisionCandidates
from app.conversation_transport import ConversationConfig


async def check_backend(api_url: str, *, client: httpx.AsyncClient | None = None) -> dict:
    """Opt-in, credential-free health read; never echo remote payloads or URLs."""
    try:
        if not isinstance(api_url, str) or len(api_url) > 2048 or any(c.isspace() or ord(c) < 32 for c in api_url):
            raise ValueError
        parsed = urlsplit(api_url)
        if (parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or parsed.path not in ('', '/') or '?' in api_url or '#' in api_url):
            raise ValueError
        parsed.port  # Validate malformed/out-of-range ports without printing them.
        target = httpx.URL(parsed._replace(path='/api/health', query='', fragment='').geturl())
    except (ValueError, TypeError, httpx.InvalidURL):
        return {'reachable': False, 'compatible': False, 'status': 'invalid_origin',
                'message': 'Use only an http(s) origin, with an optional port; no credentials, path, query, or fragment.'}
    owned = client is None
    client = client or httpx.AsyncClient(timeout=3, trust_env=False)
    response = None
    reachable = False
    try:
        async with asyncio.timeout(5):
            # Construct directly, rather than client.build_request: no inherited
            # cookies, authorization headers, or auth hooks from an injected client.
            request = httpx.Request('GET', target, headers={'Accept': 'application/json', 'Accept-Encoding': 'identity'},
                                    extensions={'timeout': httpx.Timeout(3).as_dict()})
            response = await client.send(request, stream=True, follow_redirects=False, auth=None)
            reachable = True
            base = {'reachable': True, 'compatible': False, 'http_status': response.status_code}
            if 300 <= response.status_code < 400:
                return {**base, 'status': 'redirect_rejected', 'message': 'The health endpoint redirected; redirects are not followed.'}
            if response.status_code != 200:
                return {**base, 'status': 'http_error', 'message': 'The backend health endpoint did not return HTTP 200.'}
            length = response.headers.get('content-length')
            if length is not None and (not length.isascii() or not length.isdigit() or int(length) > 16384):
                return {**base, 'status': 'response_too_large', 'message': 'Health response exceeds the 16 KiB limit or has an invalid length.'}
            if response.headers.get('content-encoding', 'identity').lower() not in ('', 'identity'):
                return {**base, 'status': 'invalid_health_response', 'message': 'Expected an uncompressed JSON health response.'}
            data = bytearray()
            async for chunk in response.aiter_bytes(chunk_size=4096):
                if len(data) + len(chunk) > 16384:
                    return {**base, 'status': 'response_too_large', 'message': 'Health response exceeds the 16 KiB limit.'}
                data.extend(chunk)
            try:
                payload = json.loads(data.decode('utf-8'))
                if not isinstance(payload, dict):
                    raise ValueError
            except (ValueError, UnicodeError, RecursionError):
                return {**base, 'status': 'invalid_health_response', 'message': 'The health endpoint did not return a JSON object.'}
            raw_version = payload.get('version')
            version = raw_version if isinstance(raw_version, str) and re.fullmatch(r'\d+\.\d+\.\d+(?:[-+][A-Za-z0-9._-]{1,48})?', raw_version) else None
            raw = payload.get('conversation')
            raw = raw if isinstance(raw, dict) else {}
            expected_route, expected_input = '/api/conversation/ws', 'pcm_s16le_24000_mono'
            conversation = {'websocket': expected_route if raw.get('websocket') == expected_route else None,
                            'input': expected_input if raw.get('input') == expected_input else None}
            for name, allowed in (
                ('reasoning_provider', ('deepseek', 'anthropic', 'gemini')),
                ('analysis_mode', ('reviewed_catalog', 'product_search')),
                ('evidence_scope', ('reviewed_catalog', 'unverified_catalog_search')),
                ('transcription_provider', ('elevenlabs', 'local', 'openai', 'disabled')),
                ('speech_model', ('eleven_flash_v2_5', 'eleven_multilingual_v2', 'eleven_v4_turbo')),
                ('speech_provider', ('elevenlabs', 'native', 'openai', 'disabled'))):
                value = raw.get(name)
                conversation[name] = value if isinstance(value, str) and value in allowed else None
            raw_providers = payload.get('providers')
            raw_providers = raw_providers if isinstance(raw_providers, dict) else {}
            # Mac integration retains the existing product provider contract;
            # additive voice capabilities are reported in their own namespace.
            conversation_providers = raw.get('providers_configured')
            if isinstance(conversation_providers, dict):
                raw_providers = {**raw_providers, **conversation_providers}
            providers = {name: raw_providers[name] for name in
                ('perceptron', 'gemini', 'deepseek', 'anthropic', 'elevenlabs', 'openai_optional')
                if type(raw_providers.get(name)) is bool}
            compatible = conversation['websocket'] is not None and conversation['input'] is not None
            return {**base, 'compatible': compatible, 'status': 'compatible' if compatible else 'incompatible',
                    'version': version, 'conversation': conversation, 'providers_configured': providers,
                    'message': ('Expected conversation route and audio format are advertised; provider access and hardware remain unverified.'
                        if compatible else 'Running backend does not advertise /api/conversation/ws with pcm_s16le_24000_mono. Restart the intended backend from the latest project before testing voice.')}
    except (httpx.HTTPError, TimeoutError, ValueError):
        return {'reachable': reachable, 'compatible': False, 'status': 'unavailable',
                'message': 'Backend health could not be read within its bounded request; check the address, server, and network.'}
    finally:
        if response is not None:
            try:
                await response.aclose()
            except Exception:
                pass
        if owned:
            await client.aclose()


async def check(live: bool, warm_schemas: bool = False, api_url: str | None = None):
    settings, config = BrainSettings.from_env(), ConversationConfig.from_env()
    provider = settings.provider
    reasoning_key = settings.reasoning_key
    result = {'reasoning_provider': provider, 'reasoning_model': settings.model,
              'reasoning_probe_scope': 'reviewed_catalog_only',
              'active_analysis_mode': (getattr(config, 'product_mode', 'reviewed_catalog')
                  if getattr(config, 'product_mode', 'reviewed_catalog') in
                     ('reviewed_catalog', 'product_search') else None),
              'product_search_text_provider': (os.getenv('TEXT_PROVIDER', 'gemini').strip().lower()
                  if os.getenv('TEXT_PROVIDER', 'gemini').strip().lower() in
                     ('gemini', 'deepseek', 'anthropic') else None),
              'deepseek_key_configured': bool(settings.deepseek_key),
              'anthropic_key_configured': bool(os.getenv('ANTHROPIC_API_KEY', '').strip()),
              'perceptron_key_configured': bool(settings.perceptron_key),
              'elevenlabs_key_configured': bool(os.getenv('ELEVENLABS_API_KEY', '').strip()),
              'elevenlabs_voice_id': os.getenv('ELEVENLABS_VOICE_ID', 'JBFqnCBsd6RMkjVDRZzb'),
              'transcription_provider': config.transcription_provider,
              'speech_provider': config.speech_provider,
              'elevenlabs_tts_model': (getattr(config, 'elevenlabs_tts_model', None)
                  if getattr(config, 'elevenlabs_tts_model', None) in
                     ('eleven_flash_v2_5', 'eleven_multilingual_v2', 'eleven_v4_turbo') else None),
              'local_transcription_dependency': importlib.util.find_spec('faster_whisper') is not None,
              'cloud_requests_made': 0, 'hardware_verified': False}
    if api_url is not None:
        result['running_backend'] = await check_backend(api_url)
    try:
        catalog = ReviewedCatalog.load(settings.catalog_path)
        catalog.exemplars()
        result['reviewed_product_count'] = len(catalog.entries)
        result['catalog_valid'] = True
    except Exception:
        result['catalog_valid'] = False
        result['catalog_message'] = 'Run the catalog validator; no package facts were printed.'
    if live or warm_schemas:
        if not reasoning_key:
            result['live_reasoning'] = 'not_run_missing_key'
        else:
            brain = ShoppingBrain({'consented': True, 'goals': [], 'avoid_ingredients': []}, settings=settings)
            try:
                probes = [(Intent, 'shopping_intent', 'Classify this greeting as greeting. Do not generate advice or perform any action.', 'Hello, Shelf Aware.')]
                if warm_schemas:
                    probes.append((VisionCandidates, 'product_candidates', 'No products or image supplied. Return an empty candidate_ids list.', 'No image. No candidates.'))
                checks = []
                for schema, name, prompt, text in probes:
                    started = time.perf_counter()
                    probe = {'schema': schema.__name__}
                    result['cloud_requests_made'] += 1
                    try:
                        value = await brain._json(schema, name, prompt,
                            [{'type': 'input_text', 'text': text}], 30)
                        valid = value.action == 'greeting' if schema is Intent else value.candidate_ids == []
                        probe['status'] = 'passed' if valid else 'unexpected_result'
                    except (TimeoutError, httpx.TimeoutException):
                        probe['status'] = 'timed_out_possible_cold_schema_or_network'
                    except Exception:
                        # Provider exceptions can contain request details; never print them.
                        probe['status'] = 'failed_check_key_model_quota_or_network_on_provider_dashboard'
                    probe['request_seconds'] = round(time.perf_counter() - started, 3)
                    checks.append(probe)
                result['schema_checks'] = checks
                result['live_reasoning'] = 'passed' if all(p['status'] == 'passed' for p in checks) else 'failed'
            finally:
                await brain.close()
    print(json.dumps(result, indent=2))
    print('reasoning_provider/model and --live/--warm-schemas describe only the optional reviewed-catalog brain; they do not validate product-search wording or its separately selected TEXT_PROVIDER.')
    print('This does not test glasses, microphone routing, package recognition, or audible response latency.')
    print('Schema warm-up uses neutral text only; it does not test vision or change live-turn deadlines.')
    print('Check ElevenLabs: python scripts/elevenlabs_preflight.py --check-account --check-stt')
    print('Optional cached local transcription: cd backend; python -m app.local_transcription')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    probes = parser.add_mutually_exclusive_group()
    probes.add_argument('--live', action='store_true', help='Make one paid neutral intent request (up to 30 seconds).')
    probes.add_argument('--warm-schemas', action='store_true', help='Make two paid neutral requests to warm intent and candidate schemas (up to 30 seconds each).')
    parser.add_argument('--api-url', help='Optionally check a running backend origin, e.g. http://127.0.0.1:8000. No token, redirects, or provider calls.')
    args = parser.parse_args()
    asyncio.run(check(args.live, args.warm_schemas, args.api_url))
