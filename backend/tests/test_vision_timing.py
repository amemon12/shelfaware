"""Per-turn timing must survive errors without mixing concurrent users' calls."""
import asyncio
import base64
import io
import json
from collections import Counter

import httpx
import pytest
from PIL import Image

from app import product_guidance
from app.product_guidance import ProductGuide, capture_vision_timings
from app.providers import ProviderError, Providers, Settings


SECRET = 'TEST_ONLY_PRIVATE_PROVIDER_KEY'
PRIVATE_RESPONSE = 'TEST_ONLY_PRIVATE_MODEL_RESPONSE'
LEFT = {'box': {'x1': 100, 'y1': 200, 'x2': 400, 'y2': 800},
        'confidence': 0.95, 'hand_contact': 'clearly_held'}
RIGHT = {'box': {'x1': 600, 'y1': 200, 'x2': 900, 'y2': 800},
         'confidence': 0.95, 'hand_contact': 'clearly_held'}


def image_url(color='red'):
    buffer = io.BytesIO()
    Image.new('RGB', (100, 100), color).save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()


def completion(content):
    return {'choices': [{'finish_reason': 'stop', 'message': {
        'content': content if isinstance(content, str) else json.dumps(content)}}]}


def identity():
    return completion(dict(name='Demo juice', brand='Demo', variant=None, package_size=None,
        category='drink', visible_text=PRIVATE_RESPONSE, confidence=0.95, uncertainties=[]))


def provider(handler):
    return Providers(Settings(perceptron_key=SECRET), httpx.MockTransport(handler))


def assert_private_safe(timings):
    assert all(set(item) == {'stage', 'milliseconds'} for item in timings)
    assert all(type(item['milliseconds']) is int and item['milliseconds'] >= 0 for item in timings)
    serialized = json.dumps(timings)
    assert SECRET not in serialized and PRIVATE_RESPONSE not in serialized
    assert 'data:image' not in serialized and 'base64' not in serialized


async def test_concurrent_turns_keep_selection_and_child_crop_timings_isolated():
    crops_started = 0
    all_crops_started = asyncio.Event()

    async def handle(request):
        nonlocal crops_started
        payload = json.loads(request.content)
        assert 'stage' not in payload  # Local telemetry is not a provider parameter.
        schema = payload['response_format']['json_schema']['name']
        if schema == 'held_product_selections':
            data = payload['messages'][1]['content'][0]['image_url']['url'].split(',')[1]
            with Image.open(io.BytesIO(base64.b64decode(data))) as image:
                red, _, blue = image.getpixel((50, 50))
            return httpx.Response(200, json=completion({'selections': [LEFT, RIGHT] if red > blue else [LEFT]}))
        crops_started += 1
        if crops_started == 3:
            all_crops_started.set()
        await asyncio.wait_for(all_crops_started.wait(), timeout=1)
        return httpx.Response(200, json=identity())

    providers = provider(handle)
    guide = ProductGuide(providers)

    async def turn(color):
        with capture_vision_timings() as timings:
            results = await guide.identify_many(image_url(color))
        assert product_guidance._vision_trace.get() is None
        return results, timings

    try:
        (pair, pair_times), (single, single_times) = await asyncio.gather(turn('red'), turn('blue'))
        assert len(pair) == 2 and len(single) == 1
        assert pair_times is not single_times
        assert Counter(item['stage'] for item in pair_times) == {'selection': 1, 'crop_identity': 2}
        assert Counter(item['stage'] for item in single_times) == {'selection': 1, 'crop_identity': 1}
        assert_private_safe(pair_times + single_times)
    finally:
        await providers.close()


@pytest.mark.parametrize('failure', ['http', 'invalid_response'])
async def test_failed_call_retains_one_safe_timing_and_restores_context(failure):
    async def handle(request):
        if failure == 'http':
            return httpx.Response(500, json={'error': SECRET + PRIVATE_RESPONSE})
        return httpx.Response(200, json={'invalid': PRIVATE_RESPONSE})

    providers = provider(handle)
    try:
        with pytest.raises(ProviderError):
            with capture_vision_timings() as timings:
                await ProductGuide(providers)._vision(
                    [{'role': 'user', 'content': image_url()}], stage='selection')
        assert len(timings) == 1 and timings[0]['stage'] == 'selection'
        assert_private_safe(timings)
        assert product_guidance._vision_trace.get() is None
    finally:
        await providers.close()


async def test_cancelled_pair_retains_selection_and_both_cancelled_crop_timings():
    crops_started = 0
    both_crops_started = asyncio.Event()
    traces, resets = [], []

    async def handle(request):
        nonlocal crops_started
        payload = json.loads(request.content)
        if payload['response_format']['json_schema']['name'] == 'held_product_selections':
            return httpx.Response(200, json=completion({'selections': [LEFT, RIGHT]}))
        crops_started += 1
        if crops_started == 2:
            both_crops_started.set()
        await asyncio.Event().wait()

    providers = provider(handle)

    async def turn():
        try:
            with capture_vision_timings() as timings:
                traces.append(timings)
                await ProductGuide(providers).identify_many(image_url())
        finally:
            resets.append(product_guidance._vision_trace.get() is None)

    task = asyncio.create_task(turn())
    try:
        await asyncio.wait_for(both_crops_started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert resets == [True]
        assert Counter(item['stage'] for item in traces[0]) == {'selection': 1, 'crop_identity': 2}
        assert_private_safe(traces[0])
        # identify_many waits for cancelled children; no late timing can leak into the next turn.
        before = list(traces[0])
        await asyncio.sleep(0)
        assert traces[0] == before
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await providers.close()


async def test_nested_capture_restores_parent_and_exited_capture_is_not_reused():
    providers = provider(lambda request: httpx.Response(200, json=completion(PRIVATE_RESPONSE)))
    guide = ProductGuide(providers)
    try:
        with capture_vision_timings() as outer:
            await guide._vision([], stage='outer_before')
            with pytest.raises(RuntimeError):
                with capture_vision_timings() as inner:
                    await guide._vision([], stage='inner')
                    raise RuntimeError('test exception')
            await guide._vision([], stage='outer_after')
        await guide._vision([], stage='untraced')
        with capture_vision_timings() as later:
            await guide._vision([], stage='later')
        assert [item['stage'] for item in outer] == ['outer_before', 'outer_after']
        assert [item['stage'] for item in inner] == ['inner']
        assert [item['stage'] for item in later] == ['later']
        assert product_guidance._vision_trace.get() is None
        assert_private_safe(outer + inner + later)
    finally:
        await providers.close()


async def test_selection_retry_records_distinct_stages_without_model_content():
    calls = 0

    def handle(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=completion(PRIVATE_RESPONSE if calls == 1 else {'selections': []}))

    providers = provider(handle)
    try:
        with capture_vision_timings() as timings:
            assert await ProductGuide(providers).identify_many(image_url()) == []
        assert [item['stage'] for item in timings] == ['selection', 'selection_retry']
        assert_private_safe(timings)
    finally:
        await providers.close()


async def test_missing_configuration_does_not_record_a_provider_call():
    providers = Providers(Settings(), httpx.MockTransport(lambda request: pytest.fail('No configured provider')))
    try:
        with capture_vision_timings() as timings:
            with pytest.raises(ProviderError, match='not configured'):
                await ProductGuide(providers)._vision([], stage='selection')
        assert timings == []
        assert product_guidance._vision_trace.get() is None
    finally:
        await providers.close()
