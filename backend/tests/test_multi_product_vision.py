import asyncio
import base64
import io
import json

import httpx
import pytest
from PIL import Image

from app.models import Box
from app.product_guidance import ProductGuide
from app.providers import ProviderError, Providers, Settings


LEFT = {'box': {'x1': 100, 'y1': 200, 'x2': 400, 'y2': 800},
        'confidence': 0.95, 'hand_contact': 'clearly_held'}
RIGHT = {'box': {'x1': 600, 'y1': 200, 'x2': 900, 'y2': 800},
         'confidence': 0.95, 'hand_contact': 'clearly_held'}


def image_url():
    image = Image.new('RGB', (100, 100), 'red')
    image.paste('blue', (50, 0, 100, 100))
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()


def selected(selections=None, arguments=None):
    content = arguments if arguments is not None else json.dumps({
        'selections': selections if selections is not None else [LEFT, RIGHT]})
    return {'choices': [{'finish_reason': 'stop', 'message': {'content': content}}]}


def constrained(selections):
    return selected(selections)


def unexpected_tool_response():
    return {'choices': [{'finish_reason': 'tool_calls', 'message': {'tool_calls': [{
        'id': 'unexpected', 'type': 'function', 'function': {'name': 'inspect_product_crops',
        'arguments': json.dumps({'selections': [LEFT, RIGHT]})}}]}}]}


def identity(name='Demo Red', **changes):
    fields = dict(name=name, brand='Demo', variant=name, package_size='250 mL',
        category='drink', visible_text=name or '', confidence=0.95, uncertainties=[])
    return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(fields | changes)}}]}


def crop_name(payload):
    assert 'tools' not in payload and 'tool_choice' not in payload
    assert payload['response_format']['json_schema']['name'] == 'held_product_identity'
    assert payload['max_completion_tokens'] == 600
    encoded = payload['messages'][1]['content'][0]['image_url']['url'].split(',')[1]
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
        assert image.size == (30, 60)
        red, _, blue = image.getpixel((15, 30))
    return 'Demo Red' if red > blue else 'Demo Blue'


def provider(handler):
    return Providers(Settings(perceptron_key='vision-test'), httpx.MockTransport(handler))


async def test_two_actual_crops_are_inspected_concurrently_and_returned_left_to_right():
    requests = []
    started = set()
    both_started = asyncio.Event()

    async def handle(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            assert 'tools' not in payload and 'tool_choice' not in payload and 'parallel_tool_calls' not in payload
            schema = payload['response_format']['json_schema']
            assert schema['name'] == 'held_product_selections'
            assert schema['strict'] is True
            assert schema['schema']['properties']['selections']['maxItems'] == 2
            assert payload['max_completion_tokens'] == 1024
            assert payload['reasoning_effort'] == 'minimal'
            encoded = payload['messages'][1]['content'][0]['image_url']['url'].split(',')[1]
            with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
                assert image.size == (100, 100)
            for excluded in ('cart', 'table', 'background'):
                assert excluded in payload['messages'][0]['content']
            return httpx.Response(200, json=selected([RIGHT, LEFT]))
        name = crop_name(payload)
        started.add(name)
        if len(started) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=1)
        if name == 'Demo Red':
            await asyncio.sleep(0)
        return httpx.Response(200, json=identity(name))

    providers = provider(handle)
    try:
        results = await ProductGuide(providers).identify_many(image_url())
        assert [result.name for result in results] == ['Demo Red', 'Demo Blue']
        assert [result.box for result in results] == [Box(x=0.1, y=0.2, w=0.3, h=0.6),
                                                     Box(x=0.6, y=0.2, w=0.3, h=0.6)]
        assert len(requests) == 3
    finally:
        await providers.close()


async def test_one_selected_package_uses_only_one_crop_call():
    requests = []
    def handle(request):
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(200, json=selected([LEFT]) if len(requests) == 1 else identity(crop_name(payload)))
    providers = provider(handle)
    try:
        results = await ProductGuide(providers).identify_many(image_url())
        assert len(results) == 1 and results[0].name == 'Demo Red'
        assert len(requests) == 2
    finally:
        await providers.close()


async def test_empty_selections_abstains_without_crop_calls():
    requests = []
    providers = provider(lambda request: requests.append(request) or httpx.Response(200, json=selected([])))
    try:
        assert await ProductGuide(providers).identify_many(image_url()) == []
        assert len(requests) == 1
    finally:
        await providers.close()


@pytest.mark.parametrize('arguments', [
    '{"selections":[{"box":{"x1":100,y1":200,"x2":400,"y2":800}}]}',
    json.dumps({'selections': [{'box_2d': [100, 200, 400, 800]}]}),
    'NO_HELD_PRODUCT',
])
async def test_malformed_selection_uses_one_fresh_constrained_retry_then_two_crops(arguments):
    requests = []
    def handle(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            return httpx.Response(200, json=selected(arguments=arguments))
        if len(requests) == 2:
            assert 'tools' not in payload and 'tool_choice' not in payload and 'parallel_tool_calls' not in payload
            assert len(payload['messages']) == 2
            assert payload['messages'][1] == requests[0]['messages'][1]
            assert arguments not in json.dumps(payload)
            assert payload['response_format']['json_schema']['strict'] is True
            assert payload['response_format']['json_schema']['name'] == 'held_product_selections'
            assert payload['max_completion_tokens'] == 1024
            return httpx.Response(200, json=constrained([RIGHT, LEFT]))
        return httpx.Response(200, json=identity(crop_name(payload)))
    providers = provider(handle)
    try:
        results = await ProductGuide(providers).identify_many(image_url())
        assert [result.name for result in results] == ['Demo Red', 'Demo Blue']
        assert len(requests) == 4
    finally:
        await providers.close()


async def test_constrained_retry_can_abstain():
    replies = iter([selected(arguments='broken'), constrained([])])
    requests = []
    providers = provider(lambda request: requests.append(request) or httpx.Response(200, json=next(replies)))
    try:
        assert await ProductGuide(providers).identify_many(image_url()) == []
        assert len(requests) == 2
    finally:
        await providers.close()


@pytest.mark.parametrize('selections', [
    [LEFT, RIGHT, LEFT], [LEFT, LEFT],
    [LEFT, LEFT | {'box': {'x1': 110, 'y1': 210, 'x2': 390, 'y2': 790}}],
    [LEFT, LEFT | {'box': {'x1': 250, 'y1': 200, 'x2': 550, 'y2': 800}}],
    [LEFT | {'confidence': 0.79}, RIGHT], [LEFT, RIGHT | {'confidence': 0.79}],
])
async def test_excess_duplicate_overlapping_or_low_confidence_selection_never_crops(selections):
    requests = []
    providers = provider(lambda request: requests.append(request) or httpx.Response(200, json=selected(selections)))
    try:
        with pytest.raises(ProviderError):
            await ProductGuide(providers).identify_many(image_url())
        assert len(requests) == 1
    finally:
        await providers.close()


@pytest.mark.parametrize('bad', [
    LEFT | {'box': [100, 200, 400, 800]},
    LEFT | {'box': {'x1': -1, 'y1': 200, 'x2': 400, 'y2': 800}},
    LEFT | {'box': {'x1': 400, 'y1': 200, 'x2': 100, 'y2': 800}},
    LEFT | {'box': {'x1': 100.5, 'y1': 200, 'x2': 400, 'y2': 800}},
    LEFT | {'hand_contact': 'on_table'}, LEFT | {'confidence': True},
])
async def test_invalid_selections_stay_rejected_after_one_retry(bad):
    requests = []
    replies = iter([selected([bad, RIGHT]), constrained([bad, RIGHT])])
    providers = provider(lambda request: requests.append(request) or httpx.Response(200, json=next(replies)))
    try:
        with pytest.raises(ProviderError):
            await ProductGuide(providers).identify_many(image_url())
        assert len(requests) == 2
    finally:
        await providers.close()


@pytest.mark.parametrize('kind', ['tool_finish', 'unexpected_tool', 'unexpected_function', 'incomplete'])
async def test_invalid_selection_protocol_does_not_get_a_retry(kind):
    result = selected()
    choice = result['choices'][0]
    if kind == 'tool_finish':
        result = unexpected_tool_response()
    elif kind == 'unexpected_tool':
        choice['message']['tool_calls'] = unexpected_tool_response()['choices'][0]['message']['tool_calls']
    elif kind == 'unexpected_function':
        choice['message']['function_call'] = {'name': 'search', 'arguments': '{}'}
    else:
        choice['finish_reason'] = 'length'
    requests = []
    providers = provider(lambda request: requests.append(request) or httpx.Response(200, json=result))
    try:
        with pytest.raises(ProviderError):
            await ProductGuide(providers).identify_many(image_url())
        assert len(requests) == 1
    finally:
        await providers.close()


@pytest.mark.parametrize('changes', [{'brand': None}, {'name': None}, {'confidence': 0.64}])
async def test_unknown_second_crop_never_returns_only_the_valid_first_package(changes):
    requests = []
    def handle(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            return httpx.Response(200, json=selected())
        name = crop_name(payload)
        return httpx.Response(200, json=identity(**({'name': name} | (changes if name == 'Demo Blue' else {}))))
    providers = provider(handle)
    try:
        with pytest.raises(ProviderError, match='both product names and brands'):
            await ProductGuide(providers).identify_many(image_url())
        assert len(requests) == 3
    finally:
        await providers.close()


async def test_unknown_single_crop_returns_empty_list():
    replies = iter([selected([LEFT]), identity(brand=None)])
    providers = provider(lambda _: httpx.Response(200, json=next(replies)))
    try:
        assert await ProductGuide(providers).identify_many(image_url()) == []
    finally:
        await providers.close()


@pytest.mark.parametrize('result', [unexpected_tool_response(), constrained([LEFT, RIGHT, LEFT]),
    {'choices': [{'finish_reason': 'stop', 'message': {'content': '{broken'}}]}])
async def test_invalid_fallback_is_rejected_without_crop_or_third_selection_call(result):
    calls = []
    replies = iter([selected(arguments='broken'), result])
    providers = provider(lambda request: calls.append(request) or httpx.Response(200, json=next(replies)))
    try:
        with pytest.raises(ProviderError):
            await ProductGuide(providers).identify_many(image_url())
        assert len(calls) == 2
    finally:
        await providers.close()


@pytest.mark.parametrize('content', ['{broken', 'NO_HELD_PRODUCT', json.dumps({'selections': [{}]})])
async def test_two_malformed_selection_responses_fail_without_crop_or_third_request(content):
    requests = []
    response = selected(arguments=content)
    providers = provider(lambda request: requests.append(request) or httpx.Response(200, json=response))
    try:
        with pytest.raises(ProviderError):
            await ProductGuide(providers).identify_many(image_url())
        assert len(requests) == 2
    finally:
        await providers.close()


async def test_pair_crops_keep_existing_size_uncertainty_gate():
    requests = []
    def handle(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            return httpx.Response(200, json=selected())
        return httpx.Response(200, json=identity(crop_name(payload), uncertainties=['Size is not visible']))
    providers = provider(handle)
    try:
        results = await ProductGuide(providers).identify_many(image_url())
        assert len(results) == 2
        assert all(result.package_size is None and result.uncertainties == ['Size is not visible'] for result in results)
    finally:
        await providers.close()


async def test_crop_protocol_failure_cancels_the_sibling_without_partial_output():
    both_started = asyncio.Event()
    sibling_cancelled = asyncio.Event()
    started = set()
    calls = []

    async def handle(request):
        payload = json.loads(request.content)
        calls.append(payload)
        if len(calls) == 1:
            return httpx.Response(200, json=selected())
        name = crop_name(payload)
        started.add(name)
        if len(started) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=1)
        if name == 'Demo Red':
            return httpx.Response(200, json=identity(carbohydrate_g=27))
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            sibling_cancelled.set()
            raise

    providers = provider(handle)
    try:
        with pytest.raises(ProviderError):
            await ProductGuide(providers).identify_many(image_url())
        assert sibling_cancelled.is_set() and len(calls) == 3
    finally:
        await providers.close()


async def test_parent_cancellation_stops_both_crop_requests():
    both_started = asyncio.Event()
    started, cancelled, calls = set(), set(), []

    async def handle(request):
        payload = json.loads(request.content)
        calls.append(payload)
        if len(calls) == 1:
            return httpx.Response(200, json=selected())
        name = crop_name(payload)
        started.add(name)
        if len(started) == 2:
            both_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.add(name)
            raise

    providers = provider(handle)
    task = asyncio.create_task(ProductGuide(providers).identify_many(image_url()))
    try:
        await asyncio.wait_for(both_started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled == {'Demo Red', 'Demo Blue'} and len(calls) == 3
    finally:
        task.cancel()
        await providers.close()
