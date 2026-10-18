import asyncio
import base64
import io
import json

import httpx
import pytest
from PIL import Image
from pydantic import ValidationError

from app.models import Box
from app.product_guidance import CatalogMatch, CatalogSource, DetectedProduct, ProductGuide
from app.providers import ProviderError, Providers, Settings


def image_url(size=(100, 100)):
    buffer = io.BytesIO()
    Image.new('RGB', size, 'blue').save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()


def identity(**changes):
    values = dict(name='Demo cereal', brand='Demo', variant=None, package_size=None,
                  category='Cereal', visible_text='Demo cereal', confidence=0.9, uncertainties=[])
    return {**values, **changes}


def detected(**changes):
    return DetectedProduct(**identity(**changes), box=Box(x=0.2, y=0.1, w=0.5, h=0.7))


def selection(**changes):
    arguments = dict(box={'x1': 200, 'y1': 100, 'x2': 700, 'y2': 800}, confidence=0.95, hand_contact='clearly_held')
    arguments.update(changes)
    return {'choices': [{'finish_reason': 'tool_calls', 'message': {'role': 'assistant', 'tool_calls': [{
        'id': 'held-1', 'type': 'function', 'function': {'name': 'inspect_held_product', 'arguments': json.dumps(arguments)}}]}}]}


def identity_response(**changes):
    return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(identity(**changes))}}]}


def constrained_selection(**changes):
    arguments = json.loads(selection(**changes)['choices'][0]['message']['tool_calls'][0]['function']['arguments'])
    return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({'selection': arguments})}}]}


def interaction(text, *, sources=True, status='completed'):
    block = {'type': 'text', 'text': text}
    if sources:
        block['annotations'] = [{'type': 'url_citation', 'title': 'Manufacturer',
            'url': 'https://manufacturer.com/products/demo', 'start_index': 0, 'end_index': len(text)}]
    return {'status': status, 'steps': [{'type': 'thought', 'summary': [{'text': 'private reasoning'}]},
        {'type': 'model_output', 'content': [block]}]}


def product(notes='Possible Demo cereal listing. 27 g carbohydrate per listed bar.'):
    return {'identity': detected().model_dump(), 'catalog': CatalogMatch(status='grounded', notes=notes,
        sources=[CatalogSource(title='Manufacturer', url='https://manufacturer.com/products/demo')]).model_dump()}


def provider(handler, **changes):
    settings = Settings(perceptron_key='perception-test', gemini_key='gemini-test', **changes)
    return Providers(settings, httpx.MockTransport(handler))


async def test_complete_guidance_is_four_calls_with_real_crop_and_no_final_tool_roundtrip():
    calls = []
    replies = [selection(), identity_response(), interaction('Possible Demo cereal listing.'),
        interaction(json.dumps({'spoken_text': 'This looks like Demo cereal, a possible catalog match. Compare the available options for your preferences; the exact package formulation is unverified.'}), sources=False)]

    def handle(request):
        payload = json.loads(request.content)
        calls.append(payload)
        if len(calls) == 1:
            assert request.url.host == 'api.perceptron.inc'
            assert payload['model'] == 'perceptron-mk1.5'
            assert payload['tools'][0]['function']['name'] == 'inspect_held_product'
        elif len(calls) == 2:
            assert 'tools' not in payload and 'tool_choice' not in payload
            schema = payload['response_format']['json_schema']['schema']
            assert not {'nutrition', 'carbohydrate_g', 'barcode', 'ingredients'} & schema['properties'].keys()
            assert set(schema['required']) == set(schema['properties'])
            encoded = payload['messages'][1]['content'][0]['image_url']['url'].split(',')[1]
            with Image.open(io.BytesIO(base64.b64decode(encoded))) as crop:
                assert crop.size == (50, 70)
        else:
            assert str(request.url) == 'https://generativelanguage.googleapis.com/v1beta/interactions'
            assert payload['store'] is False and payload['model'] == 'gemini-3.8-flash'
            assert 'thinking_level' not in payload['generation_config']
        return httpx.Response(200, json=replies[len(calls) - 1])

    providers = provider(handle)
    try:
        guide = ProductGuide(providers)
        found = await guide.identify(image_url())
        assert found.box == Box(x=0.2, y=0.1, w=0.5, h=0.7)
        match = await guide.search(found)
        text = await guide.respond({'identity': found.model_dump(), 'catalog': match.model_dump()},
            'What about this?', {'recipient_id': 'self', 'consented': True, 'goals': ['More fiber']}, [])
        assert 'possible catalog match' in text and len(text.split()) <= 65
        assert len(calls) == 4
        assert calls[2]['tools'] == [{'type': 'google_search'}]
        assert 'tools' not in calls[3]
    finally:
        await providers.close()


@pytest.mark.parametrize('response', [
    {'choices': [{'finish_reason': 'stop', 'message': {'content': 'NO_HELD_PRODUCT'}}]},
    selection(confidence=0.79),
])
async def test_no_held_or_low_confidence_selection_stops_after_one_call(response):
    calls = []
    providers = provider(lambda request: calls.append(request) or httpx.Response(200, json=response))
    try:
        assert await ProductGuide(providers).identify(image_url()) is None
        assert len(calls) == 1
    finally:
        await providers.close()


@pytest.mark.parametrize('changes', [
    {'box': [200, 100, 700, 800]},
    {'box': {'x1': 200.1, 'y1': 100, 'x2': 700, 'y2': 800}},
    {'box': {'x1': 900, 'y1': 100, 'x2': 700, 'y2': 800}},
    {'box': {'x1': -1, 'y1': 100, 'x2': 700, 'y2': 800}},
    {'hand_contact': 'near_a_hand'}, {'confidence': True},
])
async def test_twice_invalid_selection_is_rejected_after_one_repair_without_a_crop_call(changes):
    calls = []
    replies = iter([selection(**changes), constrained_selection(**changes)])
    providers = provider(lambda request: calls.append(request) or httpx.Response(200, json=next(replies)))
    try:
        with pytest.raises(ProviderError, match='one valid held product'):
            await ProductGuide(providers).identify(image_url())
        assert len(calls) == 2
        assert 'tools' in json.loads(calls[0].content)
        fallback = json.loads(calls[1].content)
        assert 'tools' not in fallback and 'tool_choice' not in fallback
        assert fallback['response_format']['json_schema']['strict'] is True
    finally:
        await providers.close()


@pytest.mark.parametrize('kind', ['multiple', 'wrong_tool', 'incomplete', 'empty', 'null_content'])
async def test_selection_protocol_errors_fail_closed(kind):
    response = selection()
    choice = response['choices'][0]
    if kind == 'multiple':
        choice['message']['tool_calls'] *= 2
    elif kind == 'wrong_tool':
        choice['message']['tool_calls'][0]['function']['name'] = 'send_email'
    elif kind == 'incomplete':
        choice['finish_reason'] = 'length'
    else:
        choice.update(finish_reason='stop', message={'content': None if kind == 'null_content' else ''})
    calls = []
    providers = provider(lambda request: calls.append(request) or httpx.Response(200, json=response))
    try:
        with pytest.raises(ProviderError):
            await ProductGuide(providers).identify(image_url())
        assert len(calls) == 1
    finally:
        await providers.close()


@pytest.mark.parametrize('arguments', [
    json.dumps({'box': {'box_2d': [175, 445, 885, 880], 'confidence': 0.95, 'hand_contact': 'clearly_held'}, 'label': 'bottle'}),
    '{"box": {"x1":390, y1":490, "x2":560,"y2":810},"confidence":0.95,"hand_contact":"clearly_held"}',
])
async def test_malformed_live_arguments_use_fresh_constrained_selection_before_crop_inspection(arguments):
    malformed = selection()
    malformed['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = arguments
    responses = [malformed, constrained_selection(), identity_response()]
    calls = []

    def handle(request):
        payload = json.loads(request.content)
        calls.append(payload)
        if len(calls) == 1:
            assert '{"box":{"x1":100,"y1":200,"x2":600,"y2":800},"confidence":0.95,"hand_contact":"clearly_held"}' in payload['messages'][0]['content']
        elif len(calls) == 2:
            assert len(payload['messages']) == 2
            assert payload['messages'][1] == calls[0]['messages'][1]
            assert arguments not in json.dumps(payload)
            assert 'tools' not in payload and 'tool_choice' not in payload and 'parallel_tool_calls' not in payload
            schema = payload['response_format']['json_schema']
            assert schema['name'] == 'held_product_selection' and schema['strict'] is True
            assert schema['schema']['required'] == ['selection']
            assert {'type': 'null'} in schema['schema']['properties']['selection']['anyOf']
            assert '{"selection":null}' in payload['messages'][0]['content']
        else:
            assert 'tools' not in payload and 'tool_choice' not in payload
            encoded = payload['messages'][1]['content'][0]['image_url']['url'].split(',')[1]
            with Image.open(io.BytesIO(base64.b64decode(encoded))) as crop:
                assert crop.size == (50, 70)
        return httpx.Response(200, json=responses[len(calls) - 1])

    providers = provider(handle)
    try:
        result = await ProductGuide(providers).identify(image_url())
        assert result.box == Box(x=0.2, y=0.1, w=0.5, h=0.7)
        assert len(calls) == 3
    finally:
        await providers.close()


@pytest.mark.parametrize('response', [
    {'choices': [{'finish_reason': 'stop', 'message': {'content': '{"selection":null}'}}]},
    constrained_selection(confidence=0.79),
])
async def test_constrained_fallback_can_abstain_without_running_crop_inspection(response):
    replies = iter([selection(box=[175, 445, 885, 880]),
                    response])
    calls = []
    providers = provider(lambda request: calls.append(request) or httpx.Response(200, json=next(replies)))
    try:
        assert await ProductGuide(providers).identify(image_url()) is None
        assert len(calls) == 2
    finally:
        await providers.close()


@pytest.mark.parametrize('kind', ['invalid_json', 'extra_field', 'tool_call', 'function_call', 'incomplete', 'missing_selection'])
async def test_malformed_constrained_fallback_fails_without_another_retry(kind):
    response = constrained_selection()
    choice = response['choices'][0]
    if kind == 'invalid_json':
        choice['message']['content'] = '{"selection": {"box": {"x1":390, y1":490}}}'
    elif kind == 'extra_field':
        data = json.loads(choice['message']['content'])
        data['label'] = 'bottle'
        choice['message']['content'] = json.dumps(data)
    elif kind == 'tool_call':
        response = selection()
    elif kind == 'function_call':
        choice['message']['function_call'] = {'name': 'inspect_held_product', 'arguments': '{}'}
    elif kind == 'incomplete':
        choice['finish_reason'] = 'length'
    else:
        choice['message']['content'] = '{}'
    replies = iter([selection(box=[]), response])
    calls = []
    providers = provider(lambda request: calls.append(request) or httpx.Response(200, json=next(replies)))
    try:
        with pytest.raises(ProviderError):
            await ProductGuide(providers).identify(image_url())
        assert len(calls) == 2
    finally:
        await providers.close()


@pytest.mark.parametrize('changes', [
    {'name': None}, {'name': '   '}, {'confidence': 0.64},
    {'brand': None}, {'brand': '   '}, {'brand': 'Unknown'}, {'brand': 'Not visible'},
    {'name': 'Unknown product'}, {'name': 'Beverage', 'category': 'Beverage'},
    {'name': 'Organic Coconut Water', 'brand': None, 'package_size': '250 mL',
     'confidence': 0.95, 'uncertainties': ['Brand not visible']},
])
async def test_unidentified_crop_does_not_become_a_catalog_candidate(changes):
    replies = iter([selection(), identity_response(**changes)])
    calls = []
    providers = provider(lambda request: calls.append(request) or httpx.Response(200, json=next(replies)))
    try:
        assert await ProductGuide(providers).identify(image_url()) is None
        assert len(calls) == 2
        prompt = json.loads(calls[1].content)['messages'][0]['content']
        assert 'BOTH its product name and brand visibly' in prompt
        assert 'infer package size from a serving' in prompt
    finally:
        await providers.close()


async def test_direct_catalog_search_without_brand_makes_no_provider_request():
    providers = provider(lambda _: pytest.fail('Missing brand must not trigger catalog search'))
    try:
        result = await ProductGuide(providers).search(detected(brand=None, confidence=0.99))
        assert result.status == 'no_match'
        assert result.sources == []
        assert 'front' in result.notes
    finally:
        await providers.close()


async def test_branded_product_name_can_share_its_category_without_being_container_only():
    replies = iter([selection(), identity_response(name='Coconut Water', brand='Harmless Harvest',
                    category='Coconut Water', visible_text='Harmless Harvest Coconut Water')])
    providers = provider(lambda _: httpx.Response(200, json=next(replies)))
    try:
        result = await ProductGuide(providers).identify(image_url())
        assert result is not None
        assert result.brand == 'Harmless Harvest'
    finally:
        await providers.close()


@pytest.mark.parametrize('uncertainties, expected_size', [
    (['Size inferred from standard packaging, not explicitly visible on front'], None),
    (['Variant text is partly obscured'], None),
    ([], '8.4 fl oz'),
])
async def test_uncertain_package_size_is_omitted_without_losing_identity_or_uncertainty(uncertainties, expected_size):
    replies = iter([selection(), identity_response(name='Sugar Free', brand='Red Bull',
                    package_size='8.4 fl oz', uncertainties=uncertainties)])
    providers = provider(lambda _: httpx.Response(200, json=next(replies)))
    try:
        result = await ProductGuide(providers).identify(image_url())
        assert result is not None
        assert result.name == 'Sugar Free' and result.brand == 'Red Bull'
        assert result.package_size == expected_size
        assert result.uncertainties == uncertainties
    finally:
        await providers.close()


async def test_crop_cannot_return_nutrition_fields_as_identity():
    replies = iter([selection(), identity_response(carbohydrate_g=27)])
    providers = provider(lambda _: httpx.Response(200, json=next(replies)))
    try:
        with pytest.raises(ProviderError, match='identify the held package'):
            await ProductGuide(providers).identify(image_url())
    finally:
        await providers.close()


@pytest.mark.parametrize('url', ['not an image', 'data:image/jpeg;base64,!!!', 'data:image/gif;base64,YQ==',
                                'data:image/jpeg;base64,' + base64.b64encode(b'not an image').decode()])
async def test_invalid_image_never_calls_a_provider(url):
    providers = provider(lambda _: pytest.fail('Invalid image must not be uploaded'))
    try:
        with pytest.raises(ProviderError, match='valid JPEG'):
            await ProductGuide(providers).identify(url)
    finally:
        await providers.close()


async def test_over_twenty_megapixels_is_rejected_before_pixel_decode(monkeypatch):
    class HugeImage:
        width, height, format = 5000, 4001, 'PNG'
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def load(self): pytest.fail('Oversized image pixels must not be decoded')
    url = image_url()
    monkeypatch.setattr('app.product_guidance.Image.open', lambda _: HugeImage())
    providers = provider(lambda _: pytest.fail('Oversized image must not be uploaded'))
    try:
        with pytest.raises(ProviderError, match='20 megapixels'):
            await ProductGuide(providers).identify(url)
    finally:
        await providers.close()


@pytest.mark.parametrize('url', ['http://localhost/x', 'https://camera.local/x', 'http://127.0.0.1/',
    'http://[::1]/', 'http://10.0.0.1/', 'https://169.254.169.254/', 'http://2130706433/',
    'http://0x7f.0.0.1/', 'https://user:secret@manufacturer.com/x', 'file:///private/image',
    'https://manufacturer.com/\nunsafe', 'https://manufacturer.com:99999/'])
def test_catalog_sources_reject_nonpublic_and_unsafe_urls(url):
    with pytest.raises(ValidationError):
        CatalogSource(title='Invalid', url=url)


async def test_search_keeps_only_cited_spans_and_never_sends_question_or_health():
    cited = 'Possible Demo cereal.'
    response = interaction(cited + ' Uncited claim: 999 g carbohydrate.')
    block = response['steps'][1]['content'][0]
    block['annotations'][0]['end_index'] = len(cited)
    block['annotations'].append({'type': 'url_citation', 'url': 'http://localhost/private', 'title': 'Private'})
    requests = []
    providers = provider(lambda request: requests.append(json.loads(request.content)) or httpx.Response(200, json=response))
    try:
        result = await ProductGuide(providers).search(detected(visible_text='private medical notes', uncertainties=['private uncertainty']))
        assert result.status == 'grounded' and result.notes == cited
        assert len(result.sources) == 1
        sent = json.loads(requests[0]['input'])
        assert set(sent) == {'visual_product_query'}
        assert set(sent['visual_product_query']) == {'name', 'brand', 'variant', 'package_size', 'category'}
        assert 'private' not in requests[0]['input']
        assert 'private reasoning' not in result.model_dump_json()
    finally:
        await providers.close()


@pytest.mark.parametrize('status', ['failed', 'incomplete', 'in_progress', 'requires_action', None])
async def test_incomplete_search_cannot_become_grounded(status):
    providers = provider(lambda _: httpx.Response(200, json=interaction('Possible match', status=status)))
    try:
        with pytest.raises(ProviderError, match='did not complete'):
            await ProductGuide(providers).search(detected())
    finally:
        await providers.close()


async def test_uncited_search_text_is_not_forwarded_as_catalog_facts():
    providers = provider(lambda _: httpx.Response(200, json=interaction('Made up 27 g carbs.', sources=False)))
    try:
        result = await ProductGuide(providers).search(detected())
        assert result.status == 'no_match' and result.sources == [] and '27' not in result.notes
    finally:
        await providers.close()


@pytest.mark.parametrize('recipient,consented,expects_health', [('self', True, True), ('self', False, False), ('household', True, False)])
async def test_wording_receives_only_minimal_consented_context(recipient, consented, expects_health):
    requests = []
    reply = 'This looks like a possible Demo cereal match. Compare options for your stated preferences; this package is unverified.'
    providers = provider(lambda request: requests.append(json.loads(request.content)) or
        httpx.Response(200, json=interaction(json.dumps({'spoken_text': reply}), sources=False)))
    context = {'recipient_id': recipient, 'consented': consented, 'goals': ['Count carbohydrate'],
        'avoid_ingredients': ['Peanuts'], 'display_name': 'Private Person', 'patient_id': 'private-id',
        'nutrition_context': {'synthetic': True, 'conditions': ['Type 1 diabetes'], 'allergies': ['Peanuts'],
            'focus': ['Carbohydrate awareness'], 'dietary_preferences': [], 'medications': ['Private medicine'],
            'observations': {'cgm': 'private-CGM'}, 'name': 'Private Person'}}
    try:
        assert await ProductGuide(providers).respond(product(), 'What do you recommend?', context, []) == reply
        payload = requests[0]
        sent = json.loads(payload['input'])
        assert payload['store'] is False and 'tools' not in payload
        assert ('nutrition_context' in sent['context']) is expects_health
        assert ('goals' in sent['context']) is expects_health
        assert 'Private' not in payload['input'] and 'private-CGM' not in payload['input'] and 'private-id' not in payload['input']
        if expects_health:
            assert sent['context']['nutrition_context']['conditions'] == ['Type 1 diabetes']
            assert sent['context']['nutrition_context']['synthetic'] is True
        assert 'box' not in sent['product']['identity'] and 'visible_text' not in sent['product']['identity']
    finally:
        await providers.close()


@pytest.mark.parametrize('reply', [
    'The online listing reports 80 g carbohydrate per bar.',
    'This contains 27 g carbohydrate per bar.',
    'The online listing reports 27 g carbohydrate per 100 g.',
    'The online listing reports twenty-seven grams of carbohydrate per bar.',
    'This 296 mL bottle has 17 carbs.',
    'This has 17 calories.',
    'This will spike your glucose.',
    'This is peanut-free and safe for you.',
    'Take a dose of insulin.',
    ' '.join(['word'] * 66),
])
async def test_unsupported_numeric_treatment_and_allergy_claims_are_not_spoken(reply):
    providers = provider(lambda _: httpx.Response(200, json=interaction(json.dumps({'spoken_text': reply}), sources=False)))
    try:
        with pytest.raises(ProviderError):
            await ProductGuide(providers).respond(product(), 'What about this?', {}, [])
    finally:
        await providers.close()


async def test_attributed_amount_from_cited_listing_is_allowed_as_an_online_estimate():
    reply = 'This looks like a possible Demo cereal match. The online listing reports 27 g carbohydrate per bar, an estimate that does not verify this package.'
    providers = provider(lambda _: httpx.Response(200, json=interaction(json.dumps({'spoken_text': reply}), sources=False)))
    try:
        assert await ProductGuide(providers).respond(product(), 'What about this?', {}, []) == reply
    finally:
        await providers.close()


@pytest.mark.parametrize('reply', [
    'With Type 1 diabetes, compare total carbohydrate when considering this possible cereal match.',
    'This looks like a 296 mL bottle. For carbohydrate awareness, compare the cited options without assuming the exact formulation.',
    'With Type 1 diabetes, this possible cereal match warrants portion awareness. The online listing reports 27 g carbohydrate per bar, an estimate for an unverified package.',
])
async def test_diagnosis_and_package_numbers_are_not_mistaken_for_nutrient_amounts(reply):
    providers = provider(lambda _: httpx.Response(200, json=interaction(json.dumps({'spoken_text': reply}), sources=False)))
    try:
        assert await ProductGuide(providers).respond(product(), 'What about this?', {}, []) == reply
    finally:
        await providers.close()


async def test_qualified_catalog_amount_cannot_be_spoken_as_an_exact_amount():
    reply = 'The online listing reports 1 g carbohydrate per bar.'
    providers = provider(lambda _: httpx.Response(200, json=interaction(json.dumps({'spoken_text': reply}), sources=False)))
    try:
        with pytest.raises(ProviderError, match='nutrition amounts'):
            await ProductGuide(providers).respond(product('Possible cereal. Less than 1 g carbohydrate per listed bar.'), 'What about this?', {}, [])
    finally:
        await providers.close()


async def test_cancellation_propagates_without_retry_or_speech():
    started = asyncio.Event()
    calls = []
    async def handle(request):
        calls.append(request)
        started.set()
        await asyncio.Event().wait()
    providers = provider(handle)
    task = asyncio.create_task(ProductGuide(providers).identify(image_url()))
    try:
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(calls) == 1
    finally:
        await providers.close()


async def test_cancellation_during_constrained_fallback_prevents_crop_call():
    started = asyncio.Event()
    calls = []

    async def handle(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(200, json=selection(box=[]))
        assert 'tools' not in json.loads(request.content)
        started.set()
        await asyncio.Event().wait()

    providers = provider(handle)
    task = asyncio.create_task(ProductGuide(providers).identify(image_url()))
    try:
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(calls) == 2
    finally:
        await providers.close()
