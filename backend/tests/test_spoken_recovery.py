import asyncio
import json
import re

import httpx
import pytest

from app.models import Box
from app.product_guidance import CatalogMatch, CatalogSource, DetectedProduct, ProductGuide
from app.providers import ProviderError, Providers, Settings


VALID_REPLY = 'For your avoid-sugar goal, Demo Sugarfree looks like the better option. The cited manufacturer information describes it as sugar-free; this is a possible catalog match.'
OVERLONG = 'DRAFT_MARKER ' + ' '.join(['word'] * 65)
UNSUPPORTED_NUTRIENT = 'REJECTED_NUMERIC_MARKER: The online listing reports 27 g carbohydrate per bar.'


def product(name='Demo Sugarfree', notes='The manufacturer describes Demo Sugarfree as sugar-free.'):
    identity = DetectedProduct(name=name, brand='Demo', variant=name, package_size=None,
        category='drink', visible_text='PRIVATE_IMAGE_TEXT', confidence=0.95, uncertainties=[],
        box=Box(x=0.1, y=0.2, w=0.3, h=0.4))
    match = CatalogMatch(status='grounded', notes=notes,
        sources=[CatalogSource(title=name, url='https://manufacturer.example/' + name.split()[-1].lower())])
    return {'identity': identity.model_dump(), 'catalog': match.model_dump()}


def response(provider, value):
    content = value if isinstance(value, str) else json.dumps(value)
    if provider == 'deepseek':
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': content}}]}
    return {'status': 'completed', 'steps': [{'type': 'model_output',
        'content': [{'type': 'text', 'text': content}]}]}


def setup(provider, replies):
    requests = []
    remaining = iter(replies)

    def handle(request):
        assert request.url.host == ('api.deepseek.com' if provider == 'deepseek' else 'generativelanguage.googleapis.com')
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=response(provider, next(remaining)))

    client = Providers(Settings(text_provider=provider, deepseek_key='test-deepseek', gemini_key='test-gemini'),
                       httpx.MockTransport(handle))
    return client, ProductGuide(client), requests


def sent_data(provider, payload):
    return json.loads(payload['messages'][-1]['content'] if provider == 'deepseek' else payload['input'])


def sent_instruction(provider, payload):
    return payload['messages'][0]['content'] if provider == 'deepseek' else payload['system_instruction']


@pytest.mark.parametrize('provider', ['deepseek', 'gemini'])
async def test_valid_short_answer_does_not_request_a_rewrite(provider):
    client, guide, requests = setup(provider, [{'spoken_text': VALID_REPLY}])
    try:
        assert await guide.respond(product(), 'Does this fit?', {}, []) == VALID_REPLY
        assert len(requests) == 1
    finally:
        await client.close()


@pytest.mark.parametrize('provider', ['deepseek', 'gemini'])
@pytest.mark.parametrize('draft', [OVERLONG, '', ' \n\t ', ' '.join(['x' * 125] * 5)])
async def test_length_or_blank_draft_gets_one_regeneration(provider, draft):
    client, guide, requests = setup(provider, [{'spoken_text': draft}, {'spoken_text': VALID_REPLY}])
    try:
        result = await guide.respond(product(), 'Does this fit?', {'recipient_id': 'self',
            'consented': True, 'goals': ['Avoid sugar']}, [])
        assert result == VALID_REPLY
        assert len(result.split()) <= 65 and len(result) <= 600
        assert len(requests) == 2
    finally:
        await client.close()


@pytest.mark.parametrize('provider', ['deepseek', 'gemini'])
@pytest.mark.parametrize('draft', [OVERLONG, '  '])
async def test_second_invalid_length_fails_after_two_calls_with_provider_neutral_error(provider, draft):
    client, guide, requests = setup(provider, [{'spoken_text': draft}, {'spoken_text': draft}])
    try:
        with pytest.raises(ProviderError) as error:
            await guide.respond(product(), 'Does this fit?', {}, [])
        assert len(requests) == 2
        assert 'gemini' not in str(error.value).lower() and 'deepseek' not in str(error.value).lower()
        assert 'DRAFT_MARKER' not in str(error.value)
    finally:
        await client.close()


@pytest.mark.parametrize('provider', ['deepseek', 'gemini'])
@pytest.mark.parametrize('claim', [
    'Take an insulin dose.',
    'This is peanut-free and safe for you.',
    'This will spike your glucose.',
    'The online listing reports 27 g carbohydrate per bar. Take an insulin dose.',
    'The online listing reports 27 g carbohydrate per bar. This is safe for your peanut allergy.',
    'The online listing reports 27 g carbohydrate per bar. This will spike your glucose.',
])
async def test_unsafe_overlong_draft_fails_semantics_without_regeneration(provider, claim):
    draft = claim + ' ' + OVERLONG
    client, guide, requests = setup(provider, [{'spoken_text': draft}])
    try:
        with pytest.raises(ProviderError):
            await guide.respond(product(), 'Does this fit?', {}, [])
        assert len(requests) == 1
    finally:
        await client.close()


@pytest.mark.parametrize('provider', ['deepseek', 'gemini'])
@pytest.mark.parametrize('value', [
    'not JSON', {'spoken_text': 'Hello.', 'extra': 'not allowed'}, {'spoken_text': 12},
    {'spoken_text': None}, {}, {'spoken_text': 'x' * 4001},
])
async def test_malformed_draft_does_not_trigger_length_recovery(provider, value):
    client, guide, requests = setup(provider, [value])
    try:
        with pytest.raises(ProviderError):
            await guide.respond(product(), 'Does this fit?', {}, [])
        assert len(requests) == 1
    finally:
        await client.close()


@pytest.mark.parametrize('provider', ['deepseek', 'gemini'])
@pytest.mark.parametrize('claim', [
    'The online listing reports 27 g carbohydrate per bar.',
    'This is safe for your peanut allergy.',
    'Take an insulin dose.',
])
async def test_regenerated_answer_still_must_pass_all_semantic_guards(provider, claim):
    client, guide, requests = setup(provider, [{'spoken_text': OVERLONG}, {'spoken_text': claim}])
    try:
        with pytest.raises(ProviderError):
            await guide.respond(product(), 'Does this fit?', {}, [])
        assert len(requests) == 2
    finally:
        await client.close()


@pytest.mark.parametrize('provider', ['deepseek', 'gemini'])
async def test_regeneration_cannot_change_supported_nutrient_serving_basis(provider):
    client, guide, requests = setup(provider, [{'spoken_text': OVERLONG},
        {'spoken_text': 'The online listing reports 27 g carbohydrate per bottle.'}])
    try:
        with pytest.raises(ProviderError, match='serving basis'):
            await guide.respond(product(notes='The online listing reports 27 g carbohydrate per bar.'),
                                'Does this fit?', {}, [])
        assert len(requests) == 2
    finally:
        await client.close()


@pytest.mark.parametrize('provider', ['deepseek', 'gemini'])
@pytest.mark.parametrize('recipient', ['self', 'household'])
@pytest.mark.parametrize('draft', [OVERLONG, UNSUPPORTED_NUTRIENT])
async def test_regeneration_reuses_sanitized_evidence_without_sending_rejected_draft(provider, recipient, draft):
    client, guide, requests = setup(provider, [{'spoken_text': draft}, {'spoken_text': VALID_REPLY}])
    prior = product('Demo Peach', notes='Manufacturer says Demo Peach contains 10 g total sugar per 100 mL.')
    context = {'recipient_id': recipient, 'consented': True, 'goals': ['Avoid sugar'],
               'display_name': 'PRIVATE_NAME', 'private_record': 'PRIVATE_RECORD',
               'nutrition_context': {'synthetic': True, 'conditions': ['PRIVATE_CONDITION'],
                                     'medications': ['PRIVATE_MEDICATION']}}
    try:
        await guide.respond(product(), 'Compare their ingredients', context, [prior])
        assert len(requests) == 2
        first, second = [sent_data(provider, payload) for payload in requests]
        assert first == second
        assert first['product']['catalog']['sources'] == product()['catalog']['sources']
        assert first['previous_products'][0]['catalog']['sources'] == prior['catalog']['sources']
        assert first['product']['identity']['name'] == 'Demo Sugarfree'
        assert first['previous_products'][0]['identity']['name'] == 'Demo Peach'
        for payload in requests:
            serialized = json.dumps(payload)
            assert 'DRAFT_MARKER' not in serialized and 'PRIVATE_IMAGE_TEXT' not in serialized
            assert 'REJECTED_NUMERIC_MARKER' not in serialized
            assert 'PRIVATE_NAME' not in serialized and 'PRIVATE_RECORD' not in serialized
            assert 'PRIVATE_MEDICATION' not in serialized
            assert 'tools' not in payload
        if recipient == 'household':
            assert first['context'] == {'recipient_id': 'household'}
            assert 'PRIVATE_CONDITION' not in json.dumps(requests)
        else:
            assert first['context']['goals'] == ['Avoid sugar']
    finally:
        await client.close()


@pytest.mark.parametrize('provider', ['deepseek', 'gemini'])
async def test_comparison_instruction_requests_named_recommendation_supported_by_evidence(provider):
    client, guide, requests = setup(provider, [{'spoken_text': VALID_REPLY}])
    try:
        await guide.respond(product(), 'Compare their ingredients. Which is better?',
            {'recipient_id': 'self', 'consented': True, 'goals': ['Avoid sugar']}, [product('Demo Peach', notes='Total sugar 10 g per 100 mL.')])
        instruction = sent_instruction(provider, requests[0]).lower()
        assert re.search(r'previous|prior', instruction)
        assert re.search(r'first sentence|lead with|start with', instruction)
        assert 'name' in instruction and re.search(r'winner|better|recommend', instruction)
        assert re.search(r'evidence supports|supported by|evidence is sufficient', instruction)
    finally:
        await client.close()


@pytest.mark.parametrize('provider', ['deepseek', 'gemini'])
@pytest.mark.parametrize('status', [401, 402, 403, 429, 500])
async def test_provider_errors_never_trigger_wording_retry(provider, status):
    requests = []
    client = Providers(Settings(text_provider=provider, deepseek_key='test', gemini_key='test'),
        httpx.MockTransport(lambda request: requests.append(request) or httpx.Response(status, text='private error')))
    try:
        with pytest.raises(ProviderError):
            await ProductGuide(client).respond(product(), 'Does this fit?', {}, [])
        assert len(requests) == 1
    finally:
        await client.close()


@pytest.mark.parametrize('provider', ['deepseek', 'gemini'])
async def test_cancellation_during_regeneration_propagates_without_further_request(provider):
    requests = []
    started = asyncio.Event()

    async def handle(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(200, json=response(provider, {'spoken_text': OVERLONG}))
        started.set()
        await asyncio.Event().wait()

    client = Providers(Settings(text_provider=provider, deepseek_key='test', gemini_key='test'), httpx.MockTransport(handle))
    task = asyncio.create_task(ProductGuide(client).respond(product(), 'Does this fit?', {}, []))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(requests) == 2
    finally:
        task.cancel()
        await client.close()


@pytest.mark.parametrize('provider', ['deepseek', 'gemini'])
@pytest.mark.parametrize('draft,notes', [
    (UNSUPPORTED_NUTRIENT, 'The manufacturer describes Demo Sugarfree as sugar-free.'),
    ('The online listing reports 27 g carbohydrate per bottle.',
     'The online listing reports 27 g carbohydrate per bar.'),
    ('The online listing reports about 99 g sugar per bottle.',
     'Nutrition Facts\n1 bottle (296 mL)\nTotal Sugars17g'),
    (UNSUPPORTED_NUTRIENT + ' ' + OVERLONG, 'The manufacturer describes Demo Sugarfree as sugar-free.'),
])
async def test_numeric_or_basis_failure_allows_one_fresh_qualitative_response(provider, draft, notes):
    # This numeric-recovery case needs a consistent identity. A Sugarfree title
    # paired with 17 g total sugar is now rejected by the earlier variant gate.
    contains_sugar = 'Total Sugars17g' in notes
    name = 'Demo juice' if contains_sugar else 'Demo Sugarfree'
    valid = ('The cited listing shows this juice contains sugar. Compare total sugar with the other option '
             'before choosing for your avoid-sugar goal.') if contains_sugar else VALID_REPLY
    client, guide, requests = setup(provider, [{'spoken_text': draft}, {'spoken_text': valid}])
    try:
        assert await guide.respond(product(name=name, notes=notes), 'Which is better?',
            {'recipient_id': 'self', 'consented': True, 'goals': ['Avoid sugar']}, []) == valid
        assert len(requests) == 2
        assert sent_data(provider, requests[0]) == sent_data(provider, requests[1])
        instruction = sent_instruction(provider, requests[1]).lower()
        assert 'qualitative' in instruction
        assert re.search(r'no numeric|without numeric|omit (?:all )?(?:numeric|numerical)|do not (?:use|include).*\b(?:numbers|quantities)', instruction)
        assert draft not in json.dumps(requests[1])
        assert 'REJECTED_NUMERIC_MARKER' not in json.dumps(requests[1])
        assert 'tools' not in requests[1]
    finally:
        await client.close()


@pytest.mark.parametrize('provider', ['deepseek', 'gemini'])
@pytest.mark.parametrize('first,second', [
    (UNSUPPORTED_NUTRIENT, UNSUPPORTED_NUTRIENT),
    (OVERLONG, UNSUPPORTED_NUTRIENT),
    (UNSUPPORTED_NUTRIENT, OVERLONG),
    (UNSUPPORTED_NUTRIENT, 'The online listing reports 27 g carbohydrate per bottle.'),
])
async def test_length_and_nutrition_recovery_share_one_two_call_budget(provider, first, second):
    client, guide, requests = setup(provider, [{'spoken_text': first}, {'spoken_text': second}])
    try:
        with pytest.raises(ProviderError) as error:
            await guide.respond(product(), 'Which is better?', {}, [])
        assert len(requests) == 2
        assert 'REJECTED_NUMERIC_MARKER' not in str(error.value)
        assert 'DRAFT_MARKER' not in str(error.value)
        assert 'gemini' not in str(error.value).lower() and 'deepseek' not in str(error.value).lower()
    finally:
        await client.close()


@pytest.mark.parametrize('provider', ['deepseek', 'gemini'])
@pytest.mark.parametrize('second', [
    'Take an insulin dose.',
    'This is safe for your peanut allergy.',
    'This will spike your glucose.',
])
async def test_qualitative_regeneration_retains_treatment_allergy_glucose_guards(provider, second):
    client, guide, requests = setup(provider, [{'spoken_text': UNSUPPORTED_NUTRIENT}, {'spoken_text': second}])
    try:
        with pytest.raises(ProviderError, match='treatment, allergy-safety, or glucose'):
            await guide.respond(product(), 'Which is better?', {}, [])
        assert len(requests) == 2
    finally:
        await client.close()


@pytest.mark.parametrize('provider', ['deepseek', 'gemini'])
@pytest.mark.parametrize('recovers', [True, False])
async def test_pair_comparison_keeps_numeric_validation_and_shared_retry_budget(provider, recovers):
    unsupported = 'The online listing says Demo Alpha has 99 g fat per 20 g.'
    valid = 'For equal amounts, the product listings show Demo Beta has less fat than Demo Alpha.'
    client, guide, requests = setup(provider, [
        {'spoken_text': unsupported}, {'spoken_text': valid if recovers else unsupported}])
    current = product('Demo Alpha', notes='Fat 4g per 20g.')
    previous = product('Demo Beta', notes='Fat 6g per 100g.')
    try:
        if recovers:
            assert await guide.respond(current, 'Which has less fat?', {}, [previous]) == valid
        else:
            with pytest.raises(ProviderError, match='nutrition amounts'):
                await guide.respond(current, 'Which has less fat?', {}, [previous])
        assert len(requests) == 2
        first, second = [sent_data(provider, payload) for payload in requests]
        assert first == second
        check = first['checked_nutrition_comparisons'][0]
        assert check['eligible'] and check['lower_index'] == 1
        assert first['product']['catalog'] == current['catalog']
        assert first['previous_products'][0]['catalog'] == previous['catalog']
        assert 'Omit ALL numerical' in sent_instruction(provider, requests[1])
        assert unsupported not in json.dumps(requests[1])
    finally:
        await client.close()
