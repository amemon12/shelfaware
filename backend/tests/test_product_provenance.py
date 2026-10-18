"""Box/source identity and per-product numeric attribution; no provider calls."""
import asyncio
import json

import httpx
import pytest

from app.models import Box
from app.product_guidance import ProductGuide, _product_evidence, _validate_spoken
from app.providers import ProviderError, Providers, Settings
from test_multi_product_turns import paired, request


STAMP = '2026-10-18T14:00:00Z'


def product(name='Alpha', amount=2, basis='100 mL', *, left=.1, product_id='alpha', stamp=STAMP):
    return {'product_id': product_id, 'captured_at': stamp,
        'identity': {'name': name, 'brand': 'Demo', 'variant': None, 'package_size': None,
            'category': 'drink', 'visible_text': 'PRIVATE CROPPED TEXT', 'confidence': .95,
            'uncertainties': [], 'box': Box(x=left, y=.2, w=.2, h=.5).model_dump()},
        'catalog': {'status': 'grounded', 'notes': f'{name}: total sugar {amount} g per {basis}.',
            'sources': [{'title': name, 'url': f'https://manufacturer.example/{product_id}'}]}}


def pair():
    return [_product_evidence(product()),
            _product_evidence(product('Beta', 9, left=.6, product_id='beta'))]


@pytest.mark.parametrize('reply', [
    'The online listings say Alpha has 2 g total sugar per 100 mL and Beta has 9 g total sugar per 100 mL.',
    'The manufacturer lists Beta with 9 g total sugar per 100 mL; Alpha has 2 g total sugar per 100 mL.',
])
def test_each_named_amount_uses_its_own_product_regardless_of_payload_order(reply):
    assert _validate_spoken(reply, pair()) == reply
    assert _validate_spoken(reply, pair()[::-1]) == reply


@pytest.mark.parametrize('reply', [
    'According to online listings, Alpha has 9 g total sugar per 100 mL and Beta has 2 g total sugar per 100 mL.',
    'The online listings say Demo has 2 g sugar per 100 mL.',
    'The online listings say Alpha and Beta have 9 g sugar per 100 mL.',
    'The online listings say Alpha, Beta: 9 g sugar per 100 mL.',
    'The online listings say "Alpha" and "Beta" have 9 g sugar per 100 mL.',
    'The online listings say Alpha has 2 g sugar per 100 mL and the other one has 2 g sugar per 100 mL.',
    'The online listings say Alpha has 2 g sugar per 100 mL and the other has 2 g sugar per 100 mL.',
    'Choose Alpha. The online listing says it has 2 g sugar per 100 mL.',
    'The online listings say Gamma has 2 g sugar per 100 mL.',
])
def test_swapped_shared_brand_grouped_or_unresolved_claims_fail_closed(reply):
    with pytest.raises(ProviderError, match='nutrition amounts'):
        _validate_spoken(reply, pair())


def test_duplicate_product_names_are_not_unique_numeric_attribution():
    candidates = [_product_evidence(product()),
                  _product_evidence(product('Alpha', 9, left=.6, product_id='second-alpha'))]
    with pytest.raises(ProviderError, match='attributed'):
        _validate_spoken('The online listing says Alpha has 2 g sugar per 100 mL.', candidates)


def test_longer_full_name_wins_over_another_products_shorter_name():
    candidates = [_product_evidence(product()),
                  _product_evidence(product('Alpha Plus', 9, left=.6, product_id='plus'))]
    reply = 'The online listings say Alpha Plus has 9 g sugar per 100 mL and Alpha has 2 g sugar per 100 mL.'
    assert _validate_spoken(reply, candidates) == reply


def test_serving_basis_cannot_be_borrowed_from_the_other_product():
    candidates = [_product_evidence(product(basis='bottle')),
                  _product_evidence(product('Beta', 2, basis='240 mL', left=.6, product_id='beta'))]
    with pytest.raises(ProviderError, match='serving basis'):
        _validate_spoken('The online listings say Alpha has 2 g sugar per 240 mL.', candidates)
    reply = 'The online listings say Alpha has 2 g sugar per bottle and Beta has 2 g sugar per 240 mL.'
    assert _validate_spoken(reply, candidates) == reply


@pytest.mark.parametrize('name', ['Formula 2', 'Demo Zero Sugar'])
def test_a_number_in_the_full_product_name_is_not_a_nutrition_claim(name):
    candidates = [_product_evidence(product(name)), pair()[1]]
    reply = f'{name} and Beta are different products; the listings do not establish the requested difference.'
    assert _validate_spoken(reply, candidates) == reply
    # A zero-sugar name with positive sugar is a real variant conflict, not a
    # name-masking fixture. Use a supported non-sugar nutrient for that case.
    nutrient = 'protein' if name == 'Demo Zero Sugar' else 'total sugar'
    if name == 'Demo Zero Sugar':
        candidates[0]['catalog']['notes'] = 'Protein 2 g per 100 mL.'
    numeric = f'The online listings say {name} has 2 g {nutrient} per 100 mL.'
    assert _validate_spoken(numeric, candidates) == numeric


def test_caffeine_amounts_are_also_bound_to_the_named_product():
    candidates = pair()
    candidates[0]['catalog']['notes'] = 'Alpha: caffeine 20 mg per 100 mL.'
    candidates[1]['catalog']['notes'] = 'Beta: caffeine 80 mg per 100 mL.'
    with pytest.raises(ProviderError, match='nutrition amounts'):
        _validate_spoken('The online listings say Alpha has 80 mg caffeine per 100 mL.', candidates)
    reply = 'The online listing says Alpha has 20 mg caffeine per 100 mL.'
    assert _validate_spoken(reply, candidates) == reply


def test_spatial_claims_resolve_same_photo_boxes_not_payload_order():
    reply = 'The online listings say the left product has 2 g sugar per 100 mL and the right product has 9 g sugar per 100 mL.'
    assert _validate_spoken(reply, pair()) == reply
    assert _validate_spoken(reply, pair()[::-1]) == reply
    with pytest.raises(ProviderError, match='nutrition amounts'):
        _validate_spoken(reply.replace('left product', 'right product', 1), pair())


@pytest.mark.parametrize('change', ['different_capture', 'missing_id', 'same_id', 'overlapping_boxes'])
def test_spatial_claims_require_two_distinct_products_in_one_unambiguous_photo(change):
    candidates = pair()
    origin = candidates[1]['provenance']
    if change == 'different_capture': origin['captured_at'] = '2026-10-18T14:00:01Z'
    elif change == 'missing_id': origin['product_id'] = None
    elif change == 'same_id': origin['product_id'] = candidates[0]['provenance']['product_id']
    else: origin['source_box']['x'] = .2
    with pytest.raises(ProviderError, match='attributed'):
        _validate_spoken('The online listings say the left product has 2 g sugar per 100 mL.', candidates)


async def test_out_of_order_search_completion_keeps_box_id_source_mapping():
    service, guide, sid = paired()
    original = guide.search
    right_finished = asyncio.Event()
    async def search(identity):
        if identity.name == 'Sugarfree':
            await asyncio.wait_for(right_finished.wait(), timeout=1)
        result = await original(identity)
        if identity.name == 'Peach': right_finished.set()
        return result
    guide.search = search
    result = await service.turn(sid, request())
    by_name = {item.name: item for item in result.session.visual_products}
    for payload, name, x in [(guide.current, 'Peach', .6), (guide.previous[0], 'Sugarfree', .1)]:
        assert payload['product_id'] == by_name[name].id
        assert payload['captured_at'] == by_name[name].captured_at
        assert payload['identity']['box']['x'] == x
        assert payload['catalog']['sources'][0]['url'] == 'https://example.com/' + name
    assert by_name['Peach'].captured_at == by_name['Sugarfree'].captured_at


async def test_swapped_draft_uses_one_qualitative_rewrite_with_unchanged_product_provenance():
    calls = []
    bad = 'The online listings say Alpha has 9 g sugar per 100 mL and Beta has 2 g sugar per 100 mL.'
    good = 'The supplied listings describe less sugar in Alpha than Beta on the same serving basis.'
    def handle(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={'stop_reason': 'end_turn', 'content': [
            {'type': 'text', 'text': json.dumps({'spoken_text': bad if len(calls) == 1 else good})}]})
    providers = Providers(Settings(text_provider='anthropic', anthropic_key='test'), httpx.MockTransport(handle))
    try:
        answer = await ProductGuide(providers).respond(product(), 'How much sugar in each?', {},
            [product('Beta', 9, left=.6, product_id='beta')])
        assert answer == good and len(calls) == 2
        assert calls[0]['messages'] == calls[1]['messages']
        payload = json.loads(calls[0]['messages'][0]['content'][0]['text'])
        assert payload['product']['provenance']['product_id'] == 'alpha'
        assert payload['product']['provenance']['source_box']['x'] == .1
        assert payload['previous_products'][0]['provenance']['product_id'] == 'beta'
        assert payload['previous_products'][0]['provenance']['source_box']['x'] == .6
        assert 'PRIVATE CROPPED TEXT' not in json.dumps(calls)
        assert bad not in calls[1]['system']
        assert 'same capture timestamp' in calls[0]['system']
    finally:
        await providers.close()
