"""Regression cases for incompatible catalog versions, without paid calls."""
import json

import httpx
import pytest

from app.models import Box
from app.product_guidance import CatalogMatch, CatalogSource, DetectedProduct, ProductGuide
from app.providers import ProviderError, Providers, Settings


def identity(name='The Peach Edition', variant='White Peach'):
    return DetectedProduct(name=name, brand='Red Bull', variant=variant, package_size='250 ml',
                           category='drink', visible_text=name, confidence=.95, uncertainties=[],
                           box=Box(x=.2, y=.2, w=.2, h=.4))


def entry(title, content, url='https://manufacturer.example/product'):
    return {'title': title, 'content': content, 'url': url}


def provider(results, *, text_provider='deepseek'):
    seen = []
    def handle(request):
        seen.append(request)
        if text_provider == 'gemini':
            return httpx.Response(200, json={'status': 'completed', 'outputs': [{
                'type': 'text', 'text': results[0]['content'], 'annotations': [{
                    'type': 'url_citation', 'title': item['title'], 'url': item['url']}
                    for item in results]}]})
        return httpx.Response(200, json={'results': results})
    return Providers(Settings(text_provider=text_provider, deepseek_key='test-only', gemini_key='test-only'),
                     httpx.MockTransport(handle)), seen


def saved_mixed_results():
    # The real replay mixed a sugar-free regional edition with a sugared product sheet.
    return [entry('Red Bull Pink Edition Sugarfree: White Peachsmak',
                  'Red Bull Peach Edition Sugarfree. Sukralos & Acesulfam K.',
                  'https://www.redbull.com/se-sv/energydrink/products/red-bull-peach-edition-sugarfree'),
            entry('PRODUKT - INFORMATION',
                  'Red Bull The Peach Edition White Peach 250 ml\nPer 100 Milliliter\n'
                  'Kohlenhydrate, davon Zucker (ungefähr 11.0Gramm)\nInhaltsstoffe: Wasser, Saccharose, Glucose.',
                  'https://retailer.example/products/peach.pdf'),
            entry('Red Bull Peach Edition Singles - 250 ml',
                  'Total Sugars 0g\nAdded Sugars 0g\nSUCRALOSE, ACESULFAME K.',
                  'https://retailer.example/red-bull-summer-edition-sugarfree')]


@pytest.mark.parametrize('name,variant', [('THE PEAC EDITION', 'White Peach'),
                                          ('The Peach Edition Sugarfree', 'White Peach')])
async def test_real_mixed_regional_peach_results_preserve_all_cited_versions(name, variant):
    p, seen = provider(saved_mixed_results())
    try:
        found = await ProductGuide(p).search(identity(name, variant))
        assert found.status == 'grounded' and len(found.sources) == 3
        assert all(item['url'] in found.notes and item['content'] in found.notes for item in saved_mixed_results())
        assert not found.notes.startswith('Catalog variant conflict:')
        assert len(seen) == 1
    finally:
        await p.close()


@pytest.mark.parametrize('source', [
    entry('Brand Sugar-Free Peach', 'Product ingredients.', 'https://manufacturer.example/peach'),
    entry('Brand Peach', 'Product ingredients.', 'https://manufacturer.example/peach-sugarfree'),
    entry('Brand Peach Zero Sugar', 'Product ingredients.'),
])
async def test_catalog_preserves_sugarfree_source_without_changing_visual_identity(source):
    p, seen = provider([source])
    try:
        visual = identity()
        match = await ProductGuide(p).search(visual)
        assert match.status == 'grounded' and len(match.sources) == 1
        assert visual.name == 'The Peach Edition' and visual.variant == 'White Peach'
        assert source['content'] in match.notes
        assert len(seen) == 1
    finally:
        await p.close()


@pytest.mark.parametrize('title,notes', [
    ('Brand Coconut Water No Added Sugar', 'Total sugars 17 g per bottle. Added sugars 0 g per bottle.'),
    ('Brand Peach', 'Total sugars 11 g per 100 ml. Our sugar-free version is also available.'),
    ('Brand Peach is not sugar-free', 'Total sugars 11 g per 100 ml.'),
    ('Brand Zero Caffeine', 'Total sugars 11 g per 100 ml.'),
    ('Brand Zero Alcohol', 'Total sugars 11 g per 100 ml.'),
])
async def test_no_added_sugar_footer_negation_and_other_zero_claims_are_not_sugarfree_variants(title, notes):
    p, seen = provider([entry(title, notes)])
    try:
        found = await ProductGuide(p).search(identity())
        assert found.status == 'grounded'
        assert len(seen) == 1
    finally:
        await p.close()


async def test_consistent_explicit_sugarfree_sources_remain_usable():
    p, seen = provider([entry('Brand Sugarfree', 'Total sugars 0 g per 250 ml.'),
                           entry('Brand Sugarfree', 'Sucralose and acesulfame K.', 'https://retailer.example/sugarfree')])
    try:
        found = await ProductGuide(p).search(identity('Brand Sugarfree', None))
        assert found.status == 'grounded' and len(found.sources) == 2
        assert len(seen) == 1
    finally:
        await p.close()


async def test_positive_sugar_in_explicit_related_products_footer_is_not_this_products_nutrition():
    p, _ = provider([entry('Brand Sugarfree', 'Total sugars 0 g per 250 ml.\n'
                          '## Related products\nBrand Original: 11 g total sugars per 100 ml.')])
    try:
        found = await ProductGuide(p).search(identity('Brand Sugarfree', None))
        assert found.status == 'grounded'
    finally:
        await p.close()


async def test_first_results_footer_cannot_hide_the_second_results_conflicting_nutrition():
    p, _ = provider([entry('Brand Sugarfree', 'Total sugars 0 g per 250 ml.\nRelated products: other drinks.'),
                     entry('Brand Original', '11 g total sugars per 100 ml.', 'https://retailer.example/original')])
    try:
        found = await ProductGuide(p).search(identity('Brand Sugarfree', None))
        assert found.status == 'grounded' and len(found.sources) == 2
        assert '11 g total sugars' in found.notes
    finally:
        await p.close()


async def test_different_positive_juice_amounts_do_not_create_a_sugarfree_variant_conflict():
    p, _ = provider([entry('Vegetable Juice', 'Total sugars 5 g per bottle (354 ml).'),
                     entry('Vegetable Juice', 'Total sugars 8 g per bottle (354 ml).', 'https://retailer.example/juice')])
    try:
        found = await ProductGuide(p).search(identity('Vegetable Juice', None))
        assert found.status == 'grounded' and '5 g' in found.notes and '8 g' in found.notes
    finally:
        await p.close()


async def test_conflicting_zero_and_positive_total_sugar_remains_cited_evidence():
    p, _ = provider([entry('Brand Juice', 'Total sugars 0 g per 100 ml.'),
                     entry('Brand Juice', 'Total sugars 11 g per 100 ml.', 'https://retailer.example/juice')])
    try:
        found = await ProductGuide(p).search(identity('Brand Juice', None))
        assert found.status == 'grounded' and len(found.sources) == 2
        assert 'Total sugars 0 g' in found.notes and 'Total sugars 11 g' in found.notes
    finally:
        await p.close()


async def test_gemini_search_also_preserves_cited_variant_evidence():
    p, seen = provider([entry('Brand Peach Sugarfree', 'Possible Brand Peach Sugarfree listing.')],
                       text_provider='gemini')
    try:
        found = await ProductGuide(p).search(identity())
        assert found.status == 'grounded' and len(found.sources) == 1 and len(seen) == 1
    finally:
        await p.close()


def product(name, notes, *, variant=None, title=None):
    return {'identity': identity(name, variant).model_dump(), 'catalog': CatalogMatch(status='grounded',
            notes=notes, sources=[CatalogSource(title=title or name, url='https://manufacturer.example/product')]).model_dump()}


async def test_saved_mixed_evidence_answers_actual_ingredient_question_with_uncertainty():
    calls = []
    def respond(request):
        body = json.loads(request.content)
        data = json.loads(body['messages'][-1]['content'])
        calls.append(data)
        assert data['variant_uncertainties'][0]['name'] == 'THE PEAC EDITION'
        assert data['product']['catalog']['sources']
        assert 'not the pictured package' in body['messages'][0]['content']
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps({'spoken_text': 'The Peach sugar-free listing names sucralose and acesulfame K; that does not establish which version you have.'})}}]})
    p = Providers(Settings(text_provider='deepseek', deepseek_key='test-only'), httpx.MockTransport(respond))
    stored = product('THE PEAC EDITION', '\n'.join(item['content'] for item in saved_mixed_results()),
                     title='Peach Edition Sugarfree')
    try:
        answer = await ProductGuide(p).respond(stored, 'What sweeteners are in the Peach?',
            {'recipient_id': 'self', 'consented': True, 'goals': ['Avoid sugar']}, [])
        assert 'sucralose' in answer and 'which version' in answer and len(calls) == 1
    finally:
        await p.close()


async def test_equal_explicit_sugarfree_products_pass_checked_tie_to_wording_model():
    calls = []
    expected = 'Brand Peach Sugarfree and Brand Original Sugarfree are tied for avoiding sugar according to their product listings; their different sweeteners do not establish a sugar advantage.'
    def respond(request):
        body = json.loads(request.content)
        calls.append(json.loads(body['messages'][-1]['content']))
        assert 'If both are sugar-free, state a tie' in body['messages'][0]['content']
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps({'spoken_text': expected})}}]})
    p = Providers(Settings(text_provider='deepseek', deepseek_key='test-only'), httpx.MockTransport(respond))
    try:
        answer = await ProductGuide(p).respond(
            product('Brand Peach Sugarfree', 'Sugar-free; sucralose and acesulfame K.'), 'Compare these',
            {'recipient_id': 'self', 'consented': True, 'goals': ['Avoid sugar']},
            [product('Brand Original Sugarfree', 'Total sugars 0 g per 250 ml.')])
        assert answer == expected and len(calls) == 1
        check = calls[0]['checked_nutrition_comparisons'][0]
        assert check['eligible'] and check['equal'] and check['lower_index'] is None
        assert not calls[0]['variant_uncertainties']
        assert len(answer.split()) <= 65 and len(answer) <= 600
    finally:
        await p.close()


@pytest.mark.parametrize('changes', [
    {'context': {'recipient_id': 'household', 'consented': True, 'goals': ['Avoid sugar']}},
    {'context': {'recipient_id': 'self', 'consented': False, 'goals': ['Avoid sugar']}},
    {'question': 'Compare their caffeine'},
    {'question': 'Which is cheaper?'},
    {'question': 'Compare their protein'},
    {'question': 'Which has less cholesterol?'},
    {'context': {'recipient_id': 'self', 'consented': True, 'goals': ['Choose more protein'],
                 'conversation_request': 'Choose more protein'}},
    {'names': ('Brand Juice No Added Sugar', 'Other Juice No Added Sugar')},
])
async def test_tie_does_not_inject_personal_goal_or_answer_an_unrelated_question(changes):
    seen = []
    def handle(request):
        seen.append(json.loads(json.loads(request.content)['messages'][-1]['content']))
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps({'spoken_text': 'The supplied information does not resolve that comparison.'})}}]})
    p = Providers(Settings(text_provider='deepseek', deepseek_key='test-only'), httpx.MockTransport(handle))
    names = changes.get('names', ('Brand Peach Sugarfree', 'Brand Original Sugarfree'))
    try:
        result = await ProductGuide(p).respond(product(names[0], 'No added sugar.'),
            changes.get('question', 'Compare these'), changes.get('context', {
                'recipient_id': 'self', 'consented': True, 'goals': ['Avoid sugar']}),
            [product(names[1], 'No added sugar.')])
        guarded = ('names' in changes or changes.get('question') in {
            'Compare their caffeine', 'Compare their protein', 'Which has less cholesterol?'}
            or changes.get('context', {}).get('conversation_request') == 'Choose more protein')
        assert 'does not resolve' in result and len(seen) == 1 and 'tied' not in result
        if guarded:
            assert seen[0]['checked_nutrition_comparisons']
            assert all(not check['eligible'] for check in seen[0]['checked_nutrition_comparisons'])
        if changes.get('context', {}).get('recipient_id') == 'household':
            assert seen[0]['context'] == {'recipient_id': 'household'}
        if changes.get('context', {}).get('consented') is False:
            assert 'goals' not in seen[0]['context']
    finally:
        await p.close()


async def test_explicit_conversation_protein_comparison_sends_current_goal_and_checked_direction():
    seen = []
    def handle(request):
        body = json.loads(request.content)
        seen.append(body)
        data = json.loads(body['messages'][-1]['content'])
        context = data['context']
        assert context['goals'] == ['Prefer more protein']
        assert context['conversation_request'] == 'Which gives more protein?'
        check = data['checked_nutrition_comparisons'][0]
        assert check['criterion'] == 'protein' and check['eligible'] and check['lower_index'] == 0
        assert 'current question and explicit conversation_request take precedence' in body['messages'][0]['content']
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps({'spoken_text': 'For equal amounts, the product listings show Brand Original Sugarfree has more protein than Brand Peach Sugarfree.'})}}]})
    p = Providers(Settings(text_provider='deepseek', deepseek_key='test-only'), httpx.MockTransport(handle))
    try:
        reply = await ProductGuide(p).respond(product('Brand Peach Sugarfree', 'Sugar-free. Protein 2g per 100g.'),
            'Which gives more protein?', {'recipient_id': 'self', 'consented': True,
                'goals': ['Prefer more protein'], 'conversation_request': 'Which gives more protein?'},
            [product('Brand Original Sugarfree', 'Sugar-free. Protein 4g per 100g.')])
        assert len(seen) == 1
        assert 'Brand Original Sugarfree has more protein than Brand Peach Sugarfree' in reply
    finally:
        await p.close()


@pytest.mark.parametrize('question', ['Compare these', 'Which has more sugar?'])
async def test_mixed_versions_reach_wording_model_with_explicit_uncertainties(question):
    calls = []
    expected = 'Blue Sugarfree is listed as sugar-free, while the Peach listings cover both regular and sugar-free versions; the Peach can’s version determines the sugar comparison.'
    def respond(request):
        body = json.loads(request.content)
        calls.append(json.loads(body['messages'][-1]['content']))
        assert 'do not state exact nutrient amounts, an unconditional winner or tie' in body['messages'][0]['content']
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps({'spoken_text': expected})}}]})
    p = Providers(Settings(text_provider='deepseek', deepseek_key='test-only'), httpx.MockTransport(respond))
    try:
        mixed = product('Peach', 'Total sugars 0 g per 250 ml. Total sugars 11 g per 100 ml.', title='Peach Sugarfree')
        known = product('Blue Sugarfree', 'Total sugars 0 g per 250 ml.')
        answer = await ProductGuide(p).respond(mixed, question,
            {'recipient_id': 'self', 'consented': True, 'goals': ['Avoid sugar']}, [known])
        assert answer == expected and len(calls) == 1
        assert calls[0]['variant_uncertainties'][0]['name'] == 'Peach'
        assert len(calls[0]['variant_uncertainties']) == 1
        assert 'Total sugars 11 g' in calls[0]['product']['catalog']['notes']
        assert 'choose' not in answer.lower() and '11' not in answer
    finally:
        await p.close()


@pytest.mark.parametrize('name,notes', [
    ('Peach Sugarfree', 'Total sugars 0 g per 250 ml. Total sugars 11 g per 100 ml.'),
    ('Peach Sugarfree', 'Sugarfree. Kohlenhydrate, davon Zucker (ungefähr 11.0Gramm)'),
])
def test_direct_sugarfree_tie_cannot_bypass_conflicting_positive_version(name, notes):
    from app.product_guidance import _product_evidence, _sugar_free_tie
    evidence = [_product_evidence(product(name, notes)),
                _product_evidence(product('Blue Sugarfree', 'Total sugars 0 g per 250 ml.'))]
    assert _sugar_free_tie(evidence, 'Compare these',
        {'recipient_id': 'self', 'consented': True, 'goals': ['Avoid sugar']}) is None


@pytest.mark.parametrize('companion', [
    product('Juice No Added Sugar', 'No added sugar. Total sugars 5 g per bottle.'),
    product('Other Peach', 'Total sugars 0 g per bottle. Total sugars 10 g per bottle.', title='Other Peach Sugarfree'),
])
async def test_no_added_sugar_or_two_ambiguous_products_cannot_supply_known_zero_anchor(companion):
    calls = []
    expected = 'The Peach listings mix regular and sugar-free versions, so this comparison remains unresolved from the available product information.'
    def respond(request):
        calls.append(json.loads(json.loads(request.content)['messages'][-1]['content']))
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps({'spoken_text': expected})}}]})
    p = Providers(Settings(text_provider='deepseek', deepseek_key='test-only'), httpx.MockTransport(respond))
    try:
        answer = await ProductGuide(p).respond(
            product('Peach', 'Total sugars 0 g per bottle. Total sugars 11 g per bottle.', title='Peach Sugarfree'),
            'Compare these', {'recipient_id': 'self', 'consented': True, 'goals': ['Avoid sugar']}, [companion])
        assert answer == expected and len(calls) == 1
        assert len(calls[0]['variant_uncertainties']) == (2 if companion['identity']['name'] == 'Other Peach' else 1)
        assert not calls[0]['checked_nutrition_comparisons'][0]['eligible']
        assert 'comparison remains unresolved' in answer and 'they tie' not in answer and 'lower' not in answer
        assert 'show' not in answer.lower()
    finally:
        await p.close()


async def test_ambiguous_numeric_formulation_retries_once_as_qualified_description():
    calls = []
    def reply(request):
        calls.append(json.loads(request.content))
        text = ('The online Peach listing has total sugars 11 g per 100 ml.' if len(calls) == 1 else
                'The Peach listings cover regular and sugar-free versions, so they do not establish the sugar amount in your can.')
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps({'spoken_text': text})}}]})
    p = Providers(Settings(text_provider='deepseek', deepseek_key='test-only'), httpx.MockTransport(reply))
    try:
        answer = await ProductGuide(p).respond(product('Peach', 'Total sugars 0 g per 100 ml. Total sugars 11 g per 100 ml.'),
            'How much sugar does Peach have?', {}, [])
        assert len(calls) == 2 and '11' not in answer and 'versions' in answer
        assert 'Omit ALL numerical' in calls[1]['messages'][0]['content']
    finally:
        await p.close()


async def test_restored_real_search_passes_service_pair_gate_to_wording_model(monkeypatch):
    from test_multi_product_turns import paired
    from test_product_turns import request
    service, fake, sid = paired()
    fake.items = [identity('Blue Sugarfree', 'Sugarfree'), identity('Peach', 'White Peach')]
    wording_calls = []
    expected = 'Blue Sugarfree is listed as sugar-free; the Peach listings cover both regular and sugar-free versions, so its exact version determines whether these differ on sugar.'
    def search(request):
        body = json.loads(request.content)
        if request.url.path.endswith('/chat/completions'):
            wording_calls.append(json.loads(body['messages'][-1]['content']))
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {
                'content': json.dumps({'spoken_text': expected})}}]})
        query = body['query']
        results = ([entry('Blue Sugarfree', 'Total sugars 0 g per 250 ml.')]
                   if 'Blue Sugarfree' in query else saved_mixed_results())
        return httpx.Response(200, json={'results': results})
    p = Providers(Settings(text_provider='deepseek', deepseek_key='test-only'), httpx.MockTransport(search))
    real = ProductGuide(p)
    monkeypatch.setattr(fake, 'search', real.search)
    monkeypatch.setattr(fake, 'respond', real.respond)
    try:
        result = await service.turn(sid, request('restored-pair', text='Which has less sugar?'))
        assert result.status == 'ready' and len(result.session.current_visual_product_ids) == 2
        assert all(item.catalog_status == 'grounded' for item in result.session.visual_products)
        assert result.reply == expected and len(wording_calls) == 1
        assert wording_calls[0]['variant_uncertainties'][0]['name'] == 'Peach'
        sent = [wording_calls[0]['product'], *wording_calls[0]['previous_products']]
        assert {item['provenance']['product_id'] for item in sent} == set(result.session.current_visual_product_ids)
        assert len(next(item for item in sent if item['identity']['name'] == 'Peach')['catalog']['sources']) == 3
    finally:
        await p.close()


async def test_more_protein_goal_supplies_direction_only_when_current_question_does_not_override():
    calls = []
    def respond(request):
        data = json.loads(json.loads(request.content)['messages'][-1]['content'])
        calls.append(data)
        text = ('Brand Peach has less protein than Brand Original for equal amounts, according to the product listings.'
                if 'less protein' in data['question'] else
                'Brand Original has more protein than Brand Peach for equal amounts, according to the product listings.')
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps({'spoken_text': text})}}]})
    p = Providers(Settings(text_provider='deepseek', deepseek_key='test-only'), httpx.MockTransport(respond))
    try:
        for question, expected in [('Compare these', 'Brand Original has more protein'),
                                   ('Which has less protein?', 'Brand Peach has less protein')]:
            answer = await ProductGuide(p).respond(product('Brand Peach', 'Protein 2g per 100g.'), question,
                {'recipient_id': 'self', 'consented': True, 'goals': ['Prefer more protein']},
                [product('Brand Original', 'Protein 4g per 100g.')])
            assert expected in answer
        assert len(calls) == 2
        assert [data['question'] for data in calls] == ['Compare these', 'Which has less protein?']
        assert all(data['context']['goals'] == ['Prefer more protein'] for data in calls)
        assert all(data['checked_nutrition_comparisons'][0]['lower_index'] == 0 for data in calls)
    finally:
        await p.close()


async def test_bare_compare_them_with_saved_mixed_results_uses_fresh_model_wording_and_same_evidence():
    results = saved_mixed_results()
    peach = product('THE PEAC EDITION', '\n\n'.join(
        f'Possible catalog result: {item["title"]}\nSource: {item["url"]}\n{item["content"]}'
        for item in results), variant='White Peach')
    peach['catalog']['sources'] = [{'title': item['title'], 'url': item['url']} for item in results]
    blue = product('Red Bull Sugarfree', 'Total sugars 0 g per 250 ml.')
    for value, product_id, left in [(peach, 'saved-peach', .6), (blue, 'saved-blue', .1)]:
        value.update(product_id=product_id, captured_at='2026-10-18T11:00:00Z')
        value['identity']['package_size'] = None
        value['identity']['box']['x'] = left
    responses = [
        'Red Bull Sugarfree is listed as sugar-free, while Peach has both regular and sugar-free listings; the Peach version determines the difference for your goal.',
        'The useful difference is the Peach version: its regular listing contains sugar, while its sugar-free listing matches Red Bull Sugarfree for avoiding sugar.',
    ]
    calls = []
    def respond(request):
        assert request.url.host == 'api.deepseek.com'
        body = json.loads(request.content)
        calls.append(json.loads(body['messages'][-1]['content']))
        assert 'tools' not in body
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps({'spoken_text': responses[len(calls) - 1]})}}]})
    p = Providers(Settings(text_provider='deepseek', deepseek_key='test-only'), httpx.MockTransport(respond))
    context = {'recipient_id': 'self', 'consented': True, 'goals': ['Avoid sugar']}
    try:
        guide = ProductGuide(p)
        first = await guide.respond(peach, 'Compare them', context, [blue])
        memory = [{'question': 'Compare them', 'answer': first}]
        second = await guide.respond(peach, 'Compare them', {**context, 'recent_product_exchanges': memory}, [blue])
        assert [first, second] == responses and len(calls) == 2
        assert all('if yours contains sugar' not in answer for answer in responses)
        for data in calls:
            assert data['question'] == 'Compare them'
            assert data['product']['catalog'] == peach['catalog']
            assert data['previous_products'][0]['catalog'] == blue['catalog']
            assert data['product']['provenance']['product_id'] == 'saved-peach'
            assert data['previous_products'][0]['provenance']['product_id'] == 'saved-blue'
            assert data['product']['provenance']['source_box']['x'] == .6
            assert data['product']['identity']['package_size'] is None
            assert data['previous_products'][0]['identity']['package_size'] is None
            assert data['variant_uncertainties'][0]['name'] == 'THE PEAC EDITION'
            assert len(data['variant_uncertainties']) == 1
            assert data['checked_nutrition_comparisons'][0]['criterion'] == 'sugar'
            # The English numeric rows say zero; the German positive-sugar
            # version is still a separate material variant uncertainty.
            assert data['checked_nutrition_comparisons'][0]['equal']
        assert calls[1]['context']['recent_product_exchanges'] == memory
    finally:
        await p.close()


@pytest.mark.parametrize('question', ['Is Peach sugar-free?', 'Does Peach contain sugar?', 'Does it have any sugar?',
    'Hey, uh, is Peach sugar-free?', 'Can you tell me whether Peach is sugar-free?',
    'Tell me if Peach is sugar-free.'])
async def test_ambiguous_sugar_status_never_becomes_an_unqualified_physical_claim(question):
    calls = []
    def wrong_answer(request):
        calls.append(request)
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps({'spoken_text': 'Peach is sugar-free.'})}}]})
    p = Providers(Settings(text_provider='deepseek', deepseek_key='test-only'), httpx.MockTransport(wrong_answer))
    try:
        answer = await ProductGuide(p).respond(product('Peach', 'Total sugars 0 g per 100 ml. Total sugars 11 g per 100 ml.'),
            question, {}, [])
        assert 'version is unresolved' in answer and not calls
        assert 'show' not in answer.lower() and 'Peach is sugar-free' not in answer
    finally:
        await p.close()


async def test_known_named_sugar_status_still_uses_own_listing_despite_ambiguous_companion():
    calls = []
    def answer(request):
        calls.append(request)
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps({'spoken_text': 'The Blue Sugarfree listings describe it as sugar-free.'})}}]})
    p = Providers(Settings(text_provider='deepseek', deepseek_key='test-only'), httpx.MockTransport(answer))
    try:
        reply = await ProductGuide(p).respond(product('Peach', 'Total sugars 0 g per 100 ml. Total sugars 11 g per 100 ml.'),
            'Is Blue Sugarfree sugar-free?', {}, [product('Blue Sugarfree', 'Total sugars 0 g per 100 ml.')])
        assert len(calls) == 1 and 'Blue Sugarfree' in reply and 'unresolved' not in reply
    finally:
        await p.close()
