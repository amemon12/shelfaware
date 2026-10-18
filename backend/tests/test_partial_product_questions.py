from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.product_guidance import ProductGuide
from app.product_questions import grounded_named_target
from test_conversation_product_search import setup, photo


def unresolved_pair():
    service, guide, sid, brain = setup()
    guide.items = [guide.items[0].model_copy(update={'name': 'Red Bull Sugar Free', 'brand': 'Red Bull', 'variant': 'Sugar Free'}),
                   guide.items[1].model_copy(update={'name': 'Red Bull Peach Edition', 'brand': 'Red Bull', 'variant': 'White Peach'})]
    original = guide.search
    async def search(identity):
        if identity.name == 'Red Bull Peach Edition':
            await guide.gate('search')
            return SimpleNamespace(status='no_match', sources=[], notes='Catalog variant conflict: regular and sugar-free versions conflict. Show the front.')
        match = await original(identity)
        match.sources = [SimpleNamespace(model_dump=lambda: {'title': identity.name, 'url': 'https://example.com/sugarfree'})]
        return match
    guide.search = search
    return service, guide, sid, brain


@pytest.mark.parametrize('question', ['What are these?', 'Um, what are these two?', 'What am I holding?'])
async def test_identification_uses_only_visual_names_even_with_unresolved_catalog(question):
    service, guide, sid, brain = unresolved_pair()
    await brain.respond('Compare these', photo)
    # Use the service for the already selected photo; the voice fast path may
    # request a fresh view for an explicit "what are these" capture action.
    from test_product_turns import request
    result = await service.turn(sid, request('identify-current', image=False, text=question))
    assert result.status == 'ready' and result.reply.startswith('These appear to be ')
    assert 'Red Bull Sugar Free' in result.reply and 'Red Bull Peach Edition' in result.reply
    assert 'better' not in result.reply and '0 g' not in result.reply
    assert 'respond' not in guide.calls and len(result.session.current_visual_product_ids) == 2
    assert [p.catalog_status for p in result.session.visual_products] == ['grounded', 'no_match']


async def test_fresh_identification_still_succeeds_when_lookup_fails():
    service, guide, _, brain = unresolved_pair()
    guide.failed_name = 'Red Bull Sugar Free'
    result = await brain.respond('What are these?', photo)
    assert result['status'] == 'ready' and 'appear to be' in result['reply']
    assert len(result['snapshot']['products']) == 2 and 'respond' not in guide.calls


@pytest.mark.parametrize('question', ['What sweeteners are in Red Bull Sugar Free?',
    'Does Sugarfree contain caffeine?', 'Which ingredients are in Sugar Free?'])
async def test_named_known_product_question_isolated_without_changing_pair_or_memory_scope(question):
    service, guide, sid, brain = unresolved_pair()
    await brain.respond('Compare these', photo)
    ids = list(service.get(sid).session.current_visual_product_ids)
    scope = service.get(sid).product_memory_scope
    service.get(sid).product_memory = [{'question': 'Earlier Peach question', 'answer': 'Earlier unknown companion claim.'}]
    result = await brain.respond(question)
    assert result['status'] == 'ready' and guide.question == question
    assert guide.current['identity']['name'] == 'Red Bull Sugar Free' and guide.previous == []
    assert 'recent_product_exchanges' not in guide.context
    assert service.get(sid).session.current_visual_product_ids == ids
    assert service.get(sid).product_memory_scope == scope
    assert len(result['snapshot']['products']) == 2
    assert result['snapshot']['products'][1]['catalog_status'] == 'no_match'
    assert guide.calls == ['identify', 'search', 'search', 'respond']


@pytest.mark.parametrize('question', ['Which is better?', 'Does Sugarfree have less sugar?',
    'Compare Red Bull Sugar Free and Red Bull Peach Edition.', 'What about the other one?',
    'Does Red Bull contain caffeine?', 'Does the left one have caffeine?',
    'What sweeteners are in Peach Edition?', 'What sweeteners are in Sugarfree and Peach Edition?',
    'Is Sugarfree better than the other?', 'Which one would you pick, Sugarfree?',
    'Does Sugarfree contain caffeine, and does that one?',
    'What ingredients are in Sugarfree and this product?', 'Is Sugarfree the winner?'])
async def test_comparisons_shared_brand_unknown_item_and_ambiguous_references_stay_unresolved(question):
    service, guide, sid, brain = unresolved_pair()
    await brain.respond('Compare these', photo)
    result = await brain.respond(question)
    assert result['status'] == 'unknown' and 'respond' not in guide.calls
    assert 'Peach Edition' in result['reply'] and 'conflicting variants' in result['reply']
    assert 'show' not in result['reply'].casefold() and '?' not in result['reply']
    assert len(service.get(sid).session.current_visual_product_ids) == 2


async def test_duplicate_variant_alias_is_not_assigned_to_the_known_product():
    service, _, sid, brain = unresolved_pair()
    await brain.respond('Compare these', photo)
    pictured = service.get(sid).session.visual_products
    pictured[1].variant = 'Sugar Free'
    assert grounded_named_target('Does Sugarfree contain caffeine?', pictured) is None


async def test_real_productguide_receives_no_conflicted_companion_or_history(monkeypatch):
    service, guide, sid, brain = unresolved_pair()
    await brain.respond('Compare these', photo)
    service.providers.settings = replace(service.providers.settings, text_provider='anthropic', anthropic_key='fake-test-key')
    calls = []
    async def answer(prompt, data, schema, max_tokens):
        calls.append(data)
        return {'spoken_text': 'The supplied Sugarfree listing does not name the sweeteners.'}
    monkeypatch.setattr(service.providers, 'anthropic_json', answer)
    monkeypatch.setattr(guide, 'respond', ProductGuide(service.providers).respond)
    result = await brain.respond('What sweeteners are in Sugarfree?')
    assert result['status'] == 'ready' and len(calls) == 1
    assert calls[0]['product']['identity']['name'] == 'Red Bull Sugar Free'
    assert calls[0]['previous_products'] == []
    assert 'recent_product_exchanges' not in calls[0]['context']
    assert calls[0]['product']['catalog']['sources']
    assert len(service.get(sid).session.current_visual_product_ids) == 2
