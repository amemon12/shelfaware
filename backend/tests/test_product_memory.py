import asyncio
from copy import deepcopy
from dataclasses import replace

import pytest

from app.models import ContextRequest
from app.product_guidance import ProductGuide, _minimal_context
from app.product_memory import compact_exchanges, product_scope
from test_conversation_product_search import setup, photo
from test_multi_product_turns import item
from test_product_turns import request


async def test_followup_receives_last_useful_exchange_with_current_pair_evidence():
    service, guide, sid, brain = setup()
    first = await brain.respond('Compare these', photo)
    assert 'recent_product_exchanges' not in guide.context
    await brain.respond('Why did you choose that one?')
    assert guide.context['recent_product_exchanges'] == [{'question': 'Compare these', 'answer': first['reply']}]
    assert guide.question == 'Why did you choose that one?'
    assert guide.current['identity']['name'] == 'Peach'
    assert guide.previous[0]['identity']['name'] == 'Sugarfree'
    pictured = [p for p in service.get(sid).session.visual_products if p.id in service.get(sid).session.current_visual_product_ids]
    assert len({p.captured_at for p in pictured}) == 1
    assert pictured[0].captured_at == service.get(sid).frame.captured_at
    assert guide.calls == ['identify', 'search', 'search', 'respond', 'respond']
    assert len(service.get(sid).product_memory) == 2


async def test_fresh_photo_of_same_pair_keeps_memory_despite_new_ids_reversed_order_and_boxes():
    service, guide, sid, brain = setup()
    first = await brain.respond('Compare these', photo)
    old_ids = set(service.get(sid).session.current_visual_product_ids)
    scope = service.get(sid).product_memory_scope
    guide.items = [item('Peach', .1), item('Sugarfree', .6)]
    await brain.respond('Look at these again', photo)
    assert set(service.get(sid).session.current_visual_product_ids).isdisjoint(old_ids)
    assert service.get(sid).product_memory_scope == scope
    assert guide.context['recent_product_exchanges'][0]['answer'] == first['reply']
    assert guide.current['identity']['name'] == 'Sugarfree'


async def test_reversed_new_photo_discards_old_positional_references_but_keeps_named_product_memory():
    service, guide, sid, brain = setup()
    await brain.respond('Compare these', photo)
    service.get(sid).product_memory = [
        {'question': 'Why Sugarfree?', 'answer': 'Sugarfree was the previous named choice.'},
        {'question': 'What about the left one?', 'answer': 'The left one was the previous choice.'}]
    guide.items = [item('Peach', .1), item('Sugarfree', .6)]
    await brain.respond('Have a look at these', photo)
    assert guide.context['recent_product_exchanges'] == [
        {'question': 'Why Sugarfree?', 'answer': 'Sugarfree was the previous named choice.'}]
    assert guide.current['identity']['name'] == 'Sugarfree' and guide.current['identity']['box']['x'] == .6
    assert guide.previous[0]['identity']['name'] == 'Peach' and guide.previous[0]['identity']['box']['x'] == .1
    assert guide.current['captured_at'] == guide.previous[0]['captured_at']


@pytest.mark.parametrize('field,value', [('name', 'Different product'), ('brand', 'Another brand'),
    ('variant', 'Different variant'), ('package_size', '500 mL')])
async def test_changed_identity_clears_old_exchange_before_wording(field, value):
    service, guide, sid, brain = setup()
    await brain.respond('Compare these', photo)
    guide.items[0] = guide.items[0].model_copy(update={field: value})
    await brain.respond('Look at these', photo)
    assert 'recent_product_exchanges' not in guide.context
    assert [row['question'] for row in service.get(sid).product_memory] == ['Look at these']


async def test_changed_identity_clears_old_memory_even_if_new_lookup_fails():
    service, guide, sid, brain = setup()
    await brain.respond('Compare these', photo)
    guide.items = [item('Another product', .2)]
    guide.failed_name = 'Another product'
    result = await brain.respond('Look at this', photo)
    assert result['status'] == 'unknown' and not service.get(sid).product_memory


async def test_failed_photo_or_wording_never_appends_or_exposes_an_old_selection():
    service, guide, sid, brain = setup()
    await brain.respond('Compare these', photo)
    previous = deepcopy(service.get(sid).product_memory)
    calls = list(guide.calls)
    assert (await brain.respond('Look at these'))['status'] == 'needs_image'
    assert not brain.snapshot()['products'] and guide.calls == calls
    assert service.get(sid).product_memory == previous
    guide.reply_error = True
    assert (await brain.respond('Look at these', photo))['status'] == 'unknown'
    assert guide.context['recent_product_exchanges'] == previous
    assert service.get(sid).product_memory == previous


async def test_recipient_changes_and_new_session_never_receive_previous_personal_exchange():
    service, guide, sid, brain = setup()
    await brain.respond('Compare these for my private preference', photo)
    brain.set_recipient('household')
    assert service.get(sid).product_memory == []
    await brain.respond('Which is better?')
    assert 'recent_product_exchanges' not in guide.context and guide.context['recipient_id'] == 'household'
    service.context(sid, ContextRequest(recipient_id='self'))
    await service.turn(sid, request('after-context', image=False, text='Why that one?'))
    assert 'recent_product_exchanges' not in guide.context
    _, second, _, new_brain = setup()
    await new_brain.respond('Compare these', photo)
    assert 'recent_product_exchanges' not in second.context


async def test_social_errors_and_repeated_automatic_question_do_not_fill_memory():
    service, guide, sid, brain = setup()
    await brain.respond_automatically(photo)
    first = deepcopy(service.get(sid).product_memory)
    for utterance in ['Thanks', 'Hello', 'Are you still there?', 'What did you say?']:
        await brain.respond(utterance)
    await brain.respond_automatically()
    await brain.respond_automatically()
    assert service.get(sid).product_memory == first
    await brain.respond_automatically(photo)
    assert len(service.get(sid).product_memory) == 1
    assert service.get(sid).product_memory == first
    for question in ['Why that one?', 'What is different?', 'What about the other option?', 'Which ingredients are listed?']:
        await brain.respond(question)
    history = service.get(sid).product_memory
    assert len(history) <= 3
    assert sum(len(row['question']) + len(row['answer']) for row in history) <= 1800
    assert history[-1]['question'] == 'Which ingredients are listed?'


async def test_cancelled_wording_cannot_append_an_exchange():
    service, guide, sid, brain = setup()
    await brain.respond('Compare these', photo)
    before = deepcopy(service.get(sid).product_memory)
    guide.pause = 'respond'
    guide.started.clear()
    pending = asyncio.create_task(brain.respond('Why that one?'))
    await guide.started.wait()
    brain.cancel()
    guide.release.set()
    assert (await pending)['status'] == 'cancelled'
    assert service.get(sid).product_memory == before


async def test_real_wording_payload_carries_only_bounded_memory_alongside_current_sources(monkeypatch):
    service, guide, _, brain = setup()
    first = await brain.respond('Compare these', photo)
    service.providers.settings = replace(service.providers.settings, text_provider='anthropic', anthropic_key='fake-test-key')
    requests = []
    async def json_answer(prompt, data, schema, max_tokens):
        requests.append((prompt, data))
        return {'spoken_text': 'The current product listings determine the comparison.'}
    monkeypatch.setattr(service.providers, 'anthropic_json', json_answer)
    monkeypatch.setattr(guide, 'respond', ProductGuide(service.providers).respond)
    result = await brain.respond('What about the other one?')
    assert result['status'] == 'ready' and len(requests) == 1
    prompt, data = requests[0]
    assert data['context']['recent_product_exchanges'] == [{'question': 'Compare these', 'answer': first['reply']}]
    assert data['question'] == 'What about the other one?'
    assert data['product']['catalog']['sources'] and data['previous_products'][0]['catalog']['sources']
    assert 'current cited evidence, which overrides any earlier answer' in prompt
    assert not any('image' in key or 'audio' in key for key in data['context'])


async def test_old_memory_numbers_cannot_validate_a_claim_missing_from_current_sources(monkeypatch):
    service, guide, sid, brain = setup()
    await brain.respond('Compare these', photo)
    service.get(sid).product_memory = [{'question': 'How much sugar?', 'answer': 'An earlier answer said 99 grams of sugar per bottle.'}]
    service.providers.settings = replace(service.providers.settings, text_provider='anthropic', anthropic_key='fake-test-key')
    calls = []
    async def json_answer(prompt, data, schema, max_tokens):
        calls.append(data)
        return {'spoken_text': ('Based on the product listing, Peach has 99 grams of sugar per bottle.' if len(calls) == 1
                                else 'The current listings do not establish a sugar amount.')}
    monkeypatch.setattr(service.providers, 'anthropic_json', json_answer)
    monkeypatch.setattr(guide, 'respond', ProductGuide(service.providers).respond)
    result = await brain.respond('How much sugar did you say?')
    assert result['status'] == 'ready' and '99' not in result['reply']
    assert len(calls) == 2
    assert calls[0]['context']['recent_product_exchanges'][0]['answer'].startswith('An earlier answer')


def test_memory_bounds_reject_nested_data_and_keep_complete_latest_answers():
    entries = [{'question': str(i) + 'q' * 400, 'answer': str(i) + 'a' * 599} for i in range(6)]
    bounded = compact_exchanges(entries)
    assert len(bounded) == 2 and sum(len(v) for row in bounded for v in row.values()) == 1800
    assert bounded[-1]['answer'] == entries[-1]['answer']
    assert not compact_exchanges([{'question': 'hello', 'answer': 'a' * 601},
        {'question': 'hello', 'answer': 'answer', 'image': 'private'}])
    assert _minimal_context({'recipient_id': 'self', 'consented': False, 'recent_product_exchanges': entries}) == {'recipient_id': 'self'}
    minimized = _minimal_context({'recipient_id': 'household', 'consented': True,
        'recent_product_exchanges': entries, 'nutrition_context': {'conditions': ['private']}})
    assert 'nutrition_context' not in minimized and minimized['recent_product_exchanges'] == bounded


def test_identity_scope_preserves_duplicate_count_and_normalizes_only_text_presentation():
    first = item('Same Product', .1)
    normalized = first.model_copy(update={'name': ' same   PRODUCT ', 'brand': ' DEMO '})
    assert product_scope('self', [first]) == product_scope('self', [normalized])
    assert product_scope('self', [first]) != product_scope('self', [first, first])
    assert product_scope('self', [first]) != product_scope('household', [first])


@pytest.mark.parametrize('change', [{'name': 'Different product'}, {'variant': 'Different variant'},
    {'brand': 'Different brand'}, {'package_size': '500 mL'}])
async def test_repeat_cannot_replay_previous_identity_after_new_product_wording_fails(change):
    service, guide, sid, brain = setup()
    first = await brain.respond('Compare these', photo)
    guide.items[0] = guide.items[0].model_copy(update=change)
    guide.reply_error = True
    result = await brain.respond('Look at these', photo)
    assert result['status'] == 'unknown' and brain._last_reply is None
    before = list(guide.calls)
    repeated = await brain.respond('Repeat that')
    assert repeated['status'] == 'needs_clarification' and repeated['reply'] != first['reply']
    assert guide.calls == before and service.get(sid).product_memory == []


async def test_same_identity_new_photo_failure_preserves_repeat_but_not_old_identity_after_return():
    service, guide, sid, brain = setup()
    first = await brain.respond('Compare these', photo)
    original = list(guide.items)
    guide.items = original[::-1]
    guide.reply_error = True
    await brain.respond('Look at these', photo)
    assert (await brain.respond('Repeat that'))['reply'] == first['reply']
    guide.items = [item('Entirely new item', .2)]
    await brain.respond('Look at this', photo)
    assert (await brain.respond('Repeat that'))['status'] == 'needs_clarification'
    guide.items = original
    await brain.respond('Look at these', photo)
    assert (await brain.respond('Repeat that'))['status'] == 'needs_clarification'
    assert service.get(sid).product_memory == []


async def test_repeat_after_failed_capture_retains_scope_so_later_new_identity_still_clears_it():
    service, guide, _, brain = setup()
    first = await brain.respond('Compare these', photo)
    await brain.respond('Look at these')
    assert (await brain.respond('Repeat that'))['reply'] == first['reply']
    assert brain._last_reply_scope is not None
    guide.items = [item('New item', .2)]
    guide.reply_error = True
    await brain.respond('Look at this', photo)
    assert (await brain.respond('Repeat that'))['status'] == 'needs_clarification'


async def test_rest_product_or_recipient_change_invalidates_voice_repeat_before_reading_cache():
    service, guide, sid, brain = setup()
    first = await brain.respond('Compare these', photo)
    guide.items = [item('REST new item', .2)]
    guide.reply_error = True
    await service.turn(sid, request('rest-new', text='Look at this'))
    assert (await brain.respond('Repeat that'))['status'] == 'needs_clarification'
    guide.reply_error = False
    await brain.respond('Look at this', photo)
    service.context(sid, ContextRequest(recipient_id='household'))
    assert (await brain.respond('Repeat that'))['status'] == 'needs_clarification'
    assert brain._last_reply is None and brain._last_reply_scope is None
