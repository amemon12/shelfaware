import asyncio
import base64
from datetime import datetime, timedelta, timezone
import io
from types import SimpleNamespace

import pytest
from PIL import Image

from app.conversation_product_search import ProductSearchBrain
from app.models import Profile, TurnRequest
from test_multi_product_turns import item, paired
from test_product_turns import IMAGE


def setup(pause=None):
    service, guide, sid = paired(pause)
    service.get(sid).session.profile = Profile(consented=True, goals=['Avoid sugar'])
    brain = ProductSearchBrain(service, {}, session_id=sid)
    return service, guide, sid, brain


def frame(**changes):
    values = dict(image_data_url=IMAGE, width=1, height=1,
                  received_at=datetime.now(timezone.utc).isoformat())
    values.update(changes)
    return SimpleNamespace(**values)


async def photo():
    return frame()


@pytest.mark.parametrize('text', ['Compare these', 'Please compare these two for my avoid sugar goal.',
    'Can you look at this?', 'Read this product', 'Which of these is better?', 'What about this one?'])
async def test_new_view_uses_working_pair_pipeline_without_reviewed_catalog(text):
    service, guide, sid, brain = setup()
    result = await brain.respond(text, photo)
    assert result['status'] == 'ready'
    assert guide.calls == ['identify', 'search', 'search', 'respond']
    expected_context = {'recipient_id': 'self', 'consented': True, 'goals': ['Avoid sugar'], 'avoid_ingredients': []}
    if text == 'Please compare these two for my avoid sugar goal.':
        expected_context.update(goals=[text], conversation_request=text)
    assert guide.context == expected_context
    assert len(guide.previous) == 1
    assert guide.question == text
    snapshot = result['snapshot']
    assert snapshot['analysis_mode'] == 'product' and snapshot['evidence_scope'] == 'catalog_search'
    assert snapshot['catalog_ready'] is False
    assert [p['name'] for p in snapshot['products']] == ['Sugarfree', 'Peach']
    assert all(p['verified'] is False for p in snapshot['products'])
    assert snapshot['comparison'] is None and snapshot['portions'] == {}
    assert not service.get(sid).session.products
    assert not service.get(sid).session.cart


async def test_followups_reuse_pair_and_cached_search_without_camera_or_second_intent_call():
    service, guide, sid, brain = setup()
    await brain.respond('Compare these', photo)
    async def unexpected_photo():
        pytest.fail('Follow-up captured a second image')
    for text in ['Which is better for me?', 'Compare again', 'Why that one?']:
        result = await brain.respond(text, unexpected_photo)
        assert result['status'] == 'ready'
    assert guide.calls == ['identify', 'search', 'search', 'respond', 'respond', 'respond', 'respond']
    assert len(guide.previous) == 1


@pytest.mark.parametrize('question', ['Compare these', 'Compare them.', 'Now compare them.'])
@pytest.mark.parametrize('failure', ['missing', 'none', 'exception', 'stale', 'invalid'])
async def test_failed_new_capture_invalidates_old_pair_before_request(question, failure):
    service, guide, sid, brain = setup()
    await brain.respond('Compare these', photo)
    async def failed_photo():
        assert service.get(sid).session.current_visual_product_ids == []
        if failure == 'none': return None
        if failure == 'exception': raise RuntimeError('private camera detail')
        if failure == 'stale':
            value = frame()
            value.received_at = (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat()
            return value
        value = frame()
        value.image_data_url = 'https://example.com/private.jpg'
        return value
    result = await brain.respond(question, None if failure == 'missing' else failed_photo)
    assert result['status'] == 'needs_image'
    assert result['snapshot']['products'] == []
    assert service.get(sid).frame is None
    assert service.get(sid).latest is None
    assert service.get(sid).product_reply is None
    result = await brain.respond('Which is better?')
    assert result['status'] == 'needs_image'
    assert guide.calls == ['identify', 'search', 'search', 'respond']


@pytest.mark.parametrize('stage', ['identify', 'search', 'respond'])
async def test_cancel_invalidates_service_generation_and_late_provider_results(stage):
    service, guide, sid, brain = setup(stage)
    pending = asyncio.create_task(brain.respond('Compare these', photo))
    await guide.started.wait()
    brain.cancel()
    guide.release.set()
    result = await pending
    assert result['status'] == 'cancelled'
    assert service.get(sid).latest is None
    assert service.get(sid).frame is None
    assert not any(message.role == 'assistant' for message in service.get(sid).session.messages)


async def test_context_switch_during_capture_rejects_old_photo_and_clears_personal_goal():
    service, guide, sid, brain = setup()
    started, release = asyncio.Event(), asyncio.Event()
    async def delayed():
        started.set()
        await release.wait()
        return frame()
    pending = asyncio.create_task(brain.respond('Compare these', delayed))
    await started.wait()
    brain.set_recipient('household')
    release.set()
    result = await pending
    assert result['status'] == 'cancelled'
    assert guide.calls == []
    assert result['snapshot']['shopping_goals'] == []
    await brain.respond('Compare these', photo)
    assert guide.context == {'recipient_id': 'household', 'consented': True}


async def test_household_command_clears_cached_reply_and_reuses_pair_without_personal_goal():
    service, guide, sid, brain = setup()
    await brain.respond('Compare these', photo)
    result = await brain.respond('For my family')
    assert result['snapshot']['recipient_id'] == 'household'
    assert result['snapshot']['shopping_goals'] == []
    await brain.respond('Which is better?')
    assert guide.context == {'recipient_id': 'household', 'consented': True}
    assert guide.calls.count('identify') == 1
    assert guide.calls.count('respond') == 2


@pytest.mark.parametrize('text', ['Half of each bottle', 'Whole bottles', 'Both 30 grams', '250 mL each', 'I will drink 250 ml'])
async def test_portions_never_enter_legacy_grams_arithmetic_or_model_guessing(text):
    service, guide, sid, brain = setup()
    await brain.respond('Compare these', photo)
    before = list(guide.calls)
    result = await brain.respond(text)
    assert result['status'] == 'needs_review'
    assert result['snapshot']['portions'] == {} and result['snapshot']['comparison'] is None
    assert service.get(sid).session.portion_grams is None
    assert guide.calls == before


async def test_product_search_add_never_invents_reviewed_facts_or_cart_confirmation():
    service, guide, sid, brain = setup()
    await brain.respond('Compare these', photo)
    result = await brain.respond('Add the first product')
    assert result['status'] == 'unknown'
    assert result['snapshot']['cart'] == [] and result['snapshot']['pending_action'] is None
    assert not service.get(sid).session.products
    assert guide.calls == ['identify', 'search', 'search', 'respond']


@pytest.mark.parametrize('hello', ['Hello', 'Hello Shelf Aware', 'Hi ShelfAware!', 'Hey, Shelf Aware.'])
async def test_local_greeting_repeat_stop_do_not_invoke_providers(hello):
    service, guide, sid, brain = setup()
    greeting = await brain.respond(hello)
    assert (await brain.respond('Repeat that'))['reply'] == greeting['reply']
    assert (await brain.respond('Stop'))['reply'] == 'Stopped.'
    assert guide.calls == []


@pytest.mark.parametrize('utterance', [
    'Sorry, what did you say? Could you re- like, repeat?',
    'What did you say?', 'Sorry, what did you just say?',
    'Could you say it again?', 'Tell me that once again, please.',
    "I didn't catch that.", 'I couldn’t hear you.', 'Can you re-peat that?',
])
async def test_natural_repeat_preserves_completed_answer_after_failed_automatic_capture(utterance):
    service, guide, sid, brain = setup()
    first = await brain.respond('Compare these', photo)
    before = list(guide.calls)
    failed = await brain.respond_automatically()
    assert failed['status'] == 'needs_image' and failed['snapshot']['products'] == []
    async def forbidden_photo():
        pytest.fail('Repeat requested a new photo')
    repeated = await brain.respond(utterance, forbidden_photo)
    assert repeated['reply'] == first['reply']
    assert repeated['snapshot']['products'] == []
    assert guide.calls == before
    assert service.get(sid).frame is None and service.get(sid).product_reply is None
    assert (await brain.respond('Which is better?'))['status'] == 'needs_image'
    assert (await brain.respond('Say that again'))['reply'] == first['reply']
    assert guide.calls == before


async def test_repeat_survives_cancelled_automatic_capture_and_keeps_stale_evidence_cleared():
    service, guide, sid, brain = setup()
    first = await brain.respond('Compare these', photo)
    started = asyncio.Event()
    async def delayed():
        started.set()
        await asyncio.Event().wait()
    capture = asyncio.create_task(brain.respond_automatically(delayed))
    await started.wait()
    capture.cancel()
    with pytest.raises(asyncio.CancelledError):
        await capture
    assert (await brain.respond('What did you say?'))['reply'] == first['reply']
    assert not brain.snapshot()['products']
    assert service.get(sid).frame is None and service.get(sid).latest is None
    assert guide.calls == ['identify', 'search', 'search', 'respond']


async def test_repeat_memory_does_not_cross_recipient_or_session_and_no_answer_needs_no_provider():
    service, guide, sid, brain = setup()
    first = await brain.respond('Compare these', photo)
    brain.set_recipient('household')
    assert (await brain.respond('What did you say?'))['status'] == 'needs_clarification'
    assert guide.calls == ['identify', 'search', 'search', 'respond']
    brain.set_recipient('self')
    assert (await brain.respond('Repeat'))['reply'] != first['reply']
    await brain.close()
    assert brain._last_reply is None
    _, second_guide, _, fresh_brain = setup()
    assert (await fresh_brain.respond('Say it again'))['status'] == 'needs_clarification'
    assert second_guide.calls == []


async def test_failed_automatic_provider_reply_cannot_replace_last_useful_answer():
    service, guide, sid, brain = setup()
    first = await brain.respond('Compare these', photo)
    guide.reply_error = True
    result = await brain.respond_automatically(photo)
    assert result['status'] != 'ready'
    before = list(guide.calls)
    assert (await brain.respond('Repeat that'))['reply'] == first['reply']
    assert guide.calls == before


async def test_question_before_first_photo_uses_voice_guidance_without_button_instruction():
    service, guide, sid, brain = setup()
    result = await brain.respond('Can you help me shop?')
    assert result['status'] == 'needs_image'
    assert 'steady' in result['reply'] and 'compare these' in result['reply']
    assert 'tap' not in result['reply'].lower() and 'look & ask' not in result['reply'].lower()
    assert guide.calls == []


async def test_spoken_criterion_overrides_default_for_automatic_photo_and_followups():
    service, guide, sid, brain = setup()
    first = await brain.respond('I want less caffeine')
    assert first['status'] == 'needs_image'
    assert first['snapshot']['comparison_request'] == 'I want less caffeine'
    assert service.get(sid).session.profile.goals == ['Avoid sugar']
    result = await brain.respond_automatically(photo)
    assert result['status'] == 'ready'
    assert guide.context['goals'] == ['I want less caffeine']
    assert guide.context['conversation_request'] == 'I want less caffeine'
    assert 'I want less caffeine' in guide.question
    await brain.respond('Why that one?')
    assert guide.context['goals'] == ['I want less caffeine']
    assert guide.calls.count('identify') == 1
    await brain.respond('Now prioritize protein instead')
    await brain.respond_automatically(photo)
    assert guide.context['goals'] == ['Now prioritize protein instead']
    assert brain.snapshot()['comparison_request'] == 'Now prioritize protein instead'
    assert service.get(sid).session.profile.goals == ['Avoid sugar']


async def test_automatic_photo_uses_default_until_spoken_criterion_and_resets_on_recipient_change():
    service, guide, sid, brain = setup()
    await brain.respond_automatically(photo)
    assert guide.context['goals'] == ['Avoid sugar']
    assert brain.snapshot()['comparison_request'] is None
    await brain.respond('Compare these by price', photo)
    assert guide.context['goals'] == ['Compare these by price']
    await brain.respond('For my family')
    assert brain.snapshot()['comparison_request'] is None
    await brain.respond_automatically(photo)
    assert guide.context == {'recipient_id': 'household', 'consented': True}


async def test_profile_consent_and_mode_scope_are_authoritative_from_current_session():
    service, guide, sid, brain = setup()
    service.get(sid).session.profile.consented = False
    result = await brain.respond('Compare these', photo)
    assert result['status'] == 'needs_consent' and guide.calls == []


async def test_adapter_does_not_close_shared_provider_or_reuse_deleted_session():
    service, guide, sid, brain = setup()
    service.delete(sid)
    await brain.close()
    assert brain._closed


async def test_task_cancellation_clears_service_frame_and_late_publication():
    service, guide, sid, brain = setup('respond')
    pending = asyncio.create_task(brain.respond('Compare these', photo))
    await guide.started.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert service.get(sid).latest is None and service.get(sid).frame is None
    assert not any(message.role == 'assistant' for message in service.get(sid).session.messages)


async def test_existing_tap_pair_is_available_to_voice_followup_without_new_image():
    service, guide, sid, brain = setup()
    await service.turn(sid, TurnRequest(text='Which is better?', client_turn_id='tap',
                                      analysis_mode='product', image_data_url=IMAGE))
    result = await brain.respond('Which is better?')
    assert result['status'] == 'ready'
    assert guide.calls == ['identify', 'search', 'search', 'respond', 'respond']


async def test_short_recipient_commands_do_not_mislabel_personal_context():
    service, guide, sid, brain = setup()
    assert (await brain.respond('Family'))['snapshot']['recipient_id'] == 'household'
    assert (await brain.respond('Me'))['snapshot']['recipient_id'] == 'self'
    assert guide.calls == []


async def test_real_transport_frame_request_to_product_service_and_followup():
    from app.conversation_transport import ConversationConfig, ConversationCoordinator
    from test_conversation_transport import Browser, cleanup

    service, guide, sid, _ = setup()
    owner = ConversationCoordinator(service,
        config=ConversationConfig(transcription_provider='disabled', speech_provider='disabled'),
        brain_factory=lambda profile, recipient: ProductSearchBrain(service, profile, recipient))
    browser = Browser()
    connection = await owner.attach(browser, sid)
    task = asyncio.create_task(connection.run())
    try:
        await browser.send('session.start', session_id=sid, voice_consented=True, client_epoch=1)
        await browser.until('session.ready')
        await browser.send('input.text', text='Compare these', client_epoch=2)
        request = await browser.until('camera.request')
        buffer = io.BytesIO()
        Image.new('RGB', (100, 80), 'white').save(buffer, format='PNG')
        await browser.send('camera.response', client_epoch=2, request_id=request['request_id'],
            image_data_url='data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode(),
            width=100, height=80, captured_at=datetime.now(timezone.utc).isoformat())
        response = await browser.until('response.text')
        assert response['text']
        snapshot = (await browser.until('snapshot'))['snapshot']
        assert len(snapshot['products']) == 2
        assert snapshot['evidence_scope'] == 'catalog_search' and snapshot['catalog_ready'] is False
        await browser.send('input.text', text='Which is better for me?', client_epoch=3)
        await browser.until('response.text')
        assert guide.calls == ['identify', 'search', 'search', 'respond', 'respond']
        await browser.send('context.update', recipient_id='household', client_epoch=4)
        await browser.until('snapshot', lambda item: item['snapshot']['recipient_id'] == 'household')
        await browser.send('input.text', text='Which is better?', client_epoch=5)
        await browser.until('response.text')
        assert guide.context == {'recipient_id': 'household', 'consented': True}
    finally:
        await cleanup(owner, task)


_EXPLICIT_BOTH_VIEW_REQUESTS = [
    'What about the both of these?',
    'Compare the both of these and let me know which has lower fat.',
    'Hey, uh, could you compare both of those for lower fat?',
    'Hey, what about both of those?',
]

_REFERENTIAL_VIEW_REQUESTS = [
    'Compare them.',
    'Now compare them.',
    'Hey, uh, could you compare the two?',
    'Please compare them.',
    'Now, could you please compare them?',
    'Could you now compare the two?',
    'Please now compare them again.',
    'Compare them again.',
    'Compare them now please.',
    'Contrast them.',
]


async def _unresolved_old_pair():
    service, guide, sid, brain = setup()
    guide.items = [guide.items[0].model_copy(update={'name': 'Suja Uber Greens'}),
                   guide.items[1].model_copy(update={'name': 'Harmless Harvest'})]
    guide.missing_name = 'Harmless Harvest'
    first = await brain.respond('Compare these', photo)
    assert first['status'] == 'unknown'
    old_ids = list(service.get(sid).session.current_visual_product_ids)
    assert len(old_ids) == 2
    guide.items = [guide.items[0].model_copy(update={'name': 'New chips A'}),
                   guide.items[1].model_copy(update={'name': 'New chips B'})]
    return service, guide, sid, brain, old_ids


@pytest.mark.parametrize('question', _EXPLICIT_BOTH_VIEW_REQUESTS)
async def test_explicit_both_view_replaces_unresolved_old_pair_once_without_intent_model(monkeypatch, question):
    import app.conversation_product_search as adapter
    async def forbidden_router(*args, **kwargs):
        pytest.fail('An explicit shown-pair request must not need another intent call')
    monkeypatch.setattr(adapter, 'route_missing_view', forbidden_router)
    service, guide, sid, brain, old_ids = await _unresolved_old_pair()
    before = list(guide.calls)
    captures = []
    async def new_view():
        assert service.get(sid).session.current_visual_product_ids == []
        captures.append(1)
        return frame()
    result = await brain.respond(question, new_view)
    assert result['status'] == 'ready' and captures == [1]
    assert guide.calls[len(before):] == ['identify', 'search', 'search', 'respond']
    assert guide.question == question  # Preserve the shopper's actual criterion.
    assert [p['name'] for p in result['snapshot']['products']] == ['New chips A', 'New chips B']
    assert set(result['snapshot']['pair']).isdisjoint(old_ids)
    assert 'Harmless Harvest' not in result['reply']


@pytest.mark.parametrize('question', _EXPLICIT_BOTH_VIEW_REQUESTS[:2])
async def test_explicit_both_capture_failure_cannot_answer_from_old_pair(question):
    service, guide, sid, brain, old_ids = await _unresolved_old_pair()
    before = list(guide.calls)
    async def failed_photo():
        assert service.get(sid).session.current_visual_product_ids == []
        raise RuntimeError('Camera unavailable')
    result = await brain.respond(question, failed_photo)
    assert result['status'] == 'needs_image' and result['snapshot']['pair'] == []
    assert result['snapshot']['products'] == [] and 'Harmless' not in result['reply']
    assert guide.calls == before and service.get(sid).frame is None
    assert service.get(sid).product_reply is None


@pytest.mark.parametrize('question', _EXPLICIT_BOTH_VIEW_REQUESTS[:2] + ['Compare them.', 'Now compare them.'])
async def test_cancelled_explicit_both_capture_rejects_late_frame_and_old_pair(question):
    service, guide, sid, brain, old_ids = await _unresolved_old_pair()
    before = list(guide.calls)
    started, release = asyncio.Event(), asyncio.Event()
    async def delayed_photo():
        assert service.get(sid).session.current_visual_product_ids == []
        started.set()
        await release.wait()
        return frame()
    pending = asyncio.create_task(brain.respond(question, delayed_photo))
    await started.wait()
    brain.cancel()
    release.set()
    result = await pending
    assert result['status'] == 'cancelled' and result['snapshot']['pair'] == []
    assert result['snapshot']['products'] == [] and 'Harmless' not in result['reply']
    assert guide.calls == before and service.get(sid).frame is None


@pytest.mark.parametrize('question', ['Which one has lower fat between these two?',
    'Compare again', 'Why that one?', 'Do both of these have caffeine?'])
async def test_ordinary_current_pair_followups_still_reuse_view_and_sources(monkeypatch, question):
    import app.conversation_product_search as adapter
    service, guide, sid, brain = setup()
    await brain.respond('Compare these', photo)
    ids = list(service.get(sid).session.current_visual_product_ids)
    before = list(guide.calls)
    async def forbidden(*args, **kwargs):
        pytest.fail('A follow-up must reuse the selected pair, without a photo or intent call')
    monkeypatch.setattr(adapter, 'route_missing_view', forbidden)
    result = await brain.respond(question, forbidden)
    assert result['status'] == 'ready' and result['snapshot']['pair'] == ids
    assert guide.calls[len(before):] == ['respond'] and guide.question == question


@pytest.mark.parametrize('question,criterion', [
    ('Compare them.', False),
    ('Hey, uh, could you compare the two?', False),
    ('I would like to compare the sugar.', True),
    ("I'd like to compare their total fat content.", True),
    ('Please compare the added sugars.', True),
])
async def test_missing_view_explicit_comparison_captures_and_remembers_only_explicit_topic(monkeypatch, question, criterion):
    import app.conversation_product_search as adapter
    service, guide, sid, brain = setup()
    async def forbidden_router(*args, **kwargs):
        pytest.fail('Clear referential/nutrient comparison should not pay for classification')
    monkeypatch.setattr(adapter, 'route_missing_view', forbidden_router)
    captures = []
    async def fresh():
        captures.append(1)
        return frame()
    result = await brain.respond(question, fresh)
    assert result['status'] == 'ready' and captures == [1]
    assert guide.calls == ['identify', 'search', 'search', 'respond']
    assert guide.question == question and service.get(sid).session.profile.goals == ['Avoid sugar']
    assert result['snapshot']['comparison_request'] == (question if criterion else None)
    if criterion:
        assert guide.context['conversation_request'] == question
        assert question in brain.automatic_question()
    assert not service.get(sid).session.cart


@pytest.mark.parametrize('failure', ['empty', 'exception', 'no_products'])
async def test_missing_view_comparison_failure_retains_criterion_and_retries_clear_request(monkeypatch, failure):
    import app.conversation_product_search as adapter
    service, guide, sid, brain = setup()
    async def forbidden_router(*args, **kwargs):
        pytest.fail('Retrying an explicit comparison cannot need a classifier')
    monkeypatch.setattr(adapter, 'route_missing_view', forbidden_router)
    original_items = list(guide.items)
    if failure == 'no_products':
        guide.items = []
    async def failed():
        if failure == 'exception': raise RuntimeError('No still available')
        return None if failure == 'empty' else frame()
    question = 'I would like to compare the sugar.'
    result = await brain.respond(question, failed)
    assert result['status'] == 'needs_image' and result['snapshot']['products'] == []
    assert result['snapshot']['comparison_request'] == question
    assert guide.calls == (['identify'] if failure == 'no_products' else [])
    guide.items = original_items
    result = await brain.respond('Compare them.', photo)
    assert result['status'] == 'ready' and len(result['snapshot']['products']) == 2
    assert guide.context['conversation_request'] == question


@pytest.mark.parametrize('question', _REFERENTIAL_VIEW_REQUESTS)
async def test_presented_pair_comparison_replaces_old_products_and_memory_with_fresh_view(monkeypatch, question):
    import app.conversation_product_search as adapter
    service, guide, sid, brain = setup()
    guide.items = [item('Red Bull Sugar Free', .1), item('Red Bull THE PEACH EDITION', .6)]
    criterion = 'I would like to compare the sugar.'
    await brain.respond(criterion, photo)
    old_ids = list(service.get(sid).session.current_visual_product_ids)
    assert service.get(sid).product_memory
    guide.items = [item('Suja Uber Greens', .1), item('Harmless Harvest', .6)]
    before = list(guide.calls)
    captures = []
    async def fresh():
        assert service.get(sid).session.current_visual_product_ids == []
        captures.append(1)
        return frame()
    async def forbidden(*args, **kwargs):
        pytest.fail('An explicit presented-pair comparison must not need an intent call')
    monkeypatch.setattr(adapter, 'route_missing_view', forbidden)
    monkeypatch.setattr(adapter, 'route_current_view', forbidden)
    result = await brain.respond(question, fresh)
    assert result['status'] == 'ready' and captures == [1]
    assert guide.calls[len(before):] == ['identify', 'search', 'search', 'respond']
    assert set(result['snapshot']['pair']).isdisjoint(old_ids)
    assert [p['name'] for p in result['snapshot']['products']] == ['Suja Uber Greens', 'Harmless Harvest']
    assert guide.current['identity']['name'] == 'Harmless Harvest'
    assert [p['identity']['name'] for p in guide.previous] == ['Suja Uber Greens']
    assert 'recent_product_exchanges' not in guide.context
    assert guide.context['conversation_request'] == criterion
    assert result['snapshot']['comparison_request'] == criterion
    assert guide.question == question


@pytest.mark.parametrize('question', ['Compare them.', 'Now compare them.'])
async def test_presented_pair_with_only_one_new_product_never_borrows_old_companion(question):
    service, guide, sid, brain = setup()
    await brain.respond('Compare these', photo)
    old_ids = list(service.get(sid).session.current_visual_product_ids)
    guide.items = [item('New single product', .2)]
    before = list(guide.calls)
    result = await brain.respond(question, photo)
    assert result['status'] == 'needs_image'
    assert 'Show both products together' in result['reply']
    assert [p['name'] for p in result['snapshot']['products']] == ['New single product']
    assert set(result['snapshot']['pair']).isdisjoint(old_ids)
    assert guide.calls[len(before):] == ['identify', 'search']
    assert service.get(sid).product_memory == []


async def test_presented_pair_with_explicit_nutrient_refreshes_and_updates_criterion():
    service, guide, sid, brain = setup()
    await brain.respond('Compare these', photo)
    old_ids = list(service.get(sid).session.current_visual_product_ids)
    guide.items = [item('New can A', .1), item('New can B', .6)]
    before = list(guide.calls)
    question = 'Compare them for sugar.'
    result = await brain.respond(question, photo)
    assert result['status'] == 'ready'
    assert set(result['snapshot']['pair']).isdisjoint(old_ids)
    assert guide.calls[len(before):] == ['identify', 'search', 'search', 'respond']
    assert guide.context['conversation_request'] == question
    assert result['snapshot']['comparison_request'] == question


@pytest.mark.parametrize('question', ['I would like to compare the sugar.', 'Please compare their total fat content.'])
async def test_nutrient_comparison_with_selected_pair_reuses_photo_and_criterion(monkeypatch, question):
    import app.conversation_product_search as adapter
    service, guide, sid, brain = setup()
    await brain.respond('Compare these', photo)
    ids = list(service.get(sid).session.current_visual_product_ids)
    async def forbidden(*args, **kwargs):
        pytest.fail('Selected-pair comparison should reuse the view, without camera or classifier')
    monkeypatch.setattr(adapter, 'route_missing_view', forbidden)
    monkeypatch.setattr(adapter, 'route_current_view', forbidden)
    result = await brain.respond(question, forbidden)
    assert result['status'] == 'ready' and result['snapshot']['pair'] == ids
    assert guide.calls == ['identify', 'search', 'search', 'respond', 'respond']
    assert guide.question == question
    assert guide.context['conversation_request'] == question


@pytest.mark.parametrize('question,followup', [("Don't compare them.", False), ('Do not compare the two.', False),
    ('Compare them with the previous products.', True), ('Compare the two from the previous photo.', False)])
async def test_negated_or_historical_comparison_does_not_force_a_new_view(monkeypatch, question, followup):
    import app.conversation_product_search as adapter
    from app.conversation_intent_router import ConversationRoute
    service, guide, sid, brain = setup()
    async def clarify(*args, **kwargs):
        return ConversationRoute(route='brief_clarification')
    async def forbidden():
        pytest.fail('Negated or historical comparison must not authorize a new photo')
    monkeypatch.setattr(adapter, 'route_missing_view', clarify)
    monkeypatch.setattr(adapter, 'route_current_view', clarify)
    await brain.respond(question, forbidden)
    assert guide.calls == []
    await brain.respond('Compare these', photo)
    ids = list(service.get(sid).session.current_visual_product_ids)
    before = list(guide.calls)
    await brain.respond(question, forbidden)
    assert guide.calls[len(before):] == (['respond'] if followup else [])
    assert service.get(sid).session.current_visual_product_ids == ids


@pytest.mark.parametrize('question', ['Compare sugar and salt production.', 'Compare walking and running.',
    'I would like to compare phone plans.', 'Do not compare the sugar.'])
async def test_non_product_or_negated_comparisons_do_not_use_missing_view_fastpath(monkeypatch, question):
    import app.conversation_product_search as adapter
    from app.conversation_intent_router import ConversationRoute
    _, guide, _, brain = setup()
    calls = []
    async def clarify(*args, **kwargs):
        calls.append(1)
        return ConversationRoute(route='brief_clarification')
    async def forbidden(): pytest.fail('Unrelated comparison cannot authorize a photo')
    monkeypatch.setattr(adapter, 'route_missing_view', clarify)
    result = await brain.respond(question, forbidden)
    assert result['status'] == 'needs_clarification' and calls == [1] and guide.calls == []
    assert result['snapshot']['comparison_request'] is None
