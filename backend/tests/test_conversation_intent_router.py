import asyncio
from dataclasses import replace

import pytest

from app import conversation_intent_router as router
from app.providers import ProviderError
from test_conversation_product_search import setup, photo


def configured(monkeypatch, outcome=None):
    service, guide, sid, brain = setup()
    service.providers.settings = replace(service.providers.settings, text_provider='anthropic', anthropic_key='fake-test-key')
    calls = []
    async def classify(prompt, data, schema, max_tokens):
        calls.append((prompt, data, schema, max_tokens))
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome or {'route': 'inspect_new_view'}
    monkeypatch.setattr(service.providers, 'anthropic_json', classify)
    return service, guide, sid, brain, calls


@pytest.mark.parametrize('question', ['Help me decide between what I am holding.',
    'Which would you go with?', 'Can you tell what is inside the one in my hand?'])
async def test_ambiguous_missing_view_routes_once_then_uses_real_product_service(monkeypatch, question):
    service, guide, sid, brain, calls = configured(monkeypatch)
    result = await brain.respond(question, photo)
    assert result['status'] == 'ready'
    assert guide.calls == ['identify', 'search', 'search', 'respond']
    assert len(calls) == 1 and calls[0][1] == {'utterance': question, 'awaiting_view': False}
    assert calls[0][3] == 128
    assert set(calls[0][2]['properties']) == {'route'}
    assert guide.question == question
    assert service.get(sid).session.profile.goals == ['Avoid sugar']
    assert not service.get(sid).session.cart
    async def forbidden(): pytest.fail('Selected follow-up requested another view')
    await brain.respond('Which sweeteners do they use?', forbidden)
    assert len(calls) == 1 and guide.question == 'Which sweeteners do they use?'
    assert guide.calls == ['identify', 'search', 'search', 'respond', 'respond']


@pytest.mark.parametrize('question', ['Have a look at these.', 'Could you take a look at this?',
    "Hey, uh, which one's better for me?", 'Compare these two', 'Does this have added sugar?'])
async def test_clear_visual_requests_skip_router(monkeypatch, question):
    _, guide, _, brain, calls = configured(monkeypatch)
    assert (await brain.respond(question, photo))['status'] == 'ready'
    assert calls == [] and guide.calls.count('identify') == 1


@pytest.mark.parametrize('question', ['Thanks.', 'Hello.', 'Are you still with me?', 'Stop.', "Don't stop.",
    'Repeat that', 'What is my goal?', 'No, thank you.', 'For my family', 'Add this', 'Shopping summary'])
async def test_controls_social_and_transactions_never_call_router_or_camera(monkeypatch, question):
    service, guide, sid, brain, calls = configured(monkeypatch)
    async def forbidden(): pytest.fail('Local utterance requested a photo')
    await brain.respond(question, forbidden)
    assert calls == [] and guide.calls == [] and not service.get(sid).session.cart


@pytest.mark.parametrize('route,question', [('social', 'Wonderful, much appreciated.'),
    ('brief_clarification', 'How is the weather?'), ('follow_up', 'Why did you choose that earlier?')])
async def test_nonvisual_routes_are_local_without_invented_product_facts(monkeypatch, route, question):
    _, guide, _, brain, calls = configured(monkeypatch, {'route': route})
    async def forbidden(): pytest.fail('Nonvisual route requested a photo')
    result = await brain.respond(question, forbidden)
    assert len(calls) == 1 and guide.calls == []
    assert result['snapshot']['products'] == [] and brain._last_reply is None
    assert not result.get('end_conversation')
    if route == 'follow_up':
        assert 'Ask me to look' in result['reply'] and '?' not in result['reply']


@pytest.mark.parametrize('question', ['Hello.', 'Hi ShelfAware!', 'Hey, Shelf Aware.'])
async def test_greeting_is_brief_without_repeating_capture_instructions(monkeypatch, question):
    _, _, _, brain, calls = configured(monkeypatch)
    result = await brain.respond(question)
    assert result['reply'] == "I'm listening. What would you like to check?"
    assert not calls


@pytest.mark.parametrize('question', ['Are you still with me?', 'Are you there?', 'Can you hear me?'])
async def test_presence_is_local_and_preserves_last_product_answer(monkeypatch, question):
    _, guide, _, brain, calls = configured(monkeypatch)
    previous = await brain.respond('Compare these', photo)
    before = list(guide.calls)
    assert (await brain.respond(question))['reply'] == "Yes, I'm here."
    assert (await brain.respond('Repeat'))['reply'] == previous['reply']
    assert not calls and guide.calls == before


async def test_explicit_help_can_explain_capability_without_camera_or_router(monkeypatch):
    _, guide, _, brain, calls = configured(monkeypatch)
    for question in ['Help', 'Help!', 'What can you do?']:
        result = await brain.respond(question)
        assert 'compare them for your shopping goal' in result['reply']
    assert not calls and guide.calls == []


@pytest.mark.parametrize('outcome', [{'route': 'buy'}, {'route': 'inspect_new_view', 'goal': 'invented'},
    {}, {'route': 1}, ProviderError('private upstream error')])
async def test_invalid_or_failed_router_clarifies_without_fallback_capture(monkeypatch, outcome):
    # Empty dict must remain invalid rather than use the test helper's default.
    service, guide, _, brain, calls = configured(monkeypatch, outcome)
    if outcome == {}:
        async def invalid(*args, **kwargs): return {}
        monkeypatch.setattr(service.providers, 'anthropic_json', invalid)
    async def forbidden(): pytest.fail('Invalid route requested a photo')
    result = await brain.respond('Can you help me decide about something?', forbidden)
    assert result['status'] == 'needs_clarification'
    assert 'private' not in result['reply'] and guide.calls == []


async def test_router_timeout_is_bounded_and_does_not_capture_or_retry(monkeypatch):
    service, guide, _, brain, _ = configured(monkeypatch)
    started = 0
    async def delayed(*args, **kwargs):
        nonlocal started
        started += 1
        await asyncio.Event().wait()
    monkeypatch.setattr(router, 'ROUTING_TIMEOUT_SECONDS', .01)
    monkeypatch.setattr(service.providers, 'anthropic_json', delayed)
    result = await brain.respond('Could you assist me with what I have?', photo)
    assert result['status'] == 'needs_clarification' and started == 1 and guide.calls == []


async def test_context_change_during_routing_cannot_capture_or_apply_old_personal_goals(monkeypatch):
    service, guide, _, brain, _ = configured(monkeypatch)
    started, release = asyncio.Event(), asyncio.Event()
    async def delayed(*args, **kwargs):
        started.set()
        await release.wait()
        return {'route': 'inspect_new_view'}
    monkeypatch.setattr(service.providers, 'anthropic_json', delayed)
    pending = asyncio.create_task(brain.respond('Help me pick from what I am holding.', photo))
    await started.wait()
    brain.set_recipient('household')
    release.set()
    result = await pending
    assert result['status'] == 'cancelled' and guide.calls == []
    assert result['snapshot']['shopping_goals'] == []


async def test_cancelled_router_does_not_swallow_cancellation(monkeypatch):
    service, guide, _, brain, _ = configured(monkeypatch)
    started = asyncio.Event()
    async def delayed(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()
    monkeypatch.setattr(service.providers, 'anthropic_json', delayed)
    pending = asyncio.create_task(brain.respond('Help me choose from what I have.', photo))
    await started.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError): await pending
    assert guide.calls == []


async def test_unconsented_profile_never_reaches_router(monkeypatch):
    service, guide, sid, brain, calls = configured(monkeypatch)
    service.get(sid).session.profile.consented = False
    assert (await brain.respond('Help me choose what I have.', photo))['status'] == 'needs_consent'
    assert calls == [] and guide.calls == []


async def test_automatic_guidance_deduplicates_with_deadline_and_manual_success_context_resets(monkeypatch):
    import app.conversation_product_search as adapter
    _, guide, _, brain, calls = configured(monkeypatch)
    clock = [100.0]
    monkeypatch.setattr(adapter, 'monotonic', lambda: clock[0])
    first = await brain.respond_automatically()
    second = await brain.respond_automatically()
    assert first['status'] == 'needs_image' and not first.get('silent')
    assert second.get('silent') is True and calls == []
    clock[0] += 46
    assert not (await brain.respond_automatically()).get('silent')
    assert (await brain.respond_automatically()).get('silent')
    await brain.respond('Thanks')
    assert not (await brain.respond_automatically()).get('silent')
    await brain.respond_automatically(photo)
    assert not (await brain.respond_automatically()).get('silent')
    assert (await brain.respond_automatically()).get('silent')
    brain.set_recipient('household')
    assert not (await brain.respond_automatically()).get('silent')
    assert guide.calls.count('respond') == 1 and calls == []


async def test_different_automatic_failure_and_provider_diagnostics_are_not_silenced(monkeypatch):
    _, guide, _, brain, _ = configured(monkeypatch)
    await brain.respond_automatically()
    async def failure(): raise RuntimeError('capture failed')
    assert not (await brain.respond_automatically(failure)).get('silent')
    assert (await brain.respond_automatically(failure)).get('silent')
    guide.reply_error = True
    result = await brain.respond_automatically(photo)
    assert result['status'] == 'unknown' and not result.get('silent')
    assert result['snapshot']['provider_failure']['stage'] == 'product_response'


def deepseek_configured(monkeypatch, outcome=None):
    service, guide, sid, brain = setup()
    service.providers.settings = replace(service.providers.settings, text_provider='deepseek',
        deepseek_key='fake-deepseek-key', anthropic_key='must-not-be-used')
    calls = []
    async def classify(prompt, data, max_tokens):
        calls.append((prompt, data, max_tokens))
        if isinstance(outcome, BaseException): raise outcome
        return outcome if outcome is not None else {'route': 'inspect_new_view'}
    async def forbidden(*args, **kwargs): pytest.fail('DeepSeek configuration called Anthropic')
    monkeypatch.setattr(service.providers, 'deepseek_json', classify)
    monkeypatch.setattr(service.providers, 'anthropic_json', forbidden)
    return service, guide, sid, brain, calls


async def test_configured_deepseek_routes_one_ambiguous_missing_view_and_skips_selected_followups(monkeypatch):
    service, guide, _, brain, calls = deepseek_configured(monkeypatch)
    question = 'Help me pick between what I am holding.'
    assert (await brain.respond(question, photo))['status'] == 'ready'
    assert len(calls) == 1 and calls[0][1] == {'utterance': question, 'awaiting_view': False}
    assert calls[0][2] == 128 and 'exactly one key, "route"' in calls[0][0]
    await brain.respond('Why did you pick that one?')
    assert len(calls) == 1 and guide.calls == ['identify', 'search', 'search', 'respond', 'respond']


@pytest.mark.parametrize('question', ['Thanks', 'Hello', 'Are you still with me?', 'What is my goal?', 'Compare these'])
async def test_deepseek_controls_and_clear_captures_skip_semantic_router(monkeypatch, question):
    _, _, _, brain, calls = deepseek_configured(monkeypatch)
    await brain.respond(question, photo)
    assert calls == []


@pytest.mark.parametrize('output', [{'route': 'buy'}, {'route': 'inspect_new_view', 'facts': 'invented'},
    {}, ProviderError('upstream failure')])
async def test_deepseek_invalid_or_failed_route_clarifies_without_another_provider(monkeypatch, output):
    _, guide, _, brain, calls = deepseek_configured(monkeypatch, output)
    result = await brain.respond('Help me decide about what I have.', photo)
    assert result['status'] == 'needs_clarification' and len(calls) == 1 and guide.calls == []


async def test_deepseek_router_deadline_does_not_retry_or_fall_back(monkeypatch):
    service, guide, _, brain, _ = deepseek_configured(monkeypatch)
    called = 0
    async def slow(*args, **kwargs):
        nonlocal called
        called += 1
        await asyncio.Event().wait()
    monkeypatch.setattr(router, 'ROUTING_TIMEOUT_SECONDS', .01)
    monkeypatch.setattr(service.providers, 'deepseek_json', slow)
    result = await brain.respond('Help me choose what I have.', photo)
    assert result['status'] == 'needs_clarification' and called == 1 and guide.calls == []


@pytest.mark.parametrize('utterance', [
    "You're a good man, like, you know? That's fucking it. You're gonna take...",
    'There will be two levels, exchange numbers, and compare products based on an actual fortune and preferences, with preferences',
    'The train was late and the meeting starts after lunch.',
    'They were saying the comparison depends on everyone’s preferences.',
])
async def test_ambiguous_nonrequest_does_not_answer_products_or_pollute_goal(monkeypatch, utterance):
    service, guide, sid, brain, calls = deepseek_configured(monkeypatch, {'route': 'ignore'})
    answer = await brain.respond('Compare these for less sugar', photo)
    original_request = brain.snapshot()['comparison_request']
    original_pair = list(brain.snapshot()['pair'])
    original_memory = list(service.get(sid).product_memory)
    before = list(guide.calls)
    async def forbidden(): pytest.fail('Unrelated speech requested a camera frame')
    result = await brain.respond(utterance, forbidden)
    assert result['status'] == 'ignored' and result['silent'] is True
    assert result['snapshot']['comparison_request'] == original_request
    assert result['snapshot']['pair'] == original_pair
    assert guide.calls == before and service.get(sid).product_memory == original_memory
    assert (await brain.respond('Repeat that'))['reply'] == answer['reply']
    assert len(calls) == 1 and calls[0][1] == {
        'utterance': utterance, 'awaiting_view': False, 'has_current_view': True}
    assert calls[0][2] == 128
    assert 'There IS a current selected product view' in calls[0][0]
    assert 'Never infer who' in calls[0][0]


@pytest.mark.parametrize('question', [
    'Which is better for me?', 'Why that one?', 'What sweeteners do they use?',
    'Thanks, which has less sugar?', 'Actually I want less caffeine',
    'Now prioritize protein instead', 'Less sugar, please.', 'Is Sugarfree the winner?',
    'Which ingredients are listed?', 'How much sugar did you say?',
    'Compare again', 'Do both of these have caffeine?',
])
async def test_clear_selected_product_followups_and_goal_changes_add_no_router_call(monkeypatch, question):
    _, guide, _, brain, calls = deepseek_configured(monkeypatch, {'route': 'ignore'})
    await brain.respond('Compare these', photo)
    before = list(guide.calls)
    async def forbidden(): pytest.fail('Follow-up unnecessarily requested a photo')
    result = await brain.respond(question, forbidden)
    assert result['status'] == 'ready' and calls == []
    assert guide.calls[len(before):] == ['respond'] and guide.question == question
    if question in ('Actually I want less caffeine', 'Now prioritize protein instead', 'Less sugar, please.'):
        assert brain.snapshot()['comparison_request'] == question


@pytest.mark.parametrize('outcome,expected', [
    ({'route': 'social'}, 'ready'), ({'route': 'brief_clarification'}, 'needs_clarification'),
    ({'route': 'buy'}, 'needs_clarification'), (ProviderError('private error'), 'needs_clarification'),
])
async def test_unclear_selected_speech_never_reaches_guidance_on_social_or_failure(monkeypatch, outcome, expected):
    _, guide, _, brain, calls = deepseek_configured(monkeypatch, outcome)
    await brain.respond('Compare these', photo)
    before = list(guide.calls)
    result = await brain.respond('Something is different about today.')
    assert result['status'] == expected and len(calls) == 1 and guide.calls == before
    assert brain.snapshot()['comparison_request'] is None


async def test_ambiguous_but_actual_followup_can_reach_existing_guidance_once(monkeypatch):
    _, guide, _, brain, calls = deepseek_configured(monkeypatch, {'route': 'follow_up'})
    await brain.respond('Compare these', photo)
    result = await brain.respond('Help me make sense of the choice you just gave.')
    assert result['status'] == 'ready' and len(calls) == 1
    assert guide.calls == ['identify', 'search', 'search', 'respond', 'respond']


async def test_cancelled_current_view_classifier_cannot_change_criterion_or_answer(monkeypatch):
    service, guide, _, brain, _ = deepseek_configured(monkeypatch)
    await brain.respond('Compare these', photo)
    entered, release = asyncio.Event(), asyncio.Event()
    async def classify(*args, **kwargs):
        entered.set()
        await release.wait()
        return {'route': 'follow_up'}
    monkeypatch.setattr(service.providers, 'deepseek_json', classify)
    pending = asyncio.create_task(brain.respond('I was wondering about the choice you made.'))
    await entered.wait()
    brain.set_recipient('household')
    release.set()
    result = await pending
    assert result['status'] == 'cancelled' and brain.snapshot()['comparison_request'] is None
    assert guide.calls == ['identify', 'search', 'search', 'respond']


async def test_missing_view_narrative_ignore_is_silent_and_does_not_establish_criterion(monkeypatch):
    _, guide, _, brain, calls = deepseek_configured(monkeypatch, {'route': 'ignore'})
    result = await brain.respond('There will be two levels, and compare products based on preferences.')
    assert result['status'] == 'ignored' and result['silent'] is True
    assert brain.snapshot()['comparison_request'] is None and guide.calls == [] and len(calls) == 1
