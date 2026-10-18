import pytest

from app.models import ContextRequest
from app.service import local_intent
from test_conversation_product_search import setup, photo
from test_multi_product_turns import item
from test_product_turns import request


@pytest.mark.parametrize('question', [
    "Hey, uh, which one's better amongst these two?",
    "Hey, uh, which one's better for me?",
    "Which one’s better for us?", 'Which one is better for me?',
    'Could you tell me which one would be better for us?',
    'Which one is best among those two?',
    'Does this have added sugar?', 'Is this low sugar?', "What's in this?",
])
async def test_natural_missing_product_reference_captures_once_but_selected_followup_reuses(question):
    _, guide, _, brain = setup()
    result = await brain.respond(question, photo)
    assert result['status'] == 'ready'
    assert guide.calls == ['identify', 'search', 'search', 'respond']
    async def forbidden(): pytest.fail('Existing pair follow-up took another photo')
    await brain.respond(question, forbidden)
    assert guide.calls == ['identify', 'search', 'search', 'respond', 'respond']


async def test_here_only_requests_image_when_waiting_for_a_view_and_clears_on_success_and_recipient_change():
    _, guide, _, brain = setup()
    async def forbidden(): pytest.fail('Unprompted Here took a photo')
    # The first bare Here cannot authorize a photo until a view was requested.
    assert (await brain.respond('Here.', forbidden))['status'] == 'needs_image'
    assert guide.calls == []
    assert (await brain.respond('Here you go.', photo))['status'] == 'ready'
    assert not brain._awaiting_view
    await brain.respond('Here.', forbidden)
    assert guide.calls.count('identify') == 1
    await brain.respond('Compare these')
    assert brain._awaiting_view
    brain.set_recipient('household')
    assert not brain._awaiting_view


@pytest.mark.parametrize('question', ['Which route is better to Costco?', 'Which is better for exercise?',
    'Which one is better, walking or running?', 'Which phone should I buy?',
    'How does photosynthesis work?', 'Can you help me shop?'])
async def test_general_questions_do_not_authorize_a_camera_capture(question):
    _, guide, _, brain = setup()
    async def forbidden(): pytest.fail('General question requested camera')
    await brain.respond(question, forbidden)
    assert guide.calls == []


@pytest.mark.parametrize('acknowledgement', ['Thank you.', 'Thanks so much!', 'Okay, thank you.', 'Got it.', 'Sounds good.', 'No, thank you.', 'No thanks.'])
async def test_acknowledgements_are_local_and_preserve_repeat_without_ending(acknowledgement):
    _, guide, _, brain = setup()
    first = await brain.respond('Compare these', photo)
    before = list(guide.calls)
    async def forbidden(): pytest.fail('Acknowledgment requested camera')
    result = await brain.respond(acknowledgement, forbidden)
    assert result['reply'] in ("You're welcome.", 'Okay.')
    assert not result.get('end_conversation')
    assert (await brain.respond('What did you say?'))['reply'] == first['reply']
    assert guide.calls == before


async def test_mixed_thanks_question_and_ingredient_summary_questions_stay_substantive():
    _, guide, _, brain = setup()
    await brain.respond('Compare these', photo)
    for question in ['Thanks, which has less sugar?', 'Did they add sugar?', 'Summarize why that one is better']:
        await brain.respond(question)
        assert question in guide.question
    assert guide.calls == ['identify', 'search', 'search', 'respond', 'respond', 'respond', 'respond']


@pytest.mark.parametrize('control', ['Stop', 'Goodbye.', 'Bye for now!', 'Please end the conversation.', "That's all.", 'Thanks, goodbye.', "Thank you, that's all."])
async def test_explicit_end_is_local_flagged_and_preserves_last_answer(control):
    _, guide, _, brain = setup()
    first = await brain.respond('Compare these', photo)
    before = list(guide.calls)
    assert (await brain.respond(control))['end_conversation'] is True
    assert (await brain.respond('Repeat that'))['reply'] == first['reply']
    assert guide.calls == before


@pytest.mark.parametrize('control', ["Don't stop.", 'Do not repeat that.', 'No need to say that again.'])
async def test_negated_controls_do_not_end_or_replay(control):
    _, guide, _, brain = setup()
    first = await brain.respond('Compare these', photo)
    before = list(guide.calls)
    response = await brain.respond(control)
    assert response['reply'] == 'Okay.' and not response.get('end_conversation')
    assert (await brain.respond('Repeat'))['reply'] == first['reply']
    assert guide.calls == before


@pytest.mark.parametrize('confirmation', ['Yes.', 'No.', 'Sure.'])
async def test_unbound_confirmation_neither_invents_action_nor_calls_provider(confirmation):
    service, guide, sid, brain = setup()
    assert (await brain.respond(confirmation))['status'] == 'needs_clarification'
    assert guide.calls == [] and not service.get(sid).session.cart


async def test_goal_query_is_local_and_does_not_expose_personal_goal_in_household():
    _, guide, _, brain = setup()
    assert 'Avoid sugar' in (await brain.respond('What is my goal?'))['reply']
    assert brain._comparison_request is None
    await brain.respond('I want less caffeine')
    assert 'less caffeine' in (await brain.respond("What's my current goal?"))['reply']
    brain.set_recipient('household')
    answer = (await brain.respond('What is my goal?'))['reply']
    assert 'Personal goals are off' in answer and 'sugar' not in answer and 'caffeine' not in answer
    assert guide.calls == []


@pytest.mark.parametrize('text,expected', [
    ('Not for me, for my family', 'household'),
    ('Not for my family, for me', 'self'),
    ('This is for my family, 60 grams', 'household'),
    ('This is for me', 'self'),
])
def test_recipient_correction_uses_affirmative_context(text, expected):
    assert local_intent(text).recipient_id == expected


@pytest.mark.parametrize('text,expected', [
    ('Did they add sugar?', 'question'), ('Does this put sugar in the ingredients?', 'question'),
    ('Add this', 'add'), ('Please put this on my list', 'add'), ('Could you add it?', 'add'),
    ('Shopping summary', 'summary'), ('Summarize my shopping list', 'summary'),
    ('Summarize why that one is better', 'compare'), ("Don't repeat that", 'question'),
])
def test_shared_controls_require_actual_control_requests(text, expected):
    assert local_intent(text).intent == expected


async def test_corrected_household_request_never_sends_personal_goal_to_provider():
    service, guide, sid, brain = setup()
    await brain.respond('Compare these', photo)
    await brain.respond('Not for me, for my family. Which is better?')
    assert guide.context == {'recipient_id': 'household', 'consented': True}
    assert service.get(sid).session.recipient_id == 'household'


async def test_pure_context_correction_is_local_and_pardon_repeats_without_new_work():
    _, guide, _, brain = setup()
    first = await brain.respond('Compare these', photo)
    assert (await brain.respond('Pardon?'))['reply'] == first['reply']
    before = list(guide.calls)
    answer = await brain.respond('Not for me, for my family.')
    assert answer['snapshot']['recipient_id'] == 'household'
    answer = await brain.respond('Not for my family, for me.')
    assert answer['snapshot']['recipient_id'] == 'self'
    assert guide.calls == before


async def test_negated_recipient_without_replacement_requires_context_before_auto_advice():
    _, guide, _, brain = setup()
    assert (await brain.respond('This is not for me'))['status'] == 'needs_clarification'
    async def forbidden(): pytest.fail('Unresolved recipient requested a camera')
    assert (await brain.respond_automatically(forbidden))['status'] == 'needs_clarification'
    assert guide.calls == []
    await brain.respond('For my family')
    assert not brain._awaiting_recipient
    await brain.respond_automatically(photo)
    assert guide.context == {'recipient_id': 'household', 'consented': True}


async def test_explicit_current_pair_never_borrows_unpictured_history():
    service, guide, sid, brain = setup()
    await brain.respond('Compare these', photo)
    guide.items = [item('One new can', .2)]
    before = guide.calls.count('respond')
    result = await brain.respond('Hey, which is better amongst these two?', photo)
    # A valid selected pair is a follow-up. Explicitly request a new photo next.
    assert result['status'] == 'ready'
    result = await brain.respond('Compare these two', photo)
    assert result['status'] == 'needs_image' and 'one of the two' in result['reply']
    assert guide.calls.count('respond') == before + 1
    assert service.get(sid).session.current_visual_product_ids == []
    # REST/tap shares the same guard, including an existing one-item selection.
    await service.turn(sid, request('single'))
    calls = list(guide.calls)
    result = await service.turn(sid, request('explicit-pair', image=False, text='Which of these two is better?'))
    assert result.status == 'needs_image' and guide.calls == calls
