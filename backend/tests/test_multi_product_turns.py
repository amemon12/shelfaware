import asyncio
from types import SimpleNamespace

import pytest

from app.models import Box, ContextRequest, Session, TurnRequest
from app.product_guidance import DetectedProduct
from app.providers import ProviderError
from app.service import Record
from test_product_turns import FakeGuide, request, setup


def item(name, left):
    return DetectedProduct(name=name, brand='Demo', variant=name, package_size=None,
        category='drink', visible_text=name, confidence=0.95, uncertainties=[],
        box=Box(x=left, y=0.2, w=0.2, h=0.6))


class PairGuide(FakeGuide):
    def __init__(self, pause=None):
        super().__init__(pause)
        self.items = [item('Sugarfree', 0.1), item('Peach', 0.6)]
        self.failed_name = None
        self.missing_name = None

    async def identify_many(self, image):
        await self.gate('identify')
        return self.items

    async def search(self, identity):
        await self.gate('search')
        if identity.name == self.failed_name:
            raise ProviderError('Product lookup unavailable.')
        if identity.name == self.missing_name:
            return SimpleNamespace(status='no_match', notes='No cited result.', sources=[])
        return SimpleNamespace(status='grounded', notes=f'{identity.name} cited catalog information.',
            sources=[SimpleNamespace(model_dump=lambda: {'title': identity.name,
                'url': 'https://example.com/' + identity.name})])

    async def respond(self, product, question, context, previous):
        self.question = question
        self.current = product
        return await super().respond(product, question, context, previous)


def paired(pause=None):
    service, _, sid = setup()
    guide = PairGuide(pause)
    service.product_guide = guide
    return service, guide, sid


@pytest.mark.parametrize('question', ['Is this a good option?', 'Is this a good option for me?'])
async def test_one_photo_two_products_automatically_compares_default_question(question):
    service, guide, sid = paired()
    result = await service.turn(sid, request(text=question))
    assert result.status == 'ready'
    assert len(result.session.current_visual_product_ids) == 2
    assert len(set(result.session.current_visual_product_ids)) == 2
    assert result.session.current_visual_product_id == result.session.current_visual_product_ids[-1]
    assert [e.box.x for e in result.events if e.type == 'product.detected'] == [0.1, 0.6]
    assert len([e for e in result.evidence if e.label == 'Held product candidate']) == 2
    assert guide.current['identity']['name'] == 'Peach'
    assert [p['identity']['name'] for p in guide.previous] == ['Sugarfree']
    assert 'Compare the two products' in guide.question
    assert question in guide.question
    assert guide.calls == ['identify', 'search', 'search', 'respond']
    assert result.session.products == [] and result.comparison is None
    assert not result.session.cart and result.pending_action is None
    assert all(not e.verified for e in result.evidence)


@pytest.mark.parametrize('question', ['Which has caffeine?', 'What are these?', 'Have a look at these.',
    'Which ingredients are in these?', 'What is the difference between these?',
    'Which one is better for me?', 'Compare these products for my current shopping goal.'])
async def test_explicit_fresh_photo_question_keeps_its_intent_with_both_sources(question):
    service, guide, sid = paired()
    result = await service.turn(sid, request('fresh-explicit', text=question))
    assert result.status == 'ready'
    if question == 'What are these?':
        assert 'appear to be' in result.reply and 'Sugarfree' in result.reply and 'Peach' in result.reply
        assert guide.calls == ['identify', 'search', 'search']
        return
    assert guide.question == question
    assert guide.current['identity']['name'] == 'Peach'
    assert guide.previous[0]['identity']['name'] == 'Sugarfree'
    assert guide.current['catalog']['sources'] and guide.previous[0]['catalog']['sources']
    assert guide.calls == ['identify', 'search', 'search', 'respond']


async def test_pair_excludes_unrelated_history_and_survives_text_followup():
    service, guide, sid = paired()
    guide.items = [item('Unrelated cereal', 0.2)]
    await service.turn(sid, request('old'))
    guide.items = [item('Sugarfree', 0.1), item('Peach', 0.6)]
    await service.turn(sid, request('pair'))
    await service.turn(sid, request('followup', image=False, text='Which is better?'))
    assert [p['identity']['name'] for p in guide.previous] == ['Sugarfree']
    assert guide.calls.count('identify') == 2
    assert guide.calls.count('search') == 3
    assert guide.calls.count('respond') == 3
    assert len(service.get(sid).session.visual_products) == 3


@pytest.mark.parametrize('question', [
    'Which sweeteners are in these?',
    'Why did you recommend that one?',
    'Did they add sugar?',
    'Summarize why that one is better',
    'Thanks, which has less sugar?',
    'Compare again for lower caffeine',
])
async def test_saved_pair_followup_preserves_exact_question_and_both_product_sources(question):
    service, guide, sid = paired()
    await service.turn(sid, request('capture'))
    selected = list(service.get(sid).session.current_visual_product_ids)
    result = await service.turn(sid, request('followup', image=False, text=question))
    assert result.status == 'ready'
    assert guide.question == question
    assert guide.current['identity']['name'] == 'Peach'
    assert [item['identity']['name'] for item in guide.previous] == ['Sugarfree']
    assert guide.current['catalog']['sources'] and guide.previous[0]['catalog']['sources']
    assert result.session.current_visual_product_ids == selected
    assert guide.calls == ['identify', 'search', 'search', 'respond', 'respond']


async def test_saved_pair_criterion_correction_overrides_goal_without_forced_ranking_instruction():
    service, guide, sid = paired()
    await service.turn(sid, request('capture'))
    question = 'Actually, focus on caffeine instead. Explain what each listing says.'
    result = await service.turn(sid, request('criterion', image=False, text=question,
        conversation_request='Focus on caffeine instead'))
    assert result.status == 'ready' and guide.question == question
    assert guide.context['goals'] == ['Focus on caffeine instead']
    assert len(guide.previous) == 1
    assert guide.calls == ['identify', 'search', 'search', 'respond', 'respond']


async def test_new_single_photo_replaces_pair_without_implicit_old_comparison():
    service, guide, sid = paired()
    await service.turn(sid, request())
    guide.items = [item('New product', 0.4)]
    result = await service.turn(sid, request('single'))
    assert len(result.session.current_visual_product_ids) == 1
    assert guide.previous == []
    assert len([e for e in result.events if e.type == 'product.detected']) == 1


@pytest.mark.parametrize('question', ['Which one is better?', 'Compare this with the previous one',
    'What is the difference?', 'What about the first one?'])
async def test_new_single_comparison_never_borrows_from_an_unrelated_historical_pair(question):
    service, guide, sid = paired()
    await service.turn(sid, request('old-pair'))
    guide.items = [item('New product', .4)]
    result = await service.turn(sid, request('new-single', text=question))
    assert result.status == 'needs_image' and 'New product' in result.reply
    assert len(result.session.current_visual_product_ids) == 1
    assert guide.calls.count('respond') == 1
    assert service.get(sid).product_memory == []
    again = await service.turn(sid, request('single-followup', image=False, text=question))
    assert again.status == 'needs_image' and guide.calls.count('respond') == 1
    assert 'Sugarfree' not in again.reply and 'Peach' not in again.reply


async def test_new_single_what_about_this_uses_only_current_product_and_can_establish_a_new_pair():
    service, guide, sid = paired()
    await service.turn(sid, request('old-pair'))
    guide.items = [item('New product', .4)]
    result = await service.turn(sid, request('new-single', text='What about this?'))
    assert result.status == 'ready' and not guide.previous
    assert guide.current['identity']['name'] == 'New product'
    assert 'recent_product_exchanges' not in guide.context
    await service.turn(sid, request('single-followup', image=False, text='Which ingredients does it have?'))
    assert not guide.previous and guide.calls.count('identify') == 2
    guide.items = [item('New product', .1), item('Another new product', .6)]
    assert (await service.turn(sid, request('new-pair', text='Which is better?'))).status == 'ready'
    assert [p['identity']['name'] for p in guide.previous] == ['New product']
    assert guide.current['identity']['name'] == 'Another new product'


async def test_two_physical_packages_with_same_identity_keep_two_boxes():
    service, guide, sid = paired()
    guide.items = [item('Same variant', 0.1), item('Same variant', 0.6)]
    result = await service.turn(sid, request())
    assert len(result.session.visual_products) == 2
    assert len(set(result.session.current_visual_product_ids)) == 2
    assert len(guide.previous) == 1


@pytest.mark.parametrize('failure', ['failed_name', 'missing_name'])
async def test_missing_evidence_for_either_package_cannot_produce_a_winner(failure):
    service, guide, sid = paired()
    setattr(guide, failure, 'Peach')
    result = await service.turn(sid, request())
    assert result.status == 'unknown'
    assert 'respond' not in guide.calls
    assert len(result.session.current_visual_product_ids) == 2
    again = await service.turn(sid, request('followup', image=False, text='Which is better?'))
    assert again.status == 'unknown'
    assert 'respond' not in guide.calls


async def test_variant_conflict_is_explained_on_initial_comparison_and_followup():
    service, guide, sid = paired()
    original_search = guide.search
    async def search(identity):
        if identity.name == 'Peach':
            await guide.gate('search')
            return SimpleNamespace(status='no_match', sources=[], notes=(
                'Catalog variant conflict: Peach has both regular and sugar-free listings. '
                'Show whether its front says Sugarfree.'))
        return await original_search(identity)
    guide.search = search
    for turn in [request(), request('followup', image=False, text='Which is better?')]:
        result = await service.turn(sid, turn)
        assert result.status == 'unknown'
        assert 'conflicting variants' in result.reply and 'Peach' in result.reply
        assert 'show' not in result.reply.casefold()
        assert 'Catalog variant conflict:' not in result.reply
    assert 'respond' not in guide.calls
    assert len(result.session.current_visual_product_ids) == 2


@pytest.mark.parametrize('stage', ['identify', 'search', 'respond'])
async def test_pair_cancel_at_each_stage_cannot_publish_late_comparison(stage):
    service, guide, sid = paired(stage)
    pending = asyncio.create_task(service.turn(sid, request()))
    await guide.started.wait()
    service.cancel(sid)
    guide.release.set()
    result = await pending
    assert result.status == 'cancelled'
    assert service.get(sid).latest is None
    assert service.get(sid).frame is None
    assert not any(m.role == 'assistant' for m in service.get(sid).session.messages)
    if stage != 'respond':
        assert service.get(sid).session.current_visual_product_ids == []


async def test_pair_household_followup_omits_personal_goal_and_health():
    service, guide, sid = paired()
    await service.turn(sid, request())
    service.context(sid, ContextRequest(recipient_id='household'))
    await service.turn(sid, request('household', image=False, text='Which is better?'))
    assert guide.context == {'recipient_id': 'household', 'consented': True}
    assert len(guide.previous) == 1


async def test_failed_new_photo_and_mode_switch_clear_pair_selection():
    service, guide, sid = paired()
    await service.turn(sid, request())
    guide.items = []
    result = await service.turn(sid, request('empty'))
    assert result.status == 'needs_image'
    assert result.session.current_visual_product_ids == []
    assert result.session.current_visual_product_id is None


async def test_snapshot_restores_pair_for_text_only_reuse_when_mode_preserved():
    service, guide, sid = paired()
    await service.turn(sid, request())
    saved = service.get(sid).session.model_dump_json()
    service.records[sid] = Record(session=Session.model_validate_json(saved), analysis_mode='product')
    result = await service.turn(sid, request('restored', image=False, text='Compare again'))
    assert result.status == 'ready'
    assert len(guide.previous) == 1
    assert guide.calls.count('identify') == 1


async def test_pair_duplicate_turn_does_not_repeat_provider_calls():
    service, guide, sid = paired()
    first = await service.turn(sid, request())
    second = await service.turn(sid, request())
    assert first == second
    assert guide.calls == ['identify', 'search', 'search', 'respond']


async def test_pair_response_failure_keeps_both_products_in_visible_evidence():
    service, guide, sid = paired()
    guide.reply_error = True
    result = await service.turn(sid, request())
    assert result.status == 'unknown'
    assert 'Sugarfree' in result.reply and 'Peach' in result.reply
    assert len([e for e in result.evidence if e.label == 'Held product candidate']) == 2


async def test_mode_switch_clears_both_pair_ids_and_prevents_reuse():
    service, guide, sid = paired()
    await service.turn(sid, request())
    result = await service.turn(sid, TurnRequest(text='Repeat that', client_turn_id='label', analysis_mode='label'))
    assert result.session.current_visual_product_ids == []
    assert result.session.current_visual_product_id is None
    again = await service.turn(sid, request('product-again', image=False))
    assert again.status == 'needs_image'
