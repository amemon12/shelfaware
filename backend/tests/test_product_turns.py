import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.models import Box, ContextRequest, CreateSession, NutritionContext, Profile, TurnRequest
from app.providers import ProviderError, Providers, Settings
from app.service import ShoppingService

IMAGE = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aOC8AAAAASUVORK5CYII='


def candidate(name='Coconut Water'):
    box = Box(x=0.2, y=0.3, w=0.4, h=0.5)
    return SimpleNamespace(box=box, model_dump=lambda: dict(name=name, brand='Example Brand',
        variant='Original', package_size='296 mL', category='drink', visible_text=name,
        confidence=0.94, uncertainties=[], box=box))


class FakeGuide:
    def __init__(self, pause=None):
        self.detected = candidate()
        self.pause = pause
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = []
        self.search_error = False
        self.reply_error = False

    async def gate(self, stage):
        self.calls.append(stage)
        if self.pause == stage:
            self.started.set()
            await self.release.wait()

    async def identify(self, image):
        await self.gate('identify')
        return self.detected

    async def identify_many(self, image):
        result = await self.identify(image)
        return [result] if result is not None else []

    async def search(self, identity):
        await self.gate('search')
        if self.search_error:
            raise ProviderError('Catalog search is unavailable.')
        return SimpleNamespace(status='grounded', notes='Product listing; bottle formulation not verified.',
            sources=[SimpleNamespace(model_dump=lambda: {'title':'Manufacturer product page','url':'https://example.com/product'})])

    async def respond(self, product, question, context, previous):
        await self.gate('respond')
        self.context = context
        self.previous = previous
        if self.reply_error:
            raise ProviderError('The short response is unavailable.')
        return 'This appears to be coconut water. Compare the listed total carbohydrate for the amount you intend to drink.'


def setup(pause=None):
    service = ShoppingService(Providers(Settings()))
    service.product_guide = FakeGuide(pause)
    profile = Profile(consented=True, goals=['Compare carbohydrate'], avoid_ingredients=['Peanuts'],
        nutrition_context=NutritionContext(synthetic=True, conditions=['Type 1 diabetes mellitus'],
                                            allergies=['Peanuts'], focus=['Total carbohydrates']))
    session = service.create(CreateSession(mode='live_glasses', profile=profile))
    return service, service.product_guide, session.id


def request(turn_id='one', image=True, **kwargs):
    return TurnRequest(text=kwargs.pop('text', 'Is this a good option?'), client_turn_id=turn_id,
                       analysis_mode='product', image_data_url=IMAGE if image else None, **kwargs)


@pytest.mark.asyncio
async def test_product_flow_keeps_catalog_separate_and_reuses_without_more_photos():
    service, guide, sid = setup()
    result = await service.turn(sid, request())
    assert result.status == 'ready'
    assert result.session.products == []
    assert result.comparison is None
    assert result.session.visual_products[0].package_size == '296 mL'
    assert result.session.visual_products[0].sources[0].url == 'https://example.com/product'
    assert all(not item.verified for item in result.evidence)
    assert any(e.type == 'product.detected' and e.box is not None for e in result.events)
    assert guide.context['nutrition_context']['conditions'] == ['Type 1 diabetes mellitus']
    assert 'portion_grams' not in guide.context
    result2 = await service.turn(sid, request('two', image=False))
    assert result2.status == 'ready'
    assert guide.calls == ['identify', 'search', 'respond', 'respond']
    duplicate = await service.turn(sid, request('two', image=False))
    assert duplicate == result2
    assert guide.calls.count('respond') == 2


@pytest.mark.asyncio
async def test_sequential_comparison_does_not_assume_an_older_detection_is_still_selected():
    service, guide, sid = setup()
    await service.turn(sid, request())
    repeated = await service.turn(sid, request('two', text='Compare this with the first'))
    assert repeated.status == 'needs_image'
    assert len(service.get(sid).session.visual_products) == 1
    assert not guide.previous
    guide.detected = candidate('Green Juice')
    result = await service.turn(sid, request('three', text='Compare this with the first'))
    assert len(result.session.visual_products) == 2
    assert result.status == 'needs_image' and 'Green Juice' in result.reply
    assert guide.calls.count('respond') == 1 and not guide.previous
    assert result.comparison is None  # No gram conversion or verified numeric arithmetic.


@pytest.mark.asyncio
async def test_household_context_does_not_share_personal_context_or_previous_reply():
    service, guide, sid = setup()
    await service.turn(sid, request())
    service.context(sid, ContextRequest(recipient_id='household'))
    await service.turn(sid, request('two', image=False, text='Repeat that'))
    assert guide.calls.count('respond') == 2
    assert guide.context == {'recipient_id':'household', 'consented':True}


@pytest.mark.asyncio
async def test_no_held_product_clears_current_selection_without_search():
    service, guide, sid = setup()
    await service.turn(sid, request())
    guide.detected = None
    result = await service.turn(sid, request('two'))
    assert result.status == 'needs_image'
    assert result.session.current_visual_product_id is None
    assert guide.calls == ['identify', 'search', 'respond', 'identify']


@pytest.mark.parametrize('stage', ['identify', 'search', 'respond'])
@pytest.mark.asyncio
async def test_cancel_during_each_model_stage_prevents_late_reply(stage):
    service, guide, sid = setup(pause=stage)
    pending = asyncio.create_task(service.turn(sid, request()))
    await guide.started.wait()
    service.cancel(sid)
    guide.release.set()
    result = await pending
    assert result.status == 'cancelled'
    assert service.get(sid).latest is None
    assert service.get(sid).frame is None
    assert not any(m.role == 'assistant' for m in service.get(sid).session.messages)


@pytest.mark.asyncio
async def test_provider_error_is_visible_and_never_a_verified_label():
    service, guide, sid = setup()
    guide.search_error = True
    result = await service.turn(sid, request())
    assert result.status == 'unknown' and 'unavailable' in result.reply
    assert guide.calls == ['identify', 'search']
    assert result.session.current_product_id is None
    assert result.session.products == []


@pytest.mark.asyncio
async def test_product_add_does_not_write_or_propose_list_action():
    service, guide, sid = setup()
    await service.turn(sid, request())
    result = await service.turn(sid, request('two', image=False, text='Add this to my list'))
    assert result.pending_action is None and not result.session.cart
    assert guide.calls == ['identify', 'search', 'respond']


def test_demo_health_context_requires_explicit_profile_consent():
    service = ShoppingService(Providers(Settings()))
    with pytest.raises(HTTPException) as err:
        service.create(CreateSession(mode='live_glasses', profile=Profile(
            nutrition_context=NutritionContext(synthetic=True, conditions=['Type 1 diabetes']))))
    assert err.value.status_code == 400


@pytest.mark.asyncio
async def test_mode_switch_and_household_repeat_do_not_resurrect_personal_product_answer():
    service, guide, sid = setup()
    await service.turn(sid, request())
    service.context(sid, ContextRequest(recipient_id='household'))
    response = await service.turn(sid, TurnRequest(text='Repeat that', client_turn_id='label-repeat'))
    assert 'No label answer' in response.reply
    assert response.session.current_visual_product_id is None
    assert 'coconut water' not in response.reply
    again = await service.turn(sid, request('back-to-product', image=False))
    assert again.status == 'needs_image'
    assert guide.calls == ['identify','search','respond']
