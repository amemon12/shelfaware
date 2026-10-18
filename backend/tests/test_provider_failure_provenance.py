"""Safe failure provenance survives disconnect without retaining a photo or body."""
import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

from app.conversation_product_search import ProductSearchBrain
from app.conversation_transport import ConversationConfig, ConversationCoordinator
from app.models import ContextRequest, CreateSession, Profile, TurnRequest
from app.product_guidance import ProductGuide
from app.providers import ProviderError, Providers, Settings
from app.service import ShoppingService
from test_vision_timing import LEFT, completion, image_url

SECRET = 'PRIVATE_TEST_KEY'
BODY = 'PRIVATE_PROVIDER_BODY health record data:image/private'


@pytest.mark.parametrize('status,category', [(400,'http_error'),(401,'authorization'),
    (402,'balance'),(403,'authorization'),(429,'rate_limited'),(500,'http_error'),(503,'http_error')])
async def test_http_failure_metadata_is_bounded_and_never_contains_remote_data(status, category):
    providers = Providers(Settings(), httpx.MockTransport(lambda request: httpx.Response(status,text=BODY)))
    try:
        with pytest.raises(ProviderError) as caught:
            await providers._post('https://api.perceptron.inc/v1/chat/completions',
                {'Authorization':'Bearer '+SECRET}, {'private':'data:image/private'})
        assert caught.value.diagnostic() == {'provider':'perceptron','stage':'unknown',
            'category':category,'http_status':status}
        assert BODY not in str(caught.value)
        assert SECRET not in json.dumps(caught.value.diagnostic())
    finally: await providers.close()


@pytest.mark.parametrize('case,category', [('network','network'),('timeout','timeout'),
    ('invalid_json','invalid_response'),('not_object','invalid_response'),('too_large','response_too_large')])
async def test_transport_and_parse_failures_are_distinct_without_printing_details(case,category):
    def handle(request):
        if case=='network': raise httpx.ConnectError(BODY,request=request)
        if case=='timeout': raise httpx.ReadTimeout(BODY,request=request)
        if case=='invalid_json': return httpx.Response(200,text=BODY)
        if case=='not_object': return httpx.Response(200,json=['private'])
        return httpx.Response(200,text='x'*(1024*1024+1))
    providers=Providers(Settings(),httpx.MockTransport(handle))
    try:
        with pytest.raises(ProviderError) as caught:
            await providers._post('https://api.anthropic.com/v1/messages',{}, {})
        diagnostic=caught.value.diagnostic()
        assert diagnostic['provider']=='anthropic' and diagnostic['category']==category
        assert diagnostic['http_status']==(200 if case in ('invalid_json','not_object','too_large') else None)
        assert BODY not in json.dumps(diagnostic) and BODY not in str(caught.value)
    finally: await providers.close()


def test_arbitrary_metadata_and_boolean_status_are_never_exposed():
    error=ProviderError(BODY,provider=SECRET,stage=BODY,category=BODY,http_status=True)
    assert error.diagnostic(provider=SECRET,stage=BODY)=={
        'provider':'unknown','stage':'unknown','category':'unknown','http_status':None}


@pytest.mark.parametrize('failed_stage', ['selection','crop_identity'])
async def test_actual_vision_path_tags_the_failed_request(failed_stage):
    calls=[]
    def handle(request):
        payload=json.loads(request.content); calls.append(payload['response_format']['json_schema']['name'])
        if failed_stage=='crop_identity' and len(calls)==1:
            return httpx.Response(200,json=completion({'selections':[LEFT]}))
        return httpx.Response(503,text=BODY)
    providers=Providers(Settings(perceptron_key=SECRET),httpx.MockTransport(handle))
    try:
        with pytest.raises(ProviderError) as caught:
            await ProductGuide(providers).identify_many(image_url())
        assert caught.value.diagnostic()=={'provider':'perceptron','stage':failed_stage,
            'category':'http_error','http_status':503}
        assert len(calls)==(1 if failed_stage=='selection' else 2)
    finally: await providers.close()


class Socket:
    async def close(self,code): pass


def service_with_failure():
    providers=Providers(Settings(perceptron_key=SECRET),httpx.MockTransport(
        lambda request: httpx.Response(502,text=BODY)))
    service=ShoppingService(providers)
    session=service.create(CreateSession(mode='live_glasses',profile=Profile(consented=True,goals=['Avoid sugar'])))
    return providers,service,session.id


async def test_disconnected_protected_snapshot_keeps_only_failure_metadata_and_no_frame():
    providers,service,sid=service_with_failure()
    coordinator=ConversationCoordinator(service,config=ConversationConfig(product_mode='product_search',
        transcription_provider='disabled',speech_provider='disabled'),
        brain_factory=lambda profile,recipient:ProductSearchBrain(service,profile,recipient))
    connection=await coordinator.attach(Socket(),sid)
    async def frame():
        return SimpleNamespace(image_data_url=image_url(),width=100,height=100,
            received_at=datetime.now(timezone.utc).isoformat())
    try:
        result=await connection.runtime.brain.respond('Compare this',frame)
        assert result['status']=='needs_image'
        assert result['reply']=='I couldn’t finish recognizing the products. Please try again.'
        expected={'provider':'perceptron','stage':'selection','category':'http_error','http_status':502}
        assert result['snapshot']['provider_failure']==expected
        assert service.get(sid).frame is not None
        await connection.shutdown()
        snapshot=coordinator.snapshot(sid)
        assert snapshot['connected'] is False
        assert snapshot['snapshot']['provider_failure']==expected
        assert service.get(sid).frame is None and service.get(sid).latest is None
        with pytest.raises(HTTPException): service.get_frame(sid)
        assert not service.get(sid).session.visual_products
        serialized=json.dumps(snapshot['snapshot']['provider_failure'])
        assert not any(value in serialized for value in (BODY,SECRET,'data:image','Avoid sugar'))
        snapshot['snapshot']['provider_failure']['category']='mutated'
        assert service.get(sid).provider_failure==expected
        # A new turn clears the old provenance even when handled locally.
        response=await connection.runtime.brain.respond('Thanks')
        assert response['snapshot']['provider_failure'] is None
    finally:
        await connection.shutdown()
        await providers.close()


async def test_context_switch_clears_failure_without_restoring_photo():
    providers,service,sid=service_with_failure()
    try:
        await service.turn(sid,TurnRequest(text='Identify this',client_turn_id='one',
            analysis_mode='product',image_data_url=image_url()))
        assert service.get(sid).provider_failure
        service.context(sid,ContextRequest(recipient_id='household'))
        assert service.get(sid).provider_failure is None and service.get(sid).frame is None
    finally: await providers.close()


async def test_late_cancelled_failure_cannot_overwrite_new_turn_diagnostic():
    providers,service,sid=service_with_failure()
    started,release=asyncio.Event(),asyncio.Event()
    class DelayedGuide:
        async def identify_many(self,image):
            started.set(); await release.wait()
            raise ProviderError('Safe old failure',provider='perceptron',stage='crop_identity',
                category='http_error',http_status=500)
    service.product_guide=DelayedGuide()
    request=lambda key:TurnRequest(text='Identify this',client_turn_id=key,analysis_mode='product',image_data_url=image_url())
    pending=asyncio.create_task(service.turn(sid,request('old')))
    try:
        await started.wait()
        service.cancel(sid)
        service.product_guide=ProductGuide(providers)
        current=await service.turn(sid,request('new'))
        expected=dict(service.get(sid).provider_failure)
        assert current.status=='needs_image' and expected['stage']=='selection'
        release.set()
        old=await pending
        assert old.status=='cancelled' and service.get(sid).provider_failure==expected
    finally:
        release.set(); await pending
        service.cancel(sid)
        await providers.close()


@pytest.mark.parametrize('status,expected', [
    (429,'Product recognition is busy. Please try again shortly.'),
    (401,'Product recognition is unavailable. Check the server’s recognition access before trying again.'),
    (402,'Product recognition is unavailable until the service account has credit again.'),
])
async def test_spoken_provider_failure_is_actionable_without_internal_names(status,expected):
    providers,service,sid=service_with_failure()
    providers.transport=httpx.MockTransport(lambda request:httpx.Response(status,text=BODY))
    try:
        result=await service.turn(sid,TurnRequest(text='Identify this',client_turn_id='one',
            analysis_mode='product',image_data_url=image_url()))
        assert result.reply==expected
        assert service.get(sid).provider_failure['http_status']==status
        assert service.get(sid).provider_failure['provider']=='perceptron'
        assert 'Perceptron' not in result.reply and 'selection' not in result.reply
    finally: await providers.close()
