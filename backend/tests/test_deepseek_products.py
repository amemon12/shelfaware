import json

import httpx
import pytest

from app.models import Box
from app.product_guidance import CatalogMatch, CatalogSource, DetectedProduct, ProductGuide
from app.providers import ProviderError, Providers, Settings


def identity(**changes):
    fields = dict(name='Demo cereal', brand='Demo', variant='Original', package_size=None,
                  category='cereal', visible_text='PRIVATE IMAGE TEXT', confidence=0.95,
                  uncertainties=['PRIVATE UNCERTAINTY'], box=Box(x=0.1,y=0.2,w=0.3,h=0.4))
    return DetectedProduct(**(fields | changes))


def provider(handler, **changes):
    return Providers(Settings(**(dict(text_provider='deepseek', deepseek_key='deepseek-secret',
        gemini_key='gemini-secret') | changes)), httpx.MockTransport(handler))


def entry(**changes):
    return dict(title='Demo Original cereal', url='https://manufacturer.example/cereal',
                content='Demo Original cereal manufacturer product information.') | changes


async def test_search_uses_deepseek_only_public_identity_and_real_result_sources():
    seen = []
    def handle(request):
        seen.append(request)
        assert str(request.url) == 'https://api.deepseek.com/v1/tools/search'
        assert request.headers['Authorization'] == 'Bearer deepseek-secret'
        body = json.loads(request.content)
        assert set(body) == {'query','count','need_content'}
        assert body['need_content'] is True and body['count'] == 3
        assert len(body['query']) <= 128
        assert 'Demo' in body['query'] and 'Original' in body['query']
        assert 'PRIVATE' not in body['query'] and 'gemini-secret' not in request.content.decode()
        return httpx.Response(200,json={'results':[entry()]})
    p=provider(handle)
    try:
        found=await ProductGuide(p).search(identity())
        assert found.status == 'grounded' and len(found.sources)==1
        assert found.sources[0].url=='https://manufacturer.example/cereal'
        assert 'manufacturer product information' in found.notes
        assert len(seen)==1
    finally: await p.close()


@pytest.mark.parametrize('results', [[], [entry(url='http://127.0.0.1/private')],
    [entry(content='',summary='',snippet='')], [entry(url='https://user:pass@example.com/x')],
    [entry(url='file:///tmp/private')], [None]])
async def test_unusable_search_results_cannot_become_catalog_evidence(results):
    p=provider(lambda _:httpx.Response(200,json={'results':results}))
    try:
        found=await ProductGuide(p).search(identity())
        assert found.status=='no_match' and found.sources==[]
    finally: await p.close()


async def test_search_bounds_query_evidence_and_deduplicates_sources():
    calls=[]
    def handle(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200,json={'results':[entry(content='x'*20000),entry(),
            entry(url='https://retailer.example/cereal', content='', summary='Search summary')]
        })
    p=provider(handle)
    try:
        result=await ProductGuide(p).search(identity(name='n'*180,brand='b'*100))
        assert len(calls[0]['query'])<=128 and len(result.notes)<=6000
        assert len(result.sources)==2 and 'Search summary' in result.notes
        assert 'x'*1601 not in result.notes
    finally: await p.close()


@pytest.mark.parametrize('changes', [{'deepseek_key':''}, {'text_provider':'unknown'}])
async def test_invalid_configuration_never_falls_back_to_google(changes):
    def forbidden(_): raise AssertionError('No provider call is allowed')
    p=provider(forbidden,**changes)
    try:
        with pytest.raises(ProviderError): await ProductGuide(p).search(identity())
    finally: await p.close()


async def test_malformed_search_envelope_is_an_actionable_failure():
    p=provider(lambda _:httpx.Response(200,json={'results':{'url':'invented'}}))
    try:
        with pytest.raises(ProviderError,match='invalid product results'):
            await ProductGuide(p).search(identity())
    finally: await p.close()


def product():
    return {'identity':identity().model_dump(),'catalog':CatalogMatch(status='grounded',
        notes='Manufacturer describes Demo Original cereal.',
        sources=[CatalogSource(title='Manufacturer',url='https://manufacturer.example/cereal')]).model_dump()}


async def test_spoken_response_routes_to_deepseek_with_minimal_consented_context():
    seen=[]
    text='This looks like Demo cereal. Compare the cited information with your avoid-sugar goal; this exact package is unverified.'
    def handle(request):
        seen.append(request)
        assert request.url.host=='api.deepseek.com'
        body=json.loads(request.content)
        assert body['model']=='deepseek-flash' and body['thinking']=={'type':'disabled'}
        assert body['response_format']=={'type':'json_object'} and 'tools' not in body
        data=json.loads(body['messages'][-1]['content'])
        assert data['context']['goals']==['Avoid sugar']
        assert 'private_record' not in data['context']
        assert 'PRIVATE IMAGE TEXT' not in json.dumps(data)
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'spoken_text':text})}}]})
    p=provider(handle)
    try:
        result=await ProductGuide(p).respond(product(),'Does this fit?',
            {'recipient_id':'self','consented':True,'goals':['Avoid sugar'],'private_record':'NEVER SEND'},[])
        assert result==text and len(seen)==1
    finally: await p.close()


@pytest.mark.parametrize('reply', [
    {'spoken_text':'This has 27 g carbohydrate per bar according to an online listing.'},
    {'spoken_text':'This is safe for your peanut allergy.'},
    {'spoken_text':'Hello.', 'unexpected':'field'},
])
async def test_deepseek_still_uses_existing_spoken_evidence_guards(reply):
    p=provider(lambda _:httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(reply)}}]}))
    try:
        with pytest.raises(ProviderError):
            await ProductGuide(p).respond(product(),'Question',{'recipient_id':'household','consented':True},[])
    finally: await p.close()


async def test_live_service_optional_intent_uses_deepseek_without_a_gemini_key():
    from app.models import CreateSession, TurnRequest
    from app.service import ShoppingService
    calls=[]
    def handle(request):
        calls.append(request)
        assert request.url.host=='api.deepseek.com'
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'intent':'summary'})}}]})
    p=provider(handle,gemini_key='')
    try:
        service=ShoppingService(p)
        sid=service.create(CreateSession(mode='live_phone')).id
        result=await service.turn(sid,TurnRequest(text='Explain what I have looked at',client_turn_id='deepseek-only'))
        assert len(calls)==1
        assert any(e.type=='intent.parsed' and 'DeepSeek' in e.message for e in result.events)
    finally: await p.close()


async def test_health_reports_only_the_selected_text_provider_without_credentials():
    from app.main import create_app
    settings=Settings(text_provider='deepseek',deepseek_key='deepseek-secret',gemini_key='gemini-secret')
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(settings)),base_url='http://test') as client:
        response=await client.get('/api/health')
        assert response.json()['providers']=={'perceptron':False,'gemini':False,'deepseek':True,'anthropic':False}
        assert 'deepseek-secret' not in response.text and 'gemini-secret' not in response.text
