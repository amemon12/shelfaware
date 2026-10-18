import json

import httpx
import pytest

from app.main import create_app
from app.product_guidance import ProductGuide, _SpokenResponse
from app.providers import Intent, ProviderError, Providers, Settings
from test_deepseek_products import identity, product


def completion(text='The manufacturer lists sucralose and acesulfame K as the sweeteners.'):
    return {'stop_reason': 'end_turn', 'content': [{'type': 'text', 'text': json.dumps({'spoken_text': text})}]}


def provider(handler, **changes):
    return Providers(Settings(**(dict(text_provider='anthropic', anthropic_key='private-test-key',
        deepseek_key='separate-search-key', catalog_search_provider='deepseek') | changes)), httpx.MockTransport(handler))


async def test_text_only_sonnet_and_deepseek_search_are_independent():
    calls = []
    def handle(request):
        calls.append(request)
        payload = json.loads(request.content)
        if request.url.host == 'api.deepseek.com':
            assert str(request.url) == 'https://api.deepseek.com/v1/tools/search'
            assert request.headers['authorization'] == 'Bearer separate-search-key'
            assert 'x-api-key' not in request.headers
            assert set(payload) == {'query', 'count', 'need_content'}
            return httpx.Response(200, json={'results': [{'title': 'Demo Original cereal',
                'url': 'https://manufacturer.example/cereal', 'content': 'Demo Original cereal product information.'}]})
        assert str(request.url) == 'https://api.anthropic.com/v1/messages'
        assert request.headers['x-api-key'] == 'private-test-key'
        assert request.headers['anthropic-version'] == '2023-06-01'
        assert request.headers['anthropic-workspace-id'] == 'workspace-test'
        assert 'authorization' not in request.headers
        assert payload['model'] == 'claude-sonnet-5-5'
        assert payload['thinking'] == {'type': 'between_tools'}
        assert payload['output_config']['effort'] == 'low'
        schema = payload['output_config']['format']['schema']
        assert schema['additionalProperties'] is False and schema['required'] == ['spoken_text']
        assert 'maxLength' not in schema['properties']['spoken_text']
        assert payload['max_tokens'] == 800 and 'tools' not in payload
        content = payload['messages'][0]['content']
        assert len(content) == 1 and content[0]['type'] == 'text'
        data = json.loads(content[0]['text'])
        assert data['question'] == 'What sweeteners are used?'
        assert data['context'] == {'recipient_id': 'self', 'goals': ['Avoid sugar'], 'avoid_ingredients': []}
        assert 'PRIVATE IMAGE TEXT' not in json.dumps(data) and 'PRIVATE' not in data['context']
        assert 'do not repeat the previous winner recommendation' in payload['system']
        assert 'listing is silent' in payload['system']
        assert 'Do not rank raw amounts on different serving bases' in payload['system']
        return httpx.Response(200, json=completion())
    p = provider(handle, anthropic_workspace_id='workspace-test')
    try:
        assert (await ProductGuide(p).search(identity())).status == 'grounded'
        reply = await ProductGuide(p).respond(product(), 'What sweeteners are used?',
            {'recipient_id': 'self', 'consented': True, 'goals': ['Avoid sugar'], 'PRIVATE': 'do not send'}, [])
        assert reply == 'The manufacturer lists sucralose and acesulfame K as the sweeteners.'
        assert len(calls) == 2
    finally:
        await p.close()


def test_settings_read_explicit_anthropic_and_independent_search(monkeypatch):
    for key, value in {'TEXT_PROVIDER': ' Anthropic ', 'CATALOG_SEARCH_PROVIDER': ' DeepSeek ',
        'ANTHROPIC_API_KEY': ' test-key ', 'ANTHROPIC_MODEL': ' claude-sonnet-5-5 ',
        'ANTHROPIC_WORKSPACE_ID': ' workspace '}.items(): monkeypatch.setenv(key, value)
    settings = Settings.from_env()
    assert settings.text_provider == 'anthropic' and settings.search_provider == 'deepseek'
    assert settings.anthropic_key == 'test-key' and settings.anthropic_model == 'claude-sonnet-5-5'
    assert settings.anthropic_workspace_id == 'workspace'
    assert 'test-key' not in repr(settings)
    assert Settings(text_provider='deepseek').search_provider == 'deepseek'
    assert Settings().search_provider == 'gemini'
    assert Settings(text_provider='anthropic').search_provider == ''


@pytest.mark.parametrize('reason', ['max_tokens', 'tool_use', 'pause_turn', 'refusal', 'stop_sequence', None])
async def test_incomplete_anthropic_output_never_retries_or_falls_back(reason):
    calls=[]; response=completion(); response['stop_reason']=reason
    p=provider(lambda r: calls.append(r) or httpx.Response(200,json=response))
    try:
        with pytest.raises(ProviderError,match='complete valid JSON'):
            await ProductGuide(p).respond(product(),'Question',{},[])
        assert len(calls)==1 and calls[0].url.host=='api.anthropic.com'
    finally: await p.close()


@pytest.mark.parametrize('content', [[], None, [{'type':'tool_use','name':'buy','input':{}}],
    [{'type':'thinking','thinking':'hidden'}], [{'type':'text','text':None}],
    [{'type':'text','text':'{}'}, {'type':'tool_use','name':'buy'}],
    [{'type':'text','text':'[]'}], [{'type':'text','text':'```json\n{}\n```'}],
    [{'type':'text','text':'{"x":NaN}'}], [{'type':'text','text':'x'*32001}]])
async def test_malformed_or_action_bearing_anthropic_output_is_rejected(content):
    p=provider(lambda _:httpx.Response(200,json={'stop_reason':'end_turn','content':content}))
    try:
        with pytest.raises(ProviderError,match='complete valid JSON'):
            await p.anthropic_json('Return JSON.',{},_SpokenResponse.model_json_schema())
    finally: await p.close()


@pytest.mark.parametrize('changes', [{'anthropic_key':''}, {'anthropic_model':'other'}, {'anthropic_model':'../invalid'}])
async def test_invalid_anthropic_configuration_does_not_call_another_provider(changes):
    p=provider(lambda _:pytest.fail('Must not send a request'),**changes)
    try:
        with pytest.raises(ProviderError): await ProductGuide(p).respond(product(),'Question',{},[])
    finally: await p.close()


@pytest.mark.parametrize('changes', [{'catalog_search_provider':''}, {'catalog_search_provider':'anthropic'}, {'catalog_search_provider':'unknown'}])
async def test_anthropic_requires_a_supported_explicit_search_provider(changes):
    p=provider(lambda _:pytest.fail('Must not send a request'),**changes)
    try:
        with pytest.raises(ProviderError,match='CATALOG_SEARCH_PROVIDER'):
            await ProductGuide(p).search(identity())
    finally: await p.close()


@pytest.mark.parametrize('status', [401,403,429,500])
async def test_anthropic_auth_rate_limit_and_server_errors_are_never_retried(status):
    calls=[]
    p=provider(lambda r:calls.append(r) or httpx.Response(status,text='private-provider-error'))
    try:
        with pytest.raises(ProviderError) as error: await ProductGuide(p).respond(product(),'Question',{},[])
        assert len(calls)==1 and 'private-provider-error' not in str(error.value)
    finally: await p.close()


@pytest.mark.parametrize('draft', [' '.join(['word']*66), '', 'The online listing reports 27 g carbohydrate per bar.'])
async def test_anthropic_shares_exactly_one_length_or_numeric_rewrite(draft):
    calls=[]
    def handle(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200,json=completion(draft if len(calls)==1 else 'The supplied listings do not establish that detail.'))
    p=provider(handle)
    try:
        assert await ProductGuide(p).respond(product(),'Question',{},[])=='The supplied listings do not establish that detail.'
        assert len(calls)==2
        assert calls[0]['messages']==calls[1]['messages']
        assert draft not in calls[1]['system'] if draft else True
    finally: await p.close()


async def test_shared_retry_budget_does_not_allow_third_attempt():
    calls=[]
    p=provider(lambda r:calls.append(r) or httpx.Response(200,json=completion(' '.join(['word']*66))))
    try:
        with pytest.raises(ProviderError): await ProductGuide(p).respond(product(),'Question',{},[])
        assert len(calls)==2
    finally: await p.close()


@pytest.mark.parametrize('reply', [{'spoken_text':'This is safe for your peanut allergy.'},
    {'spoken_text':'Hello.','extra':'must reject'}, {'spoken_text':False}])
async def test_unsafe_or_schema_invalid_guidance_is_not_retried(reply):
    calls=[]
    p=provider(lambda r:calls.append(r) or httpx.Response(200,json={'stop_reason':'end_turn',
        'content':[{'type':'text','text':json.dumps(reply)}]}))
    try:
        with pytest.raises(ProviderError): await ProductGuide(p).respond(product(),'Question',{},[])
        assert len(calls)==1
    finally: await p.close()


async def test_anthropic_intent_still_validates_local_schema():
    p=provider(lambda _:httpx.Response(200,json={'stop_reason':'end_turn',
        'content':[{'type':'text','text':json.dumps({'intent':'compare','recipient_id':None,'portion_grams':None})}]}))
    try: assert (await p.interpret('Compare them')).intent=='compare'
    finally: await p.close()


@pytest.mark.parametrize('max_tokens',[0,-1,True,1.5,'800'])
async def test_anthropic_invalid_token_bounds_fail_before_request(max_tokens):
    p=provider(lambda _:pytest.fail('Must not send a request'))
    try:
        with pytest.raises(ProviderError,match='token limit'):
            await p.anthropic_json('Return JSON.',{},_SpokenResponse.model_json_schema(),max_tokens=max_tokens)
    finally: await p.close()


@pytest.mark.parametrize('system,data,schema',[('',{},{}),('Return JSON.',[],{}),
    ('Return JSON.',{'bad':float('nan')},{}),('x'*20001,{},{}),
    ('Return JSON.',{'text':'x'*128001},{}),('Return JSON.',{},[])])
async def test_anthropic_invalid_input_fails_before_request(system,data,schema):
    p=provider(lambda _:pytest.fail('Must not send a request'))
    try:
        with pytest.raises(ProviderError,match='request data'):
            await p.anthropic_json(system,data,schema)
    finally: await p.close()


async def test_health_reports_actual_product_reasoner_and_independent_search(monkeypatch):
    monkeypatch.setenv('CONVERSATION_PRODUCT_MODE','product_search')
    monkeypatch.setenv('CONVERSATION_REASONING_PROVIDER','deepseek')
    settings=Settings(text_provider='anthropic',anthropic_key='private-anthropic-key',
        deepseek_key='private-search-key',catalog_search_provider='deepseek')
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(settings)),base_url='http://test') as client:
        response=await client.get('/api/health'); result=response.json()
    assert result['text_provider']=='anthropic' and result['catalog_search_provider']=='deepseek'
    assert result['providers']=={'perceptron':False,'gemini':False,'deepseek':True,'anthropic':True}
    assert result['conversation']['reasoning_provider']=='anthropic'
    assert result['conversation']['configured_not_provider_tested'] is True
    assert result['conversation']['providers_configured']['anthropic'] is True
    assert result['conversation']['providers_configured']['deepseek'] is False
    assert 'private-' not in response.text
