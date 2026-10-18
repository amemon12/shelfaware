"""Cross-layer regression tests: real brain, transport, VAD and local STT adapter.

Provider output and speech inference are fakes; package records are synthetic.
These verify application integration, not real microphone/model accuracy.
"""
import base64
import json
import struct

import httpx
import pytest

from app.conversation_brain import BrainSettings, GoalDraft, Intent, ReviewedCatalog, ShoppingBrain
from app.conversation_transport import ConversationConfig
from app.local_transcription import LocalTranscription
from app.speech_detection import SpeechDetector
from test_conversation_brain import camera, catalog_payload, png, response
from test_conversation_transport import cleanup, connected


def snack_catalog():
    payload = catalog_payload()
    for index, item in enumerate(payload['products']):
        item.update(name=f'Synthetic Test Chips {index + 1}', category='chips',
                    basis={'value': 30, 'unit': 'g'}, net_quantity={'value': 150, 'unit': 'g'})
        item['nutrients'].update(total_carbohydrate_g=20 if index == 0 else 15)
    return ReviewedCatalog.from_payload(payload)


class YesEngine:
    """Injected inference only; the real LocalTranscription handles lifecycle."""
    async def process(self, pcm):
        return None if pcm is None else 'yes'


async def finish_playback(browser, answer, epoch):
    # This fake native player acknowledges the completed response before the
    # next utterance. audio.done/server response completion alone is insufficient.
    await browser.send('playback.finished', context_id=answer['context_id'], client_epoch=epoch)
    await browser.until('input.enabled', lambda event: event['enabled'] and event['client_epoch'] == epoch)


async def ask(browser, text, epoch):
    await browser.send('input.text', text=text, client_epoch=epoch)
    answer = await browser.until('response.text')
    assert answer['client_epoch'] == epoch
    assert len(answer['text']) <= 600
    snapshot = (await browser.until('snapshot'))['snapshot']
    await finish_playback(browser, answer, epoch)
    return answer, snapshot


async def capture(browser, epoch):
    await browser.send('input.text', text='compare these', client_epoch=epoch)
    request = await browser.until('camera.request')
    await browser.send('camera.response', request_id=request['request_id'],
        image_data_url='data:image/png;base64,' + base64.b64encode(png()).decode(), width=100, height=80)
    answer = await browser.until('response.text')
    assert answer['client_epoch'] == epoch and len(answer['text']) <= 600
    snapshot = (await browser.until('snapshot'))['snapshot']
    await finish_playback(browser, answer, epoch)
    return snapshot


async def say_yes(browser, connection, epoch, pending_kind):
    # Native emits PCM, server VAD emits speech.started, local inference returns yes.
    voice = base64.b64encode(struct.pack('<480h', *([1000] * 480))).decode()
    silence = base64.b64encode(bytes(960)).decode()
    for _ in range(5):
        await browser.send('audio.append', audio=voice, client_epoch=epoch)
    started = await browser.until('speech.started')
    assert started['client_epoch'] == epoch
    # The next question preserves the proposed action after output completed.
    assert connection.runtime.brain.snapshot()['pending_action']['kind'] == pending_kind
    for _ in range(33):
        await browser.send('audio.append', audio=silence, client_epoch=epoch)
    transcript = await browser.until('transcript.final')
    assert transcript['text'] == 'yes'
    answer = await browser.until('response.text')
    assert answer['client_epoch'] == epoch and len(answer['text']) <= 600
    snapshot = (await browser.until('snapshot'))['snapshot']
    await finish_playback(browser, answer, epoch)
    return snapshot


@pytest.mark.asyncio
async def test_real_brain_vad_confirms_identity_goals_cart_then_household_and_unknown_clear():
    calls = []
    def provider(request):
        payload = json.loads(request.content)
        calls.append(payload)
        if 'Task: package_candidates' in payload['messages'][0]['content']:
            count = sum('Task: package_candidates' in p['messages'][0]['content'] for p in calls)
            return response({'candidate_ids': ['a', 'b'] if count == 1 else []})
        return response(Intent(action='set_goals', goals=[GoalDraft(
            nutrient='total_carbohydrate_g', comparator='min')]).model_dump())
    client = httpx.AsyncClient(transport=httpx.MockTransport(provider))
    def factory(profile, recipient):
        return ShoppingBrain({**profile, 'goals': []}, recipient,
            settings=BrainSettings(deepseek_key='test-only'), client=client, catalog=snack_catalog())
    owner, session, browser, connection, task, ready = await connected(
        brain_factory=factory, config=ConversationConfig(transcription_provider='local', speech_provider='native'),
        transcriber_factory=lambda callback, error: LocalTranscription(callback, error, engine=YesEngine()),
        vad_factory=lambda: SpeechDetector(classify=lambda pcm: pcm != bytes(640)))
    try:
        assert ready['audio_input_available'] and ready['speech_provider'] == 'native'
        await browser.until('snapshot')
        snapshot = await capture(browser, 3)
        assert snapshot['pending_action']['kind'] == 'identity' and snapshot['pair'] == []
        snapshot = await say_yes(browser, connection, 3, 'identity')
        assert snapshot['pair'] == ['a', 'b'] and snapshot['comparison'] is None
        _, snapshot = await ask(browser, '30 grams each', 4)
        assert snapshot['portions']['a']['quantity'] == {'value': '30.0', 'unit': 'g'}
        assert snapshot['goals'] is None and snapshot['comparison'] is None
        _, snapshot = await ask(browser, 'prefer lower carbs', 5)
        assert snapshot['pending_action']['kind'] == 'goals'
        assert snapshot['goals'] is None  # NLP proposes; the wearer confirms.
        snapshot = await say_yes(browser, connection, 5, 'goals')
        assert snapshot['comparison']['candidate_product_id'] == 'b'
        assert snapshot['comparison']['facts'][0]['nutrients']['total_carbohydrate_g'] == '20'
        assert snapshot['comparison']['facts'][1]['nutrients']['total_carbohydrate_g'] == '15'
        _, snapshot = await ask(browser, 'add the recommended one', 6)
        assert snapshot['pending_action']['kind'] == 'cart' and not snapshot['cart']
        snapshot = await say_yes(browser, connection, 6, 'cart')
        assert len(snapshot['cart']) == 1 and snapshot['cart'][0]['product_id'] == 'b'
        assert len(owner.service.get(session.id).session.cart) == 1
        _, snapshot = await ask(browser, 'yes', 7)
        assert len(snapshot['cart']) == 1  # One-use action remains one-use across speech/text.
        await browser.send('context.update', recipient_id='household', client_epoch=8)
        snapshot = (await browser.until('snapshot'))['snapshot']
        assert snapshot['recipient_id'] == 'household'
        assert snapshot['goals'] is None and snapshot['portions'] == {} and snapshot['comparison'] is None
        assert snapshot['cart'][0]['recipient_id'] == 'self'
        snapshot = await capture(browser, 9)
        assert snapshot['pair'] == [] and snapshot['pending_action'] is None
        assert snapshot['comparison'] is None
        _, snapshot = await ask(browser, 'add it', 10)
        assert len(snapshot['cart']) == 1 and snapshot['pending_action'] is None
        assert len(calls) == 3  # Two explicit captures; one free-form goal. No model for yes.
    finally:
        await cleanup(owner, task)
        await client.aclose()


@pytest.mark.asyncio
async def test_long_names_per_one_ml_math_and_large_summary_preserve_complete_snapshot():
    payload = catalog_payload()
    for index, entry in enumerate(payload['products']):
        entry['name'] = ('First' if index == 0 else 'Second') + 'x' * (180 - (5 if index == 0 else 6))
        entry['basis'] = {'value': 1, 'unit': 'mL'}
        entry['nutrients'] = {'total_carbohydrate_g': 0.027 if index == 0 else 0.020}
        entry['net_quantity'] = {'value': 330, 'unit': 'mL'}
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response({'candidate_ids': ['a', 'b']})))
    brain = ShoppingBrain({'consented': True, 'goals': [], 'avoid_ingredients': []},
        settings=BrainSettings(deepseek_key='test-only'), client=client,
        catalog=ReviewedCatalog.from_payload(payload))
    try:
        results = [await brain.respond('compare these', camera), await brain.respond('yes'),
                   await brain.respond('both whole bottles'), await brain.respond('less carbohydrate'),
                   await brain.respond('yes')]
        result = results[-1]
        assert all(len(item['reply']) <= 600 for item in results)
        assert '8.91' in result['reply'] and '6.6' in result['reply']
        assert '330 mL' in result['reply'] and 'second product' in result['reply']
        facts = result['snapshot']['comparison']['facts']
        assert facts[0]['nutrients']['total_carbohydrate_g'] == '8.91'
        assert facts[1]['nutrients']['total_carbohydrate_g'] == '6.6'
        assert result['snapshot']['products'][0]['name'] == payload['products'][0]['name']
        brain.cart = [{'id': str(i), 'product_id': 'a', 'name': payload['products'][0]['name'],
                       'recipient_id': 'self', 'quantity': 1, 'confirmed_at': '2026-01-01T00:00:00Z'}
                      for i in range(100)]
        summary = await brain.respond('shopping list')
        assert len(summary['reply']) <= 600 and '98 more confirmed items' in summary['reply']
        assert len(summary['snapshot']['cart']) == 100
        assert summary['snapshot']['cart'][99]['name'] == payload['products'][0]['name']
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('message', [[], 'invalid', 42, None])
async def test_malformed_provider_envelope_clarifies_instead_of_crashing(message):
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200,
        json={'choices': [{'finish_reason': 'stop', 'message': message}]})))
    brain = ShoppingBrain({'consented': True}, settings=BrainSettings(deepseek_key='test-only'),
                         client=client, catalog=snack_catalog())
    try:
        brain._pending('cart', product_id='a')
        result = await brain.respond('Please compare the salt tradeoff for me')
        assert result['status'] == 'needs_clarification'
        assert result['snapshot']['comparison'] is None and result['snapshot']['pending_action'] is None
        confirmation = await brain.respond('yes')
        assert confirmation['snapshot']['cart'] == []
    finally:
        await client.aclose()
