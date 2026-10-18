import json

import httpx
import pytest

from app.models import CreateSession, Nutrition, TurnRequest
from app.providers import OCR_OUTPUT_SCHEMA, OCRResult, Providers, Settings
from app.service import ShoppingService
from test_backend import FakeProviders, IMAGE
from test_providers import ocr_body


def extracted_label(text, mass=40, carbohydrate=24, uncertain=()):
    return OCRResult(is_nutrition_label=True, name='Test label', label_regime='US',
        preparation='as_sold', serving_basis_text=text, uncertain_fields=list(uncertain),
        nutrition=Nutrition(basis_grams=mass, carbohydrate_g=carbohydrate))


@pytest.mark.parametrize('text,mass', [
    ('Serving size 1/2 cup (40 g)', 40),
    ('SERVING SIZE: 1 package (40.0 grams)', 40),
    ('Serving\nsize: 1 cup (40\u00a0g)', 40),
    ('Serving size 1 package (1,000 g)', 1000),
    ('Serving size 1 protein bar (40g)', 40),
    ('Per 100g', 100),
    ('per 100.0 grams as sold', 100),
])
def test_printed_serving_mass_matches_without_guessing(text, mass):
    assert extracted_label(text, mass).capture_issue() is None


@pytest.mark.parametrize('text,mass,issue', [
    (None, 296, 'basis'),
    ('Serving size 1 bottle (296 mL)', 296, 'volume_only'),
    ('Serving size 1 bottle (296 millilitres)', 296, 'volume_only'),
    ('Serving size 1 cup (8 fl. oz.)', 240, 'volume_only'),
    ('Serving size 1 tablet (296 mg)', 296, 'basis'),
    ('Net weight 296 g', 296, 'basis'),
    ('Per package 296 g', 296, 'basis'),
    ('Serving size 1 bottle (296 mL); net weight 296 g', 296, 'basis'),
    ('Serving size 1 bottle (296 mL); Total carbohydrate 24 g', 24, 'basis'),
    ('Serving size 1/2 cup (40g)', 296, 'basis'),
    ('Serving size 1/2 cup (40g), per 100g', 40, 'basis'),
    ('Serving size about 40 g', 40, 'basis'),
    ('Serving size <40 g', 40, 'basis'),
    ('Serving size -40 g', 40, 'basis'),
    ('Serving size −40 g', 40, 'basis'),
    ('Serving size 40,5 g', 40.5, 'basis'),
])
def test_volume_net_contents_and_ambiguous_masses_do_not_become_grams(text, mass, issue):
    assert extracted_label(text, mass, min(24, mass / 2)).capture_issue() == issue


@pytest.mark.parametrize('uncertain,issue', [
    (['basis_grams'], 'basis'),
    (['nutrition.basis_grams'], 'basis'),
    (['serving_basis_text'], 'basis'),
    (['carbohydrate_g'], 'carbohydrate'),
    (['Total carbohydrate'], 'carbohydrate'),
    (['nutrition.carbohydrate_g'], 'carbohydrate'),
])
def test_uncertain_core_fields_require_a_retake(uncertain, issue):
    assert extracted_label('Serving size 40 g', uncertain=uncertain).capture_issue() == issue


def test_missing_carbohydrate_requires_a_retake_but_absent_optional_fields_do_not():
    assert extracted_label('Serving size 40 g', carbohydrate=None).capture_issue() == 'carbohydrate'
    assert extracted_label('Serving size 40 g', uncertain=['sugars_g', 'ingredients']).capture_issue() is None


@pytest.mark.parametrize('text,mass,carbohydrate,uncertain,reply_part', [
    ('Serving size 1 bottle (296 mL)', 296, 24, [], 'volume serving'),
    (None, 40, 24, [], 'fill most of the view'),
    ('Serving size 40 g', 40, None, [], 'total carbohydrate'),
    ('Serving size 40 g', 40, 24, ['basis_grams'], 'gram-based serving'),
    ('Serving size 40 g', 40, 24, ['carbohydrate_g'], 'total carbohydrate'),
])
async def test_unusable_new_capture_clears_current_product_and_never_enters_review(text, mass, carbohydrate, uncertain, reply_part):
    provider = FakeProviders()
    service = ShoppingService(provider)
    sid = service.create(CreateSession(mode='live_glasses')).id
    first = await service.turn(sid, TurnRequest(text='Read this label', client_turn_id='first', image_data_url=IMAGE))
    assert first.status == 'needs_review'
    old_id = first.session.current_product_id
    provider.result = extracted_label(text, mass, carbohydrate, uncertain)

    response = await service.turn(sid, TurnRequest(text='Read this label', client_turn_id='retake', image_data_url=IMAGE))

    assert response.status == 'needs_image' and reply_part in response.reply
    assert response.session.current_product_id is None
    assert [product.id for product in response.session.products] == [old_id]
    assert response.comparison is None and response.pending_action is None
    assert response.evidence == []
    assert any(event.type == 'perception.retake' for event in response.events)
    assert service.get(sid).frame is not None  # The real capture remains inspectable.


async def test_provider_schema_requires_literal_basis_but_missing_output_fails_closed():
    assert 'serving_basis_text' in OCR_OUTPUT_SCHEMA['required']
    assert {'type': 'null'} in OCR_OUTPUT_SCHEMA['properties']['serving_basis_text']['anyOf']
    body = ocr_body()
    extracted = json.loads(body['choices'][0]['message']['content'])
    del extracted['serving_basis_text']
    body['choices'][0]['message']['content'] = json.dumps(extracted)
    providers = Providers(Settings(perceptron_key='mock-only'), httpx.MockTransport(
        lambda _: httpx.Response(200, json=body)))
    service = ShoppingService(providers)
    sid = service.create(CreateSession(mode='live_glasses')).id
    try:
        response = await service.turn(sid, TurnRequest(text='Read this label', client_turn_id='missing', image_data_url=IMAGE))
        assert response.status == 'needs_image'
        assert response.session.products == [] and response.comparison is None
    finally:
        await providers.close()
