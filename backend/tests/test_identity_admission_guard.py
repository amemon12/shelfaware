"""Explicit visual guesses cannot become confident catalog identities."""
import json

import httpx
import pytest

from app.models import Box
from app.product_guidance import DetectedProduct, ProductGuide
from app.providers import Providers, Settings
from test_product_guidance import image_url, identity_response, selection


@pytest.mark.parametrize('admission', [
    'product name not explicitly visible, inferred from design and color scheme',
    'The product name is inferred from the package color.',
    'Brand name was guessed from the design.',
    'Branding is not clearly readable.',
    'The name may have been assumed from a similar product.',
    'I cannot read the brand.',
    "I can't identify the product name.",
    'Name and brand are not visible.',
])
async def test_explicit_visual_guess_is_rejected_even_with_high_confidence(admission):
    calls = []
    responses = iter([selection(), identity_response(confidence=.99, uncertainties=[admission])])
    def handle(request):
        calls.append(request)
        return httpx.Response(200, json=next(responses))
    p = Providers(Settings(perceptron_key='test-only'), httpx.MockTransport(handle))
    try:
        assert await ProductGuide(p).identify(image_url()) is None
        assert len(calls) == 2
    finally:
        await p.close()


@pytest.mark.parametrize('uncertainty', [
    'Size inferred from standard packaging, not explicitly visible on front.',
    'Product name is clearly visible, but size is not readable.',
    'Brand is clear while the size is unknown.',
    'Brand name was not guessed; it is printed on the package.',
    'The product name is Uber Greens; ingredients are also printed below it.',
    'The package size is inferred from the brand name.',
])
async def test_size_or_noncontradictory_uncertainty_keeps_readable_identity(uncertainty):
    responses = iter([selection(), identity_response(uncertainties=[uncertainty])])
    p = Providers(Settings(perceptron_key='test-only'),
                  httpx.MockTransport(lambda _: httpx.Response(200, json=next(responses))))
    try:
        product = await ProductGuide(p).identify(image_url())
        assert product is not None and product.name == 'Demo cereal'
    finally:
        await p.close()


async def test_direct_search_cannot_bypass_an_admitted_guess():
    p = Providers(Settings(deepseek_key='test-only', text_provider='deepseek'),
                  httpx.MockTransport(lambda _: pytest.fail('A guessed identity must not search')))
    try:
        product = DetectedProduct(name='Demo Lemon', brand='Demo', variant='Lemon',
            package_size=None, category='chips', visible_text='Demo', confidence=.99,
            uncertainties=['Product name not explicitly visible; inferred from color.'],
            box=Box(x=.2, y=.2, w=.4, h=.5))
        result = await ProductGuide(p).search(product)
        assert result.status == 'no_match' and result.sources == []
    finally:
        await p.close()
