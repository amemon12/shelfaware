"""Pre-v6 prefix excerpt contract and literal serving declarations."""
from decimal import Decimal

import httpx
import pytest

from app.product_guidance import ProductGuide, _criterion_bases
from test_deepseek_products import identity, provider, entry


@pytest.mark.parametrize('text', [
    'Serving Size\n\nAbout 14 Chips (28g)\n\nAmount Per Serving\nTotal Fat 6g',
    'Serving sizeAbout 14 chips (28g)Amount per servingCalories140% Daily valueTotal Fat6g',
    'Nutrition Facts2.5 Servings Per ContainerServing sizeAbout 14 chips (28g)Amount per servingCalories140Total Fat6g',
    'Serving Size\n1 ONZ (28g)\nCalories160',
])
def test_literal_declared_mass_survives_page_formatting(text):
    assert _criterion_bases(text, 'fat') == {(Decimal(28), 'g')}


@pytest.mark.parametrize('text', [
    'Serving size28.00\nTotal Fat10g',
    'Serving Size\n\nTotalFat28g',
    'Serving Size\n\nFat 28g',
    'Serving Size\nnot supplied\nProtein 28g',
    'Serving size8.00\nTotal sugar14g',
    'Serving size Per 26 chi\nProtein4g',
])
def test_absent_serving_unit_cannot_borrow_a_nutrient_amount(text):
    assert _criterion_bases(text, 'fat') == set()


def long_source():
    return ('Navigation and product marketing\n' * 70 +
        '## Ingredients\nCorn, oil, salt.\n## Nutrition Facts\nServing Size\nAbout 14 chips (28g)\n'
        'Total Fat6g\nProtein2g\n' + 'More page information\n' * 70)


async def test_search_restores_original_prefix1600_and_preserves_source_label():
    text = long_source()
    client = provider(lambda _: httpx.Response(200, json={'results': [entry(content=text)]}))
    try:
        result = await ProductGuide(client).search(identity())
        assert result.status == 'grounded'
        assert result.notes == ('Possible catalog result: Demo Original cereal\n'
            'Source: https://manufacturer.example/cereal\n' + text[:1600])
        assert 'Serving Size' not in result.notes  # No newly relocated excerpt.
        assert len(result.notes) <= 6000 and len(result.sources) == 1
    finally:
        await client.close()


async def test_cited_mixed_prefix_is_retained_as_evidence_not_discarded():
    text = 'Total sugar0g\nTotal sugar10g\n' + long_source()
    client = provider(lambda _: httpx.Response(200, json={'results': [entry(content=text)]}))
    try:
        result = await ProductGuide(client).search(identity())
        assert result.status == 'grounded' and len(result.sources) == 1
        assert result.notes.endswith(text[:1600])
        assert 'Total sugar0g\nTotal sugar10g' in result.notes
    finally:
        await client.close()


async def test_search_does_not_rewrite_or_relocate_related_product_text():
    text = 'Product information.\n## Related products\nTotal sugar0g\n'
    client = provider(lambda _: httpx.Response(200, json={'results': [entry(content=text)]}))
    try:
        result = await ProductGuide(client).search(identity())
        assert result.status == 'grounded' and result.notes.endswith(text.strip())
    finally:
        await client.close()


@pytest.mark.parametrize('value', ['Dietary Fiber28g', 'Fiber 28g', 'Cholesterol28g', 'Caffeine28g', 'Vitamin D28g', 'Net wt28g', 'not specified Net wt28g', 'unknown28g'])
@pytest.mark.parametrize('gap', ['\n\n', ''])
def test_missing_serving_cannot_borrow_other_nutrients_or_net_weight(value, gap):
    assert _criterion_bases('Serving Size' + gap + value + '\nTotal Fat6g', 'fat') == set()
