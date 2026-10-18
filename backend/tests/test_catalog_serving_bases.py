"""Regression shapes from manufacturer/retailer nutrition text, with no live calls."""
import pytest

from app.product_guidance import _nutrient_facts, _serving_bases, _validate_spoken
from app.providers import ProviderError


SUJA_MANUFACTURER = '''Possible catalog result: Green juice
Source: https://manufacturer.example/green-juice
## Nutritional Facts
- 1 bottle (354 mL) % Daily Value*
- Carbohydrates 9g
- Total Sugars 5g
- Added Sugar 0g
'''
SUJA_RETAILER = '''Possible catalog result: Green juice
Source: https://retailer.example/green-juice
### Nutrition Facts
1Per recommended serving 1 bottle (354ml)
serving size
354 millilitre
Amount Per Serving
Total Carbohydrate11g
Total Sugars8g
Added Sugars0g
'''
COCONUT_RETAILER = '''Possible catalog result: Coconut water
Source: https://retailer.example/coconut-water
### Nutrition Facts
Per recommended serving bottle (296mL)
serving size
296 millilitre
Amount Per Serving
Total Carbohydrate19g
Total Sugars17g
Added Sugars0g
'''


def evidence(*notes):
    return [{'catalog': {'status': 'grounded', 'notes': value}} for value in notes]


@pytest.mark.parametrize('notes,volume', [
    (SUJA_MANUFACTURER, '354 ml'),
    (SUJA_RETAILER, '354 ml'),
    (COCONUT_RETAILER, '296 ml'),
])
def test_actual_explicit_bottle_headers_preserve_volume(notes, volume):
    assert {'bottle', volume, f'bottle ({volume})'} <= _serving_bases(notes)
    assert '100 ml' not in _serving_bases(notes)


@pytest.mark.parametrize('basis', [
    'bottle', '1 bottle', 'one bottle', 'listed bottle', '354 mL bottle',
    '354 millilitres bottle', 'bottle (354ml)', '354.00 milliliters bottle',
])
def test_exact_manufacturer_sugar_amount_accepts_equivalent_basis(basis):
    reply = f'The manufacturer lists 5 g total sugars per {basis}.'
    assert _validate_spoken(reply, evidence(SUJA_MANUFACTURER)) == reply


@pytest.mark.parametrize('basis', ['100 mL', '296 mL bottle', '2 bottles', '0.5 bottle', '100 g', 'cup'])
def test_bottle_amount_cannot_be_relabelled_to_a_different_quantity(basis):
    with pytest.raises(ProviderError, match='serving basis'):
        _validate_spoken(f'The manufacturer lists 5 g sugars per {basis}.', evidence(SUJA_MANUFACTURER))


def test_actual_comparison_plural_attribution_and_two_volume_bases():
    reply = ('Based on the product listings, green juice has 5 g total sugars per 354 mL bottle, '
             'while coconut water has 17 g total sugars per 296 mL bottle.')
    products = evidence(SUJA_MANUFACTURER, COCONUT_RETAILER)
    products[0]['identity'] = {'name': 'Green juice'}
    products[1]['identity'] = {'name': 'Coconut water'}
    assert _validate_spoken(reply, products) == reply


@pytest.mark.parametrize('attribution', ['online', 'listing', 'listings', 'catalogs', 'manufacturers', 'retailers'])
def test_plural_attribution_does_not_reject_supported_amount(attribution):
    reply = f'The {attribution} information reports 17 g sugars per bottle.'
    assert _validate_spoken(reply, evidence(COCONUT_RETAILER)) == reply


def test_manufacturer_retailer_conflict_does_not_invent_a_midpoint_or_erase_qualitative_reply():
    notes = evidence(SUJA_MANUFACTURER + SUJA_RETAILER, COCONUT_RETAILER)
    reply = 'Choose green juice for less sugar: both cited listings put it below the coconut water, although their amounts differ.'
    assert _validate_spoken(reply, notes) == reply
    with pytest.raises(ProviderError, match='nutrition amounts'):
        _validate_spoken('The online listings say 6.5 g sugars per bottle.', notes)


def test_package_volume_without_explicit_nutrition_basis_is_insufficient():
    notes = 'Green juice 354 mL bottle. Total sugars 5g.'
    assert _serving_bases(notes) == set()
    with pytest.raises(ProviderError, match='serving basis'):
        _validate_spoken('The online listing gives 5 g sugars per bottle.', evidence(notes))


def test_generic_serving_does_not_imply_a_whole_bottle():
    notes = 'Nutrition facts: 5 g sugar per serving. Bottle size 354 mL.'
    with pytest.raises(ProviderError, match='serving basis'):
        _validate_spoken('The online listing gives 5 g sugar per bottle.', evidence(notes))


def test_two_bottle_basis_is_not_silently_normalized_to_one():
    notes = 'The manufacturer lists 5 g sugar per 2 bottles.'
    assert _serving_bases(notes) == {'2 bottle'}
    with pytest.raises(ProviderError, match='serving basis'):
        _validate_spoken('The online listing gives 5 g sugar per bottle.', evidence(notes))


def test_implicit_bottle_volume_is_not_invented():
    notes = 'The manufacturer lists 5 g sugar per bottle.'
    with pytest.raises(ProviderError, match='serving basis'):
        _validate_spoken('The online listing gives 5 g sugar per 354 mL bottle.', evidence(notes))


@pytest.mark.parametrize('prefix', ['about', 'approximately', 'less than', 'more than', 'under', 'over', 'up to', '~', '≈'])
def test_qualified_unsupported_amount_cannot_disappear_from_validation(prefix):
    with pytest.raises(ProviderError, match='nutrition amounts'):
        _validate_spoken(f'The online listing gives {prefix} 99 g sugars per bottle.', evidence(SUJA_MANUFACTURER))


@pytest.mark.parametrize('prefix', ['about', 'approximately'])
def test_supported_approximate_wording_still_checks_amount_and_basis(prefix):
    reply = f'The online listing gives {prefix} 5 g sugars per bottle.'
    assert _validate_spoken(reply, evidence(SUJA_MANUFACTURER)) == reply
    with pytest.raises(ProviderError, match='serving basis'):
        _validate_spoken(f'The online listing gives {prefix} 5 g sugars per 100 mL.', evidence(SUJA_MANUFACTURER))


def test_exact_evidence_does_not_support_an_inequality():
    with pytest.raises(ProviderError, match='nutrition amounts'):
        _validate_spoken('The online listing gives less than 5 g sugars per bottle.', evidence(SUJA_MANUFACTURER))


def test_added_sugar_is_not_total_sugar_in_either_direction():
    added = 'Nutrition Facts\n1 bottle (296 mL)\nAdded Sugars 0g'
    assert ('added sugar', 0, 'g') in _nutrient_facts(added)
    assert ('sugar', 0, 'g') not in _nutrient_facts(added)
    with pytest.raises(ProviderError, match='nutrition amounts'):
        _validate_spoken('The online listing gives 0 g total sugars per bottle.', evidence(added))
    with pytest.raises(ProviderError, match='nutrition amounts'):
        _validate_spoken('The online listing gives 5 g added sugars per bottle.', evidence(SUJA_MANUFACTURER))
    reply = 'The online listing gives 0 g added sugars per bottle.'
    assert _validate_spoken(reply, evidence(added)) == reply
