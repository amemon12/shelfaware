"""Source-scoped comparison eligibility; captured chip snippet shapes, no live calls."""
from decimal import Decimal
import json

import httpx
import pytest

from app.product_guidance import (ProductGuide, _nutrient_facts, _product_evidence,
    _nutrition_comparison_checks, _unresolved_nutrition_comparison)
from app.providers import Providers, Settings
from test_product_provenance import product


def item(name, *texts):
    value = product(name, product_id=name.lower())
    sources = [{'title': name, 'url': f'https://manufacturer.example/{name.lower().replace(chr(32), chr(45))}/{i}'}
               for i in range(len(texts))]
    value['catalog']['sources'] = sources
    value['catalog']['notes'] = '\n\n'.join(
        f'Possible catalog result: {name}\nSource: {source["url"]}\n{text}'
        for source, text in zip(sources, texts))
    return value


def checks(a, b, question='Which is lower in fat?', context=None):
    return _nutrition_comparison_checks([_product_evidence(a), _product_evidence(b)], question,
        context or {'consented': True, 'goals': ['Avoid sugar']})


@pytest.mark.parametrize('text', [
    'Total Fat10g13 %', 'Total Fat 10g', '- Fat\n- 10g', '### 10g\nTotal Fat',
    '| Total Fat | 10g | 13% |',
])
def test_compact_and_markdown_total_fat(text):
    assert ('fat', Decimal(10), 'g') in _nutrient_facts(text)


def test_trans_saturated_unsaturated_never_become_total_fat():
    facts = _nutrient_facts('Trans Fat0g\nSaturated Fat1.5g\nMonounsaturated Fat2g\nPolyunsaturated Fat3g')
    assert not any(key == 'fat' for key, *_ in facts)
    assert ('trans fat', Decimal(0), 'g') in facts


def test_actual_chip_unknown_denominators_do_not_choose_or_claim_amount_missing():
    a = item('Classic', 'Amount Per Serving\nSaturated Fat1.5g\nTrans Fat0g\nProtein2g',
             '- Fat\n- 10g\n- Calories\n- 160')
    b = item('Harvest', 'Amount Per Serving\nTotal Fat6g8 %\nProtein2g',
             'Serving Size Per 26 chi\nTotal Fat11g17 %\nProtein4g')
    for question in ['Which is lower in fat?', 'Which has more protein?']:
        result = checks(a, b, question)
        assert not result[0]['eligible']
        assert result[0]['issue'] == 'unknown or incompatible serving amounts'
        assert 'not comparable serving sizes' in _unresolved_nutrition_comparison(result)
    assert checks(a, b)[0]['items'][0]['sources'][0]['amount'] == '10'


@pytest.mark.parametrize('basis_a,basis_b', [
    ('per serving', 'per serving'), ('per bottle', 'per bottle'),
    ('per 28 g', 'per 100 mL'), ('', 'per 100 g'),
    ('Serving Size Per 26 chi', 'per 100 g'),
])
def test_generic_missing_truncated_or_different_dimension_bases_abstain(basis_a, basis_b):
    result = checks(item('Alpha', f'Fat 2g {basis_a}'), item('Beta', f'Fat 9g {basis_b}'))[0]
    assert not result['eligible']


def test_no_cross_source_denominator_borrowing_or_package_size_inference():
    a = item('Alpha', 'Bag size 100g. Fat 2g.', 'Serving size 100g. Protein 4g.')
    assert not checks(a, item('Beta', 'Fat 9g per 100g'))[0]['eligible']


def test_multiple_bases_in_source_are_not_paired_by_guessing():
    a = item('Alpha', 'Per 100g and per 28g. Fat 2g.')
    assert not checks(a, item('Beta', 'Fat 9g per 100g'))[0]['eligible']


def test_same_dimension_ratios_do_not_rank_raw_amounts():
    result = checks(item('Alpha', 'Fat 4g per 20g'), item('Beta', 'Fat 6g per 100g'))[0]
    assert result['eligible'] and result['lower_index'] == 1


def test_explicit_ounce_parenthetical_serving_supports_mass_without_converting_ounces():
    result = checks(item('Alpha', 'Serving Size 1 oz (28 g)\nFat 2g'),
                    item('Beta', 'Serving Size 28 grams\nFat 6g'))[0]
    assert result['eligible'] and result['lower_index'] == 0


def test_conflicting_sources_cannot_produce_a_winner():
    result = checks(item('Alpha', 'Fat 2g per 100g', 'Fat 11g per 100g'),
                    item('Beta', 'Fat 6g per 100g'))[0]
    assert not result['eligible'] and 'conflicting' in result['issue']


def test_zero_is_not_missing_and_does_not_require_a_denominator():
    zero = item('Alpha', 'Total sugars 0g')
    positive = item('Beta', 'Total sugars 10g')
    result = checks(zero, positive, 'Which is lower sugar?')[0]
    assert result['eligible'] and result['lower_index'] == 0
    absent = item('Unknown', 'No added sugar. Ingredients: juice.')
    assert not checks(absent, positive, 'Which is lower sugar?')[0]['eligible']


def test_cited_sugarfree_zero_is_usable_but_no_added_sugar_is_not_total_zero():
    zero = item('Alpha Sugarfree', 'Alpha Sugarfree contains no sugar.')
    assert checks(zero, item('Beta', 'Total sugars 10g'), 'Which is lower sugar?')[0]['eligible']
    no_added = item('Alpha', 'No added sugar.')
    assert not checks(no_added, item('Beta', 'Total sugars 10g'), 'Which is lower sugar?')[0]['eligible']


def test_actual_juice_source_shapes_support_same_direction_on_explicit_volumes():
    a = item('Green', 'Nutrition Facts\n1 bottle (354 mL) % Daily Value\nTotal Sugars 5g',
             'Per recommended serving 1 bottle (354ml)\nTotal Sugars8g')
    b = item('Coconut', 'Per recommended serving bottle (296mL)\nTotal Sugars17g')
    result = checks(a, b, 'Which is lower sugar?')[0]
    assert result['eligible'] and result['lower_index'] == 0


@pytest.mark.parametrize('question', [
    'How much fat is listed for each?', 'How much caffeine does each have?',
    'What are these?', 'Why does Alpha have sugar?', 'Which has better ingredients?', 'Which is cheaper?',
])
def test_nonranking_factual_questions_do_not_inherit_default_goal(question):
    assert checks(item('Alpha', 'Fat 2g'), item('Beta', 'Fat 6g'), question) == []


def test_why_choice_rechecks_current_criterion_instead_of_old_profile():
    result = checks(item('Alpha', 'Fat 2g'), item('Beta', 'Fat 6g'), 'Why did you pick that?',
        {'consented': True, 'goals': ['Avoid sugar'], 'conversation_request': 'Which has less fat?'})
    assert result[0]['criterion'] == 'fat' and not result[0]['eligible']


async def test_unknown_basis_reaches_model_as_ineligible_without_invented_servings():
    seen = []
    expected = 'The Alpha and Beta listings give fat amounts without comparable serving sizes, so they do not establish a lower-fat choice.'
    def response(request):
        body = json.loads(request.content)
        seen.append(json.loads(body['messages'][-1]['content']))
        assert 'eligible=false check does not support a ranking' in body['messages'][0]['content']
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps({'spoken_text': expected})}}]})
    providers = Providers(Settings(text_provider='deepseek', deepseek_key='test'), httpx.MockTransport(response))
    try:
        reply = await ProductGuide(providers).respond(item('Alpha', 'Fat 2g'),
            'Which has less fat?', {}, [item('Beta', 'Fat 6g')])
        assert reply == expected and len(seen) == 1
        check = seen[0]['checked_nutrition_comparisons'][0]
        assert not check['eligible'] and check['lower_index'] is None
        assert check['issue'] == 'unknown or incompatible serving amounts'
        for checked, name in zip(check['items'], ('Alpha', 'Beta')):
            assert checked['name'] == name
            assert checked['sources'][0]['basis_amount'] is None
            assert checked['sources'][0]['basis_unit'] is None
        assert seen[0]['product']['identity']['package_size'] is None
        assert seen[0]['previous_products'][0]['identity']['package_size'] is None
    finally:
        await providers.close()


async def test_eligible_comparison_passes_checked_direction_to_wording_model():
    seen = []
    expected = 'For equal amounts, the product listings show Beta has less fat than Alpha.'
    def response(request):
        body = json.loads(request.content)
        seen.append(json.loads(body['messages'][-1]['content']))
        assert 'preserve its product attribution and direction' in body['messages'][0]['content']
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps({'spoken_text': expected})}}]})
    providers = Providers(Settings(text_provider='deepseek', deepseek_key='test'), httpx.MockTransport(response))
    try:
        reply = await ProductGuide(providers).respond(item('Alpha', 'Fat 4g per 20g'),
            'Which has less fat?', {}, [item('Beta', 'Fat 6g per 100g')])
        assert reply == expected and len(seen) == 1
        check = seen[0]['checked_nutrition_comparisons'][0]
        assert check['eligible'] and check['lower_index'] == 1 and not check['equal']
        assert [value['name'] for value in check['items']] == ['Alpha', 'Beta']
        assert [value['sources'][0]['basis_amount'] for value in check['items']] == ['20', '100']
    finally:
        await providers.close()


@pytest.mark.parametrize('question', ['Is this a good option for me?', 'Are these tied for fat?', 'Do these have the same protein?', 'Which has the lowest fat?'])
def test_generic_and_tie_rankings_are_guarded(question):
    result = checks(item('Alpha', 'Fat 2g. Protein 2g. Total sugar 2g.'),
                    item('Beta', 'Fat 6g. Protein 4g. Total sugar 6g.'), question)
    assert result and not result[0]['eligible']


def test_another_nutrient_cannot_donate_denominator_within_same_source():
    result = checks(item('Alpha', 'Fat 10g per serving. Protein 2g per 100g.'),
                    item('Beta', 'Fat 6g per 100g.'))[0]
    assert not result['eligible']


def test_neighboring_nutrition_rows_never_cross_assign_amounts():
    facts = _nutrient_facts('Serving Size 28 grams\nFat 6g\nAdded sugar 0g')
    assert facts == {('fat', Decimal(6), 'g'), ('added sugar', Decimal(0), 'g')}


@pytest.mark.parametrize('question', ['Which whole bag has less fat?', 'Which has lower fat per serving?', 'Which entire bottle has less sugar?'])
def test_equal_quantity_ratio_does_not_prove_whole_package_or_serving_totals(question):
    result = checks(item('Alpha', 'Fat 2g per 100g. Sugar 2g per 100g.'),
                    item('Beta', 'Fat 6g per 100g. Sugar 6g per 100g.'), question)
    assert result and not result[0]['eligible']
    assert result[0]['issue'] == 'whole-container or serving totals not established'


def test_alternating_retailer_summary_amounts_are_not_assigned_to_neighbor():
    assert _nutrient_facts('### 0g\n\nTotal Fat\n\n### 28g\n\nTotal Sugars\n\n### 80mg\n\nSodium') == {
        ('fat', Decimal(0), 'g'), ('sugar', Decimal(28), 'g'), ('sodium', Decimal(80), 'mg')}
    assert _nutrient_facts('- Sodium\n- 170mg\n- Sugar\n- 1g\n- Fat\n- 10g') == {
        ('sodium', Decimal(170), 'mg'), ('sugar', Decimal(1), 'g'), ('fat', Decimal(10), 'g')}


@pytest.mark.parametrize('question', ['Which ingredients contribute fat?', 'Which nutrients are listed?'])
def test_which_factual_question_is_not_a_ranking(question):
    assert checks(item('Alpha', 'Fat 2g'), item('Beta', 'Fat 6g'), question) == []


@pytest.mark.parametrize('question', ['Which is lower in fat, and which ingredients are listed?', 'Which ingredients are listed, and which product has less fat?'])
def test_mixed_factual_and_ranking_question_cannot_skip_gate(question):
    result = checks(item('Alpha', 'Fat 2g'), item('Beta', 'Fat 6g'), question)
    assert result and not result[0]['eligible']
