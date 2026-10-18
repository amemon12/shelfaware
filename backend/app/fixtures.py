"""Fictional labels for a transparently synthetic, credential-free demonstration."""
from .models import Nutrition, Product, Provenance

FIXTURE_DATE = '2026-10-18T00:00:00+00:00'


def fixtures() -> list[Product]:
    source = Provenance(kind='synthetic', source='Fictional demonstration label; not a real product or SKU',
                        captured_at=FIXTURE_DATE, verified=True)
    return [
        Product(id='oat-a', name='Morning Field Original Oats', brand='Fictional Demo Brand', market='US',
                ingredients=['whole grain oats'],
                nutrition=Nutrition(basis_grams=40, carbohydrate_g=28, fiber_g=4, sugars_g=1, protein_g=5, sodium_mg=0),
                provenance=source.model_copy(deep=True)),
        Product(id='oat-b', name='Morning Field Fiber Blend', brand='Fictional Demo Brand', market='US',
                ingredients=['whole grain oats', 'oat bran', 'almonds'],
                nutrition=Nutrition(basis_grams=50, carbohydrate_g=30, fiber_g=7, sugars_g=2, protein_g=8, sodium_mg=40),
                provenance=source.model_copy(deep=True)),
        Product(id='unknown', name='Unidentified Demo Package', market='US', ingredients=[],
                nutrition=Nutrition(basis_grams=1),
                provenance=Provenance(kind='synthetic', source='Fictional unknown-product fixture; basis is a placeholder, never calculated',
                                      captured_at=FIXTURE_DATE, verified=False),
                uncertain_fields=['identity', 'basis_grams', 'carbohydrate_g', 'ingredients']),
    ]
