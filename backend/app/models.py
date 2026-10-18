from __future__ import annotations

import base64
import binascii
import re
from datetime import date, datetime, timedelta, timezone
from typing import Literal
from uuid import uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

Mode = Literal['fixture', 'live_phone', 'live_glasses', 'recorded_glasses']
Recipient = Literal['self', 'household']
Kind = Literal['synthetic', 'package_ocr', 'reviewed_label', 'database']


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def uid() -> str:
    return str(uuid4())


def valid_gtin(value: str) -> bool:
    """GTIN-8/12/13/14 checksum, retaining leading zeroes and exact scanned digits."""
    if len(value) not in (8, 12, 13, 14) or not value.isascii() or not value.isdigit():
        return False
    total = sum(int(digit) * (3 if index % 2 == 0 else 1)
                for index, digit in enumerate(reversed(value[:-1])))
    return (10 - total % 10) % 10 == int(value[-1])


def canonical_gtin(value: str | None) -> str | None:
    """Compare valid GTIN identities without losing stored raw scanner digits.

    Only leading-zero padding is equivalent. A nonzero GTIN-14 packaging indicator
    remains distinct. Compressed UPC-E must be expanded by the scanner first.
    """
    return value.zfill(14) if value is not None and valid_gtin(value) else None


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False, validate_assignment=True)


class Nutrition(StrictModel):
    basis_grams: float = Field(gt=0, le=10000)
    carbohydrate_g: float | None = Field(default=None, ge=0, le=10000)
    fiber_g: float | None = Field(default=None, ge=0, le=10000)
    sugars_g: float | None = Field(default=None, ge=0, le=10000)
    protein_g: float | None = Field(default=None, ge=0, le=10000)
    sodium_mg: float | None = Field(default=None, ge=0, le=10000000)

    @model_validator(mode='after')
    def physically_plausible(self) -> Nutrition:
        for field in ('carbohydrate_g', 'fiber_g', 'sugars_g', 'protein_g'):
            value = getattr(self, field)
            if value is not None and value > self.basis_grams + 1:
                raise ValueError(f'{field} exceeds the declared food mass; review the label basis')
        if self.sodium_mg is not None and self.sodium_mg > (self.basis_grams + 1) * 1000:
            raise ValueError('sodium exceeds food mass; review its unit')
        if self.carbohydrate_g is not None:
            for field in ('fiber_g', 'sugars_g'):
                value = getattr(self, field)
                if value is not None and value > self.carbohydrate_g + 1:
                    raise ValueError(f'{field} exceeds US total carbohydrate; review the label')
        return self


class Provenance(StrictModel):
    kind: Kind
    source: str = Field(min_length=1, max_length=500)
    captured_at: str
    verified: bool


class Product(StrictModel):
    id: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=180)
    brand: str | None = Field(default=None, max_length=100)
    barcode: str | None = Field(default=None, max_length=30)
    market: str = Field(min_length=1, max_length=50)
    ingredients: list[str] = Field(default_factory=list, max_length=80)
    nutrition: Nutrition
    provenance: Provenance
    uncertain_fields: list[str] = Field(default_factory=list, max_length=30)

    @field_validator('ingredients', 'uncertain_fields')
    @classmethod
    def short_strings(cls, values: list[str]) -> list[str]:
        if any(not v.strip() or len(v) > 300 for v in values):
            raise ValueError('Each entry must contain 1–300 characters')
        return values


class GoalSource(StrictModel):
    kind: Literal['self_selected', 'reviewed_guidance', 'synthetic']
    source_date: date | None = None
    confirmed: Literal[True]
    reviewed_at: AwareDatetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator('confirmed', mode='before')
    @classmethod
    def explicit_confirmation(cls, value):
        if value is not True:
            raise ValueError('Guidance provenance must be explicitly confirmed')
        return value

    @field_validator('source_date', mode='before')
    @classmethod
    def iso_date(cls, value):
        if value is None or (isinstance(value, date) and not isinstance(value, datetime)):
            return value
        if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
            raise ValueError('Use a YYYY-MM-DD source date')
        return value

    @field_validator('source_date')
    @classmethod
    def not_future_date(cls, value: date | None) -> date | None:
        if value is not None and value > datetime.now(timezone.utc).date():
            raise ValueError('The guidance source date cannot be in the future')
        return value

    @field_validator('reviewed_at')
    @classmethod
    def not_future_review(cls, value: datetime) -> datetime:
        if value > datetime.now(timezone.utc) + timedelta(minutes=5):
            raise ValueError('The review timestamp cannot be in the future')
        return value


class NutritionContext(StrictModel):
    """Explicitly selected fictional demo context; no raw medical record or identifiers."""
    synthetic: Literal[True]
    conditions: list[str] = Field(default_factory=list, max_length=10)
    allergies: list[str] = Field(default_factory=list, max_length=20)
    focus: list[str] = Field(default_factory=list, max_length=10)
    dietary_preferences: list[str] = Field(default_factory=list, max_length=20)

    @field_validator('conditions', 'allergies', 'focus', 'dietary_preferences')
    @classmethod
    def bounded_context(cls, values: list[str]) -> list[str]:
        return Product.short_strings(values)


class Profile(StrictModel):
    display_name: str = Field(default='Shopper', min_length=1, max_length=80)
    goals: list[str] = Field(default_factory=list, max_length=10)
    avoid_ingredients: list[str] = Field(default_factory=list, max_length=20)
    consented: bool = False
    goal_source: GoalSource | None = None
    nutrition_context: NutritionContext | None = None

    @field_validator('goals', 'avoid_ingredients')
    @classmethod
    def bounded_entries(cls, values: list[str]) -> list[str]:
        if any(not v.strip() or len(v) > 150 for v in values):
            raise ValueError('Profile entries must contain 1–150 characters')
        return values


class CartItem(StrictModel):
    id: str = Field(default_factory=uid)
    product_id: str
    product_name: str
    recipient_id: Recipient
    quantity: int = Field(default=1, ge=1, le=100)
    confirmed_at: str = Field(default_factory=now)


class Action(StrictModel):
    id: str = Field(default_factory=uid)
    label: str
    product_id: str
    recipient_id: Recipient
    quantity: int = Field(default=1, ge=1, le=100)
    expires_at: str


class Message(StrictModel):
    id: str = Field(default_factory=uid)
    role: Literal['user', 'assistant']
    text: str
    created_at: str = Field(default_factory=now)


class Session(StrictModel):
    id: str = Field(default_factory=uid)
    mode: Mode
    recipient_id: Recipient = 'self'
    portion_grams: float | None = None
    generation: int = 0
    profile: Profile
    products: list[Product] = Field(default_factory=list)
    current_product_id: str | None = None
    visual_products: list[VisualProduct] = Field(default_factory=list, max_length=12)
    current_visual_product_id: str | None = None
    # The one or two candidates selected together in the latest photo.
    current_visual_product_ids: list[str] = Field(default_factory=list, max_length=2)
    cart: list[CartItem] = Field(default_factory=list)
    pending_action: Action | None = None
    messages: list[Message] = Field(default_factory=list)
    created_at: str = Field(default_factory=now)


class Evidence(StrictModel):
    label: str
    value: str
    source: str
    kind: Kind | Literal['calculation', 'visual_identity', 'catalog']
    verified: bool


class ComparisonItem(StrictModel):
    product_id: str
    name: str
    carbohydrate_g: float | None
    fiber_g: float | None
    sugars_g: float | None
    protein_g: float | None
    sodium_mg: float | None


class Comparison(StrictModel):
    portion_grams: float
    recipient_id: Recipient
    items: list[ComparisonItem]
    summary: str


class Box(StrictModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    w: float = Field(gt=0, le=1)
    h: float = Field(gt=0, le=1)

    @model_validator(mode='after')
    def within_frame(self) -> Box:
        if self.x + self.w > 1.001 or self.y + self.h > 1.001:
            raise ValueError('Box extends outside the original frame')
        return self


class CatalogSource(StrictModel):
    title: str = Field(min_length=1, max_length=500)
    url: str = Field(min_length=1, max_length=3000)


class VisualProduct(StrictModel):
    """Visual/catalog candidate, kept separate from reviewed numeric label facts."""
    id: str = Field(default_factory=uid)
    name: str = Field(min_length=1, max_length=200)
    brand: str | None = Field(default=None, max_length=200)
    variant: str | None = Field(default=None, max_length=200)
    package_size: str | None = Field(default=None, max_length=100)
    category: str | None = Field(default=None, max_length=100)
    visible_text: str = Field(default='', max_length=2000)
    confidence: float = Field(ge=0, le=1)
    uncertainties: list[str] = Field(default_factory=list, max_length=20)
    box: Box
    captured_at: str
    catalog_status: Literal['grounded', 'no_match'] = 'no_match'
    catalog_notes: str = Field(default='', max_length=6000)
    sources: list[CatalogSource] = Field(default_factory=list, max_length=12)


class Event(StrictModel):
    id: str = Field(default_factory=uid)
    type: str
    at: str = Field(default_factory=now)
    stage: str
    message: str
    media_time_s: float | None = None
    box: Box | None = None


class TurnResponse(StrictModel):
    session: Session
    reply: str
    status: Literal['ready', 'needs_image', 'needs_portion', 'needs_review', 'needs_confirmation', 'unknown', 'cancelled']
    evidence: list[Evidence] = Field(default_factory=list)
    comparison: Comparison | None = None
    events: list[Event] = Field(default_factory=list)
    pending_action: Action | None = None


class CapturedFrame(StrictModel):
    image_data_url: str
    captured_at: str
    media_time_s: float | None = None


class CreateSession(StrictModel):
    mode: Mode
    profile: Profile | None = None


class TurnRequest(StrictModel):
    text: str = Field(min_length=1, max_length=1500)
    client_turn_id: str = Field(min_length=1, max_length=100, pattern=r'^[A-Za-z0-9_.:-]+$')
    recipient_id: Recipient | None = None
    portion_grams: float | None = Field(default=None, gt=0, le=5000)
    fixture_product_id: str | None = Field(default=None, max_length=100)
    barcode: str | None = Field(default=None, max_length=14)
    image_data_url: str | None = Field(default=None, max_length=5700000)
    captured_at: AwareDatetime | None = None
    media_time_s: float | None = Field(default=None, ge=0, le=86400)
    analysis_mode: Literal['label', 'product'] = 'label'
    # Ephemeral explicit conversation criterion; never rewrites the saved profile.
    conversation_request: str | None = Field(default=None, min_length=1, max_length=500)

    @field_validator('barcode')
    @classmethod
    def checked_barcode(cls, value: str | None) -> str | None:
        if value is not None and not valid_gtin(value):
            raise ValueError('Barcode must be GTIN-8, UPC-A, EAN-13, or GTIN-14 with a valid check digit')
        return value

    @field_validator('image_data_url')
    @classmethod
    def validate_image(cls, value: str | None) -> str | None:
        if value is None:
            return None
        match = re.fullmatch(r'data:image/(jpeg|png);base64,([A-Za-z0-9+/=]+)', value)
        if not match:
            raise ValueError('Use a base64 JPEG or PNG data URL; remote URLs are not accepted')
        try:
            decoded = base64.b64decode(match[2], validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError('Invalid base64 image') from exc
        if not 16 <= len(decoded) <= 4 * 1024 * 1024:
            raise ValueError('Image must be between 16 bytes and 4 MiB')
        magic = decoded.startswith(b'\xff\xd8\xff') if match[1] == 'jpeg' else decoded.startswith(b'\x89PNG\r\n\x1a\n')
        if not magic:
            raise ValueError('Image signature does not match its MIME type')
        return value

    @model_validator(mode='after')
    def single_source(self) -> TurnRequest:
        if self.fixture_product_id and (self.image_data_url or self.barcode):
            raise ValueError('A turn may contain a fixture or a real image, not both')
        if self.analysis_mode == 'product' and (self.fixture_product_id or self.barcode):
            raise ValueError('Product search uses the held package image, not fixtures or barcodes')
        return self


class ContextRequest(StrictModel):
    recipient_id: Recipient
    portion_grams: float | None = Field(default=None, gt=0, le=5000)


class ConfirmRequest(StrictModel):
    action_id: str = Field(min_length=1, max_length=100)


class Corrections(StrictModel):
    name: str | None = Field(default=None, min_length=1, max_length=180)
    nutrition: Nutrition | None = None
    ingredients: list[str] | None = Field(default=None, max_length=80)

    @field_validator('ingredients')
    @classmethod
    def valid_ingredients(cls, values: list[str] | None) -> list[str] | None:
        return Product.short_strings(values) if values is not None else None


class ReviewRequest(StrictModel):
    product_id: str = Field(min_length=1, max_length=100)
    confirmed: bool
    corrections: Corrections | None = None


# Resolve the visual-product forward reference after all wire types are defined.
Session.model_rebuild()
