from __future__ import annotations

import asyncio
import re
import time
import hashlib
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP

from fastapi import HTTPException

from .fixtures import fixtures
from .models import (Action, CapturedFrame, CartItem, Comparison, ComparisonItem, ContextRequest, CreateSession,
                     Event, Evidence, Message, Product, Profile, Provenance, ReviewRequest,
                     Session, TurnRequest, TurnResponse, VisualProduct, canonical_gtin, now, uid, valid_gtin)
from .providers import Intent, ProviderError, Providers
from .product_guidance import ProductGuide, capture_vision_timings
from .product_memory import compact_exchanges, product_scope, without_old_positions
from .product_questions import grounded_named_target, spoken_product_name, visual_identification_question

FIELDS = ('carbohydrate_g', 'fiber_g', 'sugars_g', 'protein_g', 'sodium_mg')
GRAMS = re.compile(r'(?<![\w.])([+-]?(?:\d[\d,]*(?:\.\d*)?|\.\d+))\s*(?:g|grams?)\b', re.I)
GRAM_MENTION = re.compile(r'\b(?:g|grams?)\b|(?<=\d)g\b', re.I)
NUTRIENT_WORDS = r'(?:carbohydrates?|carbs?|fib(?:er|re)|sugars?|proteins?|sodium|salt|fats?)'
INTENT_TIMEOUT_SECONDS = 2.0
LOCAL_QUESTION_PHRASES = frozenset({'is this a good option', 'read this label', 'read the label'})


def calculate(product: Product, grams: float) -> ComparisonItem:
    """Mass-only scaling. Never assumes a serving, density, or zero for absent data."""
    if not 0 < grams <= 5000:
        raise ValueError('Portion must be between 0 and 5000 grams')
    if not product.provenance.verified or 'basis_grams' in product.uncertain_fields:
        raise ValueError('Label and mass basis must be reviewed before calculation')
    factor = Decimal(str(grams)) / Decimal(str(product.nutrition.basis_grams))
    values = {}
    for name in FIELDS:
        source = getattr(product.nutrition, name)
        values[name] = None if source is None else float((Decimal(str(source)) * factor).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
    return ComparisonItem(product_id=product.id, name=product.name, **values)


def number(value: float | None) -> str:
    return 'unknown' if value is None else f'{value:g}'


def spoken_portion(text: str) -> tuple[float | None, bool]:
    """Return a single explicit food mass, or flag gram statements needing clarification."""
    matches = list(GRAMS.finditer(text))
    if not matches:
        return None, bool(GRAM_MENTION.search(text))
    if len(matches) != 1:
        return None, True
    match = matches[0]
    before, after = text[:match.start()], text[match.end():]
    nutrient_amount = (re.match(rf'\s*(?:of\s+)?(?:(?:total|net|added|dietary|saturated|unsaturated|trans)\s+)*{NUTRIENT_WORDS}\b', after, re.I)
                       or re.search(rf'\b{NUTRIENT_WORDS}(?:\s+(?:is|are|of))?\s*[:=]?\s*$', before, re.I))
    label_amount = re.search(r'\b(?:label says|label lists|per serving|serving size|nutrition facts)\b', text, re.I)
    negated_amount = re.search(r'\b(?:not|instead of|minus|negative)\s*$|[−–—/]\s*$', before, re.I)
    if nutrient_amount or label_amount or negated_amount or ',' in match[1]:
        return None, True
    grams = float(match[1])
    return (grams, False) if 0 < grams <= 5000 else (None, True)


def ingredient_words(text: str) -> tuple[str, ...]:
    """Conservative word matching, not an allergen taxonomy or cross-contact assessment."""
    words = re.findall(r'[^\W_]+', text.casefold(), flags=re.UNICODE)
    return tuple(word[:-3] + 'y' if len(word) > 4 and word.endswith('ies') else
                 word[:-1] if len(word) > 3 and word.endswith('s') and not word.endswith(('ss', 'us', 'is')) else word
                 for word in words)


def prepared_mass_needs_clarification(text: str, has_portion: bool) -> bool:
    """The current product contract only accepts as-sold nutrition; never infer cooking yield."""
    prepared = re.search(r'\b(?:cooked|prepared|boiled|steamed|after cooking)\b', text, re.I)
    return bool(prepared and (has_portion or GRAM_MENTION.search(text)))


def fiber_goal_statement(current: ComparisonItem, prior: ComparisonItem) -> str:
    if current.fiber_g is None or prior.fiber_g is None:
        return 'Fiber comparison is unknown; neither can lead for your fiber goal.'
    if current.fiber_g == prior.fiber_g:
        return 'Fiber is equal; neither leads for your fiber goal.'
    winner = current if current.fiber_g > prior.fiber_g else prior
    return f'{winner.name} has more fiber for your selected goal.'


def spoken_comparison(comparison: Comparison, profile: Profile, mode: str) -> str:
    """Keep audio brief; complete nutrient facts and goal details stay in Comparison."""
    current = comparison.items[-1]
    prior = comparison.items[0] if len(comparison.items) > 1 else None
    recipient = 'the household' if comparison.recipient_id == 'household' else 'you'
    prefix = 'Synthetic demo. ' if mode == 'fixture' else ''
    goals = {goal.strip().casefold() for goal in profile.goals} if comparison.recipient_id == 'self' and profile.consented else set()
    fiber_goal = 'prefer more fiber' in goals
    selected = next(((name, field, unit) for goal, name, field, unit in
                     [('prefer less sodium', 'sodium', 'sodium_mg', 'milligrams'), ('compare protein', 'protein', 'protein_g', 'grams')]
                     if goal in goals), None)
    if prior:
        reply = (f'For {recipient}, at {number(comparison.portion_grams)} grams each, {current.name} has '
                 f'{number(current.carbohydrate_g)} grams total carbohydrate versus '
                 f'{number(prior.carbohydrate_g)} in {prior.name}.')
        if fiber_goal:
            reply += ' ' + fiber_goal_statement(current, prior)
            if (current.fiber_g is not None and prior.fiber_g is not None and current.fiber_g > prior.fiber_g
                    and current.sugars_g is not None and prior.sugars_g is not None and current.sugars_g > prior.sugars_g):
                reply += ' It also has more sugar.'
        elif selected:
            name, field, unit = selected
            value, before = getattr(current, field), getattr(prior, field)
            reply += (f' Your {name} goal: {number(value)} versus {number(before)} {unit}.'
                      if value is not None and before is not None else f' {name.capitalize()} comparison is unknown.')
        elif current.fiber_g is not None and prior.fiber_g is not None:
            if (current.fiber_g > prior.fiber_g and current.sugars_g is not None
                    and prior.sugars_g is not None and current.sugars_g > prior.sugars_g):
                reply += ' More fiber, but also more sugar.'
            else:
                reply += f' Fiber: {number(current.fiber_g)} versus {number(prior.fiber_g)} grams.'
        return prefix + reply
    reply = (f'For {recipient}, {number(comparison.portion_grams)} grams of {current.name} has '
             f'{number(current.carbohydrate_g)} grams total carbohydrate.')
    if fiber_goal:
        reply += f' Your fiber goal: {number(current.fiber_g)} grams.' if current.fiber_g is not None else ' Fiber is unknown for your selected goal.'
    elif selected:
        name, field, unit = selected
        value = getattr(current, field)
        reply += f' {name.capitalize()}: {number(value)} {unit}.' if value is not None else f' {name.capitalize()} is unknown.'
    elif current.fiber_g is not None:
        reply += f' Fiber: {number(current.fiber_g)} grams.'
    return prefix + reply + ' Show another label to compare.'


def local_intent(text: str) -> Intent:
    t = text.casefold().strip().replace('’', "'")
    recipient = None
    # Corrections use the latest affirmative recipient, never a negated earlier
    # phrase such as "not for me, for my family".
    for match in re.finditer(r'\b(?:for (?:my |the )?(?:family|household|kids|children)|family shopping|household|for me|for myself|this is mine)\b', t):
        if re.search(r"(?:\bnot|n't|\bnever)\s*$", t[:match.start()]):
            continue
        recipient = 'self' if match.group() in ('for me', 'for myself', 'this is mine') else 'household'
    grams, _ = spoken_portion(t)
    intent = 'question'
    if re.fullmatch(r'(?:please )?(?:stop|cancel|be quiet|stop talking)[.!]?', t):
        intent = 'stop'
    elif re.search(r'\b(?:repeat|say that again)\b', t) and not re.search(r"\b(?:do not|don't|never|no need to)\s+(?:repeat|say)\b", t):
        intent = 'repeat'
    elif re.search(r'^(?:(?:please|can you|could you|would you)\s+)*(?:add|put)\b', t):
        intent = 'add'
    elif re.fullmatch(r"(?:(?:please|can you|could you)\s+)*(?:summary|summarize|summarise|shopping summary|shopping list|(?:summarize|summarise|show|read) (?:my |the )?(?:shopping )?list|what's on (?:my |the )?(?:shopping )?list)[.!?]*", t):
        intent = 'summary'
    elif re.search(r'\b(?:compare|contrast|what about|better|difference)\b', t):
        intent = 'compare'
    return Intent(intent=intent, recipient_id=recipient, portion_grams=grams)


def explicit_current_pair(text: str) -> bool:
    """An explicitly shown pair cannot borrow a missing item from old photos."""
    return bool(re.search(r'\b(?:these two|those two|both (?:these|of these)|two (?:products|items|cans|bottles|packages) (?:here|in (?:this|the) photo))\b', text, re.I))


def recipient_needs_clarification(text: str) -> bool:
    return bool(re.search(r"\bnot for (?:me|myself|(?:my |the )?(?:family|household|kids|children))\b", text, re.I)
                and local_intent(text).recipient_id is None)


@dataclass
class Record:
    session: Session
    latest: TurnResponse | None = None
    responses: dict[str, TurnResponse] = field(default_factory=dict)
    inflight: set[str] = field(default_factory=set)
    fingerprints: dict[str, str] = field(default_factory=dict)
    action_generation: int | None = None
    touched: float = field(default_factory=time.monotonic)
    calls: deque = field(default_factory=deque)
    frame: CapturedFrame | None = None
    frame_expires: float = 0
    frame_timer: asyncio.TimerHandle | None = None
    product_reply: tuple[str, str] | None = None
    analysis_mode: str = 'label'
    last_reply_context: tuple[str, str, str] | None = None
    # Bounded error provenance survives disconnect, unlike photos/reply caches.
    # Never stores exception strings, provider bodies, headers or user data.
    provider_failure: dict | None = None
    product_memory_scope: tuple | None = None
    product_memory: list[dict[str, str]] = field(default_factory=list)


class ShoppingService:
    def __init__(self, providers: Providers):
        self.providers = providers
        self.product_guide = ProductGuide(providers)
        self.records: dict[str, Record] = {}
        self.catalog = {p.id: p for p in fixtures()}

    def create(self, request: CreateSession) -> Session:
        self._purge()
        if len(self.records) >= 64:
            raise HTTPException(429, 'Local session limit reached. End an existing session first.')
        profile = request.profile
        if profile is None:
            profile = (Profile(display_name='Demo shopper', goals=['Compare total carbohydrate', 'Prefer more fiber'], consented=True)
                       if request.mode == 'fixture' else Profile())
        if not profile.consented and (profile.goals or profile.avoid_ingredients or profile.goal_source is not None
                                      or profile.nutrition_context is not None):
            raise HTTPException(400, 'Confirm consent before saving personal goals or ingredients to avoid.')
        session = Session(mode=request.mode, profile=profile.model_copy(deep=True))
        if session.profile.goal_source is not None:
            # This records acceptance of this confirmed profile, not a claimed clinical review.
            session.profile.goal_source.reviewed_at = datetime.now(timezone.utc)
        self.records[session.id] = Record(session=session)
        return session.model_copy(deep=True)

    def _purge(self) -> None:
        cutoff = time.monotonic() - 12 * 3600
        for key in [k for k, rec in self.records.items() if rec.touched < cutoff]:
            self.clear_frame(self.records[key])
            del self.records[key]

    def clear_frame(self, record: Record) -> None:
        record.frame = None
        record.frame_expires = 0
        if record.frame_timer is not None:
            record.frame_timer.cancel()
            record.frame_timer = None

    def capture_frame(self, record: Record, request: TurnRequest) -> None:
        self.clear_frame(record)
        record.frame = CapturedFrame(image_data_url=request.image_data_url,
            captured_at=request.captured_at.isoformat() if request.captured_at else now(),
            media_time_s=request.media_time_s)
        record.frame_expires = time.monotonic() + 120
        record.frame_timer = asyncio.get_running_loop().call_later(120, self.clear_frame, record)

    def get_frame(self, session_id: str) -> CapturedFrame:
        record = self.get(session_id)
        if record.frame is None or record.frame_expires <= time.monotonic():
            self.clear_frame(record)
            raise HTTPException(404, 'No recent captured frame. Frames expire after 120 seconds.')
        return record.frame.model_copy(deep=True)

    def get(self, session_id: str) -> Record:
        self._purge()
        record = self.records.get(session_id)
        if not record:
            raise HTTPException(404, 'Shopping session not found or expired.')
        record.touched = time.monotonic()
        return record

    def delete(self, session_id: str) -> None:
        record = self.get(session_id)
        self.clear_frame(record)
        record.session.generation += 1
        record.session.pending_action = None
        del self.records[session_id]

    def invalidate(self, record: Record) -> None:
        record.session.generation += 1
        record.session.pending_action = None
        record.action_generation = None
        record.latest = None

    def context(self, session_id: str, request: ContextRequest) -> Session:
        record = self.get(session_id)
        if request.recipient_id != record.session.recipient_id:
            self.clear_product_memory(record)
        self.clear_frame(record)
        self.invalidate(record)
        record.product_reply = None
        record.last_reply_context = None
        record.provider_failure = None
        record.session.recipient_id = request.recipient_id
        record.session.portion_grams = request.portion_grams
        return record.session.model_copy(deep=True)

    def cancel(self, session_id: str) -> Session:
        record = self.get(session_id)
        self.clear_frame(record)
        self.invalidate(record)
        return record.session.model_copy(deep=True)

    @staticmethod
    def clear_product_memory(record: Record) -> None:
        record.product_memory_scope = None
        record.product_memory.clear()

    def scope_product_memory(self, record: Record, products) -> None:
        scope = product_scope(record.session.recipient_id, products)
        if record.product_memory_scope != scope:
            record.product_memory.clear()
            record.product_memory_scope = scope

    @staticmethod
    def remember_product_exchange(record: Record, question: str, answer: str) -> None:
        entry = compact_exchanges([{'question': question, 'answer': answer}])
        if not entry:
            return
        # Repeated automatic comparisons update the latest useful answer rather
        # than filling the small window with the same synthetic question.
        prior = record.product_memory
        if prior and prior[-1]['question'] == entry[0]['question']:
            prior = prior[:-1]
        record.product_memory = compact_exchanges(prior + entry)

    def event(self, kind: str, stage: str, message: str, request: TurnRequest | None = None, **kwargs) -> Event:
        return Event(type=kind, stage=stage, message=message,
                     media_time_s=request.media_time_s if request else None, **kwargs)

    def stale(self, record: Record, generation: int) -> bool:
        return (record.session.generation != generation or
                self.records.get(record.session.id) is not record)

    def cancelled(self, record: Record) -> TurnResponse:
        return TurnResponse(session=record.session.model_copy(deep=True), reply='That request was cancelled or replaced.', status='cancelled')

    def finish(self, record: Record, reply: str, status: str, *, evidence=None, comparison=None, events=None) -> TurnResponse:
        record.last_reply_context = (record.analysis_mode, record.session.recipient_id, reply)
        record.session.messages.append(Message(role='assistant', text=reply))
        record.session.messages = record.session.messages[-40:]
        response = TurnResponse(session=record.session.model_copy(deep=True), reply=reply, status=status,
                                evidence=evidence or [], comparison=comparison, events=events or [],
                                pending_action=record.session.pending_action.model_copy(deep=True) if record.session.pending_action else None)
        record.latest = response.model_copy(deep=True)
        return response

    def evidence(self, product: Product, grams: float | None = None) -> list[Evidence]:
        source = product.provenance
        items = [Evidence(label='Product evidence', value=product.name, source=source.source,
                          kind=source.kind, verified=source.verified),
                 Evidence(label='Label basis', value=f'{number(product.nutrition.basis_grams)} g',
                          source=source.source, kind=source.kind, verified=source.verified),
                 Evidence(label='Captured at', value=source.captured_at, source=source.source,
                          kind=source.kind, verified=source.verified)]
        if grams is not None and source.verified:
            items.append(Evidence(label='Portion arithmetic',
                value=f'{number(grams)} g ÷ {number(product.nutrition.basis_grams)} g × declared label values',
                source='Deterministic decimal calculation; label rounding still applies', kind='calculation', verified=True))
        return items

    def current(self, record: Record) -> Product | None:
        return next((p for p in record.session.products if p.id == record.session.current_product_id), None)

    def barcode_cache(self, record: Record, barcode: str | None) -> Product | None:
        identity = canonical_gtin(barcode)
        if identity is None:
            return None
        candidates = [p for p in reversed(record.session.products)
                      if canonical_gtin(p.barcode) == identity and p.provenance.kind == 'reviewed_label'
                      and p.provenance.verified]
        # Same GTIN does not resolve conflicting label revisions or misassociated scans.
        signatures = {(p.name.casefold().strip(), (p.brand or '').casefold().strip(),
                       p.market.casefold().strip(), p.nutrition.model_dump_json(),
                       tuple(s.casefold().strip() for s in p.ingredients)) for p in candidates}
        return candidates[0] if len(signatures) == 1 else None

    def remember(self, record: Record, product: Product) -> None:
        record.session.products = [p for p in record.session.products if p.id != product.id][-11:] + [product]
        record.session.current_product_id = product.id

    def constraint(self, record: Record, product: Product) -> str | None:
        profile = record.session.profile
        if record.session.recipient_id != 'self' or not profile.consented:
            return None
        ingredients = ingredient_words(' '.join(product.ingredients))
        for avoided in profile.avoid_ingredients:
            term = ingredient_words(avoided)
            if not term:
                return 'A saved ingredient to avoid is unclear. Please correct the avoid list before choosing this product.'
            if any(ingredients[index:index + len(term)] == term for index in range(len(ingredients) - len(term) + 1)):
                return f'Your saved avoid list includes {avoided}, and the ingredient text contains it. Please check the package before choosing it.'
        if profile.avoid_ingredients and (not product.ingredients or 'ingredients' in product.uncertain_fields):
            return 'The ingredient information is incomplete, so I cannot check your saved avoid list. Please review the ingredient label.'
        return None

    def evaluate(self, record: Record, *, intent: str = 'question', events: list[Event] | None = None) -> TurnResponse:
        events = events or []
        session = record.session
        product = self.current(record)
        if product is None:
            return self.finish(record, 'Hold one product up and capture its nutrition label, or choose a synthetic demo product.', 'needs_image', events=events)
        if product.id == 'unknown':
            return self.finish(record, 'I cannot identify this demonstration package or read its label. Show a clear product name and nutrition panel; I will not guess its values.', 'unknown', events=events)
        evidence = self.evidence(product)
        if not product.provenance.verified:
            return self.finish(record, 'I extracted a label candidate. Please check the exact product, serving grams, and nutrient values on your phone before I compare it.', 'needs_review', evidence=evidence, events=events)
        if product.nutrition.carbohydrate_g is None or 'basis_grams' in product.uncertain_fields:
            return self.finish(record, 'The total carbohydrate or serving basis is missing. Please provide a clearer label or correct those fields before comparison.', 'unknown', evidence=evidence, events=events)
        constraint = self.constraint(record, product)
        if constraint:
            return self.finish(record, constraint, 'unknown', evidence=evidence, events=events)
        if session.portion_grams is None:
            recipient = 'your family' if session.recipient_id == 'household' else 'you'
            return self.finish(record, f'How many grams do {recipient} intend to eat? The label uses {number(product.nutrition.basis_grams)} grams per serving basis.', 'needs_portion', evidence=evidence, events=events)
        grams = session.portion_grams
        evidence = self.evidence(product, grams)
        other = next((p for p in reversed(session.products) if p.id != product.id and p.provenance.verified
                      and p.nutrition.carbohydrate_g is not None and 'basis_grams' not in p.uncertain_fields
                      and not (canonical_gtin(product.barcode) is not None
                               and canonical_gtin(p.barcode) == canonical_gtin(product.barcode))
                      and not self.constraint(record, p)), None)
        selected = [other, product] if other else [product]
        items = [calculate(p, grams) for p in selected]
        current_item = items[-1]
        prefix = 'Synthetic demo. ' if session.mode == 'fixture' else ''
        context = 'For the household' if session.recipient_id == 'household' else 'For your portion'
        if other:
            prior = items[0]
            delta = round(current_item.carbohydrate_g - prior.carbohydrate_g, 2)
            direction = 'less' if delta < 0 else 'more'
            statement = (f'{number(abs(delta))} g {direction} total carbohydrate than {prior.name}'
                         if delta else f'the same total carbohydrate as {prior.name}')
            summary = f'{context}, at {number(grams)} g each, {current_item.name} has {number(current_item.carbohydrate_g)} g carbohydrate: {statement}.'
            if current_item.fiber_g is not None and prior.fiber_g is not None:
                summary += f' Fiber is {number(current_item.fiber_g)} g versus {number(prior.fiber_g)} g.'
            if current_item.sugars_g is not None and prior.sugars_g is not None and current_item.sugars_g != prior.sugars_g:
                summary += f' Total sugar is {number(current_item.sugars_g)} g versus {number(prior.sugars_g)} g.'
            evidence = self.evidence(other, grams) + evidence
        else:
            summary = f'{context}, {number(grams)} g of {product.name} contains {number(current_item.carbohydrate_g)} g total carbohydrate.'
            if current_item.fiber_g is not None:
                summary += f' It has {number(current_item.fiber_g)} g fiber.'
            summary += ' Show me another label to compare the same amount.'
        if session.recipient_id == 'self' and session.profile.consented:
            goals = {goal.strip().casefold() for goal in session.profile.goals}
            for goal, keyword, nutrient, unit in (('prefer less sodium', 'sodium', 'sodium_mg', 'mg'), ('compare protein', 'protein', 'protein_g', 'g')):
                if goal not in goals:
                    continue
                value = getattr(current_item, nutrient)
                if value is None:
                    summary += f' For your {keyword} goal, this label value is missing; I cannot compare it.'
                elif other:
                    prior_value = getattr(items[0], nutrient)
                    if prior_value is None:
                        summary += f' For your {keyword} goal, this has {number(value)} {unit}; the previous product value is unknown.'
                    else:
                        summary += f' For your {keyword} goal, this has {number(value)} {unit} versus {number(prior_value)} {unit} in the previous product.'
                else:
                    summary += f' For your {keyword} goal, this portion has {number(value)} {unit}.'
            if 'prefer more fiber' in goals and other:
                summary += ' ' + fiber_goal_statement(current_item, items[0])
        if session.recipient_id == 'household':
            summary += ' Your personal goals and avoid list are not applied to the household.'
        comparison = Comparison(portion_grams=grams, recipient_id=session.recipient_id, items=items, summary=summary)
        events.append(self.event('comparison.calculated', 'calculation', 'Verified source values scaled deterministically to the confirmed gram portion.'))
        if intent == 'add':
            if len(session.cart) >= 100:
                raise HTTPException(409, 'This local shopping list has reached its limit.')
            action = Action(label=f'Add one {product.name} for {"household" if session.recipient_id == "household" else "you"}',
                            product_id=product.id, recipient_id=session.recipient_id,
                            expires_at=(datetime.now(timezone.utc) + timedelta(seconds=120)).isoformat())
            session.pending_action = action
            record.action_generation = session.generation
            events.append(self.event('action.proposed', 'confirmation', 'No list write has happened. Explicit confirmation is required.'))
            return self.finish(record, prefix + f'Add one {product.name} to the list for {"your household" if session.recipient_id == "household" else "you"}? Confirm on your phone.',
                               'needs_confirmation', evidence=evidence, comparison=comparison, events=events)
        return self.finish(record, spoken_comparison(comparison, session.profile, session.mode), 'ready',
                           evidence=evidence, comparison=comparison, events=events)

    async def turn(self, session_id: str, request: TurnRequest) -> TurnResponse:
        record = self.get(session_id)
        fingerprint = hashlib.sha256(request.model_dump_json().encode()).hexdigest()
        existing_fingerprint = record.fingerprints.get(request.client_turn_id)
        if existing_fingerprint is not None and existing_fingerprint != fingerprint:
            raise HTTPException(409, 'A client turn ID cannot be reused with different input.')
        old = record.responses.get(request.client_turn_id)
        if old:
            return old.model_copy(deep=True) if old.session.generation == record.session.generation else self.cancelled(record)
        if request.client_turn_id in record.inflight:
            raise HTTPException(409, 'This turn is already being processed.')
        session = record.session
        if request.fixture_product_id and session.mode != 'fixture':
            raise HTTPException(400, 'Synthetic products can only be used in Synthetic demo mode.')
        if (request.image_data_url or request.barcode) and session.mode == 'fixture':
            raise HTTPException(400, 'Start a live or recorded session to analyze a real image. Synthetic mode never processes real images.')
        if request.fixture_product_id and request.fixture_product_id not in self.catalog:
            raise HTTPException(404, 'Unknown synthetic product ID.')
        current_time = time.monotonic()
        while record.calls and record.calls[0] < current_time - 60:
            record.calls.popleft()
        if len(record.calls) >= 30:
            raise HTTPException(429, 'Too many turns. Please wait a moment before retrying.')
        record.calls.append(current_time)
        record.provider_failure = None
        if record.analysis_mode != request.analysis_mode:
            self.clear_product_memory(record)
            session.current_product_id = None
            session.current_visual_product_id = None
            session.current_visual_product_ids = []
            record.product_reply = None
            record.last_reply_context = None
            self.clear_frame(record)
        record.analysis_mode = request.analysis_mode
        self.invalidate(record)
        generation = session.generation
        record.inflight.add(request.client_turn_id)
        record.fingerprints[request.client_turn_id] = fingerprint
        session.messages.append(Message(role='user', text=request.text))
        session.messages = session.messages[-40:]
        events = [self.event('turn.received', 'input', f'Explicit user request received in {session.mode} mode.', request)]
        try:
            response = await self._turn(record, request, generation, events)
            if response.status != 'cancelled' and not self.stale(record, generation):
                record.responses[request.client_turn_id] = response.model_copy(deep=True)
                while len(record.responses) > 20:
                    oldest = next(iter(record.responses))
                    del record.responses[oldest]
                    record.fingerprints.pop(oldest, None)
            return response
        finally:
            record.inflight.discard(request.client_turn_id)
            if request.client_turn_id not in record.responses:
                record.fingerprints.pop(request.client_turn_id, None)

    @staticmethod
    def visual_payload(product: VisualProduct) -> dict:
        value = product.model_dump(mode='json')
        return {
            'product_id': product.id,
            'captured_at': product.captured_at,
            'identity': {key: value[key] for key in ('name', 'brand', 'variant', 'package_size',
                         'category', 'visible_text', 'confidence', 'uncertainties', 'box')},
            'catalog': {'status': product.catalog_status, 'notes': product.catalog_notes,
                        'sources': [source.model_dump() for source in product.sources]},
        }

    @staticmethod
    def visual_evidence(product: VisualProduct) -> list[Evidence]:
        return [Evidence(label='Held product candidate', value=product.name,
                         source='Perceptron front-package crop; visual identity is not package verification',
                         kind='visual_identity', verified=False),
                Evidence(label='Captured at', value=product.captured_at, source='Submitted glasses/phone image',
                         kind='visual_identity', verified=False)] + [
                    Evidence(label='Product source', value=source.title, source=source.url,
                             kind='catalog', verified=False) for source in product.sources]

    async def _product_turn(self, record: Record, request: TurnRequest, generation: int,
                            events: list[Event], intent: Intent) -> TurnResponse:
        """Identify/search the held package; never insert catalog estimates into label arithmetic."""
        session = record.session
        recipient = intent.recipient_id or request.recipient_id
        if recipient and recipient != session.recipient_id:
            self.clear_product_memory(record)
            session.recipient_id = recipient
            session.portion_grams = None
            record.product_reply = None
            events.append(self.event('recipient.changed', 'context', 'Recipient changed; personal context does not apply to household shopping.', request))
        if not request.image_data_url and intent.intent == 'add':
            return self.finish(record, 'Product search has not verified this package for the shopping list. Use label review before adding it.', 'unknown', events=events)
        if not request.image_data_url and intent.intent == 'summary':
            names = [item.product_name for item in session.cart]
            return self.finish(record, 'Your confirmed shopping list: ' + ', '.join(names) + '.' if names else
                               'Your confirmed shopping list is empty.', 'ready', events=events)
        if not request.image_data_url and intent.intent == 'repeat':
            if record.product_reply and record.product_reply[0] == session.recipient_id:
                return self.finish(record, record.product_reply[1], 'ready', events=events)

        product = next((p for p in session.visual_products if p.id == session.current_visual_product_id), None)
        pictured = [p for p in session.visual_products if p.id in session.current_visual_product_ids]
        if request.image_data_url:
            session.current_visual_product_id = None
            session.current_visual_product_ids = []
            session.current_product_id = None
            record.product_reply = None
            self.capture_frame(record, request)
            started = time.perf_counter()
            try:
                with capture_vision_timings() as vision_timings:
                    try:
                        detected = await self.product_guide.identify_many(request.image_data_url)
                    finally:
                        for timing in vision_timings:
                            events.append(self.event('product.vision_timing', 'perception',
                                f"Perceptron {timing['stage']}: {timing['milliseconds']} ms (parallel crop times overlap).", request))
            except ProviderError as exc:
                if self.stale(record, generation):
                    return self.cancelled(record)
                record.provider_failure = exc.diagnostic(provider='perceptron', stage='identify')
                # Technical provenance stays in the protected snapshot. Speak
                # the action the wearer can take, not an internal stage name.
                category = record.provider_failure['category']
                message = str(exc)
                if category == 'rate_limited':
                    message = 'Product recognition is busy. Please try again shortly.'
                elif category == 'authorization' or 'not configured' in message.casefold():
                    message = 'Product recognition is unavailable. Check the server’s recognition access before trying again.'
                elif category == 'balance':
                    message = 'Product recognition is unavailable until the service account has credit again.'
                elif category != 'unknown' or re.search(r'\b(?:Perceptron|AI provider)\b', message, re.I):
                    message = 'I couldn’t finish recognizing the products. Please try again.'
                return self.finish(record, message, 'needs_image', events=events)
            if self.stale(record, generation):
                return self.cancelled(record)
            if not detected:
                return self.finish(record, 'Hold one or two products with their front names facing the glasses. I could not confidently select a held package in this photo.', 'needs_image', events=events)
            # An actual identity change invalidates conversational references
            # even if the new search later fails. A failed/empty capture alone
            # cannot establish new identities and never receives old memory.
            # Every item in one photo shares capture provenance. Per-item now()
            # timestamps would incorrectly make a same-photo pair look separate.
            captured_at = record.frame.captured_at if record.frame else (request.captured_at.isoformat() if request.captured_at else now())
            pictured = [VisualProduct(**item.model_dump(), captured_at=captured_at) for item in detected]
            self.scope_product_memory(record, pictured)
            record.product_memory = without_old_positions(record.product_memory)
            if len(detected) < 2 and explicit_current_pair(request.text):
                return self.finish(record, 'I could only identify one of the two products. Hold both front names clearly in the same view so I can compare this pair.',
                                   'needs_image', events=events)
            elapsed = round((time.perf_counter() - started) * 1000)
            for index, item in enumerate(pictured, 1):
                events.append(self.event('product.detected', 'perception',
                    f'Product {index}: {item.brand} {item.name}. Perceptron inspected {len(pictured)} separate crop(s) in {elapsed} ms.', request, box=item.box))
            started = time.perf_counter()
            # Independent lookups share one turn deadline. Commit the group only
            # after both finish, so cancellation cannot leave half of a pair.
            matches = await asyncio.gather(*(self.product_guide.search(item) for item in detected),
                                           return_exceptions=True)
            if self.stale(record, generation):
                return self.cancelled(record)
            search_errors = []
            for item, match in zip(pictured, matches):
                if isinstance(match, ProviderError):
                    record.provider_failure = match.diagnostic(
                        provider=self.providers.settings.search_provider, stage='catalog_search')
                    search_errors.append(f'{item.name}: {match}')
                elif isinstance(match, BaseException):
                    raise match
                else:
                    item.catalog_status = match.status
                    item.catalog_notes = match.notes
                    item.sources = [source.model_dump() for source in match.sources]
                events.append(self.event('product.search', 'search',
                    f'{item.name}: catalog search finished in {round((time.perf_counter() - started) * 1000)} ms; {len(item.sources)} cited sources. Exact package formulation remains unverified.', request))
            # Replace earlier shots, but keep two separately selected packages
            # from this photo even if they appear to be the same variant.
            identity = lambda p: tuple((getattr(p, key) or '').strip().casefold() for key in ('name', 'brand', 'variant', 'package_size'))
            selected = {identity(item) for item in pictured}
            session.visual_products = [p for p in session.visual_products if identity(p) not in selected][-(12 - len(pictured)):] + pictured
            session.current_visual_product_ids = [p.id for p in pictured]
            product = pictured[-1]
            session.current_visual_product_id = product.id
            if (search_errors and not visual_identification_question(request.text)
                    and grounded_named_target(request.text, pictured) is None):
                return self.finish(record, 'I could not complete the product lookup. ' + ' '.join(search_errors),
                                   'unknown', evidence=[e for p in pictured for e in self.visual_evidence(p)], events=events)
        if product is None:
            return self.finish(record, 'Hold one product with its front name facing the glasses, then tap Look & Ask.', 'needs_image', events=events)
        if len(pictured) < 2 and explicit_current_pair(request.text):
            return self.finish(record, 'I need a clear view of both products together to compare these two. Hold both front names in the same photo.',
                               'needs_image', events=events)
        if len(pictured) < 2 and intent.intent == 'compare' and re.search(
                r'\b(?:compare|contrast|better|differences?|previous|first|other)\b', request.text, re.I):
            # A new photo replaces the active selection. Historical detections
            # are evidence records, not an implicit second comparison target.
            # Even "the previous one" is ambiguous after a previously shown
            # pair; ask for the intended pair instead of choosing history[-1].
            return self.finish(record, f'I currently have only {spoken_product_name(product)} selected. Show both products together so I can compare the ones you mean.',
                               'needs_image', evidence=self.visual_evidence(product), events=events)

        context = {'recipient_id': session.recipient_id, 'consented': session.profile.consented}
        if session.recipient_id == 'self' and session.profile.consented:
            context.update(goals=session.profile.goals, avoid_ingredients=session.profile.avoid_ingredients)
            if request.conversation_request:
                context['goals'] = [request.conversation_request]
                context['conversation_request'] = request.conversation_request
            if session.profile.nutrition_context is not None:
                context['nutrition_context'] = session.profile.nutrition_context.model_dump()
        # Deliberately omit gram portion: front-package/catalog values may use mL, bars or servings.
        # No density conversion or automatic scaling is authorized by this search workflow.
        pair = len(pictured) == 2 and product.id in {p.id for p in pictured}
        previous = [p for p in pictured if p.id != product.id] if pair else []
        self.scope_product_memory(record, [product] + previous)
        if record.product_memory:
            context['recent_product_exchanges'] = compact_exchanges(record.product_memory)
        if visual_identification_question(request.text):
            visible = pictured or [product]
            reply = ('These appear to be ' if len(visible) > 1 else 'This appears to be ') + ' and '.join(spoken_product_name(p) for p in visible) + '.'
            record.product_reply = (session.recipient_id, reply)
            self.remember_product_exchange(record, request.text, reply)
            return self.finish(record, reply, 'ready', evidence=[e for p in visible for e in self.visual_evidence(p)], events=events)
        response_product, response_previous = product, previous
        if pair and any(p.catalog_status != 'grounded' for p in pictured):
            target = grounded_named_target(request.text, pictured)
            if target is not None:
                # The unknown companion stays visible in the snapshot, but is
                # not evidence for a question explicitly naming the known item.
                # Keep pair memory scope; do not let a prior companion statement
                # re-enter this isolated factual answer through conversation.
                response_product, response_previous = target, []
                context.pop('recent_product_exchanges', None)
            else:
                unresolved = [p for p in pictured if p.catalog_status != 'grounded']
                names = ' and '.join(spoken_product_name(p) for p in unresolved)
                conflict = any(p.catalog_notes.startswith('Catalog variant conflict:') for p in unresolved)
                detail = (f'The exact version of {names} is unresolved because the listings describe conflicting variants.'
                          if conflict else f'I do not have matched product information for {names}.')
                return self.finish(record, detail + ' I cannot reliably compare this pair yet.',
                                   'unknown', evidence=[e for p in pictured for e in self.visual_evidence(p)], events=events)
        question = request.text
        if (pair and request.image_data_url
                and request.text.strip().rstrip('.!?').casefold() in
                    ('is this a good option', 'is this a good option for me')):
            # The legacy tap action has a generic default question. Explicit
            # questions and automatic voice requests already carry their actual
            # intent/criterion; do not replace those with a winner instruction.
            question = ('Compare the two products selected together in this photo for the stated shopping goal. '
                        'Name both and choose the better option only when their cited evidence supports it. '
                        'Shopper question: ' + request.text)
        started = time.perf_counter()
        try:
            reply = await self.product_guide.respond(self.visual_payload(response_product), question,
                                                     context, [self.visual_payload(p) for p in response_previous])
        except ProviderError as exc:
            if self.stale(record, generation):
                return self.cancelled(record)
            record.provider_failure = exc.diagnostic(
                provider=self.providers.settings.text_provider, stage='product_response')
            compared = [response_product] + response_previous
            description = ('I found ' + ' and '.join(p.name for p in pictured) + '.' if pair else
                           f'The package appears to be {product.name}.')
            return self.finish(record, f'{description} {exc}',
                               'unknown', evidence=[e for p in compared for e in self.visual_evidence(p)], events=events)
        if self.stale(record, generation):
            return self.cancelled(record)
        provider_label = {'deepseek': 'DeepSeek', 'gemini': 'Gemini',
                          'anthropic': 'Anthropic Sonnet'}.get(self.providers.settings.text_provider, 'Configured provider')
        events.append(self.event('product.guidance', 'response',
            f'{provider_label} returned a bounded response in {round((time.perf_counter() - started) * 1000)} ms. Catalog estimates are separate from reviewed label facts.', request))
        record.product_reply = (session.recipient_id, reply)
        self.remember_product_exchange(record, request.text, reply)
        evidence = self.visual_evidence(response_product)
        for prior in response_previous:
            evidence.extend(self.visual_evidence(prior))
        return self.finish(record, reply, 'ready', evidence=evidence, events=events)

    async def _turn(self, record: Record, request: TurnRequest, generation: int, events: list[Event]) -> TurnResponse:
        session = record.session
        intent = local_intent(request.text)
        if recipient_needs_clarification(request.text):
            return self.finish(record, 'Is this for you or your household? Confirm that before I apply personal goals.',
                               'needs_clarification', events=events)
        if intent.intent == 'stop':
            self.clear_frame(record)
            return self.finish(record, 'Stopped. Tap or speak again when you want help.', 'ready', events=events)
        if re.search(r'\b(insulin|medication dose|drug dose|predict.{0,15}(glucose|blood sugar))\b', request.text, re.I):
            return self.finish(record, 'I can explain package nutrition and compare portions, but I cannot give medication instructions or predict your glucose response.', 'ready', events=events)
        if request.analysis_mode == 'product':
            try:
                async with asyncio.timeout(40):
                    return await self._product_turn(record, request, generation, events, intent)
            except TimeoutError:
                if self.stale(record, generation):
                    return self.cancelled(record)
                return self.finish(record, 'Product search took too long. Please try once more while keeping the app open.', 'unknown', events=events)
        if prepared_mass_needs_clarification(request.text, request.portion_grams is not None or session.portion_grams is not None):
            session.recipient_id = intent.recipient_id or request.recipient_id or session.recipient_id
            session.portion_grams = None
            self.clear_frame(record)
            if request.image_data_url or request.barcode or request.fixture_product_id:
                session.current_product_id = None
            events.append(self.event('portion.basis_clarification', 'context', 'Prepared weight was not scaled with as-sold label values; portion and pending action cleared.', request))
            return self.finish(record, 'Does your portion match the label’s as-sold basis? I cannot apply dry-food values to cooked weight. Give the matching as-sold amount, or show a matching ready-to-eat package label.',
                               'needs_portion', events=events)
        _, ambiguous_portion = spoken_portion(request.text)
        if ambiguous_portion:
            session.recipient_id = intent.recipient_id or request.recipient_id or session.recipient_id
            session.portion_grams = None
            self.clear_frame(record)
            if request.image_data_url or request.barcode or request.fixture_product_id:
                session.current_product_id = None
            events.append(self.event('portion.clarification', 'context', 'Ambiguous, invalid, or nutrient gram amounts did not become a food portion; prior portion and pending actions are cleared.', request))
            return self.finish(record, 'How many grams of the food do you intend to eat? Tell me one positive amount, such as 60 grams; label nutrient amounts are separate.',
                               'needs_portion', events=events)
        cached = self.barcode_cache(record, request.barcode)
        if request.image_data_url:
            self.capture_frame(record, request)
        needs_ocr = bool(request.image_data_url and cached is None)
        known_question = ' '.join(request.text.casefold().strip().rstrip('.!?').split()) in LOCAL_QUESTION_PHRASES
        # Explicit local controls, fixture selection, and reviewed barcode hits have no model latency.
        text_provider = self.providers.settings.text_provider
        intent_key = {'deepseek': self.providers.settings.deepseek_key,
                      'gemini': self.providers.settings.gemini_key,
                      'anthropic': getattr(self.providers.settings, 'anthropic_key', '')}.get(text_provider, '')
        provider_label = {'deepseek': 'DeepSeek', 'gemini': 'Gemini', 'anthropic': 'Anthropic Sonnet'}.get(text_provider, text_provider)
        needs_intent = bool(intent_key and intent.intent == 'question'
                            and intent.recipient_id is None and intent.portion_grams is None
                            and not request.fixture_product_id and cached is None and not known_question)

        async def measured(call):
            started = time.perf_counter()
            try:
                return await call, None, round((time.perf_counter() - started) * 1000)
            except ProviderError as exc:
                return None, exc, round((time.perf_counter() - started) * 1000)

        async def optional_intent():
            # This deadline applies only to the optional intent task, never OCR.
            # asyncio.timeout cancels that request and discards its late result.
            try:
                async with asyncio.timeout(INTENT_TIMEOUT_SECONDS):
                    return await self.providers.interpret(request.text)
            except TimeoutError as exc:
                raise ProviderError('Optional intent deadline reached; using local controls.') from exc

        calls = []
        if needs_intent:
            calls.append(measured(optional_intent()))
        if needs_ocr:
            calls.append(measured(self.providers.extract_label(request.image_data_url)))
        results = await asyncio.gather(*calls) if calls else []
        if self.stale(record, generation):
            return self.cancelled(record)
        if needs_intent:
            remote, error, elapsed = results.pop(0)
            if error is None:
                # Providers never set recipient, portions, write intent, or execute tools.
                if remote.intent in ('compare', 'summary', 'repeat'):
                    intent.intent = remote.intent
                events.append(self.event('intent.parsed', 'intent', f'{provider_label} bounded intent completed in {elapsed} ms; explicit context rules remain authoritative.', request))
            else:
                events.append(self.event('intent.fallback', 'intent', f'{provider_label} unavailable after {elapsed} ms; using explicit words and local controls.', request))
        # Native/browser clients may send their previous UI snapshot on every turn.
        # Explicit spoken context is newer evidence and must override that snapshot.
        recipient = intent.recipient_id or request.recipient_id
        spoken_recipient_change = intent.recipient_id is not None and intent.recipient_id != session.recipient_id
        portion = (intent.portion_grams if spoken_recipient_change or intent.portion_grams is not None
                   else request.portion_grams)
        if recipient and recipient != session.recipient_id:
            if not request.image_data_url:
                self.clear_frame(record)
            session.recipient_id = recipient
            session.portion_grams = None
            events.append(self.event('recipient.changed', 'context', 'Recipient changed; previous portion and pending actions cleared.', request))
        if portion is not None:
            session.portion_grams = portion
            events.append(self.event('portion.confirmed', 'context', f'User explicitly specified {number(portion)} grams.', request))
        if request.fixture_product_id:
            self.remember(record, self.catalog[request.fixture_product_id].model_copy(deep=True))
            events.append(self.event('fixture.selected', 'source', 'Explicit fictional fixture selected; no camera recognition or provider call is claimed.', request))
        elif cached is not None:
            self.remember(record, cached.model_copy(deep=True))
            events.append(self.event('product.session_cache', 'source', 'Exact GTIN identity matched a consistent user-reviewed label in this session (leading-zero representations are equivalent); OCR was skipped. Original label capture time is preserved.', request))
        elif needs_ocr:
            result, error, elapsed = results.pop(0)
            if error is not None:
                # Do not retain a prior product as current after a new capture failed.
                session.current_product_id = None
                events.append(self.event('perception.unavailable', 'perception', f'{error} ({elapsed} ms)', request))
                return self.finish(record, str(error), 'needs_image', events=events)
            events.append(self.event('perception.completed', 'perception', f'Perceptron returned structured OCR candidates from this image in {elapsed} ms.', request, box=result.box))
            capture_issue = result.capture_issue()
            if not result.is_nutrition_label or result.nutrition is None or capture_issue:
                session.current_product_id = None
                if result.is_nutrition_label and capture_issue == 'volume_only':
                    reply = 'This label uses a volume serving, not a gram weight. I cannot treat millilitres or cups as grams. This version needs a printed gram-based serving to compare.'
                else:
                    reply = 'I cannot clearly read a matching gram-based serving and total carbohydrate amount. Show one bottle or package straight on, fill most of the view with its nutrition panel, and hold still.'
                events.append(self.event('perception.retake', 'perception', 'The extracted serving basis or carbohydrate was insufficient; no review candidate or comparison was created.', request))
                return self.finish(record, reply, 'needs_image', events=events)
            if result.label_regime != 'US' or result.preparation != 'as_sold':
                session.current_product_id = None
                return self.finish(record, 'This prototype needs a clear US label for the food as sold. I cannot compare this label basis reliably yet.', 'needs_image', events=events)
            extracted_barcode = result.barcode if result.barcode and valid_gtin(result.barcode) else None
            if request.barcode and extracted_barcode and canonical_gtin(request.barcode) != canonical_gtin(extracted_barcode):
                session.current_product_id = None
                return self.finish(record, 'The scanned barcode and label extraction disagree. Capture one product at a time and verify its exact variant.', 'needs_image', events=events)
            product = Product(id=uid(), name=result.name, brand=result.brand, barcode=request.barcode or extracted_barcode,
                              market=result.market, ingredients=result.ingredients, nutrition=result.nutrition.model_copy(deep=True),
                              provenance=Provenance(kind='package_ocr', source=f'Perceptron OCR candidate · {session.mode}; user review required',
                                                    captured_at=request.captured_at.isoformat() if request.captured_at else now(), verified=False),
                              uncertain_fields=result.uncertain_fields)
            self.remember(record, product)
            return self.evaluate(record, events=events)
        elif request.barcode:
            session.current_product_id = None
            self.clear_frame(record)
            return self.finish(record, 'This barcode has no unambiguous reviewed label in this session. Show its nutrition panel; a barcode alone does not establish nutrition or local stock.', 'needs_image', events=events)
        if intent.intent == 'summary':
            counts = {'self': [], 'household': []}
            for item in session.cart:
                counts[item.recipient_id].append(f'{item.quantity} × {item.product_name}')
            segments = [f'{"You" if key == "self" else "Household"}: {", ".join(items)}.' for key, items in counts.items() if items]
            reply = ' '.join(segments) if segments else 'Your confirmed shopping list is empty. Considering a product does not add it.'
            return self.finish(record, reply, 'ready', events=events)
        if intent.intent == 'repeat':
            cached_reply = record.last_reply_context
            previous = (cached_reply[2] if cached_reply and cached_reply[:2] == ('label', session.recipient_id)
                        else 'No label answer to repeat in this context yet. Show a product label or choose product search.')
            return self.finish(record, previous, 'ready', events=events)
        if re.search(r'\b(in stock|available|where can|on (?:this|the) shelf|cheaper|price|cost)\b', request.text, re.I):
            return self.finish(record, 'This build has no verified store inventory or prices. I can compare products you explicitly capture, but cannot claim they are in stock or cheaper.', 'unknown', events=events)
        return self.evaluate(record, intent=intent.intent, events=events)

    def review(self, session_id: str, request: ReviewRequest) -> TurnResponse:
        record = self.get(session_id)
        product = next((p for p in record.session.products if p.id == request.product_id), None)
        if product is None:
            raise HTTPException(404, 'Product candidate not found in this session.')
        if product.id != record.session.current_product_id:
            raise HTTPException(409, 'Review the current candidate; this candidate was replaced.')
        if product.provenance.kind != 'package_ocr':
            raise HTTPException(409, 'This product is not an unreviewed OCR candidate.')
        if request.confirmed:
            corrections = request.corrections
            corrected_name = corrections.name if corrections else None
            reviewed_name = corrected_name if corrected_name is not None else product.name
            if (not reviewed_name.strip() or reviewed_name.strip().casefold() == 'unidentified package' or
                    ('name' in product.uncertain_fields and corrected_name is None)):
                raise HTTPException(422, 'Correct the exact product name before confirming this label.')
            if ('basis_grams' in product.uncertain_fields and
                    (corrections is None or corrections.nutrition is None)):
                raise HTTPException(422, 'Correct the nutrition using a confirmed gram-based serving before confirming this label. A volume serving is not a gram weight.')
        self.invalidate(record)
        if not request.confirmed:
            record.session.products = [p for p in record.session.products if p.id != product.id]
            record.session.current_product_id = None
            return self.finish(record, 'Candidate discarded. Capture a clearer label when ready.', 'needs_image')
        if request.corrections:
            corrections = request.corrections
            if corrections.name is not None:
                product.name = corrections.name
            if corrections.nutrition is not None:
                product.nutrition = corrections.nutrition
            if corrections.ingredients is not None:
                product.ingredients = corrections.ingredients
        # Confirmation validates present values. Missing fields remain explicitly unknown.
        product.uncertain_fields = [name for name in FIELDS if getattr(product.nutrition, name) is None]
        if not product.ingredients:
            product.uncertain_fields.append('ingredients')
        product.provenance = Provenance(kind='reviewed_label', source='Physical label values reviewed and confirmed by the user',
                                        captured_at=product.provenance.captured_at, verified=True)
        return self.evaluate(record, events=[self.event('label.reviewed', 'review', 'User confirmed the exact product and present label values; absent values remain unknown.')])

    def confirm(self, session_id: str, action_id: str) -> TurnResponse:
        record = self.get(session_id)
        session = record.session
        action = session.pending_action
        if (not action or action.id != action_id or record.action_generation != session.generation or
                action.recipient_id != session.recipient_id or action.product_id != session.current_product_id or
                datetime.fromisoformat(action.expires_at) <= datetime.now(timezone.utc)):
            raise HTTPException(409, 'That action expired, was already used, or no longer matches the current product and recipient.')
        product = self.current(record)
        if not product or not product.provenance.verified or self.constraint(record, product):
            raise HTTPException(409, 'Product evidence must be reviewed and constraints resolved before adding.')
        session.cart.append(CartItem(product_id=action.product_id, product_name=product.name,
                                     recipient_id=action.recipient_id, quantity=action.quantity))
        self.invalidate(record)
        return self.finish(record, f'Added one {product.name} for {"your household" if action.recipient_id == "household" else "you"}. This is a shopping list, not a purchase.', 'ready',
                           events=[self.event('list.confirmed', 'action', 'Explicit, single-use confirmation added one item to the local shopping list.')])
