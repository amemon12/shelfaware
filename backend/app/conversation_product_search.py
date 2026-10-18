"""Voice adapter for the existing held-product search pipeline.

Conversation supplies fresh images and cancellation; ShoppingService remains the
owner of product selection, cited evidence, personal context, and bounded advice.
Online matches never become reviewed label facts or portion arithmetic here.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import re
from time import monotonic
from typing import Awaitable, Callable, Mapping
from uuid import uuid4

from fastapi import HTTPException

from .models import ContextRequest, TurnRequest
from .conversation_intent_router import route_missing_view, route_current_view
from .service import ShoppingService, local_intent, explicit_current_pair, recipient_needs_clarification


_PREFIX = r"(?:(?:can|could|would) you\s+)?(?:please\s+)?"
_NEW_VIEW = re.compile(
    r"^" + _PREFIX + r"(?:"
    r"compare (?:these|this)(?:\b)|"
    r"(?:compare|what about) (?:the )?both of (?:these|those)\b|"
    r"(?:look at|(?:have|take) a look at|scan|capture|read|identify|check) (?:these|this|those|the product|the package)(?:\b)|"
    r"(?:what is|what's|what are|what about) (?:this|these)(?:\b)|"
    r"(?:which|what) (?:one )?of these\b|"
    r"(?:scan|capture)(?:[.!?]|$))", re.I)
# A direct request to compare the presented pair always needs a current view.
# Topic questions and "compare again" still refer to the selected products.
_REFERENTIAL_COMPARISON = re.compile(
    r"^(?:(?:now|please)[,\s]+)*(?:(?:can|could|would) you\s+)?"
    r"(?:(?:now|please)[,\s]+)*(?:compare|contrast) (?:them|the two)"
    r"(?: again)?(?: (?:for|by|on) (?:(?:lower|higher|less|more|total|added|saturated) )?"
    r"(?:sugars?|fats?|caffeine|proteins?|fib(?:er|re)|carbs?|carbohydrates?|sodium|salt|calories)"
    r"(?: (?:content|amounts?|levels?))?)?(?: now)?(?: please)?[.!?]*$", re.I)
_COMPARISON_TOPIC = re.compile(
    r"^(?:(?:i (?:would like|want|need) to|i['’]d like to)\s+)?" + _PREFIX +
    r"(?:compare|contrast)\s+(?:(?:the|their)\s+)?"
    r"(?:(?:total|added|saturated)\s+)?(?:sugars?|fats?|caffeine|proteins?|fib(?:er|re)|"
    r"carbs?|carbohydrates?|sodium|salt|calories)"
    r"(?:\s+(?:content|amounts?|levels?))?[.!?]*$", re.I)
_GREETING = re.compile(r"^(?:(?:hello|hi|hey)(?:[ ,]+(?:shelf\s*aware|there))?|help|what can you do)[.!?]*$", re.I)
_PRESENCE = re.compile(r"^(?:are you (?:still )?(?:there|with me|listening)|(?:can|do) you hear me)[.!?]*$", re.I)
_REPEAT_REQUEST = re.compile(
    r"\b(?:re[\s-]+peat|what (?:did|do) you (?:just )?say|"
    r"(?:say|tell me) (?:that|it|your (?:last )?(?:answer|response)) (?:once )?again|"
    r"(?:i )?(?:did(?:n['’]t| not)|could(?:n['’]t| not)) (?:hear|catch) (?:that|you))\b", re.I)
_THANKS = re.compile(r"^(?:thanks(?: a lot| so much)?|thank you(?: so much| very much)?)[.!?]*$", re.I)
_ACKNOWLEDGEMENT = re.compile(r"^(?:okay|ok|all right|alright|got it|understood|sounds good|that helps|that['’]s helpful)[.!?]*$", re.I)
_END = re.compile(r"^(?:please )?(?:stop|cancel|be quiet|stop talking|(?:end|stop) (?:this |the )?conversation|goodbye|bye(?: for now)?|that['’]s all|we['’]re done)[.!?]*$", re.I)
_PRESENTING = re.compile(r"^(?:here(?: you go| they are| it is)?|there you go|these (?:ones|two))[.!?]*$", re.I)
_NEGATED_CONTROL = re.compile(r"^(?:please )?(?:do not|don['’]t|no need to) (?:stop|repeat(?: that| it)?|say (?:that|it) again)[.!?]*$", re.I)
_BARE_CONFIRMATION = re.compile(r"^(?:yes|yeah|yep|no|nope|sure|confirm)[.!?]*$", re.I)
_POLITE_DECLINE = re.compile(r"^no[,\s]+(?:thanks|thank you)[.!?]*$", re.I)
_GOAL_QUERY = re.compile(r"^what(?:['’]s| is| are) (?:my|our|the current) (?:current |shopping |comparison )?(?:goals?|request|criterion|criteria)[.!?]*$", re.I)


def _local_words(text: str) -> str:
    """Strip conversational lead-ins for routing, retaining original provider input."""
    return re.sub(r"^(?:(?:hey|hello|hi|uh|um|well|okay|ok|so|sorry)\b[\s,.!?-]*)+", '', text,
                  flags=re.I).strip().replace('’', "'")


def _deictic_comparison(text: str) -> bool:
    if re.search(r"\b(?:do not|don't|never|no need to)\s+(?:compare|contrast)\b|"
                 r"\b(?:previous|earlier)\b|\b(?:old|last)\s+(?:ones?|products?|items?|pair|two|photo|time)\b|"
                 r"\b(?:from|shown|saw) before\b", text, re.I):
        return False
    implicit_choice = re.fullmatch(
        r"(?:(?:can|could|would) you (?:tell me )?)?(?:which|what)(?: one)?"
        r"(?:'s| is| would be)? (?:better|best)(?: option| choice)?"
        r"(?: for (?:me|us|myself|my goals?))?[.!?]*", text, re.I)
    return bool(implicit_choice or (
        re.search(r"\b(?:which|what)\b.{0,70}\b(?:better|best|prefer|recommend)\b|\bcompare\b", text, re.I)
        and re.search(r"\b(?:these|those|both|the two|two (?:products|items|cans|bottles|packages))\b", text, re.I)))


def _deictic_product_question(text: str) -> bool:
    return bool(re.search(r"^(?:does|do|is|are|has|have) (?:this|that|these|those)\b.{0,100}\b(?:sugars?|caffeine|protein|fib(?:er|re)|sodium|salt|calories|ingredients?|allergens?|peanuts?|milk|gluten)\b", text, re.I)
                or re.fullmatch(r"what(?:['’]s| is) in (?:this|that|these|those)[.!?]*", text, re.I))
_CONTEXT_ONLY = re.compile(
    r"^(?:(?:shop|shopping|this is|it's) )?(?:for )?(?:my |the )?"
    r"(?:family|household|kids|children|me|myself)[.!?]*$", re.I)
_CRITERION = re.compile(
    r"\b(?:avoid|prefer|prioriti[sz]e|focus on|looking for|i want|i need|my goal|instead|"
    r"less|more|lower|higher|fewer|least|most|cheaper|cheapest|healthier)\b|"
    r"\bcompar\w*\b.{0,80}\b(?:for|by|based on|according to)\b|"
    r"\b(?:which|what)\b.{0,60}\b(?:sugar|caffeine|protein|fiber|fibre|price|cost|taste|salt|sodium|calories)\b", re.I)


_PRODUCT_TOPIC = re.compile(
    r"\b(?:sugars?|fats?|caffeine|proteins?|fib(?:er|re)|carbs?|carbohydrates?|sodium|salt|calories|"
    r"ingredients?|sweeteners?|allergens?|peanuts?|milk|gluten|price|cost|taste|flavou?r|"
    r"sugar[ -]*free|nutrition|healthy|healthier|cheaper|organic|vegan|products?|packages?|labels?)\b", re.I)
_REQUEST_START = re.compile(
    r"^(?:which|what|why|how|does|do|did|is|are|has|have|can|could|would|should|will|"
    r"tell me|explain|summari[sz]e|compare|contrast|check|read|look)\b", re.I)
_DIRECT_GOAL = re.compile(
    r"^(?:(?:now|actually|instead|no)[,\s]+)*(?:i (?:want|need|prefer|am looking for|would like)|"
    r"i['’]m looking for|my goal (?:is|would be)|avoid|prefer|prioriti[sz]e|focus on|less|more|lower|higher|fewer|least|most)\b", re.I)


def _request_words(words: str) -> str:
    return re.sub(r'^(?:thanks|thank you)(?: so much| very much| a lot)?[,\s]+', '', words, flags=re.I)


def _clear_product_followup(words: str, products) -> bool:
    """Cheap routing hints, not speaker identification or product reasoning."""
    words = _request_words(words)
    if (_deictic_comparison(words) or _REFERENTIAL_COMPARISON.fullmatch(words) or _COMPARISON_TOPIC.fullmatch(words)
            or (_DIRECT_GOAL.search(words) and _PRODUCT_TOPIC.search(words))):
        return True
    if re.fullmatch(r"(?:why|how so|why so|tell me more|explain(?: why)?|compare again|try again|"
        r"what(?:['’]s| is) your (?:opinion|recommendation))[.!?]*", words, re.I):
        return True
    if not _REQUEST_START.search(words):
        return False
    if _PRODUCT_TOPIC.search(words) or re.search(r"\b(?:it|they|them|this|that|these|those|both|one|other|left|right)\b", words, re.I):
        return True
    return any(len(product.name.strip()) > 2 and re.search(
        r'(?<!\w)' + re.escape(product.name.strip()) + r'(?!\w)', words, re.I) for product in products)


def _portion_request(text: str) -> bool:
    """Leave exact scaling to reviewed labels, not catalog search or an LLM."""
    if re.search(r"\b(?:half|whole|full|quarter|double|twice)\b.{0,40}\b(?:bottles?|cans?|containers?|packages?|each|both)\b", text, re.I):
        return True
    return bool(re.search(
        r"\b(?:both|each|first|second|eat|drink|have|portion|for)\b.{0,25}"
        r"\d+(?:\.\d+)?\s*(?:g|grams?|ml|milliliters?|millilitres?)\b", text, re.I)
        or re.fullmatch(r"\s*\d+(?:\.\d+)?\s*(?:g|grams?|ml|milliliters?|millilitres?)(?:\s+(?:each|both))?[.!?]*", text, re.I))


class ProductSearchBrain:
    """ShoppingBrain-compatible interface without a second reasoning pipeline."""

    def __init__(self, service: ShoppingService, profile: Mapping,
                 recipient_id: str = 'self', session_id: str | None = None):
        self.service = service
        # The coordinator binds the real session ID immediately after construction.
        self.session_id = session_id
        self.recipient_id = recipient_id if recipient_id in ('self', 'household') else 'self'
        self.generation = 0
        self._lock = asyncio.Lock()
        self._last_reply: tuple[str, str] | None = None
        self._last_reply_scope: tuple | None = None
        self._comparison_request: str | None = None
        self._awaiting_view = False
        self._awaiting_recipient = False
        self._automatic_guidance: tuple[tuple, float] | None = None
        self._closed = False

    def _record(self):
        if not self.session_id:
            raise HTTPException(404, 'Shopping session not found.')
        return self.service.get(self.session_id)

    def cancel(self, *, clear_pending: bool = True):
        self.generation += 1
        # Task cancellation alone cannot invalidate a late provider completion.
        if self.session_id:
            try:
                self.service.cancel(self.session_id)
            except HTTPException as exc:
                if exc.status_code != 404:
                    raise

    def set_recipient(self, recipient_id: str):
        if recipient_id not in ('self', 'household'):
            raise ValueError('Unsupported recipient')
        record = self._record()
        if recipient_id != self.recipient_id or record.session.recipient_id != recipient_id:
            self.cancel()
            self.service.context(self.session_id, ContextRequest(recipient_id=recipient_id))
            self._last_reply = None
            self._last_reply_scope = None
            self._comparison_request = None
            self._awaiting_view = False
            self._automatic_guidance = None
        self.recipient_id = recipient_id
        self._awaiting_recipient = False

    async def close(self):
        self._closed = True
        self._last_reply = None
        self._last_reply_scope = None
        self._comparison_request = None
        self._awaiting_view = False
        self._awaiting_recipient = False
        self._automatic_guidance = None
        self.cancel()
        # HTTP clients belong to the shared service, not a voice connection.

    def snapshot(self) -> dict:
        record = self._record()
        session = record.session
        self.recipient_id = session.recipient_id
        selected = set(session.current_visual_product_ids) if record.analysis_mode == 'product' else set()
        if not selected and record.analysis_mode == 'product' and session.current_visual_product_id:
            selected.add(session.current_visual_product_id)
        products = [p for p in session.visual_products if p.id in selected]
        return {
            'session_id': session.id, 'recipient_id': session.recipient_id,
            'analysis_mode': 'product', 'evidence_scope': 'catalog_search',
            'catalog_ready': False, 'consented': session.profile.consented,
            'pair': [p.id for p in products],
            'products': [{'id': p.id, 'name': p.name, 'box': p.box.model_dump(),
                          'catalog_status': p.catalog_status,
                          'sources': [source.model_dump() for source in p.sources],
                          'verified': False} for p in products],
            'shopping_goals': list(session.profile.goals) if session.recipient_id == 'self' else [],
            'comparison_request': self._comparison_request,
            'portions': {}, 'goals': None, 'comparison': None, 'pending_action': None,
            'cart': [{'id': item.id, 'product_id': item.product_id, 'name': item.product_name,
                      'recipient_id': item.recipient_id, 'quantity': item.quantity,
                      'confirmed_at': item.confirmed_at} for item in session.cart],
            'generation': self.generation,
            'provider_failure': dict(record.provider_failure) if record.provider_failure else None,
        }

    def _sync_reply_scope(self, record):
        if self._last_reply and (self._last_reply[0] != record.session.recipient_id
                or (self._last_reply_scope is not None and self._last_reply_scope != record.product_memory_scope)):
            self._last_reply = None
            self._last_reply_scope = None

    def _reply(self, text: str, status: str = 'ready', *, events=None, remember=True) -> dict:
        record = self._record()
        self._sync_reply_scope(record)
        snapshot = self.snapshot()
        # A failed capture alone does not prove a new product, so Repeat can
        # recover the last useful answer. Actual changed identities invalidate
        # it even when their new lookup/wording fails. Never keep a per-item cache.
        if remember and status == 'ready':
            self._last_reply = (snapshot['recipient_id'], text)
            self._last_reply_scope = record.product_memory_scope
        if status == 'needs_image':
            self._awaiting_view = True
        return {'reply': text, 'status': status, 'snapshot': snapshot, 'events': events or []}

    def _clear_selection(self):
        record = self._record()
        self.service.cancel(self.session_id)
        record.session.current_visual_product_id = None
        record.session.current_visual_product_ids = []
        record.session.current_product_id = None
        record.product_reply = None
        record.last_reply_context = None
        record.latest = None
        self._awaiting_view = True

    @staticmethod
    def _frame_data(frame) -> str:
        # Transport validates the decoded dimensions and current request ID. Use
        # the existing 4 MiB TurnRequest gate, not the reviewed-detector's 2 MiB cap.
        stamp = datetime.fromisoformat(frame.received_at.replace('Z', '+00:00'))
        if stamp.utcoffset() is None or not -2 <= (datetime.now(timezone.utc) - stamp).total_seconds() <= .75:
            raise ValueError('Stale frame')
        if any(type(n) is not int or not 1 <= n <= 4096 for n in (frame.width, frame.height)):
            raise ValueError('Invalid frame dimensions')
        return frame.image_data_url

    def automatic_question(self) -> str:
        question = 'Compare these products for my current shopping goal.'
        if self._comparison_request:
            question = 'Compare these products according to my current request: ' + self._comparison_request
        return question

    async def respond_automatically(self, frame_provider=None) -> dict:
        result = await self.respond(self.automatic_question(), frame_provider, _automatic=True)
        # Scene changes may repeatedly show no readable package. Keep the first
        # actionable guidance, but don't narrate the same failure on every scene.
        # A new failure kind/provider status remains audible; no retry is hidden
        # longer than 45 seconds, and manual questions/useful replies reset this.
        if result.get('status') in ('needs_image', 'needs_clarification', 'unknown'):
            diagnostic = result.get('snapshot', {}).get('provider_failure') or {}
            key = (result.get('status'), result.get('reply'), tuple(sorted(diagnostic.items())))
            now = monotonic()
            prior = self._automatic_guidance
            if prior and prior[0] == key and now - prior[1] < 45:
                result['silent'] = True
            else:
                self._automatic_guidance = (key, now)
        return result

    async def respond(self, text: str,
                      frame_provider: Callable[[], Awaitable[object | None]] | None = None,
                      *, _automatic: bool = False) -> dict:
        async with self._lock:
            if self._closed:
                return self._reply('Conversation has ended.', 'cancelled', remember=False)
            record = self._record()
            self._sync_reply_scope(record)
            record.provider_failure = None
            if not record.session.profile.consented:
                return self._reply('Review and consent to your shopping profile before starting.', 'needs_consent')
            if not isinstance(text, str) or not text.strip() or len(text) > 1500:
                return self._reply('Please ask a short shopping question.', 'needs_clarification')
            text = text.strip()
            if not _automatic:
                self._automatic_guidance = None
            words = _local_words(text)
            intent = local_intent(words or text)
            context_words = re.sub(r'^(?:(?:no|actually)[,\s]+)?(?:this is )?not for (?:me|myself|(?:my |the )?(?:family|household|kids|children))[,;.\s-]+', '', words, flags=re.I)
            context_words = re.sub(r'^(?:no|actually)[,\s]+', '', context_words, flags=re.I)
            context_only = bool(_CONTEXT_ONLY.fullmatch(context_words))
            recipient = ('household' if re.search(r'\b(?:family|household|kids|children)\b', context_words, re.I) else 'self') if context_only else intent.recipient_id
            if recipient:
                self.set_recipient(recipient)
            else:
                self.recipient_id = record.session.recipient_id
            generation = self.generation
            ending_words = re.sub(r'^(?:thanks|thank you)(?: so much| very much| a lot)?[,\s]+', '', words, flags=re.I)
            if _END.fullmatch(ending_words or text):
                self.cancel()
                return {**self._reply('Stopped.', remember=False), 'end_conversation': True}
            if _THANKS.fullmatch(words or text):
                return self._reply("You're welcome.", remember=False)
            if (_ACKNOWLEDGEMENT.fullmatch(text) or _ACKNOWLEDGEMENT.fullmatch(words)
                    or _NEGATED_CONTROL.fullmatch(words or text) or _POLITE_DECLINE.fullmatch(words)):
                return self._reply('Okay.', remember=False)
            if _PRESENCE.fullmatch(words):
                return self._reply("Yes, I'm here.", remember=False)
            if (_GREETING.fullmatch(text) or _GREETING.fullmatch(words)
                    or (words.rstrip('.!?').lower() in ('shelf aware', 'shelfaware', 'there')
                        and re.match(r'^(?:hey|hello|hi)\b', text, re.I))):
                greeting = ("I can look at the products you are holding, compare them for your shopping goal, and answer follow-up questions."
                            if re.fullmatch(r'(?:help|what can you do)[.!?]*', words, re.I)
                            else "I'm listening. What would you like to check?")
                return self._reply(greeting,
                                   remember=self._last_reply is None)
            if recipient_needs_clarification(words):
                self._awaiting_recipient = True
            if self._awaiting_recipient:
                return self._reply('Is this for you or your household? Confirm that before I apply personal goals.',
                                   'needs_clarification', remember=False)
            if _GOAL_QUERY.fullmatch(words):
                if self._comparison_request:
                    return self._reply('Your current comparison request is: ' + self._comparison_request)
                if self.recipient_id == 'household':
                    return self._reply('Personal goals are off for household shopping. Tell me what you want to compare.')
                goals = list(record.session.profile.goals)
                return self._reply('Your shopping goal is: ' + '; '.join(goals) + '.' if goals else
                                   'You have not selected a shopping goal yet. Tell me what you want to compare.')
            if _BARE_CONFIRMATION.fullmatch(words or text):
                return self._reply('Tell me what you would like to compare or change.', 'needs_clarification', remember=False)
            if context_only:
                return self._reply('Shopping for your household. Personal goals are off.' if self.recipient_id == 'household'
                                   else 'Shopping for you. Your selected goals apply.')
            if (intent.intent == 'repeat' or re.fullmatch(r'(?:pardon(?: me)?|come again)[.!?]*', words, re.I)
                    or (_REPEAT_REQUEST.search(words) and not re.search(r"\b(?:do not|don't|never)\b", words, re.I))):
                if self._last_reply and self._last_reply[0] == self.recipient_id:
                    return self._reply(self._last_reply[1])
                return self._reply('I do not have an earlier answer to repeat in this shopping context yet.',
                                   'needs_clarification', remember=False)
            if _portion_request(text):
                return self._reply('I can compare the products using cited listings, but portion calculations need reviewed package amounts. I will not guess those amounts.', 'needs_review')
            comparison_topic = bool(_COMPARISON_TOPIC.fullmatch(words))
            direct_goal = bool(_DIRECT_GOAL.search(_request_words(words)) and _PRODUCT_TOPIC.search(words))

            selected = {p.id for p in record.session.visual_products} & set(record.session.current_visual_product_ids)
            if not selected and record.session.current_visual_product_id:
                selected = {p.id for p in record.session.visual_products if p.id == record.session.current_visual_product_id}
            if record.analysis_mode != 'product':
                selected = set()
            new_view = (bool(_NEW_VIEW.search(words) or _REFERENTIAL_COMPARISON.fullmatch(words))
                or (not selected and (_deictic_comparison(words) or _deictic_product_question(words)
                    or comparison_topic))
                or (len(selected) < 2 and explicit_current_pair(words))
                or (self._awaiting_view and bool(_PRESENTING.fullmatch(words))))
            # Known controls and selected-product follow-ups never pay for a
            # second reasoning call. Only a missing view plus ambiguous wording
            # benefits from semantic routing; it cannot change the user's data.
            if not _automatic and not selected and not new_view and not direct_goal and intent.intent in ('question', 'compare'):
                route = await route_missing_view(self.service.providers, text, awaiting_view=self._awaiting_view)
                if generation != self.generation:
                    return self._reply('Stopped.', 'cancelled', remember=False)
                if route:
                    if route.route == 'inspect_new_view':
                        new_view = True
                    elif route.route == 'ignore':
                        return {**self._reply('No product question detected.', 'ignored', remember=False), 'silent': True}
                    elif route.route == 'social':
                        return self._reply('Okay.', remember=False)
                    elif route.route == 'follow_up':
                        return self._reply('I do not have a current product view to answer that yet. Ask me to look at what you are holding.',
                                           'needs_image', remember=False)
                    else:
                        return self._reply('What would you like to know about the products?', 'needs_clarification', remember=False)
            # A parsed recipient correction can precede a normal question.
            # Strip that clause for relevance matching only; service still gets
            # the original utterance and the affirmative recipient above.
            followup_words = (re.sub(
                r"^(?:(?:shop|shopping|this is|it's) )?(?:for )?(?:my |the )?"
                r"(?:family|household|kids|children|me|myself)[,;.!?\s]+", '', context_words, flags=re.I)
                if recipient else words)
            if (not _automatic and selected and not new_view and intent.intent in ('question', 'compare')
                    and not _clear_product_followup(followup_words, [p for p in record.session.visual_products if p.id in selected])):
                route = await route_current_view(self.service.providers, text)
                if generation != self.generation:
                    return self._reply('Stopped.', 'cancelled', remember=False)
                if route and route.route == 'inspect_new_view':
                    new_view = True
                elif route and route.route == 'follow_up':
                    pass
                elif route and route.route == 'ignore':
                    return {**self._reply('No product question detected.', 'ignored', remember=False), 'silent': True}
                elif route and route.route == 'social':
                    return self._reply('Okay.', remember=False)
                else:
                    return self._reply('Did you want to ask something about these products?',
                                       'needs_clarification', remember=False)
            if (not _automatic and (_CRITERION.search(text) or comparison_topic)
                    and (new_view or comparison_topic or direct_goal or _REQUEST_START.search(_request_words(words)))):
                # Only an accepted direct request changes the comparison criterion.
                # Narrative fragments mentioning products/preferences are not goals.
                self._comparison_request = text[:500]
            image = None
            if new_view:
                self._clear_selection()
                if frame_provider is None:
                    return self._reply('The camera is unavailable. Connect the glasses, then ask me to look again.', 'needs_image')
                try:
                    frame = await frame_provider()
                    if generation != self.generation:
                        return self._reply('Stopped.', 'cancelled', remember=False)
                    image = self._frame_data(frame)
                    # This also rejects invalid image signatures/base64 before a provider call.
                    TurnRequest(text=text, client_turn_id='frame-check', analysis_mode='product', image_data_url=image)
                except asyncio.CancelledError:
                    self.cancel()
                    raise
                except (ValueError, TypeError, AttributeError, RuntimeError, OSError):
                    if generation != self.generation:
                        return self._reply('Stopped.', 'cancelled', remember=False)
                    return self._reply('The photo did not come through, so I cannot inspect it yet. You can ask me to look again.', 'needs_image')

            if generation != self.generation:
                return self._reply('Stopped.', 'cancelled', remember=False)
            # A new ID for each accepted utterance preserves REST idempotency.
            request = TurnRequest(text=text, client_turn_id='voice-' + str(uuid4()),
                                  analysis_mode='product', recipient_id=self.recipient_id,
                                  image_data_url=image, conversation_request=self._comparison_request)
            try:
                result = await self.service.turn(self.session_id, request)
            except asyncio.CancelledError:
                self.cancel()
                raise
            if generation != self.generation or result.status == 'cancelled':
                return self._reply('Stopped.', 'cancelled', remember=False)
            if result.status == 'needs_image' and not image:
                # A spoken follow-up before the first photo used to leak the
                # legacy button instructions into the continuous conversation.
                return self._reply('I have not seen the products yet. Hold their front names steady for a photo, or say compare these.', 'needs_image',
                                   events=[event.model_dump(mode='json') for event in result.events])
            if result.status == 'ready':
                self._awaiting_view = False
                self._automatic_guidance = None
            return self._reply(result.reply, result.status,
                               events=[event.model_dump(mode='json') for event in result.events])
