"""Small conversational references, never a second source of product facts."""
from __future__ import annotations

import re

MAX_EXCHANGES = 3
MAX_MEMORY_CHARS = 1800


def compact_exchanges(value) -> list[dict[str, str]]:
    """Keep recent complete bounded exchanges; reject arbitrary nested context."""
    if not isinstance(value, list):
        return []
    result = []
    for item in value[-MAX_EXCHANGES:]:
        if not isinstance(item, dict) or set(item) != {'question', 'answer'}:
            continue
        question, answer = item['question'], item['answer']
        if not isinstance(question, str) or not isinstance(answer, str) or not question.strip() or not answer.strip():
            continue
        # Wording responses already have a 600-character production limit. Do
        # not retain a chopped answer whose qualifying clause might be missing.
        if len(answer) > 600:
            continue
        result.append({'question': question.strip()[:300], 'answer': answer.strip()})
    while sum(len(item['question']) + len(item['answer']) for item in result) > MAX_MEMORY_CHARS:
        result.pop(0)
    return result


def product_scope(recipient_id: str, products) -> tuple:
    """Identity-based, not UUID/photo/left-right based; unknown changes clear it."""
    def identity(product):
        return tuple(' '.join((getattr(product, key) or '').casefold().split())
                     for key in ('brand', 'name', 'variant', 'package_size'))
    return recipient_id, tuple(sorted(identity(product) for product in products))


def without_old_positions(exchanges) -> list[dict[str, str]]:
    """A fresh photo may reverse positions without changing product identities.

    Preserve named-product conversation, but never replay a positional reference
    as though it described the current photo. Current boxes alone define position.
    """
    return [entry for entry in compact_exchanges(exchanges)
            if not re.search(r'\b(?:left|right)(?:-hand)?\b', entry['question'] + ' ' + entry['answer'], re.I)]
