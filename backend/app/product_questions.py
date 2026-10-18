"""Narrow question routing when one pictured product lacks catalog evidence."""
from __future__ import annotations

import re


def _words(text: str) -> str:
    text = text.strip().replace('’', "'")
    return re.sub(r"^(?:(?:hey|hello|hi|uh|um|well|okay|ok|so|sorry)\b[\s,.!?-]*)+", '', text, flags=re.I)


def visual_identification_question(text: str) -> bool:
    return bool(re.fullmatch(
        r"(?:(?:can|could|would) you (?:tell me )?)?(?:please )?(?:"
        r"what (?:are|is) (?:these|those|this|that)(?: (?:two|products?|items?|cans?|bottles?))?(?: called)?|"
        r"what (?:products?|items?|drinks?) (?:are these|am i holding)|"
        r"what am i holding|identify (?:these|those|this|that)(?: products?| items?)?)[.!?]*",
        _words(text), re.I))


_COMPARATIVE_OR_UNRESOLVED = re.compile(
    r'\b(?:compar\w*|better|best|worse|worst|prefer|recommend|choose|pick|versus|vs|than|between|'
    r'both|each|other|less|more|lower|higher|least|most|safer|safest|healthier|healthiest|preferred|winner|wins?|differences?|same|equal|tie|'
    r'left|right)\b|\b(?:this|that|these|those)\s+(?:ones?|products?|items?|cans?|bottles?|packages?)\b', re.I)


def grounded_named_target(text: str, products):
    """Resolve one explicit unique name/variant; never default to current/right.

    A shared brand, two named items, pronoun-only reference, or comparison cannot
    take this path. Normalization tolerates Sugarfree versus Sugar Free spelling.
    """
    if _COMPARATIVE_OR_UNRESOLVED.search(text):
        return None
    if not re.match(r'^(?:(?:please|can you|could you|would you)\s+)*(?:what|which|does|do|is|are|has|have|tell me|explain)\b', _words(text), re.I):
        return None
    words = ' '.join(re.findall(r'[^\W_]+', text.casefold()))
    brands = {' '.join(re.findall(r'[^\W_]+', (p.brand or '').casefold())) for p in products}
    aliases = {}
    for index, product in enumerate(products):
        name = ' '.join(re.findall(r'[^\W_]+', product.name.casefold()))
        brand = ' '.join(re.findall(r'[^\W_]+', (product.brand or '').casefold()))
        candidates = [name, ' '.join(re.findall(r'[^\W_]+', (product.variant or '').casefold()))]
        if brand and name.startswith(brand + ' '):
            candidates.append(name[len(brand):].strip())
        for alias in candidates:
            if len(alias) < 3 or alias in brands:
                continue
            aliases.setdefault(alias, set()).add(index)
    matches = []
    for alias, owners in aliases.items():
        pattern = r'(?<!\w)' + r'\s*'.join(re.escape(word) for word in alias.split()) + r'(?!\w)'
        matches.extend((match.start(), match.end(), owners) for match in re.finditer(pattern, words))
    selected = []
    for match in sorted(matches, key=lambda value: -(value[1] - value[0])):
        if not any(match[0] < prior[1] and prior[0] < match[1] for prior in selected):
            selected.append(match)
    owners = set().union(*(match[2] for match in selected)) if selected else set()
    if len(owners) != 1:
        return None
    product = products[next(iter(owners))]
    return product if product.catalog_status == 'grounded' and product.sources else None


def spoken_product_name(product) -> str:
    name = ' '.join(product.name.split())
    brand = ' '.join((product.brand or '').split())
    if brand and brand.casefold() not in name.casefold():
        name = brand + ' ' + name
    return name if len(name) <= 100 else name[:97] + '...'
