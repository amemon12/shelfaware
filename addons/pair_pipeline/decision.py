"""Isolated reviewed-label comparison reference; not connected to the v1 app.

No model calls, clinical rules, database lookups, density assumptions, or scoring.
Numbers are returned as decimal strings. A candidate fits the confirmed goals;
it is not a declaration of medical or allergen safety.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, DecimalException, ROUND_HALF_UP, localcontext
from typing import Literal, Mapping, Sequence

Number = str | int | float | Decimal
NUTRIENTS = frozenset({"total_carbohydrate_g", "total_sugars_g", "added_sugars_g",
                      "fiber_g", "protein_g", "sodium_mg", "saturated_fat_g"})


class ValidationError(ValueError):
    """A safe reason code; contains no report text or credential values."""


@dataclass(frozen=True)
class Quantity:
    value: Number
    unit: Literal["g", "mL"]


@dataclass(frozen=True)
class ConstraintCheck:
    status: Literal["passed", "failed", "unknown"]
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Product:
    product_id: str
    name: str
    category: str
    market: str
    basis: Quantity
    nutrients: Mapping[str, Number | None]
    evidence_ids: tuple[str, ...]
    identity_reviewed: bool
    label_reviewed: bool
    label_revision: int = 1
    preparation: str = "as_sold"
    label_regime: str = "US"
    constraint_checks: Mapping[str, ConstraintCheck] = field(default_factory=dict)


@dataclass(frozen=True)
class Portion:
    product_id: str
    recipient_id: str
    quantity: Quantity
    label_revision: int
    confirmed: bool
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class Goal:
    nutrient: str
    comparator: Literal["min", "max", "display"]


@dataclass(frozen=True)
class GoalSet:
    recipient_id: str
    goals: tuple[Goal, ...]
    consented: bool
    confirmed: bool
    evidence_ids: tuple[str, ...]
    revision: int = 1
    hard_constraint_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReviewedNetQuantity:
    product_id: str
    label_revision: int
    quantity: Quantity
    reviewed: bool
    evidence_ids: tuple[str, ...]


def _decimal(value: Number, *, positive: bool = False) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValidationError("invalid_number")
    if isinstance(value, str) and (len(value) > 40 or not value.strip()):
        raise ValidationError("invalid_number")
    try:
        number = Decimal(str(value))
        if not number.is_finite() or number < 0 or number > Decimal("1000000000"):
            raise ValidationError("invalid_number")
        if positive and number <= 0:
            raise ValidationError("quantity_must_be_positive")
        if number and number.adjusted() < -12:
            raise ValidationError("unsupported_number_precision")
        return number
    except DecimalException as error:
        raise ValidationError("invalid_number") from error


def _id(value: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise ValidationError("invalid_identity_or_evidence")


def _evidence(values: tuple[str, ...]) -> None:
    if not values:
        raise ValidationError("evidence_required")
    for value in values:
        _id(value)


def _quantity(quantity: Quantity) -> Decimal:
    if quantity.unit not in ("g", "mL"):
        raise ValidationError("unsupported_quantity_unit")
    return _decimal(quantity.value, positive=True)


def _text(value: Decimal) -> str:
    rendered = format(value, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def _display_value(value: Decimal) -> Decimal:
    # A transparent presentation resolution, not a clinical significance cutoff.
    with localcontext() as context:
        context.prec = 60
        return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _label_values(product: Product) -> dict[str, Decimal | None]:
    """Reject grossly inconsistent US declared facts before portion scaling.

    One gram allows label rounding; it is neither clinical significance nor
    permission to repair an extracted value. Volume is never treated as mass.
    """
    values = {name: None if product.nutrients.get(name) is None
              else _decimal(product.nutrients[name]) for name in sorted(NUTRIENTS)}
    tolerance = Decimal("1")
    for component, total in (("added_sugars_g", "total_sugars_g"),
                             ("total_sugars_g", "total_carbohydrate_g"),
                             ("added_sugars_g", "total_carbohydrate_g"),
                             ("fiber_g", "total_carbohydrate_g")):
        if values[component] is not None and values[total] is not None and values[component] > values[total] + tolerance:
            raise ValidationError("inconsistent_us_nutrient_components")
    if product.basis.unit == "g":
        mass = _quantity(product.basis)
        for nutrient, value in values.items():
            unit_multiplier = Decimal("1000") if nutrient == "sodium_mg" else Decimal("1")
            if value is not None and value > (mass + tolerance) * unit_multiplier:
                raise ValidationError("nutrient_exceeds_food_mass")
    return values


def whole_package_portion(product: Product, net: ReviewedNetQuantity, recipient_id: str,
                          *, count: Number = "1", confirmed: bool = False) -> Portion:
    """Resolve an explicitly reviewed container quantity; never infer its density.

    Raises ValidationError for invalid binding/units. Returned portion still needs
    consumption confirmation; reviewing a bottle's size is not choosing to drink it.
    """
    if net.reviewed is not True:
        raise ValidationError("net_quantity_needs_review")
    if net.product_id != product.product_id or net.label_revision != product.label_revision:
        raise ValidationError("stale_or_wrong_container")
    if net.quantity.unit != product.basis.unit:
        raise ValidationError("unit_mismatch_no_density_inference")
    _evidence(net.evidence_ids)
    with localcontext() as context:
        context.prec = 40
        amount = _quantity(net.quantity) * _decimal(count, positive=True)
    _decimal(amount, positive=True)
    return Portion(product.product_id, recipient_id, Quantity(_text(amount), net.quantity.unit),
                   product.label_revision, confirmed, net.evidence_ids)


def compare_pair(products: Sequence[Product], portions: Sequence[Portion], goals: GoalSet,
                 recipient_id: str, *, pair_confirmed: bool = False) -> dict:
    """Return facts plus candidate/tradeoff/tie/needs_clarification.

    Product IDs, rather than left/right positions, bind portions and the result.
    Constraints are upstream reviewed checks; this module implements no ontology.
    Invalid inputs fail closed with a reason code and no candidate.
    """
    result = {"status": "needs_clarification", "candidate_product_id": None,
              "recipient_id": recipient_id, "goal_revision": goals.revision,
              "goal_evidence_ids": list(goals.evidence_ids),
              "numeric_policy": "decimal_scaling; 0.01 nutrient-unit display resolution; HALF_UP; label values are approximate",
              "facts": [], "per_goal": [], "constraint_statuses": [],
              "reason_codes": [], "missing_fields": []}
    try:
        if pair_confirmed is not True:
            raise ValidationError("pair_needs_confirmation")
        if recipient_id not in ("self", "household") or goals.recipient_id != recipient_id:
            raise ValidationError("recipient_goal_mismatch")
        if goals.consented is not True or goals.confirmed is not True:
            raise ValidationError("goals_need_consent_and_confirmation")
        if not isinstance(goals.revision, int) or isinstance(goals.revision, bool) or goals.revision < 1:
            raise ValidationError("invalid_goal_revision")
        _evidence(goals.evidence_ids)
        if len(products) != 2 or len(portions) != 2:
            raise ValidationError("exactly_two_products_and_portions_required")
        identities = [p.product_id for p in products]
        if len(set(identities)) != 2:
            raise ValidationError("duplicate_product_identity")
        if len({p.product_id for p in portions}) != 2 or set(identities) != {p.product_id for p in portions}:
            raise ValidationError("portion_identity_mismatch")
        if products[0].category != products[1].category:
            raise ValidationError("category_mismatch")
        if products[0].market != products[1].market:
            raise ValidationError("market_mismatch")
        if products[0].basis.unit != products[1].basis.unit:
            raise ValidationError("mixed_quantity_dimensions")
        if len(goals.goals) > 8 or len({g.nutrient for g in goals.goals}) != len(goals.goals):
            raise ValidationError("duplicate_or_excessive_goals")
        for goal in goals.goals:
            if goal.nutrient not in NUTRIENTS or goal.comparator not in ("min", "max", "display"):
                raise ValidationError("unsupported_goal")
        if len(set(goals.hard_constraint_ids)) != len(goals.hard_constraint_ids):
            raise ValidationError("duplicate_constraint")
        for constraint in goals.hard_constraint_ids:
            _id(constraint)

        values_by_id: dict[str, dict[str, Decimal | None]] = {}
        eligible: dict[str, bool] = {}
        unknown_constraint = False
        portions_by_id = {p.product_id: p for p in portions}
        with localcontext() as context:
            context.prec = 40
            for product in products:
                for value in (product.product_id, product.name, product.category, product.market):
                    _id(value)
                if product.identity_reviewed is not True or product.label_reviewed is not True:
                    raise ValidationError("product_needs_review")
                if product.market != "US" or product.label_regime != "US" or product.preparation != "as_sold":
                    raise ValidationError("unsupported_market_or_preparation")
                if not isinstance(product.label_revision, int) or isinstance(product.label_revision, bool) or product.label_revision < 1:
                    raise ValidationError("invalid_label_revision")
                _evidence(product.evidence_ids)
                if set(product.nutrients) - NUTRIENTS:
                    raise ValidationError("unsupported_nutrient")
                portion = portions_by_id[product.product_id]
                if portion.recipient_id != recipient_id or portion.confirmed is not True:
                    raise ValidationError("portion_needs_recipient_confirmation")
                if portion.label_revision != product.label_revision:
                    raise ValidationError("stale_portion_label")
                if portion.quantity.unit != product.basis.unit:
                    raise ValidationError("unit_mismatch_no_density_inference")
                _evidence(portion.evidence_ids)
                factor = _quantity(portion.quantity) / _quantity(product.basis)
                declared = _label_values(product)
                nutrients = {name: None if value is None else _display_value(value * factor)
                             for name, value in declared.items()}
                values_by_id[product.product_id] = nutrients
                result["facts"].append({"product_id": product.product_id, "name": product.name,
                    "label_revision": product.label_revision,
                    "portion": {"value": _text(_quantity(portion.quantity)), "unit": portion.quantity.unit},
                    "basis": {"value": _text(_quantity(product.basis)), "unit": product.basis.unit},
                    "nutrients": {k: None if v is None else _text(v) for k, v in nutrients.items()},
                    "evidence_ids": list(dict.fromkeys(product.evidence_ids + portion.evidence_ids))})
                statuses = []
                for constraint in goals.hard_constraint_ids:
                    check = product.constraint_checks.get(constraint, ConstraintCheck("unknown"))
                    if check.status not in ("passed", "failed", "unknown"):
                        raise ValidationError("invalid_constraint_status")
                    if check.status != "unknown":
                        _evidence(check.evidence_ids)
                    statuses.append({"constraint_id": constraint, "status": check.status,
                                     "evidence_ids": list(check.evidence_ids)})
                overall = ("unknown" if any(s["status"] == "unknown" for s in statuses)
                           else "failed" if any(s["status"] == "failed" for s in statuses) else "passed")
                result["constraint_statuses"].append({"product_id": product.product_id,
                                                      "status": overall, "checks": statuses})
                eligible[product.product_id] = overall == "passed"
                unknown_constraint |= overall == "unknown"

        a, b = identities
        better = {a: False, b: False}
        preferences = [g for g in goals.goals if g.comparator != "display"]
        for goal in goals.goals:
            av, bv = values_by_id[a][goal.nutrient], values_by_id[b][goal.nutrient]
            relation = "unknown" if av is None or bv is None else "equal" if av == bv else "a_lower" if av < bv else "b_lower"
            result["per_goal"].append({"nutrient": goal.nutrient, "comparator": goal.comparator,
                "a_product_id": a, "b_product_id": b,
                "a": None if av is None else _text(av), "b": None if bv is None else _text(bv), "relation": relation})
            if goal.comparator == "display":
                continue
            if relation == "unknown":
                result["missing_fields"].extend(f"{pid}.{goal.nutrient}" for pid, val in ((a, av), (b, bv)) if val is None)
            elif av != bv:
                winner = a if (av < bv) == (goal.comparator == "min") else b
                better[winner] = True
        if unknown_constraint:
            result["reason_codes"].append("constraint_unknown")
        if result["missing_fields"]:
            result["reason_codes"].append("selected_goal_missing_values")
        if not preferences:
            result["reason_codes"].append("display_only_choose_a_preference")
        if result["reason_codes"]:
            return result
        allowed = [pid for pid in identities if eligible[pid]]
        if not allowed:
            result["reason_codes"] = ["both_products_fail_explicit_constraints"]
        elif len(allowed) == 1:
            result.update(status="candidate", candidate_product_id=allowed[0],
                          reason_codes=["other_product_fails_explicit_constraint"])
        elif better[a] and better[b]:
            result.update(status="tradeoff", reason_codes=["goals_conflict_no_automatic_priority"])
        elif not better[a] and not better[b]:
            result.update(status="tie", reason_codes=["equal_declared_values_for_selected_goals"])
        else:
            result.update(status="candidate", candidate_product_id=a if better[a] else b,
                          reason_codes=["pareto_fit_for_confirmed_goals_and_portions"])
        return result
    except (ValidationError, DecimalException) as error:
        result.update(status="needs_clarification", candidate_product_id=None,
                      facts=[], per_goal=[], constraint_statuses=[],
                      reason_codes=[str(error) if isinstance(error, ValidationError) else "invalid_arithmetic"])
        return result
