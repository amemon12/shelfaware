"""Synthetic facts only. No network or app/backend imports."""
import json
import unittest
from dataclasses import replace
from decimal import Decimal

from addons.pair_pipeline.decision import (ConstraintCheck, Goal, GoalSet, Portion, Product,
    Quantity, ReviewedNetQuantity, ValidationError, compare_pair, whole_package_portion)


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.a = Product("a", "Synthetic A", "drink", "US", Quantity("240", "mL"),
                         {"total_carbohydrate_g": "12", "sodium_mg": "20", "fiber_g": "1"},
                         ("label-a",), True, True)
        self.b = Product("b", "Synthetic B", "drink", "US", Quantity("100", "mL"),
                         {"total_carbohydrate_g": "6", "sodium_mg": "10", "fiber_g": "1"},
                         ("label-b",), True, True)
        self.pa = Portion("a", "self", Quantity("500", "mL"), 1, True, ("portion-a",))
        self.pb = Portion("b", "self", Quantity("330", "mL"), 1, True, ("portion-b",))
        self.goals = GoalSet("self", (Goal("sodium_mg", "min"),), True, True, ("goal-confirmation",))

    def compare(self, a=None, b=None, pa=None, pb=None, goals=None, recipient="self", confirmed=True):
        return compare_pair((a or self.a, b or self.b), (pa or self.pa, pb or self.pb),
                            goals or self.goals, recipient, pair_confirmed=confirmed)

    def test_separate_whole_bottle_volumes_scale_without_mass_conversion(self):
        result = self.compare()
        self.assertEqual(result["status"], "candidate")
        self.assertEqual(result["candidate_product_id"], "b")
        self.assertEqual(result["facts"][0]["nutrients"]["total_carbohydrate_g"], "25")
        self.assertEqual(result["facts"][1]["nutrients"]["total_carbohydrate_g"], "19.8")
        self.assertEqual(result["facts"][0]["portion"], {"value": "500", "unit": "mL"})
        json.dumps(result, allow_nan=False)

    def test_whole_package_requires_review_and_exact_identity(self):
        net = ReviewedNetQuantity("a", 1, Quantity("500", "mL"), True, ("net-a",))
        portion = whole_package_portion(self.a, net, "self", count="0.5", confirmed=True)
        self.assertEqual(portion.quantity, Quantity("250", "mL"))
        for changed in (replace(net, reviewed=False), replace(net, product_id="b"),
                        replace(net, label_revision=2), replace(net, quantity=Quantity(500, "g"))):
            with self.subTest(net=changed), self.assertRaises(ValidationError):
                whole_package_portion(self.a, changed, "self")

    def test_container_review_does_not_confirm_consumption(self):
        net = ReviewedNetQuantity("a", 1, Quantity("500", "mL"), True, ("net-a",))
        portion = whole_package_portion(self.a, net, "self")
        self.assertEqual(self.compare(pa=portion)["status"], "needs_clarification")

    def test_gram_scaling(self):
        a = replace(self.a, basis=Quantity(40, "g"))
        b = replace(self.b, basis=Quantity(50, "g"))
        result = self.compare(a=a, b=b, pa=replace(self.pa, quantity=Quantity(60, "g")),
                              pb=replace(self.pb, quantity=Quantity(60, "g")))
        self.assertEqual(result["facts"][0]["nutrients"]["total_carbohydrate_g"], "18")
        self.assertEqual(result["facts"][1]["nutrients"]["total_carbohydrate_g"], "7.2")

    def test_mixed_units_and_no_density_inference(self):
        for kwargs in ({"a": replace(self.a, basis=Quantity(240, "g"))},
                       {"pa": replace(self.pa, quantity=Quantity(500, "g"))},
                       {"pa": replace(self.pa, quantity=Quantity(500, "fl_oz"))}):
            with self.subTest(kwargs=kwargs):
                self.assertEqual(self.compare(**kwargs)["status"], "needs_clarification")

    def test_nonfinite_negative_zero_and_boolean_inputs_fail_closed(self):
        for value in ("NaN", "Infinity", "-1", "0", float("nan"), True):
            with self.subTest(value=value):
                result = self.compare(pa=replace(self.pa, quantity=Quantity(value, "mL")))
                self.assertIsNone(result["candidate_product_id"])
                self.assertEqual(result["status"], "needs_clarification")
        for value in ("NaN", "-1", True):
            self.assertEqual(self.compare(a=replace(self.a, nutrients={"sodium_mg": value}))["status"], "needs_clarification")

    def test_conflicting_goals_do_not_create_weighted_winner(self):
        a = replace(self.a, nutrients={"sodium_mg": "5", "fiber_g": "0"})
        goals = replace(self.goals, goals=(Goal("sodium_mg", "min"), Goal("fiber_g", "max")))
        result = self.compare(a=a, goals=goals)
        self.assertEqual(result["status"], "tradeoff")
        self.assertIsNone(result["candidate_product_id"])

    def test_equal_scaled_values_tie(self):
        a = replace(self.a, basis=Quantity(100, "mL"), nutrients={"sodium_mg": "10"})
        result = self.compare(a=a, pa=replace(self.pa, quantity=Quantity(330, "mL")))
        self.assertEqual(result["status"], "tie")

    def test_sub_display_resolution_does_not_manufacture_a_winner(self):
        a = replace(self.a, basis=Quantity(100, "mL"), nutrients={"sodium_mg": "10.001"})
        result = self.compare(a=a, pa=replace(self.pa, quantity=Quantity(100, "mL")),
                              pb=replace(self.pb, quantity=Quantity(100, "mL")))
        self.assertEqual(result["status"], "tie")
        self.assertEqual(result["goal_evidence_ids"], ["goal-confirmation"])
        self.assertIn("0.01", result["numeric_policy"])

    def test_missing_selected_goal_never_gets_a_winner(self):
        result = self.compare(a=replace(self.a, nutrients={"fiber_g": "20"}))
        self.assertEqual(result["status"], "needs_clarification")
        self.assertEqual(result["missing_fields"], ["a.sodium_mg"])
        self.assertIsNone(result["candidate_product_id"])

    def test_display_only_does_not_invent_preference(self):
        result = self.compare(goals=replace(self.goals, goals=(Goal("total_carbohydrate_g", "display"),)))
        self.assertEqual(result["status"], "needs_clarification")
        self.assertTrue(result["facts"])
        self.assertIsNone(result["candidate_product_id"])

    def test_missing_display_only_does_not_block_known_selected_goal(self):
        goals = replace(self.goals, goals=self.goals.goals + (Goal("added_sugars_g", "display"),))
        self.assertEqual(self.compare(goals=goals)["status"], "candidate")

    def test_unconfirmed_unconsented_and_household_isolation(self):
        for goals in (replace(self.goals, consented=False), replace(self.goals, confirmed=False),
                      replace(self.goals, consented=1), replace(self.goals, recipient_id="household")):
            with self.subTest(goals=goals):
                self.assertEqual(self.compare(goals=goals)["status"], "needs_clarification")
        self.assertEqual(self.compare(recipient="household")["status"], "needs_clarification")
        household = replace(self.goals, recipient_id="household")
        result = self.compare(goals=household, recipient="household",
                              pa=replace(self.pa, recipient_id="household"), pb=replace(self.pb, recipient_id="household"))
        self.assertEqual(result["status"], "candidate")

    def test_left_right_swap_uses_identity_not_positions(self):
        original = self.compare()
        swapped = compare_pair((self.b, self.a), (self.pa, self.pb), self.goals, "self", pair_confirmed=True)
        self.assertEqual(original["candidate_product_id"], swapped["candidate_product_id"])
        self.assertEqual(swapped["facts"][0]["product_id"], "b")

    def test_duplicate_wrong_stale_and_unreviewed_identity(self):
        cases = ({"b": self.a}, {"pb": self.pa}, {"pb": replace(self.pb, product_id="c")},
                 {"pa": replace(self.pa, label_revision=2)}, {"a": replace(self.a, identity_reviewed=False)},
                 {"a": replace(self.a, label_reviewed=False)}, {"a": replace(self.a, evidence_ids=())},
                 {"confirmed": False})
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                self.assertEqual(self.compare(**kwargs)["status"], "needs_clarification")

    def test_unknown_constraints_block_even_apparent_dominance(self):
        goals = replace(self.goals, hard_constraint_ids=("declared-ingredient-exclusion",))
        result = self.compare(goals=goals)
        self.assertEqual(result["status"], "needs_clarification")
        self.assertEqual(result["constraint_statuses"][0]["status"], "unknown")

    def test_explicit_failed_constraint_is_separate_from_nutrient_tradeoffs(self):
        goals = replace(self.goals, hard_constraint_ids=("exclusion",))
        a = replace(self.a, constraint_checks={"exclusion": ConstraintCheck("failed", ("ingredient-a",))})
        b = replace(self.b, constraint_checks={"exclusion": ConstraintCheck("passed", ("ingredient-b",))})
        result = self.compare(a=a, b=b, goals=goals)
        self.assertEqual(result["candidate_product_id"], "b")
        self.assertEqual(result["constraint_statuses"][0]["status"], "failed")
        unknown_b = replace(b, constraint_checks={"exclusion": ConstraintCheck("unknown")})
        self.assertEqual(self.compare(a=a, b=unknown_b, goals=goals)["status"], "needs_clarification")

    def test_both_fail_and_missing_values_still_prevent_candidate(self):
        goals = replace(self.goals, hard_constraint_ids=("exclusion",))
        failed = {"exclusion": ConstraintCheck("failed", ("ingredients",))}
        self.assertEqual(self.compare(a=replace(self.a, constraint_checks=failed), b=replace(self.b, constraint_checks=failed), goals=goals)["status"], "needs_clarification")
        a = replace(self.a, nutrients={}, constraint_checks=failed)
        b = replace(self.b, constraint_checks={"exclusion": ConstraintCheck("passed", ("ingredients-b",))})
        self.assertIsNone(self.compare(a=a, b=b, goals=goals)["candidate_product_id"])

    def test_market_preparation_and_category_fail_closed(self):
        for change in ({"market": "Canada"}, {"label_regime": "EU"}, {"preparation": "prepared"}, {"category": "chips"}):
            with self.subTest(change=change):
                self.assertEqual(self.compare(a=replace(self.a, **change))["status"], "needs_clarification")

    def test_unsupported_or_duplicate_goal_rejected(self):
        for goals in ((Goal("predicted_glucose", "min"),), (Goal("sodium_mg", "min"), Goal("sodium_mg", "max"))):
            self.assertEqual(self.compare(goals=replace(self.goals, goals=goals))["status"], "needs_clarification")

    def test_gross_us_component_inconsistency_requires_review(self):
        cases = ({"total_sugars_g": "5", "added_sugars_g": "8"},
                 {"total_carbohydrate_g": "5", "total_sugars_g": "8"},
                 {"total_carbohydrate_g": "5", "added_sugars_g": "8"},
                 {"total_carbohydrate_g": "5", "fiber_g": "8"})
        for nutrients in cases:
            with self.subTest(nutrients=nutrients):
                result = self.compare(a=replace(self.a, nutrients={"sodium_mg": "1", **nutrients}))
                self.assertEqual(result["reason_codes"], ["inconsistent_us_nutrient_components"])
                self.assertIsNone(result["candidate_product_id"])

    def test_label_rounding_tolerance_preserves_values_without_repair(self):
        a = replace(self.a, nutrients={"sodium_mg": "1", "total_carbohydrate_g": "5", "fiber_g": "5.5"})
        result = self.compare(a=a)
        self.assertEqual(result["status"], "candidate")
        self.assertEqual(result["facts"][0]["nutrients"]["fiber_g"], "11.46")

    def test_gram_mass_bounds_include_sodium_milligrams(self):
        for nutrients in ({"sodium_mg": "12000"}, {"sodium_mg": "1", "protein_g": "12"}):
            with self.subTest(nutrients=nutrients):
                a = replace(self.a, basis=Quantity("10", "g"), nutrients=nutrients)
                b = replace(self.b, basis=Quantity("100", "g"))
                result = self.compare(a=a, b=b, pa=replace(self.pa, quantity=Quantity("10", "g")),
                                      pb=replace(self.pb, quantity=Quantity("10", "g")))
                self.assertEqual(result["reason_codes"], ["nutrient_exceeds_food_mass"])

    def test_volume_does_not_impose_fictional_mass_bound(self):
        a = replace(self.a, basis=Quantity("10", "mL"),
                    nutrients={"sodium_mg": "1", "total_carbohydrate_g": "15", "total_sugars_g": "14"})
        result = self.compare(a=a, pa=replace(self.pa, quantity=Quantity("10", "mL")))
        self.assertEqual(result["status"], "candidate")
        self.assertEqual(result["facts"][0]["nutrients"]["total_carbohydrate_g"], "15")


if __name__ == "__main__":
    unittest.main()
