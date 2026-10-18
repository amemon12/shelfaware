"""Procedural-image tests validate mechanics, never grocery recognition accuracy."""
import json
import tempfile
import unittest
from pathlib import Path

import cv2 as cv
import numpy as np

from matcher import build_index, decide, evaluate, features, fingerprint, match_pair, query_index


def textured(seed=5):
    image = np.full((600, 440), 235, np.uint8)
    rng = np.random.default_rng(seed)
    for _ in range(150):
        x, y = rng.integers([30, 30], [400, 560])
        cv.circle(image, (int(x), int(y)), int(rng.integers(3, 12)), int(rng.integers(0, 180)), -1)
    for row in range(6):
        cv.putText(image, f'PATTERN {seed} / {row}', (35, 85+row*80), cv.FONT_HERSHEY_SIMPLEX, .8, 20, 2)
    return image


class MatcherTests(unittest.TestCase):
    def test_perspective_transform_has_distributed_geometric_evidence(self):
        ref = textured()
        source = np.float32([[0, 0], [439, 0], [439, 599], [0, 599]])
        target = np.float32([[30, 20], [420, 40], [430, 570], [15, 590]])
        query = cv.warpPerspective(ref, cv.getPerspectiveTransform(source, target), (440, 600), borderValue=250)
        hit = match_pair(features(ref), features(query))
        self.assertIsNotNone(hit)
        self.assertGreaterEqual(hit['inliers'], 24)
        self.assertGreater(hit['reference_coverage'], .2)

    def test_unrelated_and_featureless_images_abstain(self):
        self.assertIsNone(match_pair(features(textured(5)), features(textured(77))))
        self.assertIsNone(match_pair(features(textured()), features(np.full((600, 440), 255, np.uint8))))

    def test_repeated_checkerboard_abstains(self):
        grid = (np.indices((600, 440)).sum(axis=0) // 20 % 2 * 255).astype(np.uint8)
        self.assertIsNone(match_pair(features(grid), features(np.roll(grid, 20, axis=0))))

    def test_ambiguous_and_known_near_variants_abstain(self):
        products = [{'product_id': 'a', 'variant_group': 'same-box'}, {'product_id': 'b', 'variant_group': 'same-box'}]
        hits = [{'product_id': 'a', 'rank_value': 80}, {'product_id': 'a', 'rank_value': 70}]
        self.assertEqual(decide(hits, products)['reason'], 'known_near_variants_require_barcode')
        products[1]['variant_group'] = 'different-box'
        self.assertEqual(decide(hits + [{'product_id': 'b', 'rank_value': 75}], products)['reason'], 'ambiguous_competing_products')

    def test_a_single_reference_is_insufficient(self):
        result = decide([{'product_id': 'a', 'rank_value': 80}], [{'product_id': 'a', 'variant_group': 'a'}])
        self.assertEqual(result['status'], 'unknown')

    def test_two_supports_still_only_return_candidate(self):
        result = decide([{'product_id': 'a', 'rank_value': 80}, {'product_id': 'a', 'rank_value': 70}], [{'product_id': 'a', 'variant_group': 'a'}])
        self.assertEqual(result['status'], 'candidate')
        self.assertIn('verify_barcode_and_label', result['reason'])

    def test_conflicting_reference_images_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cv.imwrite(str(root/'one.png'), textured())
            manifest = {'products': [{'product_id': 'a', 'variant_group': 'a', 'references': [
                {'image': 'one.png', 'capture_session': 'reference-day'}] * 2}]}
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'Duplicate or conflicting'):
                build_index(root/'manifest.json')

    def test_holdout_capture_session_cannot_overlap_reference_session(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'holdout.json'
            path.write_text(json.dumps({'cases': [{'image': 'unused.png', 'capture_session': 'same-session', 'expected_product_id': 'a'}]}))
            with self.assertRaisesRegex(ValueError, 'separate capture session'):
                evaluate({'references': [{'image_sha256': 'x', 'capture_session': 'same-session'}]}, path)

    def test_serialized_index_query_pipeline_remains_only_a_candidate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = textured()
            second = cv.convertScaleAbs(first, alpha=.88, beta=18)
            source = np.float32([[0, 0], [439, 0], [439, 599], [0, 599]])
            target = np.float32([[25, 20], [420, 25], [425, 575], [20, 585]])
            query = cv.warpPerspective(first, cv.getPerspectiveTransform(source, target), (440, 600), borderValue=250)
            for name, image in [('one.png', first), ('two.png', second), ('query.png', query)]:
                cv.imwrite(str(root/name), image)
            manifest = {'products': [{'product_id': 'procedural-a', 'variant_group': 'procedural-a', 'references': [
                {'image': 'one.png', 'capture_session': 'reference'}, {'image': 'two.png', 'capture_session': 'reference'}]}]}
            (root/'manifest.json').write_text(json.dumps(manifest))
            index = json.loads(json.dumps(build_index(root/'manifest.json')))
            result = query_index(index, root/'query.png')
            self.assertEqual(result['status'], 'candidate')
            self.assertFalse(result['exact_identity_verified'])
            self.assertFalse(result['nutrition_lookup_allowed'])
            self.assertEqual(result['confidence'], 'not_calibrated')

    def test_identical_holdout_pixels_reject_even_after_session_rename(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = textured()
            cv.imwrite(str(root/'copy.png'), image)
            (root/'holdout.json').write_text(json.dumps({'cases': [{'image': 'copy.png', 'capture_session': 'new-session', 'expected_product_id': 'a'}]}))
            index = {'references': [{'image_sha256': fingerprint(image), 'capture_session': 'reference-session'}]}
            with self.assertRaisesRegex(ValueError, 'training leakage'):
                evaluate(index, root/'holdout.json')


if __name__ == '__main__':
    unittest.main()
