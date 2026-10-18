"""Optional local visual shortlist. Never establishes exact identity or nutrition."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2 as cv
import numpy as np

cv.setNumThreads(1)
VERSION = 1
MAX_EDGE = 1000
MIN_FEATURES = 70
MIN_INLIERS = 24


def load_image(path: Path) -> np.ndarray:
    if path.stat().st_size > 25_000_000:
        raise ValueError(f'Image exceeds 25 MB: {path}')
    image = cv.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv.IMREAD_GRAYSCALE)
    if image is None or min(image.shape) < 120 or image.size > 40_000_000:
        raise ValueError(f'Use a readable image, at least 120 pixels each side: {path}')
    scale = min(1.0, MAX_EDGE / max(image.shape))
    if scale < 1:
        image = cv.resize(image, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
    return image


def fingerprint(image: np.ndarray) -> str:
    return hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()


def features(image: np.ndarray) -> dict:
    points, descriptors = cv.ORB_create(nfeatures=1800, fastThreshold=15).detectAndCompute(image, None)
    descriptors = descriptors if descriptors is not None else np.empty((0, 32), np.uint8)
    unique_ratio = len(np.unique(descriptors, axis=0)) / max(1, len(descriptors))
    return {'shape': list(image.shape), 'points': np.float32([p.pt for p in points]).reshape(-1, 2),
            'descriptors': descriptors, 'unique_ratio': unique_ratio}


def pack(feature: dict) -> dict:
    return {'shape': feature['shape'], 'points': feature['points'].tolist(),
            'descriptors_b64': base64.b64encode(feature['descriptors'].tobytes()).decode(),
            'unique_ratio': feature['unique_ratio']}


def unpack(record: dict) -> dict:
    points = np.asarray(record['points'], np.float32).reshape(-1, 2)
    descriptors = np.frombuffer(base64.b64decode(record['descriptors_b64'], validate=True), np.uint8).reshape(-1, 32)
    if len(points) != len(descriptors) or len(points) > 1800 or not np.isfinite(points).all():
        raise ValueError('Invalid reference descriptors')
    return {'shape': record['shape'], 'points': points, 'descriptors': descriptors, 'unique_ratio': record['unique_ratio']}


def build_index(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    products = manifest['products']
    if not 1 <= len(products) <= 12:
        raise ValueError('This demo index supports 1–12 products')
    seen_ids, seen_images = set(), {}
    references, metadata = [], []
    for product in products:
        product_id = product['product_id']
        if not product_id or product_id in seen_ids:
            raise ValueError('Each product_id must be unique')
        seen_ids.add(product_id)
        if not product.get('variant_group'):
            raise ValueError('Each product needs a variant_group to flag similar packaging')
        photos = product['references']
        if not 2 <= len(photos) <= 10:
            raise ValueError('Use 2–10 independent reference photos per product')
        metadata.append({k: product.get(k) for k in ('product_id', 'name', 'barcode', 'variant_group')})
        distinct = set()
        for photo in photos:
            if not photo.get('capture_session'):
                raise ValueError('Every reference needs a capture_session for holdout separation')
            path = (manifest_path.parent / photo['image']).resolve()
            image = load_image(path)
            digest = fingerprint(image)
            if digest in seen_images:
                raise ValueError(f'Duplicate or conflicting reference image: {path}; already assigned to {seen_images[digest]}')
            seen_images[digest] = product_id
            feature = features(image)
            if len(feature['points']) < MIN_FEATURES or feature['unique_ratio'] < .65:
                raise ValueError(f'Reference has insufficient distinctive texture: {path}')
            distinct.add(digest)
            references.append({'product_id': product_id, 'image_sha256': digest,
                               'capture_session': photo['capture_session'], 'image_name': path.name, **pack(feature)})
        if len(distinct) < 2:
            raise ValueError('Need at least two distinct photos per product')
    return {'version': VERSION, 'created_at': datetime.now(timezone.utc).isoformat(),
            'algorithm': 'ORB + reciprocal ratio matching + RANSAC; thresholds uncalibrated',
            'products': metadata, 'references': references}


def match_pair(reference: dict, query: dict) -> dict | None:
    ref_d, query_d = reference['descriptors'], query['descriptors']
    if min(len(ref_d), len(query_d)) < MIN_FEATURES or min(reference['unique_ratio'], query['unique_ratio']) < .65:
        return None
    matcher = cv.BFMatcher(cv.NORM_HAMMING)
    def distinctive(a, b):
        return {m.queryIdx: m for pair in matcher.knnMatch(a, b, k=2) if len(pair) == 2
                for m, n in [pair] if m.distance < .70 * n.distance and m.distance <= 50}
    forward, reverse = distinctive(ref_d, query_d), distinctive(query_d, ref_d)
    matches = [m for i, m in forward.items() if m.trainIdx in reverse and reverse[m.trainIdx].trainIdx == i]
    if len(matches) < MIN_INLIERS:
        return None
    src = np.float32([reference['points'][m.queryIdx] for m in matches])
    dst = np.float32([query['points'][m.trainIdx] for m in matches])
    cv.setRNGSeed(2026)
    homography, mask = cv.findHomography(src, dst, cv.RANSAC, 3.0, maxIters=2000, confidence=.995)
    if homography is None or mask is None or not np.isfinite(homography).all():
        return None
    inliers = mask.ravel().astype(bool)
    count, ratio = int(inliers.sum()), float(inliers.mean())
    if count < MIN_INLIERS or ratio < .65:
        return None
    h, w = reference['shape']; qh, qw = query['shape']
    corners = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    projected = cv.perspectiveTransform(corners.reshape(-1, 1, 2), homography).reshape(-1, 2)
    if not np.isfinite(projected).all() or not cv.isContourConvex(projected):
        return None
    area = abs(cv.contourArea(projected))
    if not .035 <= area / (qw * qh) <= 1.15 or (projected < [-.1*qw, -.1*qh]).any() or (projected > [1.1*qw, 1.1*qh]).any():
        return None
    ref_coverage = abs(cv.contourArea(cv.convexHull(src[inliers]))) / (w*h)
    query_coverage = abs(cv.contourArea(cv.convexHull(dst[inliers]))) / max(area, 1)
    quadrants = {(int(x >= w/2), int(y >= h/2)) for x, y in src[inliers]}
    reprojection = cv.perspectiveTransform(src[inliers].reshape(-1, 1, 2), homography).reshape(-1, 2)
    error = float(np.median(np.linalg.norm(reprojection - dst[inliers], axis=1)))
    if ref_coverage < .20 or query_coverage < .18 or len(quadrants) < 3 or error > 2.5:
        return None
    return {'inliers': count, 'inlier_ratio': round(ratio, 3), 'reference_coverage': round(ref_coverage, 3),
            'median_reprojection_px': round(error, 3), 'rank_value': count * ratio,
            'query_polygon_normalized': (projected / [qw, qh]).round(4).tolist()}


def decide(hits: list[dict], products: list[dict]) -> dict:
    grouped = defaultdict(list)
    for hit in hits:
        grouped[hit['product_id']].append(hit)
    ranked = sorted(grouped.items(), key=lambda pair: max(h['rank_value'] for h in pair[1]), reverse=True)
    if not ranked:
        return {'status': 'unknown', 'reason': 'no_geometrically_verified_match', 'candidate': None}
    product_id, supports = ranked[0]
    best = max(supports, key=lambda hit: hit['rank_value'])
    product = next(p for p in products if p['product_id'] == product_id)
    if sum(p['variant_group'] == product['variant_group'] for p in products) > 1:
        reason = 'known_near_variants_require_barcode'
    elif len(supports) < 2:
        reason = 'needs_two_independent_reference_matches'
    elif len(ranked) > 1 and (best['rank_value'] < 1.5 * max(h['rank_value'] for h in ranked[1][1]) or best['rank_value'] - max(h['rank_value'] for h in ranked[1][1]) < 10):
        reason = 'ambiguous_competing_products'
    else:
        return {'status': 'candidate', 'reason': 'visual_shortlist_only_verify_barcode_and_label',
                'candidate': product, 'supporting_references': len(supports),
                'evidence': {k: v for k, v in best.items() if k != 'rank_value'}}
    return {'status': 'unknown', 'reason': reason, 'candidate': None}


def query_index(index: dict, image_path: Path) -> dict:
    if index.get('version') != VERSION:
        raise ValueError('Unsupported reference index version')
    start = time.perf_counter()
    query = features(load_image(image_path))
    hits = []
    for record in index['references']:
        hit = match_pair(unpack(record), query)
        if hit:
            hits.append({'product_id': record['product_id'], 'reference_image': record['image_name'], **hit})
    result = decide(hits, index['products'])
    result.update({'elapsed_ms': round((time.perf_counter() - start)*1000, 1),
                   'query_features': len(query['points']), 'confidence': 'not_calibrated',
                   'exact_identity_verified': False, 'nutrition_lookup_allowed': False})
    return result


def evaluate(index: dict, manifest_path: Path) -> dict:
    cases = json.loads(manifest_path.read_text(encoding='utf-8'))['cases']
    if not cases:
        raise ValueError('Evaluation requires independent holdout cases')
    train_hashes = {r['image_sha256'] for r in index['references']}
    train_sessions = {r['capture_session'] for r in index['references']}
    results = []
    for case in cases:
        path = (manifest_path.parent / case['image']).resolve()
        if not case.get('capture_session') or case['capture_session'] in train_sessions:
            raise ValueError('Holdout photos must come from a separate capture session, not adjacent reference video frames')
        if fingerprint(load_image(path)) in train_hashes:
            raise ValueError('Holdout contains an indexed image; this is training leakage')
        result = query_index(index, path)
        predicted = result['candidate']['product_id'] if result['candidate'] else None
        expected = case.get('expected_product_id')
        results.append({'image': case['image'], 'expected_product_id': expected, 'predicted_product_id': predicted,
                        'correct': predicted == expected, **result})
    false_candidates = sum(r['predicted_product_id'] is not None and not r['correct'] for r in results)
    known = [r for r in results if r['expected_product_id'] is not None]
    unknown = [r for r in results if r['expected_product_id'] is None]
    return {'evaluated_at': datetime.now(timezone.utc).isoformat(), 'cases': len(results),
            'false_candidates': false_candidates, 'known_candidate_recall': sum(r['correct'] for r in known)/len(known) if known else None,
            'unknown_rejection_rate': sum(r['correct'] for r in unknown)/len(unknown) if unknown else None,
            'p95_elapsed_ms': float(np.percentile([r['elapsed_ms'] for r in results], 95)),
            'real_product_accuracy_claim': 'Valid only for these supplied held-out cases; not general product identification.', 'results': results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('index'); build.add_argument('manifest', type=Path); build.add_argument('output', type=Path)
    query = sub.add_parser('query'); query.add_argument('index', type=Path); query.add_argument('image', type=Path)
    check = sub.add_parser('evaluate'); check.add_argument('index', type=Path); check.add_argument('holdout', type=Path); check.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'index':
            result = build_index(args.manifest)
            args.output.write_text(json.dumps(result, separators=(',', ':')), encoding='utf-8')
            print(json.dumps({'index': str(args.output), 'references': len(result['references']), 'products': len(result['products'])}))
            return
        index = json.loads(args.index.read_text(encoding='utf-8'))
        result = query_index(index, args.image) if args.command == 'query' else evaluate(index, args.holdout)
        rendered = json.dumps(result, indent=2)
        if getattr(args, 'output', None):
            args.output.write_text(rendered, encoding='utf-8')
        print(rendered)
    except (ValueError, OSError, KeyError, cv.error) as error:
        parser.exit(2, f'Visual matcher: {error}\n')


if __name__ == '__main__':
    main()
