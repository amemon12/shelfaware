"""No network or paid calls: provider protocol and abstention tests only."""
import asyncio
import base64
import json
import unittest
from dataclasses import replace

import httpx

from addons.pair_pipeline.perceptron_adapter import (
    ENDPOINT, MAX_RESPONSE_BYTES, Frame, PositiveExemplar, PerceptronDetector,
    build_payload, parse_response,
)


def frame(name="target", marker=b"target"):
    # This transport test fixture is a signature stub, not an image accuracy test.
    url = "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8\xff" + marker).decode()
    return Frame(name, url, 1000, 500, "2026-10-18T07:00:00Z")


def references():
    return (
        PositiveExemplar("chips-a", frame("ref-a", b"ref-a"), (0, 0, 1000, 500)),
        PositiveExemplar("chips-b", frame("ref-b", b"ref-b"), (0, 0, 1000, 500)),
    )


def detection(mention="chips-a", box=(100, 50, 300, 450)):
    x1, y1, x2, y2 = box
    return {"mention": mention, "point_box": {"top_left": {"x": x1, "y": y1}, "bottom_right": {"x": x2, "y": y2}}}


def response(*detections, finish="complete"):
    return {"detections": list(detections), "finish_reason": finish}


class PayloadTests(unittest.TestCase):
    def test_exact_exemplar_and_negative_contract(self):
        payload = build_payload(frame(), references(), (frame("negative", b"different"),))
        self.assertEqual(payload["categories"], ["chips-a", "chips-b"])
        self.assertEqual(payload["config"], {"output": "box", "reasoning_effort": "none"})
        self.assertEqual(set(payload["negative_exemplars"][0]), {"media"})
        self.assertEqual(payload["exemplars"][0]["annotations"][0]["point_box"]["bottom_right"], {"x": 1000.0, "y": 500.0})
        self.assertNotIn("model", payload)
        self.assertNotIn("profile", payload)

    def test_remote_media_and_wrong_signature_rejected(self):
        for url in ("https://example.com/image.jpg", "data:image/jpeg;base64,YmFk", "data:image/png;base64,!!!"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                build_payload(replace(frame(), image_data_url=url), references())

    def test_dimensions_timezone_and_bounds_rejected(self):
        for bad in (replace(frame(), width=0), replace(frame(), height=True), replace(frame(), received_at="2026-10-18T07:00:00")):
            with self.assertRaises(ValueError):
                build_payload(bad, references())
        with self.assertRaises(ValueError):
            build_payload(frame(), (replace(references()[0], box_xyxy_px=(0, 0, 1001, 500)),))

    def test_reference_bounds_conflicts_and_empty_rejected(self):
        for refs in ((), references() * 3, (references()[0], replace(references()[0], catalog_id="other"))):
            with self.assertRaises(ValueError):
                build_payload(frame(), refs)
        with self.assertRaises(ValueError):
            build_payload(frame(), references(), (references()[0].frame,))


class ParseTests(unittest.TestCase):
    def parse(self, data):
        return parse_response(data, frame(), {"chips-a", "chips-b"})

    def test_two_boxes_normalized_and_identity_unverified(self):
        result = self.parse(response(detection(), detection("chips-b", (600, 50, 900, 450))))
        self.assertEqual(result.status, "candidates")
        self.assertEqual(result.candidates[0].box, {"x": .1, "y": .1, "w": .2, "h": .8})
        self.assertNotEqual(result.candidates[0].occurrence_id, result.candidates[1].occurrence_id)
        for candidate in result.candidates:
            self.assertFalse(candidate.exact_identity_verified)
            self.assertFalse(candidate.nutrition_lookup_allowed)
            self.assertIsNone(candidate.confidence)
            self.assertEqual(candidate.frame_id, "target")
            self.assertEqual(candidate.timestamp_kind, "phone_received")

    def test_zero_or_three_is_unknown(self):
        self.assertEqual(self.parse(response()).reason, "no_match")
        self.assertEqual(self.parse(response(detection(), detection(), detection())).reason, "too_many_objects")

    def test_incomplete_never_salvages_partial_detections(self):
        result = self.parse(response(detection(), finish="max_tokens"))
        self.assertEqual(result.reason, "incomplete_response")
        self.assertEqual(result.candidates, ())

    def test_malformed_boxes_and_labels_are_rejected(self):
        for item in (detection(box=(0, 0, 1001, 500)), detection(box=(100, 0, 50, 500)), detection(box=(0, float("nan"), 100, 100)), detection(box=(True, 0, 100, 100)), detection("not-in-catalog"), {"mention": "chips-a", "point": {"x": 1, "y": 2}}):
            with self.subTest(item=item):
                self.assertEqual(self.parse(response(item)).reason, "invalid_detection")

    def test_conflicting_or_duplicate_boxes_abstain(self):
        for second in (detection(), detection("chips-b")):
            self.assertEqual(self.parse(response(detection(), second)).reason, "overlapping_candidates")

    def test_same_sku_two_separate_instances_remain_distinct(self):
        result = self.parse(response(detection(), detection("chips-a", (600, 50, 900, 450))))
        self.assertEqual(result.status, "candidates")
        self.assertEqual(len(result.candidates), 2)

    def test_shape_changes_fail_closed(self):
        for data in (None, [], {"detections": []}, {"detections": {}, "finish_reason": "complete"}, {**response(detection()), "confidence": .99}):
            self.assertEqual(self.parse(data).status, "unknown")


class HTTPTests(unittest.IsolatedAsyncioTestCase):
    async def run_detector(self, handler, key="test-key"):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await PerceptronDetector(key, client).detect_candidates(frame(), references())

    async def test_actual_request_contract(self):
        def handler(request):
            self.assertEqual(str(request.url), ENDPOINT)
            self.assertEqual(request.headers["authorization"], "Bearer test-key")
            self.assertEqual(json.loads(request.content)["categories"], ["chips-a", "chips-b"])
            self.assertLessEqual(request.extensions["timeout"]["read"], 4)
            return httpx.Response(200, json=response(detection()))
        self.assertEqual((await self.run_detector(handler)).status, "candidates")

    async def test_no_key_means_no_http_request(self):
        def handler(_):
            self.fail("HTTP called without a key")
        self.assertEqual((await self.run_detector(handler, key="")).reason, "not_configured")

    async def test_errors_never_retry_or_expose_provider_body(self):
        for status in (401, 429, 500, 302):
            calls = []
            def handler(request):
                calls.append(request)
                return httpx.Response(status, text="secret provider text", headers={"Location": "https://other.invalid/"})
            result = await self.run_detector(handler)
            self.assertEqual(len(calls), 1)
            self.assertEqual(result.status, "unavailable")
            self.assertNotIn("secret", repr(result))

    async def test_network_timeout_and_connection_errors(self):
        for error, reason in ((httpx.ReadTimeout("private detail"), "timeout"), (httpx.ConnectError("private detail"), "network_error")):
            def handler(_):
                raise error
            self.assertEqual((await self.run_detector(handler)).reason, reason)

    async def test_bad_and_oversized_json(self):
        for body, reason in ((b"not json", "invalid_json"), (b"x" * (MAX_RESPONSE_BYTES + 1), "response_too_large")):
            self.assertEqual((await self.run_detector(lambda _: httpx.Response(200, content=body))).reason, reason)

    async def test_cancel_is_propagated_without_stale_result(self):
        async def handler(_):
            raise asyncio.CancelledError
        with self.assertRaises(asyncio.CancelledError):
            await self.run_detector(handler)

    async def test_total_deadline_cancels_a_stalled_transport(self):
        calls = []
        async def handler(request):
            calls.append(request)
            await asyncio.sleep(5)
            return httpx.Response(200, json=response(detection()))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await PerceptronDetector("test-key", client, timeout_s=2).detect_candidates(frame(), references())
        self.assertEqual(result.reason, "timeout")
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
