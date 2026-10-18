"""Optional front-package candidate detector; deliberately disconnected from app v1.

Contract checked 2026-10-18: https://docs.perceptron.inc/capabilities/detect
The Detect endpoint returns PIXEL boxes and no confidence scores. It does not
verify a GTIN, package variant, or nutrition. No keys/media are logged here.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
from uuid import uuid4

import httpx

ENDPOINT = "https://api.perceptron.inc/v1/detect"
MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAX_RESPONSE_BYTES = 128 * 1024


@dataclass(frozen=True)
class Frame:
    """Caller supplies decoded, orientation-corrected dimensions of these bytes.

    received_at is the phone receipt timestamp, not a claimed sensor exposure
    timestamp. Selection, capture consent, generation cancellation, and age gates
    belong to the host application. Remote image URLs are intentionally rejected.
    """

    frame_id: str
    image_data_url: str = field(repr=False)
    width: int
    height: int
    received_at: str


@dataclass(frozen=True)
class PositiveExemplar:
    catalog_id: str
    frame: Frame
    box_xyxy_px: tuple[float, float, float, float]


@dataclass(frozen=True)
class DetectionCandidate:
    occurrence_id: str
    catalog_id: str
    box: dict[str, float]
    frame_id: str
    frame_received_at: str
    timestamp_kind: Literal["phone_received"] = field(default="phone_received", init=False)
    source: Literal["perceptron_detect"] = field(default="perceptron_detect", init=False)
    exact_identity_verified: Literal[False] = field(default=False, init=False)
    nutrition_lookup_allowed: Literal[False] = field(default=False, init=False)
    confidence: None = field(default=None, init=False)


@dataclass(frozen=True)
class DetectionResult:
    status: Literal["candidates", "unknown", "unavailable"]
    candidates: tuple[DetectionCandidate, ...] = ()
    reason: str = ""


def _validate_frame(frame: Frame, *, max_image_bytes: int = MAX_IMAGE_BYTES) -> None:
    if not isinstance(frame.frame_id, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,100}", frame.frame_id):
        raise ValueError("Invalid frame identifier")
    if any(type(n) is not int or not 1 <= n <= 8192 for n in (frame.width, frame.height)):
        raise ValueError("Invalid decoded frame dimensions")
    try:
        timestamp = datetime.fromisoformat(frame.received_at.replace("Z", "+00:00"))
        if timestamp.utcoffset() is None:
            raise ValueError
    except (AttributeError, TypeError, ValueError):
        raise ValueError("Frame receipt timestamp must include a timezone") from None
    if not isinstance(frame.image_data_url, str) or len(frame.image_data_url) > max_image_bytes * 4 // 3 + 100:
        raise ValueError("Image exceeds the adapter size limit")
    prefix, separator, encoded = frame.image_data_url.partition(",")
    if not separator or prefix not in ("data:image/jpeg;base64", "data:image/png;base64"):
        raise ValueError("Only inline JPEG or PNG images are accepted")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise ValueError("Invalid base64 image") from None
    expected = b"\xff\xd8\xff" if "jpeg" in prefix else b"\x89PNG\r\n\x1a\n"
    if not data.startswith(expected) or len(data) > max_image_bytes:
        raise ValueError("Invalid image signature or size")


def _checked_box(box: object, width: int, height: int) -> tuple[float, float, float, float]:
    if not isinstance(box, (tuple, list)) or len(box) != 4:
        raise ValueError("Invalid box")
    if any(type(n) not in (int, float) or not math.isfinite(n) for n in box):
        raise ValueError("Non-finite box")
    x1, y1, x2, y2 = map(float, box)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("Box outside source image")
    return x1, y1, x2, y2


def _pixel_box(box: tuple[float, float, float, float]) -> dict:
    x1, y1, x2, y2 = box
    return {"top_left": {"x": x1, "y": y1}, "bottom_right": {"x": x2, "y": y2}}


def build_payload(frame: Frame, exemplars: tuple[PositiveExemplar, ...], negative_frames: tuple[Frame, ...] = ()) -> dict:
    """One target image, at most four reviewed positive/negative references each."""
    _validate_frame(frame)
    if not 1 <= len(exemplars) <= 4 or len(negative_frames) > 4:
        raise ValueError("Provide one to four positive and at most four negative exemplars")
    categories: list[str] = []
    examples: list[dict] = []
    positive_images: set[str] = set()
    image_labels: dict[str, str] = {}
    for exemplar in exemplars:
        _validate_frame(exemplar.frame)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", exemplar.catalog_id):
            raise ValueError("Invalid catalog ID")
        image = exemplar.frame.image_data_url
        if image in image_labels and image_labels[image] != exemplar.catalog_id:
            raise ValueError("Conflicting labels for the same reference image")
        image_labels[image] = exemplar.catalog_id
        positive_images.add(image)
        box = _checked_box(exemplar.box_xyxy_px, exemplar.frame.width, exemplar.frame.height)
        if exemplar.catalog_id not in categories:
            categories.append(exemplar.catalog_id)
        examples.append({"media": {"type": "image", "image_url": image}, "annotations": [{
            "mention": exemplar.catalog_id, "point_box": _pixel_box(box),
        }]})
    payload = {
        "media": {"type": "image", "image_url": frame.image_data_url},
        "categories": categories,
        "exemplars": examples,
        "config": {"output": "box", "reasoning_effort": "none"},
    }
    if negative_frames:
        negatives = []
        for negative in negative_frames:
            _validate_frame(negative)
            if negative.image_data_url in positive_images:
                raise ValueError("A positive reference cannot also be negative")
            negatives.append({"media": {"type": "image", "image_url": negative.image_data_url}})
        payload["negative_exemplars"] = negatives
    return payload


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union


def parse_response(data: object, frame: Frame, allowed_catalog_ids: set[str]) -> DetectionResult:
    """Fail the whole pair on ambiguity; never quietly pick the first two boxes."""
    if not isinstance(data, dict) or set(data) != {"detections", "finish_reason"}:
        return DetectionResult("unknown", reason="malformed_response")
    if data["finish_reason"] != "complete":
        return DetectionResult("unknown", reason="incomplete_response")
    detections = data["detections"]
    if not isinstance(detections, list):
        return DetectionResult("unknown", reason="malformed_response")
    if not detections:
        return DetectionResult("unknown", reason="no_match")
    if len(detections) > 2:
        return DetectionResult("unknown", reason="too_many_objects")
    candidates = []
    boxes = []
    try:
        for detection in detections:
            if not isinstance(detection, dict) or set(detection) - {"mention", "point_box", "point"}:
                raise ValueError
            mention = detection.get("mention")
            if not isinstance(mention, str) or mention not in allowed_catalog_ids or detection.get("point") is not None:
                raise ValueError
            box = detection.get("point_box")
            if not isinstance(box, dict) or set(box) != {"top_left", "bottom_right"}:
                raise ValueError
            top, bottom = box["top_left"], box["bottom_right"]
            if not isinstance(top, dict) or not isinstance(bottom, dict) or set(top) != {"x", "y"} or set(bottom) != {"x", "y"}:
                raise ValueError
            x1, y1, x2, y2 = _checked_box((top["x"], top["y"], bottom["x"], bottom["y"]), frame.width, frame.height)
            boxes.append((x1, y1, x2, y2))
            candidates.append(DetectionCandidate(
                occurrence_id=str(uuid4()), catalog_id=mention,
                box={"x": x1 / frame.width, "y": y1 / frame.height, "w": (x2 - x1) / frame.width, "h": (y2 - y1) / frame.height},
                frame_id=frame.frame_id, frame_received_at=frame.received_at,
            ))
    except (ValueError, TypeError, KeyError, OverflowError):
        return DetectionResult("unknown", reason="invalid_detection")
    if len(boxes) == 2 and _iou(boxes[0], boxes[1]) > 0.6:
        return DetectionResult("unknown", reason="overlapping_candidates")
    return DetectionResult("candidates", tuple(candidates), "confirmation_required")


class PerceptronDetector:
    """Reuse the host's async HTTP client. No automatic retry or state mutation."""

    def __init__(self, api_key: str, client: httpx.AsyncClient, timeout_s: float = 3.5):
        if type(timeout_s) not in (int, float) or not 2 <= timeout_s <= 4:
            raise ValueError("Detector deadline must be between two and four seconds")
        self._api_key = api_key.strip()
        self._client = client
        self._timeout_s = float(timeout_s)

    async def detect_candidates(self, frame: Frame, exemplars: tuple[PositiveExemplar, ...], negative_frames: tuple[Frame, ...] = ()) -> DetectionResult:
        if not self._api_key:
            return DetectionResult("unavailable", reason="not_configured")
        payload = build_payload(frame, exemplars, negative_frames)
        try:
            async with asyncio.timeout(self._timeout_s):
                async with self._client.stream(
                    "POST", ENDPOINT, json=payload,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    timeout=httpx.Timeout(self._timeout_s, connect=1.0),
                    follow_redirects=False,
                ) as response:
                    if response.status_code != 200:
                        reason = "rate_limited" if response.status_code == 429 else "provider_rejected"
                        return DetectionResult("unavailable", reason=reason)
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > MAX_RESPONSE_BYTES:
                            return DetectionResult("unknown", reason="response_too_large")
                    try:
                        data = json.loads(body)
                    except (ValueError, UnicodeDecodeError):
                        return DetectionResult("unknown", reason="invalid_json")
        except (TimeoutError, httpx.TimeoutException):
            return DetectionResult("unavailable", reason="timeout")
        except httpx.HTTPError:
            return DetectionResult("unavailable", reason="network_error")
        # asyncio.CancelledError intentionally propagates to the host generation.
        return parse_response(data, frame, {item.catalog_id for item in exemplars})
