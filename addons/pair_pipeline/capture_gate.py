"""Optional scheduling reference. No camera, network, or running-app changes.

Use local monotonic arrival times here. Phone capture timestamps belong to another
clock: the phone must separately reject stale captures before upload. This gate
does not measure blur, track objects, or establish exact product identity.
"""
from dataclasses import dataclass
import math
from uuid import uuid4


@dataclass(frozen=True)
class Context:
    session_id: str
    pair_revision: int
    profile_revision: int
    recipient_id: str
    turn_id: str


@dataclass(frozen=True)
class Frame:
    frame_id: str
    arrival_mono: float
    jpeg: bytes
    scene_revision: int


@dataclass(frozen=True)
class DetectionJob:
    job_id: str
    context: Context
    frame: Frame


class CaptureGate:
    """Single-slot, single-in-flight gate; caller owns cancellation and locking.

    All methods must run on one event loop. Explicit user requests may use a
    stable or unstable frame; automatic prefetch requires a stable scene and an
    opted-in active session. Both respect the request-rate and freshness limits.
    """
    def __init__(self, max_queue_age: float = 0.75, min_interval: float = 1.0,
                 max_bytes: int = 2_000_000):
        if not all(math.isfinite(v) and v > 0 for v in (max_queue_age, min_interval)):
            raise ValueError('Positive finite timing limits required')
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
            raise ValueError('Positive byte limit required')
        self.max_queue_age = max_queue_age
        self.min_interval = min_interval
        self.max_bytes = max_bytes
        self.pending: Frame | None = None
        self.inflight: DetectionJob | None = None
        self.last_dispatch = float('-inf')
        self.latest_arrival = float('-inf')

    def offer(self, frame: Frame, now: float) -> bool:
        if (not math.isfinite(now) or not math.isfinite(frame.arrival_mono)
                or not isinstance(frame.jpeg, bytes) or not frame.jpeg
                or len(frame.jpeg) > self.max_bytes or not frame.frame_id
                or not isinstance(frame.scene_revision, int) or frame.scene_revision < 0
                or frame.arrival_mono > now
                or now - frame.arrival_mono > self.max_queue_age
                or frame.arrival_mono <= self.latest_arrival):
            return False
        self.pending = frame
        self.latest_arrival = frame.arrival_mono
        return True

    def dispatch(self, context: Context, now: float, *, session_active: bool,
                 user_requested: bool, scene_stable: bool) -> DetectionJob | None:
        if not math.isfinite(now):
            raise ValueError('Use a finite local monotonic time')
        frame = self.pending
        if (not session_active or self.inflight is not None or frame is None
                or now - self.last_dispatch < self.min_interval
                or (not user_requested and not scene_stable)):
            return None
        if (frame.scene_revision != context.pair_revision or frame.arrival_mono > now
                or now - frame.arrival_mono > self.max_queue_age):
            self.pending = None
            return None
        self.pending = None
        self.last_dispatch = now
        self.inflight = DetectionJob(str(uuid4()), context, frame)
        return self.inflight

    def finish(self, job: DetectionJob, active_context: Context | None) -> bool:
        """Release capacity; return whether this result may enter further validation.

        False means discard. True is NOT identity approval. Cancel/disconnect,
        pair swaps, recipient changes, goal changes, or a new turn must replace
        active_context. Cancel the provider task too; filtering alone saves no cost.
        """
        if self.inflight is None or self.inflight.job_id != job.job_id:
            return False
        self.inflight = None
        return active_context is not None and job.context == active_context

    def clear_pending(self) -> None:
        self.pending = None

