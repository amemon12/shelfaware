from dataclasses import replace
import unittest

from addons.pair_pipeline.capture_gate import CaptureGate, Context, Frame


class CaptureGateTests(unittest.TestCase):
    def setUp(self):
        self.gate = CaptureGate()
        self.context = Context('session', 1, 1, 'self', 'turn')

    def frame(self, identity='a', at=10.0, scene=1):
        return Frame(identity, at, b'jpeg-placeholder', scene)

    def dispatch(self, at=10.0, **options):
        return self.gate.dispatch(self.context, at, session_active=True,
                                  user_requested=True, scene_stable=True, **options)

    def test_replaces_pending_instead_of_building_queue(self):
        self.gate.offer(self.frame(), 10)
        self.gate.offer(self.frame('b', 10.1), 10.1)
        self.assertEqual(self.dispatch(10.1).frame.frame_id, 'b')

    def test_only_one_call_in_flight_and_interval_applies(self):
        self.gate.offer(self.frame(), 10)
        job = self.dispatch()
        self.gate.offer(self.frame('b', 10.2), 10.2)
        self.assertIsNone(self.dispatch(10.2))
        self.assertTrue(self.gate.finish(job, self.context))
        self.assertIsNone(self.dispatch(10.3))
        self.gate.offer(self.frame('c', 11), 11)
        self.assertIsNotNone(self.dispatch(11))

    def test_stale_future_nonfinite_and_out_of_order_frames_rejected(self):
        for at in (9.0, 10.1, float('nan'), float('inf')):
            self.assertFalse(self.gate.offer(self.frame(at=at), 10))
        self.assertTrue(self.gate.offer(self.frame(), 10))
        self.assertFalse(self.gate.offer(self.frame('old', 9.9), 10))

    def test_frame_that_expires_in_queue_is_not_dispatched(self):
        self.gate.offer(self.frame(), 10)
        self.assertIsNone(self.dispatch(10.8))
        self.assertIsNone(self.gate.pending)

    def test_scene_change_rejects_previous_frame(self):
        self.gate.offer(self.frame(scene=0), 10)
        self.assertIsNone(self.dispatch())

    def test_context_changes_discard_results_and_release_capacity(self):
        changes = [dict(pair_revision=2), dict(profile_revision=2),
                   dict(recipient_id='household'), dict(turn_id='new'),
                   dict(session_id='new')]
        for change in changes:
            gate = CaptureGate()
            gate.offer(self.frame(), 10)
            job = gate.dispatch(self.context, 10, session_active=True,
                                user_requested=True, scene_stable=True)
            self.assertFalse(gate.finish(job, replace(self.context, **change)))
            self.assertIsNone(gate.inflight)

    def test_automatic_requests_require_stability_and_active_session(self):
        self.gate.offer(self.frame(), 10)
        self.assertIsNone(self.gate.dispatch(self.context, 10, session_active=True,
                                            user_requested=False, scene_stable=False))
        self.assertIsNone(self.gate.dispatch(self.context, 10, session_active=False,
                                            user_requested=True, scene_stable=True))
        self.assertIsNotNone(self.gate.dispatch(self.context, 10, session_active=True,
                                               user_requested=False, scene_stable=True))

    def test_cancelled_and_duplicate_completions_cannot_apply(self):
        self.gate.offer(self.frame(), 10)
        job = self.dispatch()
        self.assertFalse(self.gate.finish(job, None))
        self.assertFalse(self.gate.finish(job, self.context))

    def test_size_and_configuration_limits(self):
        self.assertFalse(self.gate.offer(replace(self.frame(), jpeg=b''), 10))
        self.assertFalse(self.gate.offer(replace(self.frame(), jpeg=b'x'*2_000_001), 10))
        with self.assertRaises(ValueError):
            CaptureGate(min_interval=float('nan'))


if __name__ == '__main__':
    unittest.main()
