import copy
import hashlib
import unittest

import reconcile_pilot_v2 as m


class ReconciliationGates(unittest.TestCase):
    def test_whole_endpoint_required(self):
        m.complete_status(dict(status="complete", attempts=24, planned_attempts=24))
        for value in [dict(status="partial", attempts=12, planned_attempts=24),
                      dict(status="complete", attempts=12, planned_attempts=24),
                      dict(status="complete", attempts=24, planned_attempts=120), {}]:
            with self.assertRaises(ValueError):
                m.complete_status(value)

    def test_appendable_prefix_exact(self):
        raw = b'{"fixture":1}\n'
        receipt = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), append_only_prefix_must_match=True)
        self.assertTrue(m.prefix_check(raw + b'{"fixture":2}\n', receipt)["prefix_identical"])
        for value in [raw.replace(b"1", b"2"), raw[:-1], raw + b'{"uncommitted":']:
            with self.assertRaises(ValueError):
                m.prefix_check(value, receipt)

    def test_immutable_prefix_cannot_grow(self):
        raw = b'{}\n'
        receipt = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), must_remain_entire_file_identical=True)
        m.prefix_check(raw, receipt)
        with self.assertRaises(ValueError):
            m.prefix_check(raw + b'{}\n', receipt)

    def test_exact_successful_stage(self):
        step = dict(name="fixture", args=["--fixture"])
        events = [dict(event="stage_start", name="fixture", command=m.expected_stage_command(step), utc="2026-09-10T18:01:00+00:00"),
                  dict(event="stage_exit", name="fixture", exit_code=0, utc="2026-09-10T18:02:00+00:00")]
        m.check_stage(events, step)
        for bad in [events[:1], events + events[:1]]:
            with self.assertRaises(ValueError):
                m.check_stage(bad, step)
        bad = copy.deepcopy(events)
        bad[1]["exit_code"] = 1
        with self.assertRaises(ValueError):
            m.check_stage(bad, step)

    def lease_fixture(self):
        spec = dict(args=["--plan", "fabricated-test-only"])
        completion = dict(exit_code=0, gpu=3, elapsed_seconds=120)
        events = [dict(event="acquired", utc="2026-09-10T18:00:00+00:00", gpu=dict(index=3),
                       command=[str(m.ROOT / ".venv/bin/python"), "-u", str(m.ROOT / "code/pipeline.py"), *spec["args"]],
                       entry_sha256=m.digest(m.ROOT / "code/pipeline.py")),
                  dict(event="completed", utc="2026-09-10T18:02:00+00:00", **completion)]
        return spec, completion, events

    def test_lease_binds_terminal_and_pipeline(self):
        spec, completion, events = self.lease_fixture()
        m.check_lease(events, spec, completion, [dict(utc="2026-09-10T18:01:00+00:00")])
        for key, value in [("entry_sha256", "bad"), ("command", [])]:
            bad = copy.deepcopy(events)
            bad[0][key] = value
            with self.assertRaises(ValueError):
                m.check_lease(bad, spec, completion, [])
        bad = copy.deepcopy(completion)
        bad["elapsed_seconds"] = 121
        with self.assertRaises(ValueError):
            m.check_lease(events, spec, bad, [])

    def test_stage_outside_lease_rejected(self):
        spec, completion, events = self.lease_fixture()
        with self.assertRaises(ValueError):
            m.check_lease(events, spec, completion, [dict(utc="2026-09-10T18:03:00+00:00")])

    def test_cancelled_owner_stays_failed(self):
        spec, _, events = self.lease_fixture()
        completion = dict(exit_code=1, gpu=3, status="failed", error="KeyboardInterrupt", utc="2026-09-10T18:02:00+00:00")
        events[-1] = dict(event="failed", error="KeyboardInterrupt", utc=completion["utc"])
        m.check_lease(events, spec, completion, [], failed=True)
        with self.assertRaises(ValueError):
            m.check_lease(events, spec, completion, [], failed=False)


if __name__ == "__main__":
    unittest.main()
