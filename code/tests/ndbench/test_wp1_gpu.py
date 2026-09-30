"""`runner.gpu` (plan §3 WP1): `lease` and `budget_guard`, using a temporary `AB_GPULOCK_DIR` so
these tests never touch the real, shared GPU lock files. CPU-only: `gpulock.py` is stdlib-only
(plain `flock(2)` on regular files) and no CUDA call is ever made here."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from code.ndbench.runner.gpu import BudgetExceeded, budget_guard, lease
from code.ndbench.store import Store


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "out")


@pytest.fixture(autouse=True)
def _temp_gpulock_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AB_GPULOCK_DIR", str(tmp_path / "locks"))
    # Fix list item 8: `lease()` now *requires* CUDA_VISIBLE_DEVICES=4 to already be set (the only
    # supported way to launch a GPU job), rather than setting it itself -- so every test other than
    # the ones specifically checking that precondition simulates a correctly-launched process here.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")


def test_lease_succeeds_when_cuda_visible_devices_is_already_4_and_appends_ledger(
    store: Store,
) -> None:
    import os

    with lease("unit-test-job", est_s=1.0, store=store) as gpu:
        assert gpu == 4
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "4"  # unchanged; lease() no longer sets it

    entries = list(store.ledger_path.read_text().splitlines())
    assert len(entries) == 1
    import json

    row = json.loads(entries[0])
    assert row["job"] == "unit-test-job"
    assert row["gpu"] == 4
    assert row["status"] == "ok"
    assert row["gpu_s"] >= 0.0
    assert store.ledger_total_s() == pytest.approx(row["gpu_s"])


def test_lease_refuses_when_cuda_visible_devices_is_not_preset(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fix list item 8's core change: `lease()` must refuse -- before ever touching `gpulock` --
    unless `CUDA_VISIBLE_DEVICES == "4"` already holds when it is entered."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with (
        pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"),
        lease("job", est_s=1.0, store=store),
    ):
        pass  # pragma: no cover - must never be entered
    assert store.ledger_total_s() == 0.0  # nothing was leased; no ledger entry either


def test_lease_refuses_when_cuda_visible_devices_is_the_wrong_gpu(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")  # belongs to another user (plan §A2)
    with (
        pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"),
        lease("job", est_s=1.0, store=store),
    ):
        pass  # pragma: no cover - must never be entered
    assert store.ledger_total_s() == 0.0


def test_lease_releases_the_lock_so_a_second_lease_can_acquire_it(store: Store) -> None:
    with lease("job-1", est_s=1.0, store=store) as gpu1:
        assert gpu1 == 4
    with lease("job-2", est_s=1.0, store=store) as gpu2:
        assert gpu2 == 4
    assert store.ledger_total_s() > 0.0


def test_lease_appends_ledger_entry_even_when_the_body_raises(store: Store) -> None:
    class _Boom(Exception):
        pass

    with pytest.raises(_Boom), lease("job-error", est_s=1.0, store=store):
        raise _Boom("body failed")

    rows = [r for r in _read_ledger(store)]
    assert len(rows) == 1
    assert rows[0]["status"] == "error"


def test_lease_rejects_a_gpu_other_than_4(store: Store) -> None:
    with pytest.raises(ValueError), lease("job", est_s=1.0, store=store, gpu=0):
        pass  # pragma: no cover - must never be entered
    # nothing was leased, so no ledger entry should have been written
    assert store.ledger_total_s() == 0.0


def test_budget_guard_allows_a_small_estimate(store: Store) -> None:
    remaining = budget_guard(store, est_s=60.0, cap_h=9.5)
    assert remaining == pytest.approx(9.5)


def test_budget_guard_refuses_when_estimate_exceeds_the_cap(store: Store) -> None:
    with pytest.raises(BudgetExceeded):
        budget_guard(store, est_s=100 * 3600.0, cap_h=9.5)


def test_budget_guard_accounts_for_ledgered_spend(store: Store) -> None:
    store.ledger_append({"job": "prior", "gpu_s": 8.0 * 3600.0})
    # 8h spent; a 1h estimate needs 8 + 1.5 = 9.5h <= 9.5h cap -> allowed (boundary)
    remaining = budget_guard(store, est_s=3600.0, cap_h=9.5)
    assert remaining == pytest.approx(1.5)
    # a slightly larger estimate must be refused
    with pytest.raises(BudgetExceeded):
        budget_guard(store, est_s=3601.0, cap_h=9.5)


def _read_ledger(store: Store) -> list[dict[str, Any]]:
    import json

    return [json.loads(line) for line in store.ledger_path.read_text().splitlines() if line.strip()]
