"""Tests of lh_dpo.py and the shared scoring in lh_tune.py (TUNING.md).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
        /scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python -m pytest --confcutdir=. -p no:cacheprovider \
        -q test_lh_dpo.py

A tiny random ALiBi model (lh_tune.tiny_base, 2 x 64) on toy pairs from l19: the same proof with every premise note
in form orF versus For (equal length, so a random model starts near a tie). The real base is used, when present,
only for the log-probability check; a reiteration twin pair (hand-written, and one from c24 through lh_twin when it
is there) pins the zero gradient on the rows before the rated site.
"""

# ruff: noqa: E501
from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lh_dpo  # noqa: E402
import lh_tune as U  # noqa: E402

L19 = U.LH / "data" / "l19.jsonl.gz"
C24 = U.LH / "data" / "c24.jsonl.gz"
REAL_BASE = Path("/scratch/cowifornia/nd-rl/two_authors/bases/stage1_authors_half_s0.pt")
pytestmark = pytest.mark.skipif(not L19.exists(), reason="no l19 corpus under LH")
ABS, REL = U.ACC_ABS, U.ACC_REL
SNAPS = "5,10,20"

# One reiteration site: the A member restates the premise Q inside the inner box (N4 : R N1); the B member cites N1.
REIT_PROMPT = "THM Q SEQ ( ( ~ ( ~ R ) ) > ( S > ( ( R > S ) v ( ( ~ ( ~ P ) ) v Q ) ) ) ) PRF"
REIT_A = ("N1 Q : PR ; N2 | ( ~ ( ~ R ) ) : AS ; N3 | | S : AS ; N4 | | Q : R N1 ; N5 | | ( ( ~ ( ~ P ) ) v Q ) : ORI2 N4 ; "
          "N6 | | ( ( R > S ) v ( ( ~ ( ~ P ) ) v Q ) ) : ORI2 N5 ; N7 | ( S > ( ( R > S ) v ( ( ~ ( ~ P ) ) v Q ) ) ) : IMPI N3 N6 ; "
          "N8 ( ( ~ ( ~ R ) ) > ( S > ( ( R > S ) v ( ( ~ ( ~ P ) ) v Q ) ) ) ) : IMPI N2 N7 ; QED")  # fmt: skip
REIT_B = ("N1 Q : PR ; N2 | ( ~ ( ~ R ) ) : AS ; N3 | | S : AS ; N4 | | ( ( ~ ( ~ P ) ) v Q ) : ORI2 N1 ; "
          "N5 | | ( ( R > S ) v ( ( ~ ( ~ P ) ) v Q ) ) : ORI2 N4 ; N6 | ( S > ( ( R > S ) v ( ( ~ ( ~ P ) ) v Q ) ) ) : IMPI N3 N5 ; "
          "N7 ( ( ~ ( ~ R ) ) > ( S > ( ( R > S ) v ( ( ~ ( ~ P ) ) v Q ) ) ) ) : IMPI N2 N6 ; QED")  # fmt: skip


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    d = tmp_path_factory.mktemp("dpo")
    base = U.tiny_base(d / "tiny.pt")
    recs = U.toy_records(48, seed=1, forms=("orF", "For"))
    for r in recs:
        r.pop("corpus_proof")
    U.write_jsonl(d / "train.jsonl", recs[:32])
    U.write_jsonl(d / "held.jsonl", recs[32:])
    # the same held-out pairs with the members swapped: the tiny base's systematic preference is then the rater's
    U.write_jsonl(
        d / "rev.jsonl",
        [{**r, "preferred": r["other"], "other": r["preferred"]} for r in recs[32:]],
    )
    ab = U.toy_records(24, seed=2)
    U.write_jsonl(
        d / "kl.jsonl", [{"prompt": r["prompt"], "proof": r.pop("corpus_proof")} for r in ab]
    )
    U.write_jsonl(d / "ab.jsonl", ab)
    return d, base


def run(d: Path, base: Path, name: str, *extra: str, pairs: str = "train.jsonl", held: str = "held.jsonl",
        steps: int = 30) -> dict:  # fmt: skip
    return lh_dpo.main(["--base", str(base), "--pairs", str(d / pairs), "--heldout", str(d / held),
                        "--out", str(d / name), "--device", "cpu", "--batch", "8", "--max-steps", str(steps),
                        "--lr", "3e-3", "--eval-every", "5", "--kl-states", str(d / "kl.jsonl"), "--kl-n", "8",
                        "--snap-steps", SNAPS, *extra])  # fmt: skip


@pytest.fixture(scope="module")
def real_run(toy: tuple[Path, Path]) -> tuple[Path, dict]:
    d, base = toy
    sha = U.sha256(base)
    doc = run(d, base, "real")
    assert U.sha256(base) == sha, "the base file changed"
    return d / "real", doc


def rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def model_and_tok(base: Path):
    U.setup("cpu")
    m, tok, _ = U.load_base(base, "cpu")
    return m, tok


# ---------------------------------------------------------------------------------------------- encoding, scoring
def test_states_are_untagged_whatever_the_process_tag(toy, monkeypatch):
    d, base = toy
    monkeypatch.setenv("AUTHOR_TAG", "A")
    _, tok = model_and_tok(base)
    r = U.read_jsonl(d / "train.jsonl")[0]
    rs = U.encode_proof(tok, r["prompt"], r["preferred"])
    assert tok.author_tag is None
    for row in rs:
        ids = list(row.ids)
        assert ids[row.plen - 1] == tok.act_id and ids[-1] == tok.eos
        assert not set(ids) & set(tok.tag_ids.values())
    with pytest.raises(U.Unreadable):
        U.encode_proof(tok, r["prompt"], "N1 garbage ; QED")


def stepwise_logp(model, tok, prompt: str, proof: str) -> float:
    """log pi(proof) one step at a time, written independently of lh_tune's batching."""
    import torch
    from state_env import decompose  # type: ignore[import-not-found]

    steps, _, _ = decompose(prompt, proof, canon=True)
    tok.author_tag = "none"
    tot = 0.0
    with torch.no_grad():
        for st, act, _ in steps:
            p, q = tok.encode_toks(st), tok.encode_toks(act) + [tok.eos]
            ids = torch.tensor([p + q])
            lp = torch.log_softmax(model(ids[:, :-1]).float(), -1)[0]
            tot += sum(float(lp[len(p) - 1 + i, t]) for i, t in enumerate(q))
    tok.author_tag = None
    return tot


@pytest.mark.parametrize("which", ["tiny", "real"])
def test_proof_logp_is_the_sum_over_steps(toy, which):
    d, base = toy
    if which == "real":
        if not REAL_BASE.exists():
            pytest.skip("no real base")
        base = REAL_BASE
    m, tok = model_and_tok(base)
    ctx = U.Ctx("cpu", tok.ref0, 300, False)  # small chunks: several per proof
    for r in U.read_jsonl(d / "train.jsonl")[:3]:
        for f in ("preferred", "other"):
            got = U.proof_logp(m, U.encode_proof(tok, r["prompt"], r[f]), ctx)
            want = stepwise_logp(m, tok, r["prompt"], r[f])
            assert got == pytest.approx(want, rel=1e-5, abs=1e-3)


def test_shared_rows_cancel_in_the_margin(toy):
    d, base = toy
    m, tok = model_and_tok(base)
    ctx = U.Ctx("cpu", tok.ref0, 4000, False)
    pairs, rep = U.load_pairs(d / "ab.jsonl", tok)
    assert rep["rows_shared_share"] > 0
    cut = U.pair_margins(m, pairs, ctx)
    full = U.pair_margins(m, pairs, ctx, full=True)
    assert cut.tolist() == pytest.approx(full.tolist(), abs=2e-3)
    shifts = [min(p.room, 7) for p in pairs]  # the same offset on both proofs of a pair
    sums = []
    for flag in (False, True):
        rs, sh, owner, sign = U.flatten_pairs(pairs, shifts, full=flag)
        sums.append(
            U.pair_sums(U.apply_rows(U.logp_fn(m), rs, sh, ctx), owner, sign, len(pairs), "cpu")
        )
    assert sums[0].tolist() == pytest.approx(sums[1].tolist(), abs=2e-3)
    assert sums[0].tolist() != pytest.approx(cut.tolist(), abs=1e-3)  # the shift changes the values


def reit_twins() -> list[tuple[str, str, str]]:
    """(prompt, A member, B member): the hand-written pair, and a c24 record's reiteration twin (notes and step order
    ahead of the site) when lh_twin and c24 are there."""
    out = [(REIT_PROMPT, REIT_A, REIT_B)]
    if C24.exists():
        try:
            for r in U.twin_records(40, seed=3, habit="reit", field="proof", corpus=C24, scan=3000):
                if (
                    "N2 " in r["preferred"] and ": PR ; N2" in r["preferred"]
                ):  # premises, hence notes, ahead
                    out.append((r["prompt"], r["preferred"], r["other"]))
                    break
        except Exception:  # lh_twin / lh_profile not importable: the hand-written pair still runs
            pass
    return out


@pytest.mark.parametrize("k", [0, 1])
def test_rows_before_the_rated_site_get_no_gradient(toy, tmp_path, k):
    """PREREG Amendment 3 point 1 (audit F2): with split_common, the logits of every row before the first differing
    row (shared by both members: for c24 the notes and step order) get exactly zero gradient from the DPO loss; the
    rows that enter the trainer's forward pass are exactly the kept ones; and without the cancellation those rows
    would get a gradient (so the test can fail)."""
    import torch

    tw = reit_twins()
    if k >= len(tw):
        pytest.skip("no c24 reiteration twin with premises (lh_twin or c24 missing)")
    prompt, wa, wb = tw[k]
    d, base = toy
    m, tok = model_and_tok(base)
    rw, rl = U.encode_proof(tok, prompt, wa), U.encode_proof(tok, prompt, wb)
    first = next(i for i, (x, y) in enumerate(zip(rw, rl, strict=False)) if x != y)
    assert first >= 3 if k == 0 else first >= 2
    kw, kl, _ = U.common_mask(rw, rl)
    assert not any(kw[:first]) and not any(kl[:first]) and kw[first] and kl[first]
    U.write_jsonl(
        tmp_path / "p.jsonl", [{"key": "t", "prompt": prompt, "preferred": wa, "other": wb}]
    )
    pairs, _ = U.load_pairs(tmp_path / "p.jsonl", tok)
    trows, _, _, _ = U.flatten_pairs(pairs)
    assert trows == [r for r, x in zip(rw, kw, strict=True) if x] + [
        r for r, x in zip(rl, kl, strict=True) if x
    ]
    assert not set(rw[:first]) & set(trows)
    ctx = U.Ctx("cpu", tok.ref0, 10**6, False)
    x, plen, lens = U.pack(rw + rl, None, ctx)
    sign = torch.tensor([1.0] * len(rw) + [-1.0] * len(rl))

    def logits_grad(keep: list[bool]) -> torch.Tensor:
        with torch.no_grad():
            z = m(x[:, :-1]).float()
        z.requires_grad_(True)
        lp = torch.log_softmax(z, -1).gather(-1, x[:, 1:, None]).squeeze(-1)
        per_row = torch.where(U.action_mask(x, plen, lens), lp, 0.0).sum(1)
        dlt = (per_row * sign * torch.tensor(keep, dtype=torch.float32)).sum()
        (-torch.nn.functional.logsigmoid(0.1 * (dlt - 0.0))).backward()
        return z.grad, dlt.detach()

    g, dlt = logits_grad(kw + kl)
    assert float(dlt) == pytest.approx(float(U.pair_margins(m, pairs, ctx)[0]), abs=1e-3)
    n = len(rw)
    for i in range(first):
        assert torch.count_nonzero(g[i]) == 0 and torch.count_nonzero(g[n + i]) == 0, (
            f"row {i} got a gradient"
        )
    assert torch.count_nonzero(g[first]) > 0 and torch.count_nonzero(g[n + first]) > 0
    g_full, _ = logits_grad(
        [True] * (len(rw) + len(rl))
    )  # no cancellation: each copy gets its own gradient
    assert torch.count_nonzero(g_full[0]) > 0


def test_chunked_gradient_equals_one_pass(toy):
    import torch

    d, base = toy
    m, tok = model_and_tok(base)
    pairs, _ = U.load_pairs(d / "train.jsonl", tok)
    rs, sh, owner, sign = U.flatten_pairs(pairs[:4], [1, 0, 2, 3])

    def loss_fn(o):
        return -torch.nn.functional.logsigmoid(0.1 * U.pair_sums(o, owner, sign, 4, "cpu")).mean()

    grads = []
    for mt in (10**6, 400):
        m.zero_grad()
        ctx = U.Ctx("cpu", tok.ref0, mt, False)
        assert (len(U.chunk_indices([len(r.ids) for r in rs], mt)) == 1) == (mt == 10**6)
        U.objective_backward(U.logp_fn(m), rs, sh, loss_fn, ctx)
        grads.append(torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None]))
    assert torch.allclose(grads[0], grads[1], rtol=1e-4, atol=1e-6)


def fake_pairs(sides: dict[str, int]) -> list:
    out = []
    for side, n in sides.items():
        out += [U.Pair(f"{side}{i}", "p", [], [], [], [], 0, False, side) for i in range(n)]
    return out


def test_swap_set_is_a_seeded_half_within_each_side():
    pairs = fake_pairs({"A": 101, "B": 60, "None": 3})
    s = lh_dpo.swap_set(pairs, 0)
    by = {k: sum(pairs[i].side == k for i in s) for k in ("A", "B", "None")}
    assert by == {"A": 50, "B": 30, "None": 1}
    assert s == lh_dpo.swap_set(pairs, 0) and s != lh_dpo.swap_set(pairs, 1)


def side_counts(pairs, swap, batch):
    return {k: (sum(i in swap for i in batch if pairs[i].side == k), sum(pairs[i].side == k for i in batch))
            for k in sorted({pairs[i].side for i in batch})}  # fmt: skip


def test_couples_balance_every_batch_within_each_side():
    """PREREG Amendment 7 point 3 (audit T4): with an even batch, every batch holds exactly as many swap-set pairs as
    others within each sampled side; an odd side's extra pair sits out one epoch at a time (a different one); the
    swap set never changes."""
    import random

    pairs = fake_pairs({"A": 41, "B": 30})
    swap = lh_dpo.swap_set(pairs, 7)
    cp = lh_dpo.Couples(pairs, swap, 8, random.Random(1))
    assert cp.n_couples == 20 + 15
    seen, sat = [], set()
    for _ in range(40):  # 8 batches per epoch (70 pairs, the last 6 unused): 5 epochs
        b = cp.next()
        assert len(b) == 8 and len(set(b)) == 8
        for k, (sw, n) in side_counts(pairs, swap, b).items():
            assert 2 * sw == n, (k, sw, n)
        seen += b
        sat.update(cp.sat_out)
    assert (
        cp.epochs == 5 and len(sat) > 1 and all(pairs[i].side == "A" and i not in swap for i in sat)
    )
    assert set(seen) == set(
        range(len(pairs))
    )  # every pair trains (the sat-out one changes per epoch)


def test_couples_odd_batch_alternates_and_balances_in_twos():
    import random

    pairs = fake_pairs({"A": 10, "B": 10})
    swap = lh_dpo.swap_set(pairs, 0)
    cp = lh_dpo.Couples(pairs, swap, 5, random.Random(0))
    for _ in range(4):  # 20 pairs = 4 batches of 5 per epoch: pairs of consecutive batches
        b1, b2 = cp.next(), cp.next()
        c1, c2 = side_counts(pairs, swap, b1), side_counts(pairs, swap, b2)
        assert abs(sum(sw for sw, _ in c1.values()) - 2.5) == 0.5
        for k in set(c1) | set(c2):
            sw = c1.get(k, (0, 0))[0] + c2.get(k, (0, 0))[0]
            n = c1.get(k, (0, 0))[1] + c2.get(k, (0, 0))[1]
            assert 2 * sw == n, (k, c1, c2)


def test_the_swap_set_indexes_the_loaded_pairs(toy, tmp_path):
    """Audit T10: a pair that does not decompose is dropped before the swap set is drawn, so each side still has
    exactly floor(n_loaded / 2) swapped; swapping trades the members' rows."""
    d, base = toy
    _, tok = model_and_tok(base)
    recs = U.read_jsonl(d / "ab.jsonl")
    bad = {**recs[0], "key": "bad", "preferred": "N1 garbage ; QED"}
    U.write_jsonl(tmp_path / "p.jsonl", [bad, *recs])
    pairs, rep = U.load_pairs(tmp_path / "p.jsonl", tok, max_unreadable=0.5)
    assert rep["unreadable"] == 1 and len(pairs) == len(recs) and pairs[0].key == recs[0]["key"]
    swap = lh_dpo.swap_set(pairs, 0)
    for k, n in rep["sampled_side_loaded"].items():
        assert sum(pairs[i].side == k for i in swap) == n // 2
    sw = U.swap_pairs(pairs, swap)
    for i, (p, q) in enumerate(zip(pairs, sw, strict=True)):
        if i in swap:
            assert (
                q.swapped
                and (q.rw, q.rl, q.dw, q.dl) == (p.rl, p.rw, p.dl, p.dw)
                and q.room == p.room
            )
        else:
            assert q == p


def test_snap_plan_grid_and_drops():
    """PREREG Amendment 7 point 1: the grid 5 ... 400, 400 steps, lr 3e-5 by default."""
    a = lh_dpo.parse(["--base", "b", "--pairs", "p", "--heldout", "h", "--out", "o"])
    assert a.lr == 3e-5 and a.max_steps == 400 and a.swap_seed == a.seed == 0
    thr, steps, info = U.snap_plan(a)
    assert (
        thr == ()
        and steps == [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 400]
        and not info
    )
    a = lh_dpo.parse(
        [
            "--base",
            "b",
            "--pairs",
            "p",
            "--heldout",
            "h",
            "--out",
            "o",
            "--max-steps",
            "300",
            "--seed",
            "4",
        ]
    )
    thr, steps, info = U.snap_plan(a)
    assert steps[-1] == 300 and info["dropped_past_max_steps"] == [400] and a.swap_seed == 4
    a = lh_dpo.parse(
        ["--base", "b", "--pairs", "p", "--heldout", "h", "--out", "o", "--acc-snapshots"]
    )
    assert U.snap_plan(a)[0] == U.THRESHOLDS and U.snap_plan(a)[1] == list(U.GRID)


def test_one_pass_packs_as_the_reference_does(toy):
    """Audit T11: the policy's one-pass training forward packs the rows exactly as apply_rows (the reference's
    no-grad pass) does, so at step 0 the outputs are bitwise equal and every margin is exactly 0."""
    import torch

    d, base = toy
    m, tok = model_and_tok(base)
    pairs, _ = U.load_pairs(d / "train.jsonl", tok)
    rs, sh, owner, sign = U.flatten_pairs(pairs[:6], [1, 0, 2, 3, 0, 1])
    ctx = U.Ctx("cpu", tok.ref0, 10**6, False)
    _, o = U.objective_backward(U.logp_fn(m), rs, sh, lambda x: x.sum(), ctx)
    ref = U.apply_rows(U.logp_fn(m), rs, sh, ctx)
    assert torch.equal(o, ref)


# ---------------------------------------------------------------------------------------------- the trainer
def test_dpo_moves_the_preference_and_snapshots_on_the_grid(real_run):
    out, doc = real_run
    m = rows(out / "metrics.jsonl")
    assert m[0]["step"] == 0 and m[0][REL] == 0.0 and m[0]["kl"] == pytest.approx(0.0, abs=1e-6)
    assert (
        doc["base_heldout_acc_abs"] <= 0.5 and doc["final"][ABS] >= 0.9 and doc["final"][REL] >= 0.9
    )
    tr = rows(out / "train.jsonl")
    assert len(tr) == 30 and tr[0]["loss"] == pytest.approx(0.6931, abs=1e-3)
    # the step-0 check (PREREG Amendment 7 point 7): in fp32 on cpu the margins are exactly 0
    assert tr[0]["step0_ok"] is True and tr[0]["step0_loss"] == pytest.approx(
        U.STEP0_LOSS, abs=1e-6
    )
    assert tr[0]["step0_margin_absmax"] == 0.0 and not any("step0_loss" in r for r in tr[1:])
    sides = collections.Counter(r["sampled_side"] for r in U.read_jsonl(out.parent / "train.jsonl"))
    assert doc["step0"]["step0_ok"] is True
    assert doc["batching"]["couples_per_epoch"] == sum(k // 2 for k in sides.values())
    assert sum(r["loss"] for r in tr[-5:]) / 5 < 0.3
    assert doc["final"]["heldout_margin_rel"] > 20 and doc["final"]["kl"] > 0
    assert doc["ref_frozen"] is True and doc["thresholds"] == [] and doc["acc_snapshots"] is False
    assert doc["snap_steps"] == [5, 10, 20] and set(doc["definitions"]) >= {ABS, REL, "kl"}
    assert "affects" in doc["seed"]
    by = {r["step"]: r for r in m}
    for k in (5, 10, 20):
        s = doc["snapshots"][f"step{k}"]
        assert s["step"] == k and (out / f"snap_step{k}.pt").exists()
        r = by[
            k
        ]  # a metrics row at every snapshot step, with both accuracies, the margin, the loss and the KL
        assert r["snapshots"] == [f"step{k}"] and r["kl"] is not None and r["loss"] is not None
        assert r[ABS] is not None and r[REL] is not None and r["train_margin_rel"] is not None
    assert (
        not list(out.glob("snap_acc*.pt"))
        and (out / "final.pt").exists()
        and (out / "done.json").exists()
    )


def test_accuracy_snapshots_behind_the_flag(toy):
    d, base = toy
    out = d / "accsnap"
    doc = run(d, base, "accsnap", "--acc-snapshots", "--snap-steps", "")
    m = rows(out / "metrics.jsonl")
    assert (
        doc["thresholds"] == [60, 75, 90] and doc["not_reached"] == [] and doc["snap_metric"] == ABS
    )
    for t in (60, 75, 90):
        s = doc["snapshots"][f"acc{t}"]
        assert s["step"] > 0 and s[ABS] >= t / 100 and (out / s["file"]).exists()
        assert all(r[ABS] < t / 100 for r in m if r["step"] < s["step"]), (
            "not the first passing evaluation"
        )


def test_swap_random_control_and_its_seeds(toy, real_run):
    d, base = toy
    out_real, doc_real = real_run
    docs = {
        s: run(d, base, f"swap_s{s}", "--swap-random", "--seed", str(s), steps=20) for s in (0, 1)
    }
    sides = collections.Counter(r["sampled_side"] for r in U.read_jsonl(d / "train.jsonl"))
    for s, doc in docs.items():
        assert doc["swapped"] == sum(n // 2 for n in sides.values()) and doc["swap_seed"] == s
        assert doc["swap_by_side"] == {
            k: {"pairs": n, "swap_set": n // 2} for k, n in sides.items()
        }
        assert doc["snap_steps"] == [5, 10, 20]
        assert all((d / f"swap_s{s}" / f"snap_step{k}.pt").exists() for k in (5, 10, 20))
        # a rater with no preference: the held-out relative margin barely moves (tiny-model ties make the
        # accuracies themselves jumpy, so the margin is the quantity compared)
        assert (
            abs(doc["final"]["heldout_margin_rel"]) < 0.1 * doc_real["final"]["heldout_margin_rel"]
        )
        for r in rows(
            d / f"swap_s{s}" / "train.jsonl"
        ):  # exactly half of every batch, within each side
            assert all(2 * sw == n for sw, n in r["swapped_by_side"].values()), r["swapped_by_side"]
    # PREREG Amendment 7 point 3: the two control seeds have different orientations (swap seed = run seed)
    assert docs[0]["swapped_sha256"] != docs[1]["swapped_sha256"]
    assert (
        docs[0]["swapped_sha256"] == doc_real["swap_set_sha256"]
        and doc_real["swapped_sha256"] is None
    )
    # the rated run and its control of one seed see the same pairs in the same batches, with the same offsets
    t_real, t_swap = rows(out_real / "train.jsonl"), rows(d / "swap_s0" / "train.jsonl")
    assert [(r["rows"], r["tokens"]) for r in t_real[:20]] == [
        (r["rows"], r["tokens"]) for r in t_swap
    ]
    l0 = [r["loss"] for r in rows(d / "swap_s0" / "train.jsonl")]
    l1 = [r["loss"] for r in rows(d / "swap_s1" / "train.jsonl")]
    assert l0[1:] != l1[1:], "--seed did not change the data order"
    doc2 = run(d, base, "swap_other", "--swap-random", "--seed", "1", "--swap-seed", "0", steps=2)
    assert (
        doc2["swapped_sha256"] == docs[0]["swapped_sha256"]
    )  # --swap-seed alone sets the orientation
    doc3 = run(d, base, "swap_match", "--swap-random", "--match-steps", str(out_real), steps=12)
    assert doc3["snap_steps"] == [5, 10] and doc3["match"]["dropped_past_max_steps"] == [20]


def test_snapshots_load_like_stage1_bases(toy, real_run):
    """Through Dan's `model.load_ckpt` after `tag_core.install_dan()` (authors_measure.run / lh_measure) and
    `tag_core.ckpt_mode` / `ckpt_meta` (its checks): the same keys, tokenizer mode and tag metadata as the base."""
    import torch
    from model import load_ckpt  # type: ignore[import-not-found]

    d, base = toy
    out, doc = real_run
    b = torch.load(str(base), map_location="cpu", weights_only=False)
    for f in ("snap_step5.pt", "snap_step20.pt", "final.pt"):
        p = out / f
        ck = torch.load(str(p), map_location="cpu", weights_only=False)
        assert set(ck) == set(b) == {"cfg", "state", "tok_mode", "extra"}
        assert sorted(ck["state"]) == sorted(b["state"]) and ck["cfg"] == b["cfg"]
        m, tok, extra = load_ckpt(str(p), "cpu")
        assert tok.mode == U.T.MODE == U.T.ckpt_mode(p)
        assert U.T.ckpt_meta(p) == U.T.ckpt_meta(base)
        assert (
            extra["author_tags"] == b["extra"]["author_tags"]
            and extra["tuning"]["trainer"] == "dpo"
        )
        side = json.loads(p.with_suffix(".tag.json").read_text())
        assert side["ckpt"] == str(p) and side["tuning"]["snapshot"] == extra["tuning"]["snapshot"]
    fin = torch.load(str(out / "final.pt"), map_location="cpu", weights_only=False)["state"]
    assert any(not torch.equal(fin[k], b["state"][k]) for k in fin)


def test_snapshot_loads_in_the_grpo_package(real_run):
    """Leon's GRPO loader (tag_core.install_thinking, code.thinking.ckpt.load_ckpt), as an RL line started from a
    snapshot would load it: same weights as Dan's loader, flat state format."""
    out, _ = real_run
    f = str(out / "snap_step20.pt")
    code = (f"import sys; sys.path.insert(0, {str(HERE)!r}); import lh_tune as U; U.T.install_thinking(); "
            f"from code.thinking import ckpt; m, tok, extra = ckpt.load_ckpt({f!r}, 'cpu'); "
            f"import torch; s = sum(float(v.double().sum()) for v in m.state_dict().values()); "
            f"print(tok.mode, type(m).__name__, m.state_format, extra['tuning']['snapshot'], repr(s))")  # fmt: skip
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4"}
    p = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=str(out)
    )
    assert p.returncode == 0, p.stderr[-2000:]
    mode, cls, fmt, snap, s = p.stdout.split()[-5:]
    assert (mode, cls, fmt, snap) == (U.T.MODE, "ALiBiGPT", "flat", "step20")
    m, _ = model_and_tok(Path(f))
    assert float(s) == pytest.approx(
        sum(float(v.double().sum()) for v in m.state_dict().values()), rel=1e-9
    )


def test_base_already_past_the_thresholds(toy):
    import torch

    d, base = toy
    doc = run(d, base, "atbase", "--acc-snapshots", pairs="rev.jsonl", held="rev.jsonl", steps=2)
    assert doc["base_heldout_acc_abs"] >= 0.9 and doc["at_base"] == [60, 75, 90]
    assert all(doc["snapshots"][f"acc{t}"]["at_base"] for t in (60, 75, 90))
    b = torch.load(str(base), map_location="cpu", weights_only=False)["state"]
    s = torch.load(str(d / "atbase" / "snap_acc90.pt"), map_location="cpu", weights_only=False)[
        "state"
    ]
    assert all(torch.equal(s[k], b[k]) for k in b)


def test_refuses_an_existing_run(toy, real_run):
    d, base = toy
    with pytest.raises(SystemExit):
        run(d, base, "real")


def test_step0_check_stops_before_any_update(toy, monkeypatch):
    """PREREG Amendment 7 point 7 (audit T11): a reference that is not the policy (here another random base) gives a
    step-0 loss away from ln 2: exit 4 before the optimizer moves, the stopped train.jsonl row and step0_failed.json
    written, no snapshot and no done.json."""
    d, base = toy
    other = U.tiny_base(d / "tiny_other.pt", seed=5)
    real_load, calls = U.load_base, []

    def load(path, device):
        calls.append(path)
        return real_load(
            other if len(calls) == 2 else path, device
        )  # the second load is the reference

    monkeypatch.setattr(U, "load_base", load)
    with pytest.raises(SystemExit) as e:
        run(d, base, "step0bad", steps=5)
    assert e.value.code == U.STEP0_EXIT == 4
    out = d / "step0bad"
    tr = rows(out / "train.jsonl")
    assert len(tr) == 1 and tr[0]["step0_ok"] is False and "STOPPED" in tr[0]["stopped"]
    assert abs(tr[0]["step0_loss"] - U.STEP0_LOSS) > U.STEP0_TOL
    assert json.loads((out / "step0_failed.json").read_text())["step0_ok"] is False
    assert not (out / "done.json").exists() and not list(out.glob("*.pt"))
    assert [r["step"] for r in rows(out / "metrics.jsonl")] == [0]
