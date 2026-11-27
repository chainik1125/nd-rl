"""Tests of steer/steer_prep.py: the F2 sets (pure function), disjointness with the reach prompts, and, when the built inputs
exist on serrano, the built files (manifest hashes, texts, pid files) plus a rebuild-is-identical check.
Run: pytest -q tests/test_steer_prep.py"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "steer"))
import steer_common as C  # noqa: E402
import steer_prep as P  # noqa: E402

BUILT = P.OUT


def synth():
    arm = [f"ifbench_{i}" for i in range(218)]
    reach = arm[:42] + [
        f"ifeval_{i}" for i in range(6)
    ]  # 42 of the 48 reach prompts are arm-A prompts (as in the real data)
    pool = [
        f"ifbench_{i}" for i in range(60, 130)
    ]  # pool-train order; includes reach ids 60.. none; add two reach to test skipping
    pool[3] = reach[5]
    return arm, reach, pool


def test_make_sets_sizes_disjoint_and_reproducible():
    arm, reach, pool = synth()
    s = P.make_sets(arm, reach, pool)
    assert (
        len(s["N"]) == 176 and len(s["vec"]) == 32 and len(s["dev"]) == 64 and len(s["held"]) == 80
    )
    assert (
        reach[5] not in s["vec"]
    )  # a pool-train pid that is a reach prompt is skipped; next lines are taken
    assert s["vec"][:3] == pool[:3] and s["vec"][3] == pool[4]
    P.check_disjoint(s)
    for k in ("N", "vec", "dev", "held"):
        assert not set(reach) & set(s[k])
    assert (
        not set(s["vec"]) & set(s["dev"])
        and not set(s["dev"]) & set(s["held"])
        and not set(s["vec"]) & set(s["held"])
    )
    s2 = P.make_sets(
        list(reversed(arm)), reach, pool
    )  # the input order of the pid file does not matter
    assert s2["dev"] == s["dev"] and s2["held"] == s["held"] and s2["vec"] == s["vec"]
    # the split is RandomState(0).permutation of the sorted remaining pids
    rest = sorted(p for p in s["N"] if p not in set(s["vec"]))
    perm = np.random.RandomState(0).permutation(len(rest))
    assert s["dev"] == [rest[i] for i in perm[:64]] and s["held"] == [rest[i] for i in perm[64:144]]


def test_check_disjoint_raises_on_a_reach_prompt_in_a_set():
    arm, reach, pool = synth()
    s = P.make_sets(arm, reach, pool)
    s["dev"][0] = reach[0]
    with pytest.raises(AssertionError):
        P.check_disjoint(s)


def test_pid_files_draw_counts_and_e2_totals(tmp_path):
    man = {
        "sets": {
            "reach": [f"r{i}" for i in range(48)],
            "held": [f"h{i}" for i in range(80)],
            "dev": [f"d{i}" for i in range(64)],
        }
    }
    P.write_pids(tmp_path, man, reach_n=8, held_n=2, dev_n=2)

    def read(name):
        return dict(
            (ln.split(":")[0], int(ln.split(":")[1]))
            for ln in (tmp_path / name).read_text().split()
        )

    e1 = read("pids_e1.txt")
    assert (
        sum(e1[p] for p in man["sets"]["reach"]) == 384
        and sum(e1[p] for p in man["sets"]["held"]) == 160
    )
    assert (
        sum(read("pids_e0p_reach.txt").values()) == 384
        and sum(read("pids_dev_guard.txt").values()) == 128
    )
    e2 = [read(f"pids_e2_s{k}.txt") for k in (1, 2, 3)]
    assert sum(sum(x.values()) for x in e2) == 544 == sum(e1.values())
    for i, h in enumerate(man["sets"]["held"]):
        assert [k for k in range(3) if h in e2[k]] == [i % 3]
    # the cut values are parameters: E0' = 192 draws, held-out 1, dev 1
    P.write_pids(tmp_path, man, reach_n=8, held_n=1, dev_n=1, e0_reach_n=4)
    assert (
        sum(read("pids_e0p_reach.txt").values()) == 192
        and sum(read("pids_dev_guard.txt").values()) == 64
    )
    assert sum(read("pids_e1.txt").values()) == 384 + 80
    # no reach prompt appears in the dev-guard file; no non-reach prompt in the E0' file
    assert not set(read("pids_dev_guard.txt")) & set(man["sets"]["reach"])
    assert set(read("pids_e0p_reach.txt")) == set(man["sets"]["reach"])


@pytest.mark.skipif(not (BUILT / "manifest.json").exists(), reason="inputs not built")
def test_built_inputs_are_consistent_and_never_contain_reach_prompts_outside_the_reach_cells():
    man = json.loads((BUILT / "manifest.json").read_text())
    reach = set(P.read_pids(P.DAN / "olmo_ifcurve" / "pids_reach.txt"))
    arm = set(P.read_pids(P.DAN / "olmo_ifcurve" / "pids_armA.txt"))
    s = man["sets"]
    assert set(s["reach"]) == reach and len(s["reach"]) == 48
    assert (
        len(s["vec32"]) == 32
        and len(s["dev"]) == 64
        and len(s["held"]) == 80
        and len(s["S16"]) == 16
        and len(s["H16"]) == 16
    )
    for k in ("vec32", "dev", "held"):
        assert set(s[k]) <= arm and not set(s[k]) & reach
    assert set(s["S16"]) <= set(s["dev"]) and set(s["H16"]) <= set(s["held"])
    # vec32 = lines 0-31 of pool/train.jsonl (all are in N, so none is skipped)
    pool = [json.loads(ln)["pid"] for ln in (P.POOL / "train.jsonl").read_text().splitlines()]
    assert s["vec32"] == pool[:32]
    # the stored hashes match the files on disk
    assert man["texts_jsonl_sha256"] == C.sha256_file(BUILT / "texts.jsonl")
    for k, v in man["inputs"].items():
        assert C.sha256_file(v["path"]) == v["sha256"], k
    # texts: every record's pid is in the right set, no reach pid, segment indices sane, marker consistent with the ids
    roles = {}
    for ln in (BUILT / "texts.jsonl").read_text().splitlines():
        r = json.loads(ln)
        roles[r["role"]] = roles.get(r["role"], 0) + 1
        assert r["pid"] not in reach
        assert 1 <= r["lead_end"] <= r["trail_start"] <= r["n_prompt"]
        resp = C.ids_from_b64(r["resp_ids"])
        assert len(resp) == r["n_resp"] and C.find_marker(resp) == r["think_end"]
        assert r["n_resp"] <= (12000 if r["role"] in ("vec_dpo", "sel_dpo") else 12000)
        assert {
            "vec_dpo": s["vec32"],
            "vec_s2300": s["vec32"],
            "sel_s2300": s["S16"],
            "sel_dpo": s["S16"],
            "chk_s2300": s["H16"],
        }[r["role"]].count(r["pid"]) == 1
    assert roles == {
        "vec_dpo": 32,
        "vec_s2300": 32,
        "sel_s2300": 16,
        "sel_dpo": 16,
        "chk_s2300": 16,
    }
    # pre-flight sequences come from dev prompts; baseline is the 16-draw job
    for sq in json.loads((BUILT / "preflight_seqs.json").read_text()):
        assert sq["pid"] in s["dev"] and len(C.ids_from_b64(sq["ids"])) == sq["n"]
    base = json.loads((BUILT / "dpo_baseline.json").read_text())
    assert set(base["dev"]) == set(s["dev"]) and set(base["held"]) == set(s["held"])
    assert all(len(v) == 16 for k in base for v in base[k].values())
    # pid files agree with the sets
    e1 = dict(x.split(":") for x in (BUILT / "pids_e1.txt").read_text().split())
    assert {p for p in e1 if p in reach} == reach and {p for p in e1 if p not in reach} == set(
        s["held"]
    )
    assert not (
        set(dict(x.split(":") for x in (BUILT / "pids_dev_guard.txt").read_text().split())) & reach
    )


@pytest.mark.skipif(not (BUILT / "manifest.json").exists(), reason="inputs not built")
def test_rebuild_is_reproducible(tmp_path):
    old = json.loads((BUILT / "manifest.json").read_text())
    new = P.build(tmp_path)
    assert new["texts_jsonl_sha256"] == old["texts_jsonl_sha256"]
    assert new["sets"] == old["sets"] and new["inputs"] == old["inputs"]
    for f in ("preflight_seqs.json", "dpo_baseline.json", "pids_e1.txt", "pids_e2_s2.txt"):
        assert (tmp_path / f).read_bytes() == (BUILT / f).read_bytes(), f
