# ruff: noqa: E501
"""Amendment 9: a9.py (plan, budget decision, reading rule, push scan, adapter check) and pod/drive_a9.sh run against
stub ssh / mini_pod.sh / clock (no pod, no GPU)."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import a9
import pytest

HERE = Path(__file__).resolve().parent
V2 = Path("/scratch/cowifornia/nd-rl/olmo_elicit/v2")


# ---------------------------------------------------------------------------------------------- a9.py
def test_plan_is_the_stage1_rule_for_a_full_size_cell():
    (c,) = a9.PLAN["cells"]
    assert c["cell"] == "dpo__final__A16__n16" and c["pool"] == "dpo/final" and c["n"] == 16
    assert c["family"] == "lora:r=16:targets=all" and c["edge_extend"] is True
    assert (
        c["lrs"] == [1e-4, 3e-4, 1e-3]
        and c["steps"] == 64
        and c["eval_steps"] == [4, 8, 16, 32, 64]
    )
    assert "seed" not in c and "fixed" not in c and "recipe_from" not in c and "extend" not in c
    # the same grid, flags and options as the stage-1 plans of the reference and of the 4-example cell
    for plan, cell in (("plan_dpo_h200b.json", a9.REF), ("plan_dpo_h100b.json", a9.N4)):
        if not (V2 / plan).exists():
            pytest.skip("stage-1 plans not on this machine")
        old = next(x for x in json.loads((V2 / plan).read_text())["cells"] if x["cell"] == cell)
        assert (old["lrs"], old["family"], old["pool"], old["edge_extend"]) == (
            c["lrs"], c["family"], c["pool"], c["edge_extend"])  # fmt: skip


def test_plan_schedule_is_the_trainers_default_for_16_examples():
    mt = pytest.importorskip("mini_train")
    (c,) = a9.PLAN["cells"]
    assert mt.default_eval_steps(16, 64) == c["eval_steps"]
    assert mt.lr_neighbor(1e-4, up=False) == 3e-5 and mt.lr_neighbor(1e-3, up=True) == 3e-3


def test_plan_file_roundtrip(tmp_path):
    f = tmp_path / "plan_dpo_a9.json"
    assert a9.main(["plan", "--check", str(f)]) == 1
    assert a9.main(["plan", "--out", str(f)]) == 0 and json.loads(f.read_text()) == a9.PLAN
    assert a9.main(["plan", "--check", str(f)]) == 0 and a9.main(["plan", "--out", str(f)]) == 0
    f.write_text(f.read_text().replace("64", "128"))
    assert a9.main(["plan", "--out", str(f)]) == 1 and a9.main(["plan", "--check", str(f)]) == 1
    assert "128" in f.read_text()  # not overwritten


def test_budget_numbers():
    assert round(a9.stop_s()) == 10208  # $15 at $5.29/h = 170 min
    assert round(a9.CELL_S) == 2119  # 384 draws x 8,000 tokens at 1,450 tokens per second
    assert a9.sample_plan_s(3) == pytest.approx(3 * a9.CELL_S + 900)


def test_decide_drops_whole_cells_in_the_preregistered_order():
    three, two = [a9.RANK1, a9.N16, a9.N4], [a9.RANK1, a9.N4]
    d = a9.decide("train", 8 * 60)
    assert d["cells"] == three and d["train_n16"] and d["planned_end_s"] <= d["stop_s"]
    # before training: three cells need training + 5 min + 3 cells + 15 min = about 158 min
    edge = a9.stop_s() - (a9.TRAIN_S + a9.TRAIN_XFER_S + a9.sample_plan_s(3))
    assert a9.decide("train", edge - 1)["cells"] == three
    late = a9.decide("train", edge + 1)
    assert late["cells"] == two and not late["train_n16"] and late["dropped"] == [a9.N16]
    # before sampling: three cells fit until about minute 49, two until about minute 84
    e3, e2 = a9.stop_s() - a9.sample_plan_s(3), a9.stop_s() - a9.sample_plan_s(2)
    assert 48 * 60 < e3 < 50 * 60 and 83 * 60 < e2 < 86 * 60
    assert (
        a9.decide("sample", e3 - 1)["cells"] == three
        and a9.decide("sample", e3 + 1)["cells"] == two
    )
    assert (
        a9.decide("sample", e2 + 1)["cells"] == [] and a9.decide("sample", e2 + 1)["option"] is None
    )
    assert a9.decide("sample", 60, have_n16=False)["cells"] == two
    # the optional re-sample of the reference never fits under $15 and is the first to go
    four = a9.decide("sample", 40 * 60, want_ref=True)
    assert four["cells"] == three and four["dropped"] == [a9.REF]
    assert a9.decide("sample", 40 * 60, want_ref=True, cap_usd=18.5)["cells"] == [*three, a9.REF]
    assert a9.decide("train", 0, want_ref=True)["cells"] == three
    # the wait deadline of the sample stage leaves the last fetch inside the money stop
    d = a9.decide("sample", 2400)
    assert (
        d["sample_deadline_s"] == round(a9.stop_s() - a9.FETCH_S - 2400)
        and d["train_deadline_s"] == 0
    )
    # the wait for training ends where three cells stop fitting: about minute 46 of pod time, whenever it starts
    for e in (0, 300, 700):
        t = a9.decide("train", e)
        assert t["train_deadline_s"] == round(
            a9.stop_s() - a9.sample_plan_s(3) - a9.TRAIN_FETCH_S - e
        )
        assert 46 * 60 < e + t["train_deadline_s"] < 47 * 60 and t["train_deadline_s"] >= a9.TRAIN_S
    assert a9.decide("train", edge + 1)["train_deadline_s"] == 0
    with pytest.raises(ValueError):
        a9.decide("other", 0)


def test_decide_cli(capsys):
    assert a9.main(["decide", "--phase", "sample", "--elapsed-s", "2400"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert json.loads(out[0])["cells"] == [a9.RANK1, a9.N16, a9.N4]
    assert out[-1] == f"CELLS={a9.RANK1},{a9.N16},{a9.N4}"
    assert a9.main(["decide", "--phase", "sample", "--elapsed-s", "9000"]) == 3
    assert capsys.readouterr().out.splitlines()[-1] == "CELLS="


@pytest.mark.parametrize(
    ("diff", "lo", "hi", "want"),
    [
        (0.0, -8.0, 8.0, "holds"),
        (-4.9, -11.9, 3.0, "holds"),
        (-5.0, -11.0, 1.0, "unclear with 48 prompts"),  # the difference must be ABOVE -5
        (-3.0, -12.0, 6.0, "unclear with 48 prompts"),  # the lower end must be ABOVE -12
        (-3.0, -13.5, 7.0, "unclear with 48 prompts"),
        (-7.0, -15.0, 1.0, "unclear with 48 prompts"),
        (-10.0, -18.0, -2.0, "unclear with 48 prompts"),  # the difference must be BELOW -10
        (-10.1, -18.0, 0.0, "unclear with 48 prompts"),  # the upper end must be BELOW 0
        (-10.1, -18.0, -0.1, "falls short"),
        (-20.0, -29.0, -11.0, "falls short"),
        (6.0, -2.0, 14.0, "holds"),
    ],
)
def test_reading_rule(diff, lo, hi, want):
    assert a9.reading(diff, lo, hi) == want


def test_prereg_states_the_numbers_the_code_uses():
    note = (HERE / "PREREG.md").read_text()
    am = note[note.index("## Amendment 9 (") :]
    for cell in (a9.RANK1, a9.N16, a9.N4, a9.REF):
        assert f"`{cell}`" in am
    assert f"engine seed {a9.SEED_SAMPLE}" in am and f"`{a9.BATCH}`" in am
    assert "the difference is above −5 and the interval's lower end is above −12" in am
    assert "the difference is below −10 and the interval's upper end is below 0" in am
    assert (a9.HOLDS_DIFF, a9.HOLDS_LO, a9.SHORT_DIFF, a9.SHORT_HI) == (-5, -12, -10, 0)
    assert (
        "hard stop is $15" in am and a9.CAP_USD == 15 and "$5.29/h" in am and a9.RATE_USD_H == 5.29
    )
    assert "8,000 tokens per draw" in am and "1,450 tokens per second" in am
    assert (a9.TOKENS_PER_DRAW, a9.TOKENS_PER_S) == (8000, 1450)
    assert a9.TRAIN_ARGS in am
    assert all(m[:8] + "…" in am for m in (a9.WEIGHTS_MD5[a9.RANK1], a9.WEIGHTS_MD5[a9.N4]))


def test_prereg_changes_after_the_audit_match_the_code():
    note = (HERE / "PREREG.md").read_text()
    j = " ".join(
        note[note.index("### 9(j) Changes after the independent audit") :].split()
    )  # line breaks aside
    assert "15 minutes of overhead per batch" in j and a9.SAMPLE_OVERHEAD_S == 15 * 60
    stop = a9.stop_s()
    assert "taken by minute 49" in j and int((stop - a9.sample_plan_s(3)) / 60) == 49
    assert "set-up ends by minute 12" in j
    assert int((stop - a9.TRAIN_S - a9.TRAIN_XFER_S - a9.sample_plan_s(3)) / 60) == 12
    assert "only until minute 46" in j
    assert int((stop - a9.sample_plan_s(3) - a9.TRAIN_FETCH_S) / 60) == 46
    assert (
        "at minute 165" in j
        and round((stop - a9.FETCH_S) / 60) == 165
        and "which is 170 minutes" in j
    )
    drv = (HERE / "pod" / "drive_a9.sh").read_text()
    assert "2,000 MiB" in j and "FREE_MIB=${A9_GPU_FREE_MIB:-2000}" in drv
    assert (
        "`/workspace/wi/gpu.lock`" in j
        and "OTHER_LOCK=${A9_OTHER_LOCK-/workspace/wi/gpu.lock}" in drv
    )
    # the table of what the rule does: a normal estimate with standard error 4.3 and an interval of +-8.5
    from statistics import NormalDist

    for true, holds, unclear, short in (
        (0, 79, 20, 1),
        (-5, 36, 52, 12),
        (-10, 7, 44, 50),
        (-15, 0, 12, 88),
    ):
        d = NormalDist(true, 4.3)
        h = 1 - d.cdf(max(a9.HOLDS_DIFF, a9.HOLDS_LO + 8.5))
        f = d.cdf(min(a9.SHORT_DIFF, a9.SHORT_HI - 8.5))
        assert (round(100 * h), round(100 * f)) == (holds, short), true
        assert abs(100 * (1 - h - f) - unclear) < 1.5
        assert f"| {str(true).replace('-', '−')} | {holds} % | {unclear} % | {short} % |" in j
    pool = V2 / "pools" / "dpo" / "final" / "train.jsonl"
    if pool.exists():
        tok = [json.loads(ln)["n_tok"] for ln in pool.read_text().splitlines()]
        assert (sum(tok[:4]), sum(tok[:16]), sum(tok)) == (13110, 55747, 245487)
        assert "13,110 tokens × 16" in j and "55,747 × 4" in j and "245,487 × 2" in j


def test_scan_refuses_sealed_and_credential_names(tmp_path):
    ok = tmp_path / "ok"
    (ok / "pools/dpo/final").mkdir(parents=True)
    (ok / "pools/dpo/final/train.jsonl").write_text('{"pid": "ifbench_832x"}\n')
    (ok / "mini_train.py").write_text("x = 1\n")
    (ok / "seeds_L.csv").write_text("dpo,0,640000\n")
    assert a9.scan(ok) == [] and a9.main(["scan", "--stage-dir", str(ok)]) == 0
    for name, kind in (
        ("data/textbook72.jsonl", "sealed"), ("test_13_20.json", "sealed"), ("sets/test_21_30/a.txt", "sealed"),
        ("theorems_832.jsonl", "sealed"), ("832/x.txt", "sealed"), ("sealed_ids.txt", "sealed"),
        (".env", "credential"), ("keys/id_ecdsa", "credential"), ("hf_token.txt", "credential"),
        ("a/.ssh/config", "credential"), ("runpod.pem", "credential"),
    ):  # fmt: skip
        d = tmp_path / re.sub(r"\W", "_", name)
        (d / name).parent.mkdir(parents=True)
        (d / name).write_text("x\n")
        bad = a9.scan(d)
        assert bad and kind in bad[0], (name, bad)
    t = tmp_path / "tok"
    t.mkdir()
    (t / "notes.md").write_text("export HF=hf_" + "a" * 34 + "\n")
    assert (
        a9.scan(t) == ["token-like text: notes.md"]
        and a9.main(["scan", "--stage-dir", str(t)]) == 1
    )
    assert (
        a9.FORBIDDEN.search("pools/s0832/own") is None and a9.FORBIDDEN.search("x8321.txt") is None
    )


def test_this_folders_push_would_pass_the_scan(tmp_path):
    """The files mini_pod.sh push copies from this folder (its find rule), by name and for token-like text."""
    st = tmp_path / "st"
    (st / "pod").mkdir(parents=True)
    for p in HERE.iterdir():
        if p.is_file() and p.suffix in (".py", ".sh", ".md") and not p.name.startswith("test_"):
            shutil.copy(p, st / p.name)
    for p in (HERE / "pod").iterdir():
        if p.suffix in (".sh", ".py"):
            shutil.copy(p, st / "pod" / p.name)
    assert a9.scan(st) == []


def test_verify_reused_adapters_on_disk():
    if not (V2 / a9.SOURCES[a9.RANK1]).exists():
        pytest.skip("stage-1 runs not on this machine")
    assert a9.verify(V2, with_ref=True) == []


def test_verify_reports_problems(tmp_path):
    bad = a9.verify(tmp_path)
    assert any("pool manifest" in b for b in bad) and sum("no DONE" in b for b in bad) == 2


# ------------------------------------------------------------------------------ the driver against stubs
N0 = 1_800_000_000  # the stub clock starts here; `sleep N` advances it by N


def _fake_cell(d: Path, cell: str, r: int = 16, did_not_train: bool = False) -> str:
    (d / "adapter").mkdir(parents=True)
    (d / "adapter" / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": r}))
    (d / "adapter" / "adapter_model.safetensors").write_bytes(cell.encode())
    (d / "gate.npz").write_bytes(b"gate")
    cj = {"cell": cell, "family": "f", "P": 1, "n": 4, "source": "final", "selected": {"lr": 1e-4, "step": 4},
          "did_not_train": did_not_train, "pool_manifest_sha256": "POOLSHA", "selection": "dev loss",
          "reload_check_max_abs_diff": 0.0}  # fmt: skip
    (d / "cell.json").write_text(json.dumps(cj))
    (d / "DONE").write_text("x\n")
    return a9.file_digest(d / "adapter" / "adapter_model.safetensors", "md5")


def _exe(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def rig(tmp_path):
    """A copy of the driver, its library, a9.py and stage_cells.py in a scratch folder, with a scratch V2 holding small
    stand-in adapters, and stubs for ssh, mini_pod.sh, date +%s and sleep."""
    w, v2, fake, fbin = tmp_path / "w", tmp_path / "v2", tmp_path / "fake", tmp_path / "bin"
    for d in (w / "pod", fake, fbin, v2 / "pools/dpo/final"):
        d.mkdir(parents=True)
    shutil.copy(HERE / "stage_cells.py", w)
    for f in ("drive_a9.sh", "drive_lib.sh"):
        shutil.copy(HERE / "pod" / f, w / "pod")
    (v2 / "pools/dpo/final/MANIFEST.json").write_text("{}\n")
    md5 = {
        a9.RANK1: _fake_cell(v2 / "runs_rp-om1/dpo/train/cells" / a9.RANK1, a9.RANK1, r=1),
        a9.N4: _fake_cell(v2 / "runs_rp-om1/dpo/train/cells" / a9.N4, a9.N4),
        a9.REF: _fake_cell(v2 / "runs_rp-om2/dpo/train/cells" / a9.REF, a9.REF),
    }
    _fake_cell(fake / "n16cell", a9.N16)
    src = (HERE / "a9.py").read_text()
    for (
        cell,
        m,
    ) in (
        md5.items()
    ):  # the stand-ins' checksums in place of the recorded ones; the logic is untouched
        src = src.replace(a9.WEIGHTS_MD5[cell], m)
    src = src.replace(
        a9.POOL_MANIFEST_SHA256, a9.file_digest(v2 / "pools/dpo/final/MANIFEST.json", "sha256")
    )
    src = src.replace('"POOLSHA"', '"x"')
    (w / "a9.py").write_text(src)
    for d in (v2 / "runs_rp-om1", v2 / "runs_rp-om2"):
        for cj in d.glob("dpo/train/cells/*/cell.json"):
            cj.write_text(
                cj.read_text().replace(
                    "POOLSHA", a9.file_digest(v2 / "pools/dpo/final/MANIFEST.json", "sha256")
                )
            )
    (v2 / "plan_dpo_a9.json").write_text(a9.plan_text())
    (fake / "now").write_text(str(N0))
    (fake / "train_probes").write_text(
        "25\n"
    )  # training takes 25 stub minutes unless a test says otherwise
    real_date = shutil.which("date")
    _exe(
        fbin / "date",
        f'#!/bin/bash\nif [ "$*" = "+%s" ]; then cat "$FAKE/now"; else exec {real_date} "$@"; fi\n',
    )
    _exe(fbin / "sleep", '#!/bin/bash\necho $(( $(cat "$FAKE/now") + ${1%.*} )) > "$FAKE/now"\n')
    _exe(
        fbin / "ssh",
        """#!/bin/bash
cmd="${@: -1}"
echo "ssh: $cmd" | head -c 200 >> "$FAKE/calls"; echo >> "$FAKE/calls"
case "$cmd" in
  *"test -e /workspace/mini"*) cat "$FAKE/fresh" 2>/dev/null || echo FRESH ;;
  *"query-gpu=memory.used"*) cat "$FAKE/gpu" 2>/dev/null || echo 3 ;;
  *train_plan_dpo_a9.DONE*)  # RUNNING for $FAKE/train_probes probes (one stub minute each), then DONE
    if [ -f "$FAKE/probe_train" ]; then cat "$FAKE/probe_train"; else
      n=$(cat "$FAKE/n" 2>/dev/null || echo 0); echo $((n + 1)) > "$FAKE/n"
      if [ "$n" -lt "$(cat "$FAKE/train_probes")" ]; then echo "RUNNING 5"; else echo DONE; fi
    fi ;;
  *sample_batch_a9.DONE*) cat "$FAKE/probe_sample" 2>/dev/null || echo DONE ;;
  *) echo "unexpected ssh command" >&2; exit 9 ;;
esac
""",
    )
    _exe(
        w / "pod" / "mini_pod.sh",
        f"""#!/bin/bash
echo "mini_pod: $* TO_SAMPLE=${{TO_SAMPLE:-}} TO_TRAIN=${{TO_TRAIN:-}} GPU_FREE_MIB=${{GPU_FREE_MIB:-}} ALSO_LOCK=${{ALSO_LOCK:-}}" >> "$FAKE/calls"
cmd=$1; A=$2
case $cmd in
  push) mkdir -p "$V2/pod_stage/$A"; cp "$(dirname "$0")/../a9.py" "$V2/pod_stage/$A/"
        [ -f "$FAKE/bad_push" ] && touch "$V2/pod_stage/$A/textbook72.jsonl"; echo "+ rsync (dry)" ;;
  setup) [ -f "$FAKE/setup_fail" ] && {{ echo "SETUP FAILED"; exit 1; }}; echo "setup done" ;;
  train) [ -f "$FAKE/chain_refused" ] && {{ echo "GPU_BUSY: 131000 MiB in use"; echo "NOT STARTED: dpo on $A"; exit 1; }}
         touch "$FAKE/trained"; echo "started dpo" ;;
  sample) cp "$4/cells_manifest.json" "$FAKE/sampled_manifest.json"; touch "$FAKE/sampling"; echo "started dpo" ;;
  stop) touch "$FAKE/stopped"; echo "process groups signalled" ;;
  fetch)
    if [ -f "$FAKE/sampling" ] && [ -f "$FAKE/final_fetch_fail" ]; then echo "FETCH_MISMATCH dpo: 1 of 9 files differ"; exit 1; fi
    if [ -f "$FAKE/trained" ] && [ ! -f "$FAKE/no_n16" ]; then
      mkdir -p "$V2/runs_$A/dpo/train/cells"; cp -r "$FAKE/n16cell" "$V2/runs_$A/dpo/train/cells/{a9.N16}"
    fi
    mkdir -p "$V2/runs_$A/dpo"; echo "FETCH_OK dpo: 9 files (1M) md5-equal" ;;
  *) echo "unexpected mini_pod command $cmd"; exit 9 ;;
esac
""",
    )

    def run(alias="rp-t1", **env):
        for f in ("calls", "trained", "sampling", "stopped", "sampled_manifest.json", "n"):
            (fake / f).unlink(missing_ok=True)
        (fake / "now").write_text(str(N0))
        e = {**os.environ, "PATH": f"{fbin}:{os.environ['PATH']}", "V2": str(v2), "FAKE": str(fake),
             "PY": "python3", **{k: str(v) for k, v in env.items()}}  # fmt: skip
        r = subprocess.run(["bash", str(w / "pod" / "drive_a9.sh"), alias], env=e, capture_output=True,
                           text=True, timeout=120)  # fmt: skip
        calls = (fake / "calls").read_text() if (fake / "calls").exists() else ""
        man = fake / "sampled_manifest.json"
        cells = sorted(json.loads(man.read_text())["cells"]) if man.exists() else None
        return r.returncode, r.stdout + r.stderr, calls, cells

    run.fake, run.v2, run.w = fake, v2, w
    return run


def test_driver_happy_path(rig):
    rc, log, calls, cells = rig()
    assert rc == 0, log
    assert cells == sorted([a9.RANK1, a9.N16, a9.N4])
    assert (
        "A9 DONE on rp-t1" in log
        and "NEEDS A DECISION" not in log
        and "POD READY TO DELETE" not in log
    )
    order = [ln.split()[1] for ln in calls.splitlines() if ln.startswith("mini_pod:")]
    assert order == ["push", "setup", "train", "fetch", "sample", "fetch"]
    assert "--dry-run" in next(ln for ln in calls.splitlines() if ln.startswith("mini_pod: push"))
    train = next(ln for ln in calls.splitlines() if ln.startswith("mini_pod: train"))
    assert (
        "--plan" in train
        and "plan_dpo_a9.json" in train
        and "--train-args --ce-checkpoint --max-memory-gb 78" in train
    )
    sample = next(ln for ln in calls.splitlines() if ln.startswith("mini_pod: sample"))
    assert f"--seed {a9.SEED_SAMPLE}" in sample and "/stage/batch_a9" in sample
    # the pod-side sampling cap is later than the driver's own money stop, which is 5 min before $15
    to_sample = int(re.search(r"TO_SAMPLE=(\d+)", sample).group(1))
    assert to_sample - 1200 == int(a9.stop_s()) - a9.FETCH_S - 25 * 60 and to_sample > 7000
    # the GPU is checked before anything is pushed, before training and before the engine starts, and both chains are
    # told to check again on the pod and to hold the other job's lock
    assert calls.count("query-gpu=memory.used") == 3
    assert calls.index("query-gpu=memory.used") < calls.index("mini_pod: push")
    for ln in (train, sample):
        assert "GPU_FREE_MIB=2000" in ln and "ALSO_LOCK=/workspace/wi/gpu.lock" in ln
    for f in ("decide_train_rp-t1.json", "decide_sample_rp-t1.json"):
        assert json.loads((rig.v2 / "a9" / f).read_text())["cells"] == [a9.RANK1, a9.N16, a9.N4]


def test_driver_goes_on_without_the_new_adapter_when_training_fails(rig):
    (rig.fake / "probe_train").write_text("FAILED\n")
    (rig.fake / "no_n16").touch()
    rc, log, calls, cells = rig()
    assert rc == 0, log
    assert cells == sorted([a9.RANK1, a9.N4]) and "not trained" in log and "A9 DONE" in log


def test_driver_late_start_skips_training_and_samples_two_cells(rig):
    rc, log, calls, cells = rig(A9_T0=N0 - 55 * 60)
    assert rc == 0, log
    assert cells == sorted([a9.RANK1, a9.N4]) and "mini_pod: train" not in calls
    assert "the 16-example adapter is not trained" in log


def test_driver_stops_a_slow_training_while_the_two_reused_cells_still_fit(rig):
    # the stub clock jumps 60 s per probe; a training stage that is still running at minute 46 can no longer be
    # followed by three cells, so it is stopped there and does not eat the budget of the reused cells
    (rig.fake / "train_probes").write_text("80\n")
    rc, log, calls, cells = rig()
    assert rc == 0, log
    assert cells == sorted([a9.RANK1, a9.N4]), log
    assert (
        "OVERRUN train_plan_dpo_a9" in log
        and "the 16-example adapter is not trained (stage rc 2)" in log
    )
    order = [ln.split()[1] for ln in calls.splitlines() if ln.startswith("mini_pod:")]
    assert order == ["push", "setup", "train", "stop", "fetch", "sample", "fetch"]
    dec = json.loads((rig.v2 / "a9" / "decide_sample_rp-t1.json").read_text())
    assert (
        dec["cells"] == [a9.RANK1, a9.N4]
        and 46 * 60 <= dec["planned_end_s"] - a9.sample_plan_s(2) <= 48 * 60
    )
    assert "A9 DONE" in log


def test_driver_training_that_ends_in_time_is_sampled(rig):
    (rig.fake / "train_probes").write_text("43\n")
    rc, log, calls, cells = rig()
    assert rc == 0 and cells == sorted([a9.RANK1, a9.N16, a9.N4]) and "mini_pod: stop" not in calls


def test_driver_nothing_fits(rig):
    rc, log, calls, cells = rig(A9_T0=N0 - 100 * 60)
    assert rc == 8 and cells is None
    assert (
        "A9 STOPPED rc=8" in log
        and "mini_pod: train" not in calls
        and "mini_pod: sample" not in calls
    )


def test_driver_money_stop_during_sampling_stops_the_chain_fetches_and_reports(rig):
    (rig.fake / "probe_sample").write_text("RUNNING 5\n")
    rc, log, calls, cells = rig()
    assert rc == 7, log
    assert "OVERRUN sample_batch_a9" in log and (rig.fake / "stopped").exists()
    assert (
        "A9 STOPPED rc=7" in log
        and "money stop arrived during sampling" in log
        and "reported as bounds" in log
    )
    order = [ln.split()[1] for ln in calls.splitlines() if ln.startswith("mini_pod:")]
    assert order[-2:] == ["stop", "fetch"]
    # the stop came 5 minutes before the $15 mark of this job's pod time
    assert int((rig.fake / "now").read_text()) - N0 <= a9.stop_s() - a9.FETCH_S + 120


def test_driver_failed_sample_stage_is_not_called_a_money_stop(rig):
    (rig.fake / "probe_sample").write_text("FAILED\n")
    rc, log, calls, cells = rig()
    assert rc == 7 and "NEEDS A DECISION" in log and "it was not the money stop" in log
    assert "mini_pod: stop" not in calls and "A9 STOPPED" not in log


def test_driver_refuses_a_busy_gpu_before_anything_is_sent(rig):
    (rig.fake / "gpu").write_text("131072\n")
    rc, log, calls, cells = rig()
    assert rc == 4 and "is not idle (131072 MiB in use, limit 2000)" in log
    assert "mini_pod:" not in calls and "A9 STOPPED" in log
    assert calls.count("query-gpu=memory.used") == 13  # asked for two minutes before giving up
    (rig.fake / "gpu").write_text("")  # no answer from nvidia-smi is not "free"
    rc, log, calls, cells = rig()
    assert rc == 4 and "no answer" in log and "mini_pod:" not in calls


def test_driver_chain_refused_on_the_pod_needs_a_decision(rig):
    (rig.fake / "chain_refused").touch()
    rc, log, calls, cells = rig()
    assert rc == 5 and "train start failed" in log and "NEEDS A DECISION" in log and cells is None


def test_driver_other_lock_can_be_left_alone(rig):
    rc, log, calls, cells = rig(A9_OTHER_LOCK="")
    assert rc == 0, log
    train = next(ln for ln in calls.splitlines() if ln.startswith("mini_pod: train"))
    assert train.endswith("GPU_FREE_MIB=2000 ALSO_LOCK=")


def test_driver_refuses_a_used_pod_a_changed_plan_and_an_existing_batch(rig):
    (rig.fake / "fresh").write_text("USED\n")
    rc, log, calls, _ = rig()
    assert rc == 2 and "not a fresh pod" in log and "mini_pod:" not in calls
    (rig.fake / "fresh").unlink()
    plan = rig.v2 / "plan_dpo_a9.json"
    plan.write_text(plan.read_text().replace("0.0003", "0.0004"))
    rc, log, calls, _ = rig()
    assert rc == 3 and "is not a9.py's plan" in log and calls == ""
    plan.write_text(a9.plan_text())
    (rig.v2 / "stage" / "batch_a9").mkdir(parents=True)
    rc, log, calls, _ = rig()
    assert rc == 3 and "A9_RESTAGE=1" in log
    rc, log, calls, cells = rig(A9_RESTAGE=1)
    assert rc == 0 and len(list((rig.v2 / "stage").glob("batch_a9.old.*"))) == 1 and len(cells) == 3


def test_driver_refuses_a_push_with_a_sealed_name(rig):
    (rig.fake / "bad_push").touch()
    rc, log, calls, _ = rig()
    assert rc == 3 and "sealed-set or credential name" in log and "mini_pod: setup" not in calls


def test_driver_changed_adapter_is_refused(rig):
    f = rig.v2 / "runs_rp-om1/dpo/train/cells" / a9.N4 / "adapter" / "adapter_model.safetensors"
    f.write_bytes(b"other weights")
    rc, log, calls, _ = rig()
    assert rc == 3 and "differs from its recorded md5" in log and "ssh:" not in calls


def test_driver_final_fetch_failure_keeps_the_pod(rig):
    (rig.fake / "final_fetch_fail").touch()
    rc, log, calls, _ = rig()
    assert (
        rc == 6
        and "NEEDS A DECISION" in log
        and "final fetch is not OK" in log
        and "A9 DONE" not in log
    )


def test_driver_setup_failure_needs_a_decision(rig):
    (rig.fake / "setup_fail").touch()
    rc, log, calls, _ = rig()
    assert rc == 5 and "NEEDS A DECISION" in log and "mini_pod: train" not in calls


def test_driver_optional_reference_needs_the_raised_stop(rig):
    rc, log, calls, cells = rig(REF2=1)
    assert rc == 0 and cells == sorted([a9.RANK1, a9.N16, a9.N4])  # under $15 it is dropped first
    rc, log, calls, cells = rig("rp-t2", REF2=1, A9_CAP_USD=18.5, A9_RESTAGE=1)
    assert rc == 0, log
    assert cells == sorted([a9.RANK1, a9.N16, a9.N4, a9.REF])


def test_driver_syntax_and_no_pod_control():
    src = (HERE / "pod" / "drive_a9.sh").read_text()
    assert subprocess.run(["bash", "-n", str(HERE / "pod" / "drive_a9.sh")]).returncode == 0
    for word in ("pkill", "runpodctl", "delete-pod", "rm -rf", "pod-action"):
        assert word not in src
