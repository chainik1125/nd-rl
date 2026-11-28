"""The pod chains run locally with stubs. A temporary folder stands in for /workspace/wi; the real
job and chain runners of metrics 2 and this run's cell / write / delete scripts run unchanged; the
write runs for real on tiny checkpoints. Only the sampler, Dan's grader and the download are stubs.
No pod, no GPU."""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import cell_count as cc
import make_chain as mc
import pytest
from wi_fixtures import CONFIG, W, env_path, make_pair, write_sharded

STUBS = Path(__file__).resolve().parent / "stubs"
PIDS = [f"p{i:02d}" for i in range(48)]
NOW = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Pod:
    def __init__(self, tmp: Path, downloaded: bool = True):
        """downloaded=True: dpo and s2300 are already in the models folder (a pod that had them
        before this run). False: they sit in a stand-in hub and the stub download copies them in."""
        self.tmp = tmp
        data = tmp / "widata"
        (data / "inputs").mkdir(parents=True)
        (data / "inputs" / "pids_reach.txt").write_text("\n".join(PIDS) + "\n")
        (data / "inputs" / "spec_reach4.txt").write_text("\n".join(f"{p}:4" for p in PIDS) + "\n")
        (data / "inputs" / "prompt_sha256_reach.json").write_text(
            json.dumps({p: "h" + p for p in PIDS})
        )
        (data / "inputs" / "MANIFEST.json").write_text("{}")
        self.data = data
        r = subprocess.run(
            ["bash", str(W / "pod/wi_pod.sh"), "push", "rp-test", "--dry-run"],
            env=dict(os.environ, WI_DATA=str(data)),
            capture_output=True,
            text=True,
        )
        assert "staged 24 files" in r.stdout, r.stdout + r.stderr
        self.root = tmp / "wi"
        shutil.copytree(data / "pod_stage" / "rp-test", self.root)  # exactly what a push would send
        shutil.copy(STUBS / "m2_download.sh", self.root / "pod" / "m2_download.sh")
        (self.root / "meta").mkdir()
        (self.root / "meta" / "gpu_status").write_text("GPU_OK 1\n")
        self.models = tmp / "models"
        self.merged = self.root / "models"  # merged checkpoints: under the run's own folder
        self.weights = make_pair(self.models)
        (self.models / "tok").mkdir()
        self.hub = tmp / "hub"
        if not downloaded:
            self.hub.mkdir()
            for ck in ("dpo", "s2300"):
                shutil.move(self.models / ck, self.hub / ck)
        dan = tmp / "dan" / "olmo_ifomega"
        dan.mkdir(parents=True)
        shutil.copy(STUBS / "grade.py", dan / "grade.py")
        (dan / "problems.jsonl").write_text('{"pid": "p00"}\n')
        sbin = tmp / "bin"
        sbin.mkdir()
        smi = sbin / "nvidia-smi"  # the lab machine's real one must never decide a test
        smi.write_text(
            "#!/usr/bin/env bash\n"
            'if [[ "$*" == *compute-apps* ]]; then exit 0; fi\n'
            'if [[ "$*" == *utilization* ]]; then echo "${STUB_GPU_MIB:-0}, 0"; '
            'else echo "${STUB_GPU_MIB:-0}"; fi\n'
        )
        smi.chmod(0o755)
        self.env = dict(
            os.environ,
            M2_ROOT=str(self.root),
            MODELS_ROOT=str(self.models),
            DAN_ROOT=str(tmp / "dan"),
            GRADE_PY=sys.executable,
            WI_SAMPLER=f"{sys.executable} {STUBS / 'stub_sampler.py'}",
            HB_INTERVAL="1",
            PATH=f"{sbin}:{env_path()}",
            WI_WAIT_POLL="0.2",
            WI_WAIT_GRACE="5",
            KILL_GRACE="2",
            STOP_GRACE="5",
            OMP_NUM_THREADS="1",
            PYTHONDONTWRITEBYTECODE="1",
            WI_MIN_FREE_GB="0",
            WI_VOLUME_ROOT=str(tmp),
            STUB_HUB=str(self.hub),
        )

    def lines(self, mode="lanes", cell_timeout=30, start=NOW):
        return mc.chains(start, mode=mode, cell_timeout=cell_timeout, threads=1)

    def start_chain(self, name, lines, **env):
        d = self.root / "runs" / "_chains" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "chain.txt").write_text("\n".join(lines) + "\n")
        return subprocess.Popen(
            ["bash", "pod/m2_chain.sh", name],
            cwd=self.root,
            env=dict(self.env, **env),
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def run_chains(self, chains, timeout=180, **env):
        ps = [self.start_chain(n, ln, **env) for n, ln in chains.items()]
        return [p.wait(timeout) for p in ps]

    def state(self, job):
        st = self.root / "runs" / job / "state"
        for s in ("DONE", "FAILED", "RUNNING"):
            if (st / s).exists():
                return s, (st / s).read_text().strip()
        return "ABSENT", ""

    def rcs(self, chain):
        f = self.root / "runs" / "_chains" / chain / "state" / "rcs"
        return dict(ln.split() for ln in f.read_text().splitlines()) if f.exists() else {}

    def calls(self):
        f = self.root / "stub_calls.log"
        return [ln.split()[0] for ln in f.read_text().splitlines()] if f.exists() else []

    def count(self, tag):
        return json.loads((self.root / "runs" / f"cell_{tag}" / "out" / "count.json").read_text())

    def cell(self, tag, **env):
        e = {**self.env, "WI_POD_START_UTC": NOW, "WI_PREP_CHAIN": "prep", "WI_WAIT_S": "3", **env}
        return subprocess.run(
            ["bash", "pod/wi_cell.sh", tag], cwd=self.root, env=e, capture_output=True, text=True
        )

    def release(self, *args, **env):
        return subprocess.run(
            ["bash", "pod/wi_release.sh", *args],
            cwd=self.root,
            env=dict(self.env, **env),
            capture_output=True,
            text=True,
        )

    def download(self, ck, **env):
        return subprocess.run(
            [
                "bash",
                "pod/m2_job.sh",
                f"download_{ck}",
                "60",
                "--",
                "bash",
                "pod/wi_download.sh",
                ck,
            ],
            cwd=self.root,
            env=dict(self.env, NOGPU="1", **env),
            capture_output=True,
            text=True,
        )

    def rm(self, tag):
        return subprocess.run(
            ["bash", "pod/wi_rm.sh", tag],
            cwd=self.root,
            env=self.env,
            capture_output=True,
            text=True,
        )

    def satisfy(self, tag):
        """What the prep chain leaves behind for this cell, written by hand."""
        need = "preflight" if tag == "a100" else f"write_{tag}"
        (self.root / "runs" / need / "state").mkdir(parents=True, exist_ok=True)
        (self.root / "runs" / need / "state" / "DONE").write_text("x rc=0\n")
        if tag == "a100":
            d = self.root / "runs" / "download_s2300"
            d.mkdir(parents=True, exist_ok=True)
            (d / "download_verify.json").write_text('{"ok": true, "sha256_checked": true}\n')
            (self.models / "s2300" / ".complete").touch()
            return
        m = self.merged / f"interp_{tag}"
        m.mkdir(parents=True, exist_ok=True)
        (m / "WI_WRITE_OK").write_text(json.dumps({"alpha": cc.TAGS[tag]}))
        (m / "weights.safetensors").write_text("stand-in")
        out = self.root / "runs" / need / "out"
        out.mkdir(parents=True, exist_ok=True)
        (out / "check.json").write_text('{"ok": true}')
        (out / "MANIFEST.json").write_text("{}")


@pytest.fixture
def pod(tmp_path):
    assert (Path(sys.executable).parent / "python3").exists()
    return Pod(tmp_path)


def stray(marker="sleep 611"):
    """Live processes of the stub sampler, or of its helper (`sleep` with the argument 611). The
    command name is matched too, so a shell whose command line only mentions the marker (a `ps |
    grep` typed beside the tests, for instance) is not counted."""
    out = subprocess.run(["ps", "-eo", "comm=,args="], capture_output=True, text=True).stdout
    name = "sleep" if marker.startswith("sleep") else "python"
    return [ln for ln in out.splitlines() if marker in ln and ln.split()[0].startswith(name)]


def test_lanes_run_everything_in_order_and_delete_only_counted_checkpoints(pod):
    ch = pod.lines("lanes")
    assert pod.run_chains(ch) == [0, 0]
    assert pod.calls() == [
        "cell_a100",
        "cell_a050",
        "cell_a075",
        "cell_a025",
    ]  # the yardstick first
    for ck in ("dpo", "s2300"):  # in this fixture the weights were on the pod before the run
        assert (
            (pod.root / f"runs/download_{ck}/out/origin.txt").read_text().startswith("PREEXISTING")
        )
    assert pod.merged == pod.root / "models" and not list(pod.models.glob("interp_*"))
    for job in [ln.split()[0] for lines in ch.values() for ln in lines]:
        assert pod.state(job)[0] == "DONE", job
    assert all(v == "0" for v in {**pod.rcs("prep"), **pod.rcs("cells")}.values())
    for tag, alpha in (("a050", 0.5), ("a075", 0.75), ("a025", 0.25)):
        assert not (pod.merged / f"interp_{tag}").exists()  # deleted after its cell was counted
        rep = json.loads((pod.root / f"runs/write_{tag}/out/check.json").read_text())
        assert rep["ok"] and rep["alpha"] == alpha and rep["check"]["tensors_checked"] == 57
        assert rep["check"]["census"] is (tag == "a025")  # the alpha-free passes run once
        assert (pod.root / f"runs/write_{tag}/out/MANIFEST.json").exists()
        assert "removed" in (pod.root / f"runs/rm_{tag}/out/removed.txt").read_text()
    for d in ("dpo", "s2300"):  # the two downloaded checkpoints are never touched
        assert len(list((pod.models / d).glob("*.safetensors"))) == 3
    for tag in ("a100", "a050", "a075", "a025"):
        c = pod.count(tag)
        assert (
            c["complete"]
            and c["graded_rows"] == 192
            and c["passes"] == 96
            and c["prompt_sha_checked"] == 48
        )
        want = str(pod.models / "s2300" if tag == "a100" else pod.merged / f"interp_{tag}")
        assert c["settings"]["model_dir"] == want and c["settings"]["seed"] == 710012
    first = json.loads((pod.root / "runs/cell_a100/out/graded.jsonl").read_text().splitlines()[0])
    assert first["model"] == "s2300_same_engine" and first["job"] == "cell_a100"
    # a re-run skips every finished job: nothing is sampled twice
    assert pod.run_chains(ch) == [0, 0]
    assert len(pod.calls()) == 4
    log = (pod.root / "runs/_chains/cells/log/chain.log").read_text()
    assert log.count("SKIP_DONE") == 7


def test_a_cut_cell_keeps_rows_and_checkpoint_and_is_never_resumed(pod):
    ch = pod.lines("sequential", cell_timeout=4)
    assert pod.run_chains(ch, STUB_MODE_cell_a050="hang") == [0]  # the chain goes on after the cut
    rcs = pod.rcs("interp")
    assert (
        rcs["cell_a050"] == "124"
        and rcs["rm_a050"] == "11"
        and rcs["cell_a075"] == rcs["cell_a025"] == "0"
    )
    st, txt = pod.state("cell_a050")
    assert st == "FAILED" and "rc=124" in txt and "timeout" in txt
    c = pod.count("a050")
    assert (
        not c["complete"]
        and c["graded_rows"] == 50
        and c["n_missing"] == 142
        and not c["gzip_whole"]
    )
    out = pod.root / "runs/cell_a050/out"
    assert (
        c["repaired_copy"] == "samples_complete.jsonl.gz"
        and (out / "samples_complete.jsonl.gz").exists()
    )
    assert "CELL_CUT a050" in (out / "cell_status.txt").read_text()
    assert (
        pod.merged / "interp_a050" / "WI_WRITE_OK"
    ).exists()  # not deleted: its cell is not complete
    assert not (pod.merged / "interp_a075").exists() and pod.count("a075")["complete"]
    assert not stray()  # the stuck sampler's helper process went with it
    # re-submitting the chain must not let Dan's sampler fill only the missing draws
    assert pod.run_chains(ch) == [1]
    assert pod.calls().count("cell_a050") == 1
    assert "rows of an earlier attempt exist" in (out / "cell_status.txt").read_text()
    assert pod.state("cell_a050")[1].split()[1] == "rc=2"


def test_no_yardstick_no_run(pod):
    assert pod.run_chains(pod.lines("sequential"), STUB_MODE_cell_a100="crash") == [1]
    assert (
        pod.state("cell_a100") == ("FAILED", pod.state("cell_a100")[1])
        and "rc=1" in pod.state("cell_a100")[1]
    )
    assert pod.state("write_a050")[0] == "ABSENT" and pod.calls() == ["cell_a100"]
    assert not list(pod.merged.glob("interp_*"))


def test_failed_preflight_stops_both_lanes_before_any_draw(pod):
    d, s = pod.weights
    write_sharded(pod.models / "s2300", s, config=dict(CONFIG, hidden_size=16))
    assert pod.run_chains(pod.lines("lanes")) == [1, 1]
    assert "rc=6" in pod.state("preflight")[1] and "rc=21" in pod.state("cell_a100")[1]
    assert pod.calls() == [] and pod.state("write_a050")[0] == "ABSENT"
    assert (
        "the job this cell depends on did not finish"
        in (pod.root / "runs/cell_a100/out/cell_status.txt").read_text()
    )


def test_failed_download_stops_the_cells_lane_at_once(pod):
    t0 = time.time()
    assert pod.run_chains(pod.lines("lanes"), STUB_DOWNLOAD_FAIL="dpo") == [1, 1]
    assert time.time() - t0 < 60 and pod.calls() == []  # no hour-long wait on the GPU
    assert "rc=21" in pod.state("cell_a100")[1]


def test_cell_refuses_without_verified_weights_and_obeys_the_ledger(pod):
    r = pod.cell("a050")
    assert r.returncode == 21 and pod.calls() == []  # nothing written for it
    pod.satisfy("a050")
    (pod.merged / "interp_a050" / "WI_WRITE_OK").write_text('{"alpha": 0.75}')
    assert pod.cell("a050").returncode == 21  # the folder holds another alpha
    pod.satisfy("a050")
    old = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 2.6 * 3600))
    r = pod.cell("a050", WI_POD_START_UTC=old)  # 2.6 h x $5.29 + $7.94 > $21
    assert r.returncode == 9 and "SKIPPED by the ledger gate" in r.stdout and pod.calls() == []
    r = pod.cell("a050", WI_POD_START_UTC=old, WI_CAP_USD="40")
    assert r.returncode == 0 and pod.calls() == ["cell_a050"]
    e = dict(pod.env, WI_PREP_CHAIN="prep", WI_WAIT_S="3")
    e.pop("WI_POD_START_UTC", None)
    r = subprocess.run(
        ["bash", "pod/wi_cell.sh", "a075"], cwd=pod.root, env=e, capture_output=True, text=True
    )
    assert (
        r.returncode == 21
    )  # (not written) and, once written, the missing pod start time is fatal too
    pod.satisfy("a075")
    r = subprocess.run(
        ["bash", "pod/wi_cell.sh", "a075"], cwd=pod.root, env=e, capture_output=True, text=True
    )
    assert r.returncode == 2 and "WI_POD_START_UTC is not set" in r.stdout
    assert subprocess.run(["bash", "pod/wi_cell.sh", "a033"], cwd=pod.root, env=e).returncode == 2


def test_wrong_setting_fails_the_cell_and_blocks_deletion(pod):
    pod.satisfy("a075")
    r = pod.cell("a075", STUB_MODE="badmeta")
    assert r.returncode == 32 and "CELL_INCOMPLETE" in r.stdout
    c = pod.count("a075")
    assert (
        c["rows_complete"]
        and not c["complete"]
        and any("temperature" in p for p in c["settings_problems"])
    )
    (pod.root / "runs/cell_a075/state/DONE").write_text("forged")
    r = pod.rm("a075")
    assert r.returncode == 11 and (pod.merged / "interp_a075").exists()


def test_delete_step_guards(pod):
    assert (
        pod.rm("a100").returncode == 2
        and pod.rm("dpo").returncode == 2
        and pod.rm("../dpo").returncode == 2
    )
    pod.satisfy("a050")
    assert pod.rm("a050").returncode == 11  # the cell has not run
    assert pod.cell("a050").returncode == 0
    assert pod.rm("a050").returncode == 11  # complete, but its job has no DONE marker yet
    (pod.root / "runs/cell_a050/state/DONE").write_text("x rc=0")
    (pod.merged / "interp_a050" / "WI_WRITE_OK").unlink()
    r = pod.rm("a050")
    assert r.returncode == 2 and "does not carry the write marker" in r.stdout
    pod.satisfy("a050")
    c = pod.root / "runs/cell_a050/out/count.json"
    full = c.read_text()
    c.write_text(json.dumps(dict(json.loads(full), n_missing=1, complete=False)))
    assert pod.rm("a050").returncode == 11 and (pod.merged / "interp_a050").exists()
    c.write_text(full)
    r = pod.rm("a050")
    assert r.returncode == 0 and not (pod.merged / "interp_a050").exists()
    assert pod.rm("a050").returncode == 0  # already gone
    assert (
        (pod.models / "dpo").exists()
        and (pod.models / "s2300").exists()
        and (pod.models / "tok").exists()
    )


def test_stop_reaches_the_sampler_and_its_children(pod):
    pod.satisfy("a050")
    e = dict(pod.env, WI_POD_START_UTC=NOW, STUB_MODE="hang", STUB_ROWS="7")
    p = subprocess.Popen(
        ["bash", "pod/m2_job.sh", "cell_a050", "600", "--", "bash", "pod/wi_cell.sh", "a050"],
        cwd=pod.root,
        env=e,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    end = time.time() + 30
    while time.time() < end and not stray():
        time.sleep(0.2)
    assert stray()
    r = subprocess.run(
        ["bash", "pod/m2_stop.sh", "cell_a050"],
        cwd=pod.root,
        env=pod.env,
        capture_output=True,
        text=True,
    )
    assert "STOPPED cell_a050" in r.stdout, r.stdout + r.stderr
    p.wait(30)
    end = time.time() + 10
    while time.time() < end and stray():
        time.sleep(0.2)
    assert not stray() and not stray("stub_sampler.py")
    st, txt = pod.state("cell_a050")
    assert st == "FAILED" and "stopped" in txt


def test_wait_for(tmp_path):
    wi = tmp_path
    st = wi / "runs/write_a050/state"
    chain = wi / "runs/_chains/prep/state"

    def w(max_wait=1.0, grace=0.3):
        return cc.wait_for("a050", wi, "prep", max_wait, 0.05, grace)

    assert w()[0] == 21 and "never started" in w()[1]
    chain.mkdir(parents=True)
    (chain / "RUNNING").touch()
    rc, why = w()
    assert rc == 21 and "not DONE after" in why
    st.mkdir(parents=True)
    threading.Timer(0.3, lambda: (st / "DONE").write_text("x")).start()
    assert w(max_wait=5)[0] == 0  # appears while waiting
    (st / "DONE").unlink()
    (st / "FAILED").write_text("t rc=4")
    assert w() == (21, "write_a050 FAILED: t rc=4")
    (st / "RUNNING").touch()  # a re-run of the failed job is under way: keep waiting, then time out
    assert "not DONE after" in w()[1]
    (st / "FAILED").unlink()
    (st / "RUNNING").unlink()
    (chain / "FAILED").write_text("stopped at preflight")
    assert "has ended" in w(max_wait=30)[1]
    assert cc.wait_for("a100", wi, "prep", 0.2, 0.05, 0.1)[0] == 21  # a100 waits for preflight
    (wi / "runs/preflight/state").mkdir(parents=True)
    (wi / "runs/preflight/state/DONE").touch()
    assert cc.wait_for("a100", wi, "prep", 0.2, 0.05, 0.1)[0] == 0
    assert cc.wait_for("a033", wi, "prep", 0.2)[0] == 2


def test_chain_text_and_what_is_pushed(pod):
    lanes, seq = (
        mc.chains("2026-11-28T12:00:00Z"),
        mc.chains("2026-11-28T12:00:00Z", mode="sequential"),
    )
    jobs = lambda lines: [ln.split()[0] for ln in lines]  # noqa: E731
    assert lanes["prep"][0] == "download_s2300 3600 0 --nogpu -- bash pod/wi_download.sh s2300"
    assert jobs(lanes["prep"]) == [
        "download_s2300",
        "download_dpo",
        "preflight",
        "write_a050",
        "write_a075",
        "write_a025",
    ]
    assert jobs(lanes["cells"]) == [
        "cell_a100",
        "cell_a050",
        "rm_a050",
        "cell_a075",
        "rm_a075",
        "cell_a025",
        "rm_a025",
    ]
    assert jobs(seq["interp"]) == [
        "download_s2300",
        "download_dpo",
        "preflight",
        "cell_a100",
        "write_a050",
        "cell_a050",
        "rm_a050",
        "write_a075",
        "cell_a075",
        "rm_a075",
        "write_a025",
        "cell_a025",
        "rm_a025",
    ]
    for lines in (*lanes.values(), *seq.values()):
        for ln in lines:
            job, to, codes = ln.split()[:3]
            assert mc.LINE_RE.match(ln)
            assert ("--nogpu" in ln.split(" -- ")[0]) == job.startswith(
                ("download", "preflight", "write")
            )
            if job.startswith(
                "cell_"
            ):  # the ledger gate never falls back on another run's defaults
                for k in (
                    "WI_POD_START_UTC=2026-11-28T12:00:00Z",
                    "WI_CAP_USD=21",
                    "WI_RATE=5.29",
                    "WI_PRIOR_USD=0",
                    "WI_CELL_USD=7.94",
                    "WI_CELL_TIMEOUT=5400",
                ):
                    assert k in ln
                assert codes == ("0" if job == "cell_a100" else "0,124")
            elif job.startswith("rm_"):
                assert codes == "0,11"
            else:
                assert codes == "0"
    assert all(ln.split()[1] == "9900" for ln in lanes["cells"] if ln.startswith("cell_"))
    assert all(ln.split()[1] == "6300" for ln in seq["interp"] if ln.startswith("cell_"))
    for ln in seq["interp"]:
        if ln.startswith("write_"):
            assert (
                "ledger_ok.py --start-utc 2026-11-28T12:00:00Z --rate 5.29 --prior 0 --cap 21 "
                "--cell-usd 9.26 &&" in ln
            )
    assert (
        sum("--no-census" in ln for ln in lanes["prep"]) == 2
        and "--no-census" not in lanes["prep"][5]
    )
    with pytest.raises(SystemExit):
        mc.chains("28 Nov 2026")
    cell = (W / "pod/wi_cell.sh").read_text()
    for k in (
        "--pids @inputs/spec_reach4.txt --n 1 --chunk 1",
        "WI_SEED:-710012",
        "WI_CELL_TIMEOUT:-5400",
        '--tokenizer "$MODELS/tok"',
        "WI_CAP_USD:-21",
    ):
        assert k in cell
    assert (
        "--temperature" not in cell and "--top-p" not in cell and "--lora" not in cell
    )  # Dan's defaults: T 1.0, top_p 1.0
    # what a push sends: exactly the staged list, the unedited metrics-2 files byte for byte
    staged = sorted(
        str(p.relative_to(pod.data / "pod_stage/rp-test"))
        for p in (pod.data / "pod_stage/rp-test").rglob("*")
        if p.is_file()
    )
    listed = [
        ln.split()[1]
        for ln in (W / "pod/STAGE_LIST.txt").read_text().splitlines()
        if ln and not ln.startswith("#")
    ]
    assert staged == sorted(
        listed
        + [
            f"inputs/{f}"
            for f in (
                "MANIFEST.json",
                "pids_reach.txt",
                "prompt_sha256_reach.json",
                "spec_reach4.txt",
            )
        ]
    )
    m2 = W.parent / "olmo_metrics2_20261127"
    for rel in (
        "rank1/expo_write.py",
        "sample_wrap.py",
        "pod/m2_job.sh",
        "pod/m2_chain.sh",
        "tools/grade_job.sh",
    ):
        assert (pod.data / "pod_stage/rp-test" / rel).read_bytes() == (m2 / rel).read_bytes()
    # sealed or credential-like names stop the push before anything is sent
    for bad in ("textbook72_items.jsonl", "id_ecdsa", "hf_token.txt"):
        f = pod.data / "inputs" / bad
        f.write_text("x")
        r = subprocess.run(
            ["bash", str(W / "pod/wi_pod.sh"), "push", "rp-test", "--dry-run"],
            env=dict(os.environ, WI_DATA=str(pod.data)),
            capture_output=True,
            text=True,
        )
        assert r.returncode == 1 and "PUSH REFUSED" in r.stdout, bad
        f.unlink()
    assert (
        subprocess.run(
            ["bash", str(W / "pod/wi_pod.sh"), "push", "serrano"], capture_output=True
        ).returncode
        == 2
    )


def test_release_removes_only_what_this_run_made_and_marks_the_gpu_free(tmp_path):
    pod = Pod(tmp_path, downloaded=False)
    for ck in ("s2300", "dpo"):
        assert pod.download(ck).returncode == 0
        assert (pod.root / f"runs/download_{ck}/out/origin.txt").read_text().startswith("ABSENT ")
    assert (
        pod.download("s2300", FORCE="1").returncode == 0
    )  # a second attempt keeps the first record
    assert (pod.root / "runs/download_s2300/out/origin.txt").read_text().startswith("ABSENT ")
    r = subprocess.run(["bash", "pod/wi_download.sh", "s0500"], cwd=pod.root, env=pod.env)
    assert r.returncode == 2
    pod.satisfy("a050")
    pod.satisfy("a075")  # two merged checkpoints still on disk, as after cut cells
    other = pod.models / "someone_elses_adapter"
    other.mkdir()
    (other / "adapter.safetensors").write_text("not ours")
    r = pod.release()
    assert r.returncode == 2 and "usage" in r.stdout  # it deletes weights: never by accident
    # a job of this run is still running: nothing is removed
    st = pod.root / "runs/cell_a025/state"
    st.mkdir(parents=True)
    (st / "RUNNING").write_text(NOW)
    (st / "pid").write_text(f"{os.getpid()} {os.getpgid(0)} {NOW} cell_a025\n")
    r = pod.release("--fetched")
    assert r.returncode == 3 and "RELEASE_REFUSED" in r.stdout
    assert (pod.merged / "interp_a050").exists() and (pod.models / "s2300").exists()
    assert not (pod.root / "GPU_FREE").exists()
    (st / "RUNNING").unlink()
    # the GPU is in use (by someone else: nothing of this run runs): no marker, nothing stopped
    r = pod.release("--fetched", STUB_GPU_MIB="50000")
    assert r.returncode == 4 and "GPU_NOT_FREE" in r.stdout and not (pod.root / "GPU_FREE").exists()
    assert not pod.merged.exists() and not (pod.models / "s2300").exists()
    assert not (pod.models / "s2300.lock").exists() and (pod.models / "dpo.lock").exists()
    r = pod.release("--fetched")
    assert r.returncode == 0, r.stdout + r.stderr
    marker = (pod.root / "GPU_FREE").read_text()
    assert marker.startswith("GPU_FREE 20") and f"dpo is at {pod.models / 'dpo'}" in marker
    rep = (pod.root / "meta/release.txt").read_text()
    assert "s2300: not present" in rep and marker in rep
    assert f"left in place: {pod.models / 'dpo'} (" in rep and "commit 97604023f2602a7d" in rep
    # dpo, the tokenizer folder and another job's files are untouched
    assert len(list((pod.models / "dpo").glob("*.safetensors"))) == 3
    assert (pod.models / "tok").is_dir() and (
        other / "adapter.safetensors"
    ).read_text() == "not ours"


def test_release_keeps_an_s2300_that_this_run_did_not_download(tmp_path):
    pod = Pod(tmp_path)  # the weights were on the pod before this run
    assert pod.download("s2300").returncode == 0
    origin = pod.root / "runs/download_s2300/out/origin.txt"
    assert origin.read_text().startswith("PREEXISTING ")
    r = pod.release("--fetched")
    assert r.returncode == 0 and "it existed before this run" in r.stdout
    assert (pod.models / "s2300" / ".complete").exists() and (pod.root / "GPU_FREE").exists()
    # downloaded by this run, but the folder is not the pinned commit: kept as well
    origin.write_text("ABSENT x\n")
    (pod.models / "s2300" / ".sha").write_text("0000\n")
    r = pod.release("--fetched")
    assert r.returncode == 0 and "not the pinned commit" in r.stdout
    assert (pod.models / "s2300").exists()
    origin.unlink()
    assert "no origin record" in pod.release("--fetched").stdout and (pod.models / "s2300").exists()
