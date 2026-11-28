"""The serrano side end to end, with a stub `ssh` that runs the "remote" commands in a temporary
folder: push with md5 on both sides, both chains started through the driver, the watcher, status,
the fetch with its md5 list, then the analysis and the figure on the fetched files.
No pod, no GPU, no network."""

import json
import shutil
import subprocess

import analyze as an
import fig_interp
import pytest
from test_chain import STUBS, Pod
from wi_fixtures import W


def test_push_chains_watch_fetch_analyse(tmp_path):
    pod = Pod(tmp_path, downloaded=False)  # the stub download copies the weights in, as a hub would
    shutil.rmtree(pod.root)  # this time the driver's own push fills the pod folder
    shutil.copy(STUBS / "ssh", tmp_path / "bin" / "ssh")
    env = dict(
        pod.env,
        FAKE_WI=str(pod.root),
        WI_DATA=str(pod.data),
        POLL="1",
        WI_WATCH_WAIT="1",
        STUB_PASS_cell_a100="3",
        STUB_PASS_cell_a075="2",
        STUB_PASS_cell_a050="1",
        STUB_PASS_cell_a025="0",
    )

    def drv(*a):
        return subprocess.run(
            ["bash", str(W / "pod/wi_pod.sh"), *a],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )

    r = drv("push", "rp-test")
    assert r.returncode == 0 and "push md5 ok: 24 files equal" in r.stdout, r.stdout + r.stderr
    assert (pod.root / "interp_write.py").read_bytes() == (W / "interp_write.py").read_bytes()
    r = drv("podcheck", "rp-test")  # read only; which schedule fits (here: this machine's numbers)
    assert r.returncode in (0, 5) and "mem_bytes=" in r.stdout and "\nMODE " in r.stdout, r.stdout
    files = {}
    for name, lines in pod.lines("lanes").items():
        files[name] = tmp_path / f"chain_{name}.txt"
        files[name].write_text("# comment line\n" + "\n".join(lines) + "\n")
    r = drv("chain", "rp-test", "prep", str(files["prep"]))
    assert (
        r.returncode == 1 and "run setup first" in r.stdout
    )  # nothing starts on a pod without setup
    # what setup leaves behind (setup installs vLLM and cannot run here), and the stub download
    (pod.root / "meta").mkdir()
    (pod.root / "meta" / "gpu_status").write_text("GPU_OK 1\n")
    (pod.root / "meta" / "setup.done").write_text("now\n")
    shutil.copy(STUBS / "m2_download.sh", pod.root / "pod" / "m2_download.sh")
    bad = tmp_path / "bad.txt"
    bad.write_text("cell_a100 5400 -- bash pod/wi_cell.sh a100\n")  # no continue list
    assert drv("chain", "rp-test", "x", str(bad)).returncode == 2
    for name in ("prep", "cells"):
        r = drv("chain", "rp-test", name, str(files[name]))
        assert r.returncode == 0 and f"chain {name} started on rp-test" in r.stdout, (
            r.stdout + r.stderr
        )
    r = drv("release", "rp-test")  # nothing has been fetched yet: weights are not deleted
    assert r.returncode == 1 and "RELEASE REFUSED" in r.stdout
    w = subprocess.run(
        ["bash", str(W / "pod/wi_watch_chain.sh"), "rp-test", "cells", str(files["cells"])],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert w.returncode == 0, w.stdout + w.stderr
    log = (pod.data / "logs" / "watch_rp-test.log").read_text()
    assert (
        log.count(" EXIT state=DONE rc=0") == 7
        and "cell_a025 EXIT state=DONE" in log
        and " PT) " in log
    )
    subprocess.run(
        ["bash", str(W / "pod/wi_watch_chain.sh"), "rp-test", "prep", str(files["prep"])],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    st = drv("status", "rp-test").stdout
    assert "chain:cells state=DONE" in st and "chain:prep state=DONE" in st
    assert "cell_a050 state=DONE rc=0" in st and "progress=192/192" in st
    r = drv("fetch", "rp-test", "cell_a050")
    assert r.returncode == 0 and "FETCH_OK cell_a050" in r.stdout, r.stdout + r.stderr
    r = drv("fetchall", "rp-test")
    assert r.returncode == 0 and "FETCHALL_OK rp-test" in r.stdout, r.stdout + r.stderr
    md5 = (pod.data / "fetch_rp-test.md5").read_text()
    for tag in ("a100", "a050", "a075", "a025"):
        for f in ("samples.jsonl.gz", "graded.jsonl", "meta.jsonl", "wrap.json", "count.json"):
            assert f"  runs/cell_{tag}/out/{f}\n" in md5, (tag, f)
            assert (pod.data / "runs" / f"cell_{tag}" / "out" / f).read_bytes() == (
                pod.root / "runs" / f"cell_{tag}" / "out" / f
            ).read_bytes()
    for tag in ("a050", "a075", "a025"):
        assert (
            f"  runs/write_{tag}/out/check.json\n" in md5
            and f"  runs/rm_{tag}/out/removed.txt\n" in md5
        )
    assert "  runs/_chains/cells/log/chain.log\n" in md5 and "  meta/setup.done\n" in md5
    assert not list(pod.data.rglob("*.safetensors"))  # weights are never fetched
    # the fetched files go straight into the analysis and the figure
    r = drv("release", "rp-test", "--dry-run")
    assert r.returncode == 0 and "wi_release.sh --fetched" in r.stdout
    res = an.run(
        pod.data,
        pod.data / "analysis",
        300,
        models_root=str(pod.models),
        merged_root=str(pod.merged),
    )
    assert [res["cells"][t]["pass1"] for t in ("a025", "a050", "a075", "a100")] == [
        0.0,
        0.25,
        0.5,
        0.75,
    ]
    assert all(
        res["cells"][t]["complete"] and not res["cells"][t]["problems"] for t in res["cells"]
    )
    assert res["ratios"]["a050"]["sampled"]["r"] == pytest.approx(1 / 3)
    assert res["reading"] == {
        "verdict": "below proportional at the midpoint",
        "why": "both cells have every planned draw",
        "by_bound": dict.fromkeys(("sampled", "low", "high"), "below proportional at the midpoint"),
        "cells_complete": True,
        "near_boundary": True,
    }
    assert set(res["writes"]) == {"a025", "a050", "a075"} and all(
        v["ok"] for v in res["writes"].values()
    )
    assert res["half_gain_realised"]["axis"]["a050"] == pytest.approx(
        res["writes"]["a050"]["realised_fraction"]
    )
    assert (
        res["writes"]["a025"]["alpha1_identity"]["elements_not_equal"] == 0
        and res["writes"]["a050"]["alpha1_identity"] is None
    )
    man = json.loads((pod.data / "analysis" / "cells_manifest.json").read_text())
    assert all(m["rows"] == 192 and m["complete"] for m in man.values())
    assert (
        len(
            fig_interp.draw(
                res := json.loads((pod.data / "analysis" / "results.json").read_text()),
                tmp_path / "fig",
            )
        )
        == 3
    )
    assert res["cells"]["a100"]["passes"] == 144
    # the last step on a shared pod: this run's s2300 and merged copies go, dpo stays, and the
    # GPU is marked free
    for ck in ("s2300", "dpo"):
        assert (pod.root / f"runs/download_{ck}/out/origin.txt").read_text().startswith("ABSENT ")
        assert f"  runs/download_{ck}/out/origin.txt\n" in md5
    assert (pod.models / "s2300").exists() and not (pod.root / "GPU_FREE").exists()
    r = drv("release", "rp-test")
    assert r.returncode == 0 and "RELEASED rp-test" in r.stdout, r.stdout + r.stderr
    assert f"removed {pod.models / 's2300'} (" in r.stdout
    assert not (pod.models / "s2300").exists() and not pod.merged.exists()
    assert (
        len(list((pod.models / "dpo").glob("*.safetensors"))) == 3 and (pod.models / "tok").is_dir()
    )
    marker = (pod.root / "GPU_FREE").read_text()
    assert marker.startswith("GPU_FREE 20") and str(pod.models / "dpo") in marker
    assert marker in (pod.data / "release_rp-test.txt").read_text()
    # nothing but the alias, options and commands went through "ssh": no key or token is ever named
    sent = (tmp_path / "wi.ssh.log").read_text()
    assert "id_" not in sent and "token" not in sent.lower() and "HF_" not in sent
