"""Counting a cell's per-draw files: complete only with every planned row and the registered
settings."""

import json

import cell_count as cc
from wi_fixtures import spec48, uniform, write_cell


def count(d, tag="a050", **kw):
    model = "/workspace/models/s2300" if tag == "a100" else f"/workspace/wi/models/interp_{tag}"
    return cc.count(d, spec48(), tag, 710012, model, **kw)


def test_complete_cell(tmp_path):
    write_cell(tmp_path, "a050", uniform(2))
    r = count(tmp_path, prompt_sha=dict.fromkeys(spec48(), "x"))
    assert r["complete"] and r["planned"] == r["graded_rows"] == r["sample_rows"] == 192
    assert (
        r["passes"] == 96
        and r["n_missing"] == 0
        and r["gzip_whole"]
        and r["prompt_sha_checked"] == 48
    )


def test_missing_rows_are_listed_not_dropped(tmp_path):
    p = uniform(2)
    p["p03"] = p["p03"][:1]  # three draws of one prompt never finished
    del p["p07"]  # a prompt with no finished draw at all
    write_cell(tmp_path, "a050", p, truncate=True)
    r = count(tmp_path)
    assert not r["complete"] and not r["rows_complete"] and not r["gzip_whole"]
    assert r["n_missing"] == 7 and ["p07", 0, 0] in r["missing"] and ["p03", 1, 0] in r["missing"]
    assert r["graded_rows"] == 185 and r["repaired_copy"] == "samples_complete.jsonl.gz"
    assert (
        cc.main(
            [
                "count",
                "--dir",
                str(tmp_path),
                "--spec",
                _spec(tmp_path),
                "--tag",
                "a050",
                "--seed",
                "710012",
                "--model-dir",
                "/workspace/wi/models/interp_a050",
            ]
        )
        == 31
    )
    assert json.loads((tmp_path / "count.json").read_text())["n_missing"] == 7


def _spec(tmp_path):
    f = tmp_path / "spec.txt"
    f.write_text("\n".join(f"{p}:{n}" for p, n in spec48().items()) + "\n")
    return str(f)


def test_duplicates_extras_and_bad_strict(tmp_path):
    write_cell(tmp_path, "a050", uniform(1))
    g = tmp_path / "graded.jsonl"
    rows = [json.loads(ln) for ln in g.read_text().splitlines()]
    g.write_text("".join(json.dumps(r) + "\n" for r in rows + [rows[0], dict(rows[1], pid="zz")]))
    r = count(tmp_path)
    assert not r["complete"] and any("duplicate graded" in p for p in r["problems"])
    assert any("outside the plan" in p for p in r["problems"])
    rows[5]["strict"] = "True"
    g.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert any("boolean" in p for p in count(tmp_path)["problems"])


def test_settings_mismatch_is_exit_32(tmp_path):
    for i, (bad, word) in enumerate(
        [
            ({"temperature": 0.6}, "temperature"),
            ({"seed": 1}, "seed"),
            ({"model_dir": "/x"}, "model_dir"),
            ({"max_tokens": 12000}, "max_tokens"),
            ({"vllm": "0.12.0"}, "vllm"),
            ({"lora": "/a"}, "lora"),
        ]
    ):
        d = tmp_path / str(i)
        write_cell(d, "a050", uniform(2), meta=bad)
        r = count(d)
        assert r["rows_complete"] and not r["complete"], bad
        assert any(word in p for p in r["settings_problems"]), bad
    d = tmp_path / "x"
    write_cell(d, "a050", uniform(2))
    assert (
        cc.main(
            [
                "count",
                "--dir",
                str(d),
                "--spec",
                _spec(tmp_path),
                "--tag",
                "a050",
                "--seed",
                "1",
                "--model-dir",
                "/workspace/wi/models/interp_a050",
            ]
        )
        == 32
    )
    # two meta lines = the sampler was resumed: not one uninterrupted engine load
    (d / "meta.jsonl").write_text((d / "meta.jsonl").read_text() * 2)
    assert any("meta lines" in p for p in count(d)["settings_problems"])
    # a prompt that tokenises differently from the reference
    r = count(d, prompt_sha=dict.fromkeys(spec48(), "y"))
    assert any("tokenise differently" in p for p in r["settings_problems"])
    (d / "wrap.json").write_text(json.dumps({"status": "META_MISMATCH"}))
    assert any("wrap.json status" in p for p in count(d)["settings_problems"])


def test_precheck(tmp_path):
    wi, models = tmp_path / "wi", tmp_path / "models"
    assert cc.precheck("a100", wi, models)[0] == 21
    (models / "s2300").mkdir(parents=True)
    (models / "s2300" / ".complete").touch()
    v = wi / "runs/download_s2300/download_verify.json"
    v.parent.mkdir(parents=True)
    v.write_text('{"ok": true, "sha256_checked": false}\n')
    assert cc.precheck("a100", wi, models)[0] == 21  # downloaded but not sha256-checked
    v.write_text('{"ok": true, "sha256_checked": true}\n')
    assert cc.precheck("a100", wi, models)[0] == 0
    assert cc.precheck("a050", wi, models)[0] == 21
    (wi / "models" / "interp_a050").mkdir(parents=True)
    (wi / "models" / "interp_a050" / "WI_WRITE_OK").write_text('{"alpha": 0.75}')
    rep = wi / "runs/write_a050/out/check.json"
    rep.parent.mkdir(parents=True)
    rep.write_text('{"ok": true}')
    assert cc.precheck("a050", wi, models)[0] == 21  # the folder holds another alpha
    (wi / "models" / "interp_a050" / "WI_WRITE_OK").write_text('{"alpha": 0.5}')
    assert cc.precheck("a050", wi, models)[0] == 0
    rep.write_text('{"ok": false}')
    assert cc.precheck("a050", wi, models)[0] == 21
    assert cc.precheck("a033", wi, models)[0] == 2
    # the merged checkpoints live under the run's own folder, not beside the downloads
    assert not (models / "interp_a050").exists()
    rep.write_text('{"ok": true}')
    assert cc.precheck("a050", wi, models, tmp_path / "elsewhere")[0] == 21
    assert cc.main(["precheck", "--tag", "a050", "--wi", str(wi), "--models", str(models)]) == 0


def test_schedule_mode_from_the_pods_memory_and_disk():
    gb = 10**9
    assert cc.schedule_mode(250 * gb, 0)[0] == "lanes"
    assert cc.schedule_mode(250 * gb, 100 * gb)[0] == "lanes"  # 400 GB unused
    mode, why = cc.schedule_mode(64 * gb, 0)
    assert mode == "sequential" and "64 GB is under 96 GB" in why
    mode, why = cc.schedule_mode(250 * gb, 200 * gb)  # another job's files: 300 GB unused
    assert mode == "sequential" and "300 GB of the volume are unused" in why
    assert (
        cc.schedule_mode(None, 0)[0] == "sequential"
    )  # an unknown limit is not assumed to be enough
    mode, why = cc.schedule_mode(250 * gb, 300 * gb)
    assert mode == "none" and "200 GB" in why
    assert cc.schedule_mode(250 * gb, 0, volume_gb=300)[0] == "sequential"
    assert cc.main(["mode", "--mem-bytes", str(250 * gb), "--used-bytes", "0"]) == 0
    assert cc.main(["mode", "--mem-bytes", "unknown", "--used-bytes", str(400 * gb)]) == 5
