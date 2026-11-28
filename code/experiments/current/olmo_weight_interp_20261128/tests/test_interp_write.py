"""The write / check step on tiny bf16 fixtures: alpha arithmetic against an independent exact
reference, alpha 1.0 = s2300 and alpha 0 = dpo bit for bit, census, config, disk and corruption."""

import json
from fractions import Fraction
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from wi_fixtures import CONFIG, make_pair, step, write_sharded

# isort: split
import interp_write as iw
from common import Checkpoint


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def neighbour(o: torch.Tensor, k: int) -> float:
    """The bf16 value k steps away from the 0-dim bf16 tensor o (ordered by value)."""
    b = int(bits(o.reshape(1))[0])
    v = -(b & 0x7FFF) if b < 0 else b
    v += k
    nb = (abs(v) | 0x8000) - 0x10000 if v < 0 else v
    return float(torch.tensor([nb], dtype=torch.int16).view(torch.bfloat16)[0])


def assert_correctly_rounded(d, s, o, alpha: Fraction, name: str) -> None:
    """Independent of torch's fp32 path: o is a bf16 value nearest to the exact rational result, and
    on an exact tie the one whose last bit is even."""
    fd, fs, fo = d.reshape(-1), s.reshape(-1), o.reshape(-1)
    for i in range(fd.numel()):
        exact = Fraction(float(fd[i])) + alpha * (Fraction(float(fs[i])) - Fraction(float(fd[i])))
        err = abs(Fraction(float(fo[i])) - exact)
        for k in (-1, 1):
            e2 = abs(Fraction(neighbour(fo[i], k)) - exact)
            assert err <= e2, (name, i, float(fd[i]), float(fs[i]), float(fo[i]))
            if err == e2 and err != 0:
                assert int(bits(fo[i].reshape(1))[0]) & 1 == 0, (name, i, "tie not to even")


def run(tmp_path, alpha, name=None, **kw):
    out = tmp_path / (name or f"interp_x{int(alpha * 100):03d}")
    rc = iw.run(
        tmp_path / "dpo",
        tmp_path / "s2300",
        alpha,
        out,
        tmp_path / "rep" / out.name,
        min_free_gb=0,
        **kw,
    )
    rep = json.loads((tmp_path / "rep" / out.name / "check.json").read_text())
    return rc, out, rep


def test_alpha_one_is_s2300_and_alpha_zero_is_dpo_bit_for_bit(tmp_path):
    d, s = make_pair(tmp_path)
    rc1, out1, rep1 = run(tmp_path, 1.0)
    rc0, out0, rep0 = run(tmp_path, 0.0)
    assert rc1 == 0 and rc0 == 0 and rep1["ok"] and rep0["ok"]
    c1, c0 = Checkpoint(out1), Checkpoint(out0)
    for n in d:
        assert torch.equal(bits(c1.get(n)), bits(s[n])), n
        assert torch.equal(bits(c0.get(n)), bits(d[n])), n
    assert rep1["check"]["all"]["realised_fraction"] == pytest.approx(1.0, abs=1e-6)
    assert rep1["check"]["all"]["at_ckpt_share_of_differing"] == 1.0
    assert rep0["check"]["all"]["realised_fraction"] == 0.0
    assert rep0["check"]["all"]["at_dpo_share_of_differing"] == 1.0


@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.75])
def test_interior_alphas_are_the_correctly_rounded_exact_result(tmp_path, alpha):
    d, s = make_pair(tmp_path, seed=3)
    rc, out, rep = run(tmp_path, alpha, exact="all")
    assert rc == 0 and rep["ok"] and (out / iw.MARKER).exists()
    chk = rep["check"]
    assert chk["tensors_checked"] == chk["tensors_expected"] == len(d) == chk["exact_tensors"]
    assert chk["max_abs_diff_ulp"] == 0.0 and chk["elements_not_equal_to_reference"] == 0
    assert chk["all"]["outside"] == 0 and chk["all"]["equal_but_changed"] == 0
    ck = Checkpoint(out)
    for n in d:
        o = ck.get(n)
        assert o.dtype == torch.bfloat16 and o.shape == d[n].shape
        assert_correctly_rounded(d[n], s[n], o, Fraction(alpha), n)
        if n.startswith(
            "model.layers.1."
        ):  # tensors that do not differ are left exactly as they are
            assert torch.equal(bits(o), bits(d[n]))
    # same shard layout and index as dpo; non-weight files from dpo; manifest kept beside the report
    assert ck.wmap == Checkpoint(tmp_path / "dpo").wmap
    assert json.loads((out / "config.json").read_text()) == CONFIG
    man = json.loads((tmp_path / "rep" / out.name / "MANIFEST.json").read_text())
    assert man["gamma"] == alpha and man["tensors"] == len(d)
    assert json.loads((out / iw.MARKER).read_text())["alpha"] == alpha


def test_one_step_differences_round_as_the_note_says(tmp_path):
    """A weight one bf16 step from dpo: stays at dpo at alpha 0.25, lands on s2300 at 0.75, goes to
    the
    neighbour with the even last bit at 0.5. The realised fraction then is 0, 1 and about 0.5."""
    g = torch.Generator().manual_seed(1)
    d = {"model.layers.0.mlp.up_proj.weight": (torch.randn(64, 64, generator=g) * 0.05).bfloat16()}
    s = {k: step(v, torch.ones(v.shape, dtype=torch.int32)) for k, v in d.items()}
    write_sharded(tmp_path / "dpo", d, 1)
    write_sharded(tmp_path / "s2300", s, 1)
    n = "model.layers.0.mlp.up_proj.weight"
    res = {}
    for alpha in (0.25, 0.5, 0.75):
        rc, out, rep = run(tmp_path, alpha, exact="all")
        assert rc == 0
        res[alpha] = (Checkpoint(out).get(n), rep["check"]["all"])
    assert torch.equal(bits(res[0.25][0]), bits(d[n])) and res[0.25][1]["realised_fraction"] == 0.0
    assert torch.equal(bits(res[0.75][0]), bits(s[n]))
    assert res[0.75][1]["realised_fraction"] == pytest.approx(1.0, abs=1e-6)
    half = res[0.5][0]
    even_d = (bits(d[n]) & 1) == 0
    assert torch.equal(bits(half), torch.where(even_d, bits(d[n]), bits(s[n])))
    a = res[0.5][1]
    assert 0.4 < a["realised_fraction"] < 0.6 and a["between"] == 0
    assert a["at_dpo"] + a["at_ckpt"] == a["differ"] == 64 * 64
    assert a["steps"] == {"0": 0, "1": 4096, "2": 0, "3-4": 0, "5-8": 0, ">8": 0}
    assert a["steps_sq_share"]["1"] == pytest.approx(1.0, abs=1e-5)
    assert a["off_line_share"] > 0.3  # half the weights moved a full step, half not at all


def test_census_refuses_a_differing_non_float_tensor_and_other_mismatches(tmp_path):
    d, s = make_pair(tmp_path)
    for case in ("nonfloat", "names", "dtype"):
        root = tmp_path / case
        d2, s2 = dict(d), dict(s)
        if case == "nonfloat":
            d2["model.position_ids"] = torch.arange(4)
            s2["model.position_ids"] = torch.arange(4) + 1
        elif case == "names":
            s2["extra.weight"] = torch.zeros(2).bfloat16()
        else:
            s2["model.norm.weight"] = s2["model.norm.weight"].float()
        write_sharded(root / "dpo", d2)
        write_sharded(root / "s2300", s2)
        rc = iw.run(
            root / "dpo", root / "s2300", 0.5, root / "interp_a", root / "rep", min_free_gb=0
        )
        rep = json.loads((root / "rep" / "check.json").read_text())
        assert rc == 6 and not rep["census"]["ok"] and not (root / "interp_a").exists(), case
    # an identical non-float tensor is fine and is copied
    root = tmp_path / "same"
    write_sharded(root / "dpo", dict(d, ids=torch.arange(4)))
    write_sharded(root / "s2300", dict(s, ids=torch.arange(4)))
    assert (
        iw.run(root / "dpo", root / "s2300", 0.5, root / "interp_a", root / "rep", min_free_gb=0)
        == 0
    )
    assert torch.equal(Checkpoint(root / "interp_a").get("ids"), torch.arange(4))


def test_census_counts_classes_and_preflight_mode(tmp_path):
    d, _ = make_pair(tmp_path)
    rc = iw.run(
        tmp_path / "dpo",
        tmp_path / "s2300",
        0.5,
        tmp_path / "interp_none",
        tmp_path / "pf",
        mode="preflight",
    )
    rep = json.loads((tmp_path / "pf" / "check.json").read_text())
    assert rc == 0 and rep["ok"] and not (tmp_path / "interp_none").exists()
    assert rep["census"]["by_class"] == {"embedding": 1, "output_head": 1, "matrix": 42, "norm": 13}
    assert rep["census"]["tensors"] == len(d) == 57 and rep["census"]["dtypes"] == {"BF16": 57}
    assert rep["config"]["differing_fields"] == {"use_cache": [True, False]}


def test_config_must_differ_only_in_labels(tmp_path):
    d, s = make_pair(tmp_path)
    write_sharded(tmp_path / "s2300", s, config=dict(CONFIG, hidden_size=16))
    rc, out, rep = run(tmp_path, 0.5)
    assert (
        rc == 6 and not rep["config"]["ok"] and "hidden_size" in rep["config"]["differing_fields"]
    )
    assert not out.exists()


def test_real_configs_differ_only_in_use_cache():
    oe = Path("/scratch/cowifornia/nd-rl/olmo_elicit")
    a, b = oe / "metrics2/audit/dpo_config_fetch", oe / "tok/s2300"
    if not ((a / "config.json").exists() and (b / "config.json").exists()):
        pytest.skip("local copies of the two config.json files are not here")
    c = iw.config_check(a, b)
    assert c["ok"] and set(c["differing_fields"]) == {"use_cache"}


def test_a_corrupted_or_swapped_tensor_is_caught_and_the_marker_removed(tmp_path):
    make_pair(tmp_path)
    rc, out, _ = run(tmp_path, 0.5)
    assert rc == 0 and (out / iw.MARKER).exists()
    shard = sorted(out.glob("*.safetensors"))[0]
    t = load_file(str(shard))
    name = next(n for n in sorted(t) if "layers.0" in n and "proj" in n)
    t[name] = step(
        t[name], torch.ones(t[name].shape, dtype=torch.int32)
    )  # every element one step off
    save_file(t, str(shard), metadata={"format": "pt"})
    rc = iw.run(tmp_path / "dpo", tmp_path / "s2300", 0.5, out, tmp_path / "rep2", mode="check")
    rep = json.loads((tmp_path / "rep2" / "check.json").read_text())
    assert rc == 4 and not rep["ok"] and not (out / iw.MARKER).exists()
    why = {b["why"] for b in rep["check"]["bad"] if b["name"] == name}
    assert "not bit-equal to the recomputed fp32 combine" in why
    # a folder written for another alpha is not accepted as this alpha
    rc, out2, _ = run(tmp_path, 0.25, name="interp_q")
    assert rc == 0
    rc = iw.run(tmp_path / "dpo", tmp_path / "s2300", 0.75, out2, tmp_path / "rep3", mode="check")
    assert rc == 4


def test_marker_makes_a_second_write_a_no_op_and_a_partial_write_is_redone(tmp_path):
    make_pair(tmp_path)
    rc, out, _ = run(tmp_path, 0.75)
    stamp = (out / "model.safetensors.index.json").stat().st_mtime_ns
    rc2, _, rep2 = run(tmp_path, 0.75)
    assert rc == rc2 == 0 and "already written" in rep2["why"]
    assert (out / "model.safetensors.index.json").stat().st_mtime_ns == stamp
    rc3, _, rep3 = run(tmp_path, 0.5, name=out.name)  # same folder, other alpha: refused
    assert rc3 == 2
    (out / iw.MARKER).unlink()  # as after a write that died before its check
    rc4, _, rep4 = run(tmp_path, 0.75)
    assert rc4 == 0 and rep4["ok"] and "write_seconds" in rep4 and (out / iw.MARKER).exists()


def test_disk_budget_and_refusals(tmp_path):
    make_pair(tmp_path)
    need = iw.weight_bytes(tmp_path / "dpo") / 1e9
    b = iw.disk_budget(tmp_path, tmp_path / "dpo", volume_gb=iw.MARGIN_GB + 1)
    assert b["ok"] and b["need_gb"] == round(need, 2)
    rc, out, rep = run(tmp_path, 0.5, volume_gb=iw.MARGIN_GB / 2, volume_root=tmp_path)
    assert rc == 5 and not rep["disk"]["ok"] and not out.exists()
    (tmp_path / "precious").mkdir()
    rc = iw.run(tmp_path / "dpo", tmp_path / "s2300", 0.5, tmp_path / "precious", tmp_path / "r")
    assert rc == 2 and (tmp_path / "precious").exists()  # only interp_* folders are ever replaced
    assert (
        iw.run(tmp_path / "dpo", tmp_path / "s2300", 1.5, tmp_path / "interp_z", tmp_path / "r")
        == 2
    )
    # not a fraction with a denominator of at most 100: the exact verifier would be undefined
    assert (
        iw.run(
            tmp_path / "dpo", tmp_path / "s2300", 0.1234567, tmp_path / "interp_z", tmp_path / "r"
        )
        == 2
    )


def test_exact_sample_covers_non_matrices_and_whole_layers(tmp_path):
    d, _ = make_pair(tmp_path)
    names = sorted(d)
    pick = iw.exact_sample(names)
    assert {n for n in names if iw.tensor_class(n) != "matrix"} <= pick
    layers = {
        int(n.split(".layers.")[1].split(".")[0]) for n in pick if iw.tensor_class(n) == "matrix"
    }
    assert len(layers) == iw.EXACT_LAYERS
    for i in layers:
        assert sum(n.startswith(f"model.layers.{i}.") and n in pick for n in names) == 9
    assert pick == iw.exact_sample(names)  # seeded
    rc, _, rep = run(tmp_path, 0.5)
    assert rc == 0 and rep["check"]["exact_mode"] == "sample"
    assert (
        rep["check"]["exact_tensors"] == len(pick) < len(names) == rep["check"]["tensors_checked"]
    )


def test_bf16_steps_and_chunking(tmp_path):
    a = torch.tensor([1.0, -1.0, 0.0, 1.0], dtype=torch.bfloat16)
    one = torch.tensor([1, 1, 1, 3], dtype=torch.int32)
    b = step(a, one)
    assert iw.bf16_steps(a, b).tolist()[:2] == [1, 1] and iw.bf16_steps(a, b).tolist()[3] == 3
    tiny = torch.tensor([1], dtype=torch.int16).view(torch.bfloat16)  # the smallest positive value
    assert iw.bf16_steps(tiny, -tiny).tolist() == [2] and iw.bf16_steps(
        tiny, tiny * 0
    ).tolist() == [1]
    make_pair(tmp_path)
    r1 = run(tmp_path, 0.5, name="interp_c1", chunk=7)[2]["check"]["all"]
    r2 = run(tmp_path, 0.5, name="interp_c2")[2]["check"]["all"]
    for k in ("differ", "at_dpo", "at_ckpt", "between", "steps"):
        assert r1[k] == r2[k]
    assert r1["realised_fraction"] == pytest.approx(r2["realised_fraction"], rel=1e-5)


def test_cli_tag_uses_the_pod_layout(tmp_path, monkeypatch):
    models = tmp_path / "models"
    make_pair(models)
    monkeypatch.setenv("MODELS_ROOT", str(models))
    monkeypatch.setenv("M2_ROOT", str(tmp_path / "wi"))
    monkeypatch.setenv("WI_VOLUME_ROOT", str(tmp_path))
    monkeypatch.delenv("WI_MERGED_ROOT", raising=False)
    assert iw.main(["preflight", "--threads", "1"]) == 0
    assert (tmp_path / "wi/runs/preflight/out/check.json").exists()
    rc = iw.main(["write", "--tag", "a075", "--threads", "1", "--min-free-gb", "0", "--no-census"])
    rep = json.loads((tmp_path / "wi/runs/write_a075/out/check.json").read_text())
    assert rc == 0 and rep["alpha"] == 0.75 and rep["disk"]["volume_gb"] == 500
    assert rep["disk"]["volume_root"] == str(tmp_path) and rep["disk"]["used_gb"] >= 0  # all of it
    assert rep["check"]["census"] is False and rep["check"]["all"]["steps"] is None
    assert (
        rep["check"]["all"]["alpha1_not_ckpt"] is None and rep["check"]["alpha1_identity"] is None
    )
    # the merged copy goes under the run's own folder, not beside the downloads
    assert (tmp_path / "wi/models/interp_a075" / iw.MARKER).exists()
    assert not (models / "interp_a075").exists()
    # a shared volume that is nearly full (someone else's files count) refuses the write
    monkeypatch.setenv("WI_VOLUME_GB", "20.00001")
    assert iw.main(["write", "--tag", "a050", "--threads", "1", "--min-free-gb", "0"]) == 5
    assert not (tmp_path / "wi/models/interp_a050").exists()
    assert (tmp_path / "wi/runs/write_a075/out/MANIFEST.json").exists()


def test_exact_check_counts_mismatches_not_only_their_size(tmp_path):
    """Audit finding 3: a tensor whose elements are one step apart, written wholly at the wrong end,
    is within one step everywhere. The count of unequal elements (and the bit-for-bit recompute)
    must fail it."""
    g = torch.Generator().manual_seed(5)
    n = "model.norm.weight"
    d = {n: (torch.ones(512) + torch.randn(512, generator=g) * 0.01).bfloat16()}
    s = {n: step(d[n], torch.ones(512, dtype=torch.int32))}
    write_sharded(tmp_path / "dpo", d, 1)
    write_sharded(tmp_path / "s2300", s, 1)
    rc, out, rep = run(tmp_path, 0.25)
    assert rc == 0 and torch.equal(bits(Checkpoint(out).get(n)), bits(d[n]))
    shard = next(out.glob("*.safetensors"))
    save_file({n: s[n]}, str(shard), metadata={"format": "pt"})  # the whole tensor at the s2300 end
    v = iw.ew.verify_tensors(
        Checkpoint(tmp_path / "dpo"), Checkpoint(tmp_path / "s2300"), Checkpoint(out), 0.25, 1
    )
    assert v[0]["ok"] and v[0]["n_ne"] == 512  # metrics 2's criterion alone lets it through
    rc = iw.run(tmp_path / "dpo", tmp_path / "s2300", 0.25, out, tmp_path / "rep2", mode="check")
    bad = json.loads((tmp_path / "rep2" / "check.json").read_text())["check"]["bad"]
    assert rc == 4 and {b["why"] for b in bad} == {
        "exact reference",
        "not bit-equal to the recomputed fp32 combine",
    }
    assert next(b for b in bad if b["why"] == "exact reference")["elements_not_equal"] == 512


def test_alpha_one_identity_is_counted_in_the_census(tmp_path):
    make_pair(tmp_path)
    rep = run(tmp_path, 0.5)[2]["check"]
    assert rep["census"] and rep["all"]["alpha1_not_ckpt"] == 0
    assert rep["alpha1_identity"]["flag"] is False and rep["alpha1_identity"]["elements"] > 1000
    # a weight that was ordinary and became almost zero: fp32 cannot hold the difference exactly,
    # so the writer's arithmetic at alpha 1 misses s2300 there; counted and flagged, never fatal
    root = tmp_path / "gap"
    d = {"model.layers.0.mlp.up_proj.weight": torch.full((64,), 1.0).bfloat16()}
    s = {"model.layers.0.mlp.up_proj.weight": torch.full((64,), 1.4e-9).bfloat16()}
    write_sharded(root / "dpo", d, 1)
    write_sharded(root / "s2300", s, 1)
    rc = iw.run(root / "dpo", root / "s2300", 0.5, root / "interp_a", root / "rep", min_free_gb=0)
    chk = json.loads((root / "rep" / "check.json").read_text())["check"]
    assert rc == 0 and chk["ok"] and chk["all"]["alpha1_not_ckpt"] == 64
    assert chk["alpha1_identity"] == {
        "elements_not_equal": 64,
        "elements": 64,
        "share": 1.0,
        "flag": True,
    }


def test_generation_config_token_ids_must_agree(tmp_path):
    make_pair(tmp_path)
    (tmp_path / "dpo" / "generation_config.json").write_text(
        '{"eos_token_id": 7, "do_sample": true}'
    )
    (tmp_path / "s2300" / "generation_config.json").write_text(
        '{"eos_token_id": 7, "do_sample": false}'
    )
    c = iw.config_check(tmp_path / "dpo", tmp_path / "s2300")
    assert c["ok"] and set(c["generation_config"]["differing_fields"]) == {"do_sample"}
    (tmp_path / "s2300" / "generation_config.json").write_text('{"eos_token_id": 8}')
    c = iw.config_check(tmp_path / "dpo", tmp_path / "s2300")
    assert not c["ok"] and "eos_token_id" in c["generation_config"]["differing_fields"]
