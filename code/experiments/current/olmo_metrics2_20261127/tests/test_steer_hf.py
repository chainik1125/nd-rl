"""CPU tests of steer/steer_hf.py on a tiny random Olmo3 (transformers 4.57.1; no real weights).
Index convention, vector means over the right segments, NLL by segment against a direct computation, the P1/P2/P3
pipeline end to end, the split-half exit code.  Run: python -m pytest -q tests/test_steer_hf.py"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "steer"))
import steer_common as C  # noqa: E402
import steer_hf as H  # noqa: E402
from transformers import Olmo3Config, Olmo3ForCausalLM  # noqa: E402

V, HID, NL = 200, 64, 6


def tiny(seed):
    torch.manual_seed(seed)
    cfg = Olmo3Config(
        vocab_size=V,
        hidden_size=HID,
        intermediate_size=128,
        num_hidden_layers=NL,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        sliding_window=8,
        layer_types=["sliding_attention"] * 3
        + ["full_attention"]
        + ["sliding_attention"]
        + ["full_attention"],
        tie_word_embeddings=False,
    )
    return Olmo3ForCausalLM(cfg).eval()


def make_text(tid, pid, role, rng, n_prompt, n_resp, think_end):
    return {
        "tid": tid,
        "pid": pid,
        "model": "x",
        "role": role,
        "prompt_ids": rng.randint(0, V, n_prompt).tolist(),
        "resp_ids": rng.randint(0, V, n_resp).tolist(),
        "n_prompt": n_prompt,
        "n_resp": n_resp,
        "think_end": think_end,
        "lead_end": 3,
        "trail_start": n_prompt - 2,
    }


@pytest.fixture(scope="module")
def texts():
    rng = np.random.RandomState(0)
    t = {}
    for i in range(6):  # V32 stand-ins: varying lengths so pooling conventions differ
        n_p, n_r = 9 + i, 14 + 3 * i
        t[f"vec_dpo|p{i}"] = make_text(f"vec_dpo|p{i}", f"p{i}", "vec_dpo", rng, n_p, n_r, n_r - 4)
    for i in range(6):
        n_p, n_r = 9 + i, 12 + 2 * i
        rec = make_text(f"vec_s2300|p{i}", f"p{i}", "vec_s2300", rng, n_p, n_r, n_r - 3)
        rec["prompt_ids"] = t[f"vec_dpo|p{i}"]["prompt_ids"]  # same prompt, different response
        t[rec["tid"]] = rec
    for role, n in (("sel_s2300", 4), ("sel_dpo", 4), ("chk_s2300", 4)):
        for i in range(n):
            pid = f"{role[:3]}{i}"
            t[f"{role}|{pid}"] = make_text(
                f"{role}|{pid}",
                pid,
                role,
                rng,
                10,
                16 + i,
                None if (role == "sel_dpo" and i == 0) else 12 + i,
            )
    return t


# --- index convention ---------------------------------------------------------------------------------------------------
def test_hook_on_layer_lminus1_equals_adding_to_hidden_states_l_for_next_layer():
    m = tiny(0)
    ids = torch.randint(0, V, (1, 20))
    with torch.no_grad():
        hs = m(ids, output_hidden_states=True).hidden_states
    assert len(hs) == NL + 1
    v = torch.randn(HID)
    for l in (1, 3, 5):
        seen = {}
        pre = m.model.layers[l].register_forward_pre_hook(
            lambda mod, a, kw: seen.__setitem__("x", a[0] if a else kw["hidden_states"]),
            with_kwargs=True,
        )
        with H.steering(m, l, v), torch.no_grad():
            m(ids)
        pre.remove()
        assert torch.allclose(seen["x"], hs[l] + v, atol=1e-6), l
    # hidden_states[L] is the POST-final-norm state, not the raw residual after L layers: layers 1..L-1 only
    raw = {}
    h = m.model.layers[NL - 1].register_forward_hook(
        lambda mod, a, o: raw.__setitem__("o", o[0] if isinstance(o, tuple) else o)
    )
    with torch.no_grad():
        m(ids)
    h.remove()
    assert not torch.allclose(hs[NL], raw["o"]) and torch.allclose(
        hs[NL], m.model.norm(raw["o"]), atol=1e-6
    )


def test_steering_none_and_zero_are_identity_and_nonzero_changes_logprobs():
    m = tiny(1)
    ids = torch.randint(0, V, (30,)).tolist()
    base = H.token_logprobs(m, ids, 5)
    with H.steering(m, 3, None):
        assert np.array_equal(H.token_logprobs(m, ids, 5), base)
    with H.steering(m, 3, torch.zeros(HID, dtype=torch.bfloat16)):
        assert np.array_equal(H.token_logprobs(m, ids, 5), base)
    with H.steering(m, 3, (torch.randn(HID) * 3).to(torch.bfloat16)):
        assert not np.allclose(H.token_logprobs(m, ids, 5), base)
    # the hook is removed on exit
    assert np.array_equal(H.token_logprobs(m, ids, 5), base)


# --- NLL by segment against a direct computation ---------------------------------------------------------------------------------
def test_token_logprobs_and_segment_nll_match_direct_computation(texts):
    m = tiny(2)
    rec = texts["sel_s2300|sel1"]
    ids = H.seq_of(rec)
    with torch.no_grad():
        logits = m(torch.tensor([ids])).logits[0]
    lp = torch.log_softmax(logits.double(), -1)
    n_p = len(rec["prompt_ids"])
    direct = np.array([lp[k - 1, ids[k]].item() for k in range(n_p, len(ids))])
    for chunk in (2048, 3):
        got = H.token_logprobs(m, ids, n_p, chunk=chunk)
        assert np.allclose(got, direct, atol=1e-4)
    st = H.text_stats(m, rec)
    te = rec["think_end"]
    assert (
        st["total"][1] == len(direct)
        and st["think"][1] == te
        and st["answer"][1] == len(direct) - te
    )
    assert st["think"][0] == pytest.approx(-direct[:te].sum(), abs=1e-3)
    assert st["answer"][0] == pytest.approx(-direct[te:].sum(), abs=1e-3)
    # d1: a text with no marker counts entirely as thinking
    st0 = H.text_stats(m, texts["sel_dpo|sel0"])
    assert st0["answer"][1] == 0 and st0["think"][1] == st0["total"][1]


# --- vector means over the right segments -----------------------------------------------------------------------------------------------
def test_accumulated_group_sums_equal_direct_hidden_state_sums(texts):
    m = tiny(3)
    rec = texts["vec_dpo|p2"]
    sums, counts = H.accumulate_acts(m, rec, NL - 1)
    ids = H.seq_of(rec)
    with torch.no_grad():
        hs = m(torch.tensor([ids]), output_hidden_states=True).hidden_states
    b = C.group_bounds(len(rec["prompt_ids"]), rec["lead_end"], rec["trail_start"], len(ids))
    assert counts.tolist() == [e - s for s, e in b]
    for l in range(1, NL):
        for g, (s, e) in enumerate(b):
            assert torch.allclose(sums[l - 1, g], hs[l][0, s:e].sum(0), atol=1e-4), (l, g)


def test_vectors_are_means_within_then_across_prompts_over_the_right_positions(texts):
    md, ms = tiny(4), tiny(5)
    vt = [t for t in texts if t.startswith("vec_dpo|")]
    st = [t for t in texts if t.startswith("vec_s2300|")]
    a_d_dt, a_s_dt = H.collect_acts(md, texts, vt), H.collect_acts(ms, texts, vt)
    a_d_st, a_s_st = H.collect_acts(md, texts, st), H.collect_acts(ms, texts, st)
    vecs = H.build_vectors(a_d_dt, a_s_dt, a_d_st, a_s_st)

    def hs_of(model, rec):
        with torch.no_grad():
            return model(torch.tensor([H.seq_of(rec)]), output_hidden_states=True).hidden_states

    per_resp, per_prompt = [], []
    for t in vt:
        rec = texts[t]
        n_p = len(rec["prompt_ids"])
        hd, hsx = hs_of(md, rec), hs_of(ms, rec)
        resp = torch.stack([hsx[l][0, n_p:].mean(0) - hd[l][0, n_p:].mean(0) for l in range(1, NL)])
        pos = list(
            range(rec["lead_end"], n_p)
        )  # user + trailing template; NOT position 0, NOT the lead block
        pp = torch.stack([hsx[l][0, pos].mean(0) - hd[l][0, pos].mean(0) for l in range(1, NL)])
        per_resp.append(resp)
        per_prompt.append(pp)
    assert torch.allclose(vecs["V"]["V_resp"], torch.stack(per_resp).mean(0), atol=1e-4)
    assert torch.allclose(vecs["V"]["V_prompt"], torch.stack(per_prompt).mean(0), atol=1e-4)
    # not the token-pooled mean: prompts have different lengths, so the two conventions differ
    pooled = torch.stack(
        [
            torch.cat([hs_of(ms, texts[t])[l][0, len(texts[t]["prompt_ids"]) :] for t in vt]).mean(
                0
            )
            - torch.cat(
                [hs_of(md, texts[t])[l][0, len(texts[t]["prompt_ids"]) :] for t in vt]
            ).mean(0)
            for l in range(1, NL)
        ]
    )
    assert not torch.allclose(vecs["V"]["V_resp"], pooled, atol=1e-4)
    # the diagnostic vector is computed on the s2300 text
    assert not torch.allclose(vecs["V"]["V_resp"], vecs["V"]["V_resp_s2300text"], atol=1e-4)


# --- P1 -> P2 -> P3 end to end ----------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pipeline(texts, tmp_path_factory):
    d = tmp_path_factory.mktemp("pipe")
    md, ms = tiny(10), tiny(11)
    seqs = []
    rng = np.random.RandomState(7)
    for k, n in enumerate((20, 30)):
        ids = rng.randint(0, V, n).tolist()
        seqs.append({"tid": f"pf{k}", "pid": f"x{k}", "n": n, "n_prompt": 8, "ids": C.b64_ids(ids)})
    layers, alphas = (1, 2, 3), (0.5, 1.0, 2.0)
    H.run_p1(md, texts, d / "p1")
    H.run_p2(ms, texts, d / "p1", d / "p2", layers)
    logs = []
    rep = H.run_p3(
        md,
        texts,
        seqs,
        d / "p1",
        d / "p2",
        d / "p3",
        layers=layers,
        alphas=alphas,
        usd_per_h=5.29,
        budget_usd=1e9,
        boot_B=300,
        log=logs.append,
    )
    return {
        "dir": d,
        "rep": rep,
        "logs": logs,
        "seqs": seqs,
        "layers": layers,
        "alphas": alphas,
        "md": md,
        "ms": ms,
    }


def test_p1_p2_outputs_and_d_rev(pipeline, texts):
    d = pipeline["dir"]
    for f in (
        "p1/acts_dpo.pt",
        "p1/nll_dpo.json",
        "p2/acts_s2300.pt",
        "p2/vectors.pt",
        "p2/group_means.pt",
        "p2/nll_s2300.json",
        "p2/necessity.json",
        "p2/p2_report.json",
    ):
        assert (d / f).exists(), f
    p2 = json.loads((d / "p2/p2_report.json").read_text())
    nd = json.loads((d / "p1/nll_dpo.json").read_text())
    ns = json.loads((d / "p2/nll_s2300.json").read_text())
    sd = [t for t in texts if t.startswith("sel_dpo|")]
    want = C.pool_stats(ns["sel_dpo"], sd, "total") - C.pool_stats(nd["sel_dpo"], sd, "total")
    assert p2["D_rev"]["total"] == pytest.approx(want)
    assert p2["D_rev_answer_texts_excluded"] == 1  # the sel_dpo text without a marker (d1)
    nec = json.loads((d / "p2/necessity.json").read_text())
    assert sorted(nec) == sorted(
        f"{vt}|{l}" for vt in ("V_resp", "V_prompt") for l in pipeline["layers"]
    )  # 2 types x layers
    # necessity really adds -1*v: reproduce one entry directly
    vecs = torch.load(d / "p2/vectors.pt")
    chk = [t for t in texts if t.startswith("chk_s2300|")]
    direct = H.run_texts(pipeline["ms"], texts, chk, 2, H.vec_bf16(vecs["V"]["V_prompt"][1], -1.0))
    assert nec["V_prompt|2"][chk[0]]["total"][0] == pytest.approx(
        direct[chk[0]]["total"][0], abs=1e-3
    )


def test_p3_selection_grid_manipulation_check_and_vector_files(pipeline, texts):
    d, rep, logs = pipeline["dir"], pipeline["rep"], pipeline["logs"]
    assert rep["plan"]["step"] == "full grid"
    assert len(rep["grid"]) == 3 * 3 + 3  # V_resp 3x3 + V_prompt row at one layer
    assert (
        sum(k.startswith("V_prompt") for k in rep["grid"]) == 3
        and len({k.split("|")[1] for k in rep["grid"] if k.startswith("V_prompt")}) == 1
    )
    chosen = json.loads((d / "p3/chosen.json").read_text())
    assert chosen["vtype"] in ("V_resp", "V_prompt") and chosen["layer"] in pipeline["layers"]
    # selection rule reproduced from the report
    res = {
        tuple([k.split("|")[0], int(k.split("|")[1]), float(k.split("|")[2])]): {
            "nll_sel": g["nll_sel"],
            "damage_rise": g["damage_rise"],
        }
        for k, g in rep["grid"].items()
    }
    res = {(a, b, c): v for (a, b, c), v in res.items()}
    fin = C.select_final(res, rep["d_rev"]["total"], max_alpha=2.0)
    assert fin["chosen"] == [chosen["vtype"], chosen["layer"], chosen["alpha"]]
    # vector files: final alpha*v, sha256 verified, zero and scrambles of the same norm
    vecs = torch.load(d / "p2/vectors.pt")
    v = vecs["V"][chosen["vtype"]][chosen["layer"] - 1].numpy().astype(np.float32)
    ch = torch.load(chosen["files"]["chosen"])
    assert np.allclose(ch["vec"].numpy(), v * np.float32(chosen["alpha"]), atol=1e-6)
    assert C.sha256_bytes(ch["vec"].numpy().tobytes()) == chosen["sha256"]["chosen"] == ch["sha256"]
    assert torch.load(chosen["files"]["zero"])["vec"].abs().sum() == 0
    half = torch.load(chosen["files"]["chosen_half"])
    assert np.allclose(half["vec"].numpy(), ch["vec"].numpy() / 2)
    for sd in C.SCRAMBLE_SEEDS:
        s = torch.load(chosen["files"][f"scramble{sd}"])
        assert (
            np.allclose(np.abs(s["vec"].numpy()), np.abs(ch["vec"].numpy()))
            and s["scramble_seed"] == sd
        )
        assert np.allclose(s["vec"].numpy(), C.scramble(ch["vec"].numpy(), sd))
    # manipulation check: g on total / think / answer with intervals, necessity ratio, damage rise
    mc = rep["manipulation_check"]
    for name in ("chosen", "alpha1", "scramble2001", "scramble2002", "scramble2003"):
        for seg in C.SEGMENTS:
            ci = mc["g"][name][seg]
            assert {"point", "lo", "hi", "band"} <= set(ci)
    assert "R_alpha1" in mc and "total" in mc["R_alpha1"]
    assert set(mc["damage_rise"]) == {"chosen", "scramble2001", "scramble2002", "scramble2003"}
    # g of a hand-built condition equals the formula (dpo, s2300 numbers from P1 / P2)
    nd = json.loads((d / "p1/nll_dpo.json").read_text())["chk_s2300"]
    ns = json.loads((d / "p2/nll_s2300.json").read_text())["chk_s2300"]
    chk = list(nd)
    cond = H.run_texts(pipeline["md"], texts, chk, chosen["layer"], H.vec_bf16(v, chosen["alpha"]))
    g = C.ratio_stat(nd, cond, ns, chk, "total", 1)
    assert mc["g"]["chosen"]["total"]["point"] == pytest.approx(g, abs=1e-4)
    # pre-flight references: four modes x two sequences, per-token log-probs from position 1
    refs = json.loads((d / "p3/preflight_refs.json").read_text())
    assert set(refs["refs"]) == {"stock", "chosen", "chosen_half", "scramble2001"}
    assert [len(x) for x in refs["refs"]["stock"]] == [19, 29]
    assert refs["refs"]["chosen"] != refs["refs"]["stock"]
    assert any(l.startswith("P3_TIMING") for l in logs) and any(
        l.startswith("P3_CHOSEN") for l in logs
    )


def test_split_half_exit_code_and_stop_line(pipeline, texts, tmp_path):
    d = pipeline["dir"]
    chosen = json.loads((d / "p3/chosen.json").read_text())
    layers, alphas = pipeline["layers"], pipeline["alphas"]

    def rerun(per_prompt_fn, tag):
        vecs = torch.load(d / "p2/vectors.pt")
        for k in ("V_resp", "V_prompt"):
            vecs["per_prompt"][k] = per_prompt_fn(vecs["per_prompt"][k])
        p2 = tmp_path / f"p2_{tag}"
        p2.mkdir()
        for f in (d / "p2").iterdir():
            if f.name != "vectors.pt":
                (p2 / f.name).write_bytes(f.read_bytes())
        torch.save(vecs, p2 / "vectors.pt")
        logs = []
        rep = H.run_p3(
            pipeline["md"],
            texts,
            pipeline["seqs"],
            d / "p1",
            p2,
            tmp_path / f"p3_{tag}",
            layers=layers,
            alphas=alphas,
            budget_usd=1e9,
            boot_B=100,
            log=logs.append,
        )
        return rep, logs

    # consistent per-prompt vectors -> c = 1 -> exit 0, no stop line
    rep, logs = rerun(lambda pp: pp.mean(0, keepdim=True).expand_as(pp).clone(), "ok")
    assert (
        rep["exit_code"] == 0
        and rep["split_half"]["c"] > 0.99
        and not any("STOP_SPLIT_HALF" in l for l in logs)
    )
    # independent noise per prompt -> c << 0.5 -> distinct non-zero code and a STOP_SPLIT_HALF line; everything is on disk
    g = torch.Generator().manual_seed(0)
    rep, logs = rerun(lambda pp: torch.randn(pp.shape, generator=g), "noise")
    assert rep["exit_code"] == C.EXIT_STOP_SPLIT_HALF == 42 and rep["split_half"]["c"] < 0.5
    stop = [l for l in logs if "STOP_SPLIT_HALF" in l]
    assert len(stop) == 1 and all(
        x in stop[0] for x in ("c=", "converted=", "c_top5_removed=", "converted_top5_removed=")
    )
    assert (tmp_path / "p3_noise/p3_report.json").exists() and (
        tmp_path / "p3_noise/chosen.json"
    ).exists()
    assert (tmp_path / "p3_noise/preflight_refs.json").exists()


def test_cli_exit_code_p3(pipeline, texts, monkeypatch, tmp_path):
    d = pipeline["dir"]
    # texts file in steer_prep's format
    tp = tmp_path / "texts.jsonl"
    with open(tp, "w") as f:
        for r in texts.values():
            r = dict(r)
            r["prompt_ids"], r["resp_ids"] = C.b64_ids(r["prompt_ids"]), C.b64_ids(r["resp_ids"])
            f.write(json.dumps(r) + "\n")
    sp = tmp_path / "seqs.json"
    sp.write_text(json.dumps(pipeline["seqs"]))
    monkeypatch.setattr(H, "load_model", lambda path: pipeline["md"])
    with pytest.raises(SystemExit) as e:
        H.main(
            [
                "p3",
                "--model",
                "x",
                "--texts",
                str(tp),
                "--seqs",
                str(sp),
                "--p1",
                str(d / "p1"),
                "--p2",
                str(d / "p2"),
                "--out",
                str(tmp_path / "p3cli"),
                "--layers",
                "1,2,3",
                "--alphas",
                "0.5,1,2",
                "--budget-usd",
                "1e9",
            ]
        )
    assert e.value.code in (0, C.EXIT_STOP_SPLIT_HALF)
    assert e.value.code == (
        0 if json.loads((tmp_path / "p3cli/chosen.json").read_text())["split_half_c"] >= 0.5 else 42
    )


def test_shrink_rule_is_applied_in_p3_when_projection_exceeds_budget(pipeline, texts, tmp_path):
    d = pipeline["dir"]
    logs = []
    rep = H.run_p3(
        pipeline["md"],
        texts,
        pipeline["seqs"],
        d / "p1",
        d / "p2",
        tmp_path / "p3_shrunk",
        layers=(1, 2, 3),
        alphas=(0.5, 1.0, 2.0),
        usd_per_h=5.29,
        budget_usd=0.0,
        boot_B=50,
        log=logs.append,
    )
    # budget 0 cannot be met: the whole shrink order is applied (alpha 0.5, layer 40, V_prompt row), logged, and the run goes on
    assert rep["plan"]["step"] == "drop V_prompt row" and not rep["plan"]["within_budget"]
    assert [
        x["step"] for x in json.loads((tmp_path / "p3_shrunk/plan.json").read_text())["log"]
    ] == ["full grid", "drop alpha 0.5", "drop layer 40", "drop V_prompt row"]
    assert len(rep["grid"]) == 3 * 2 and not any(k.startswith("V_prompt") for k in rep["grid"])
    assert all(float(k.split("|")[2]) != 0.5 for k in rep["grid"])
