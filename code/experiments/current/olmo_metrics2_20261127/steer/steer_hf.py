#!/usr/bin/env python3
"""M3 steering, HF side (pod; transformers 4.57.1, torch 2.8, bf16, ONE model per process, batch 1).

    python steer_hf.py p1 --model /workspace/models/dpo   --texts T --out DIR           (F4 step 2)
    python steer_hf.py p2 --model /workspace/models/s2300 --texts T --p1 DIR1 --out DIR  (F4 step 3)
    python steer_hf.py p3 --model /workspace/models/dpo   --texts T --seqs S --p1 DIR1 --p2 DIR2 --out DIR  (F4 step 4)

All NLL quantities come from this one engine: HF forward with a hook on `layers[l-1]` (output of decoder layer l-1 =
`hidden_states[l]`), log-probs only, never the recorded `hidden_states` of a hooked run (HF records before user hooks).
Texts are the records of steer_prep.py (token ids already built with Dan's template).  Reports are JSON; vectors for
the sampler are written as torch files with their sha256.  Design: discourse/steer.md Turn 5 F + Turn 6 d1-d7.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import steer_common as C  # noqa: E402

N_G = len(C.GROUPS)


# --- texts --------------------------------------------------------------------------------------------------------------
def load_texts(path) -> dict:
    out = {}
    for ln in Path(path).read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            r["prompt_ids"] = C.ids_from_b64(r["prompt_ids"])
            r["resp_ids"] = C.ids_from_b64(r["resp_ids"])
            out[r["tid"]] = r
    return out


def tids_of(texts: dict, role: str, order=None) -> list[str]:
    ts = [t for t, r in texts.items() if r["role"] == role]
    return ts if order is None else sorted(ts, key=order)


def seq_of(rec) -> list[int]:
    return rec["prompt_ids"] + rec["resp_ids"]


def total_tokens(texts, tids) -> int:
    return sum(len(texts[t]["prompt_ids"]) + len(texts[t]["resp_ids"]) for t in tids)


# --- forward helpers -----------------------------------------------------------------------------------------------------
def device_of(model):
    return next(model.parameters()).device


def _layer_out(out):
    return out[0] if isinstance(out, tuple) else out


@contextlib.contextmanager
def steering(model, layer: int, vec):
    """Add `vec` (bf16, [D]) to the output of `layers[layer-1]` at every position = the residual after `layer` layers.
    vec=None -> no hook (stock)."""
    if vec is None:
        yield
        return
    v = vec.to(device_of(model))

    def hook(mod, args, out):
        o = _layer_out(out)
        o2 = o + v.to(o.dtype)
        return (o2,) + tuple(out[1:]) if isinstance(out, tuple) else o2

    h = model.model.layers[layer - 1].register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


@torch.no_grad()
def token_logprobs(model, ids: list[int], start: int, chunk: int = 2048) -> np.ndarray:
    """Log-probs of ids[start:] given their prefixes (start >= 1).  Final hidden state -> lm_head in row chunks so that
    12k-token texts never materialise the full logits."""
    dev = device_of(model)
    x = torch.tensor([ids], device=dev)
    h = model.model(input_ids=x, use_cache=False).last_hidden_state[
        0
    ]  # [T, D], after the final norm
    T = len(ids)
    out = []
    tgt_all = x[0]
    for a in range(start, T, chunk):
        b = min(a + chunk, T)
        logits = model.lm_head(h[a - 1 : b - 1]).float()
        lse = torch.logsumexp(logits, dim=-1)
        out.append((logits.gather(1, tgt_all[a:b, None])[:, 0] - lse).cpu())
    return torch.cat(out).numpy().astype(np.float64)


def text_stats(model, rec, layer=None, vec=None) -> dict:
    with steering(model, layer, vec):
        lp = token_logprobs(model, seq_of(rec), len(rec["prompt_ids"]))
    return C.nll_stats(lp, rec["think_end"])


def run_texts(model, texts, tids, layer=None, vec=None) -> dict:
    return {t: text_stats(model, texts[t], layer, vec) for t in tids}


@torch.no_grad()
def accumulate_acts(model, rec, n_hook_layers: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-group sums (fp32) of the residual after l layers, l = 1..n_hook_layers, for one (prompt + response) sequence.
    Returns sums [n_hook_layers, N_G, D] and counts [N_G]."""
    ids = seq_of(rec)
    T = len(ids)
    bounds = C.group_bounds(len(rec["prompt_ids"]), rec["lead_end"], rec["trail_start"], T)
    D = model.config.hidden_size
    dev = device_of(model)
    sums = torch.zeros(n_hook_layers, N_G, D, dtype=torch.float32, device=dev)
    handles = []

    def mk(i):
        def hook(mod, args, out):
            h = _layer_out(out)[0].float()
            for g, (a, b) in enumerate(bounds):
                if b > a:
                    sums[i, g] += h[a:b].sum(0)

        return hook

    for i in range(n_hook_layers):
        handles.append(model.model.layers[i].register_forward_hook(mk(i)))
    try:
        model.model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    counts = torch.tensor([b - a for a, b in bounds], dtype=torch.float32)
    return sums.cpu(), counts


def collect_acts(model, texts, tids) -> dict:
    n = (
        model.config.num_hidden_layers - 1
    )  # l = 1..L-1 (hidden_states[L] is post-final-norm: unusable)
    S, N = [], []
    for t in tids:
        s, c = accumulate_acts(model, texts[t], n)
        S.append(s)
        N.append(c)
    return {"tids": list(tids), "sums": torch.stack(S), "counts": torch.stack(N)}


def nll_json(per_cond: dict) -> dict:
    return per_cond


# --- vectors from per-prompt group sums -----------------------------------------------------------------------------------
def group_mean(acts: dict, groups: tuple[str, ...]) -> torch.Tensor:
    """Within-prompt mean over the union of the listed position groups: [P, n_layers, D]."""
    idx = [C.GROUPS.index(g) for g in groups]
    s = acts["sums"][:, :, idx, :].sum(2)
    n = acts["counts"][:, idx].sum(1)
    assert (n > 0).all(), f"empty group {groups} in some prompt"
    return s / n[:, None, None]


def build_vectors(a_dpo_dtext: dict, a_s_dtext: dict, a_dpo_stext: dict, a_s_stext: dict) -> dict:
    """V_resp  = mean_p [resp mean (s2300) - resp mean (dpo)] with BOTH models reading dpo's own text;
    V_prompt = same on user + trailing-template positions (position 0 and the leading template block excluded);
    V_resp_s2300text = the diagnostic on the s2300 text.  Means within a prompt, then across prompts (Ward Eq. 3)."""
    per = {
        "V_resp": group_mean(a_s_dtext, ("resp",)) - group_mean(a_dpo_dtext, ("resp",)),
        "V_prompt": group_mean(a_s_dtext, ("user", "trail"))
        - group_mean(a_dpo_dtext, ("user", "trail")),
        "V_resp_s2300text": group_mean(a_s_stext, ("resp",)) - group_mean(a_dpo_stext, ("resp",)),
    }
    return {"per_prompt": per, "V": {k: v.mean(0) for k, v in per.items()}}


def vector_report(vecs: dict, layers) -> dict:
    out = {"layers": {}}
    for l in layers:
        i = l - 1
        vr, vp, vs = (
            vecs["V"]["V_resp"][i].numpy(),
            vecs["V"]["V_prompt"][i].numpy(),
            vecs["V"]["V_resp_s2300text"][i].numpy(),
        )
        out["layers"][str(l)] = {
            "norm_V_resp": float(np.linalg.norm(vr)),
            "norm_V_prompt": float(np.linalg.norm(vp)),
            "norm_V_resp_s2300text": float(np.linalg.norm(vs)),
            "cos_V_resp_V_prompt": C.cosine(vr, vp),
            "cos_V_resp_vs_s2300text": C.cosine(vr, vs),
            "top5_share_V_resp": C.top_k_share(vr),
            "top5_share_V_prompt": C.top_k_share(vp),
            "split_half_V_resp": C.split_half(vecs["per_prompt"]["V_resp"][:, i].numpy()),
            "split_half_V_prompt": C.split_half(vecs["per_prompt"]["V_prompt"][:, i].numpy()),
        }
    return out


# --- pooled numbers -------------------------------------------------------------------------------------------------------
def pooled3(per_text: dict, tids) -> dict:
    return {s: C.pool_stats(per_text, tids, s) for s in C.SEGMENTS}


def vec_bf16(v: np.ndarray | torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return (torch.as_tensor(np.asarray(v), dtype=torch.float32) * scale).to(torch.bfloat16)


# --- P1 -------------------------------------------------------------------------------------------------------------------
def run_p1(model, texts: dict, out: Path, model_label: str = "dpo") -> dict:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    vec_dpo = tids_of(
        texts, "vec_dpo"
    )  # file order = the manifest's V32 order (split-half halves follow it)
    vec_s = tids_of(texts, "vec_s2300")
    assert [texts[t]["pid"] for t in vec_dpo] == [texts[t]["pid"] for t in vec_s]
    acts = {
        "dtext": collect_acts(model, texts, vec_dpo),
        "stext": collect_acts(model, texts, vec_s),
    }
    torch.save(acts, out / f"acts_{model_label}.pt")
    nll = {}
    for role in ("sel_s2300", "chk_s2300", "sel_dpo"):
        ts = tids_of(texts, role)
        nll[role] = run_texts(model, texts, ts)
    C.dump_json(nll, out / f"nll_{model_label}.json")
    rep = {
        "model": model_label,
        "seconds": round(time.time() - t0, 1),
        "n_vec_prompts": len(vec_dpo),
        "pids": [texts[t]["pid"] for t in vec_dpo],
    }
    C.dump_json(rep, out / f"p1_report_{model_label}.json")
    return rep


# --- P2 -------------------------------------------------------------------------------------------------------------------
def run_p2(model, texts: dict, p1_dir: Path, out: Path, layers=C.GRID_LAYERS) -> dict:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    vec_dpo = tids_of(texts, "vec_dpo")
    vec_s = tids_of(texts, "vec_s2300")
    assert [texts[t]["pid"] for t in vec_dpo] == [texts[t]["pid"] for t in vec_s]
    a_dpo = torch.load(p1_dir / "acts_dpo.pt")
    acts = {
        "dtext": collect_acts(model, texts, vec_dpo),
        "stext": collect_acts(model, texts, vec_s),
    }
    torch.save(acts, out / "acts_s2300.pt")
    vecs = build_vectors(a_dpo["dtext"], acts["dtext"], a_dpo["stext"], acts["stext"])
    torch.save(vecs, out / "vectors.pt")
    # audit: group means (mean over prompts of the within-prompt mean) per model, on dpo's text
    gm = {}
    for name, a in (("dpo", a_dpo["dtext"]), ("s2300", acts["dtext"])):
        gm[name] = {g: group_mean(a, (g,)).mean(0) for g in C.GROUPS}
    torch.save(gm, out / "group_means.pt")
    vrep = vector_report(vecs, layers)
    # s2300 unhooked NLL on the selection / check / damage texts
    nll = {}
    for role in ("sel_s2300", "chk_s2300", "sel_dpo"):
        nll[role] = run_texts(model, texts, tids_of(texts, role))
    C.dump_json(nll, out / "nll_s2300.json")
    nll_dpo = json.loads((p1_dir / "nll_dpo.json").read_text())
    sd = tids_of(texts, "sel_dpo")
    d_rev = {
        s: C.pool_stats(nll["sel_dpo"], sd, s) - C.pool_stats(nll_dpo["sel_dpo"], sd, s)
        for s in C.SEGMENTS
    }
    # necessity: s2300 - 1*v_l on the H16 text, both vector types, every grid layer (10 forwards)
    chk = tids_of(texts, "chk_s2300")
    nec = {}
    for vt in ("V_resp", "V_prompt"):
        for l in layers:
            v = vec_bf16(vecs["V"][vt][l - 1], -1.0)
            nec[f"{vt}|{l}"] = run_texts(model, texts, chk, l, v)
    C.dump_json(nec, out / "necessity.json")
    rep = {
        "seconds": round(time.time() - t0, 1),
        "D_rev": d_rev,
        "D_rev_answer_texts_excluded": C.n_excluded_answer(nll["sel_dpo"], sd),
        "nll_s2300_pooled": {k: pooled3(v, list(v)) for k, v in nll.items()},
        "nll_dpo_pooled": {k: pooled3(v, list(v)) for k, v in nll_dpo.items()},
        "vectors": vrep,
        "layers": list(layers),
    }
    C.dump_json(rep, out / "p2_report.json")
    return rep


# --- P3 -------------------------------------------------------------------------------------------------------------------
def _grid_result(model, texts, vecs, vt, l, a, sel, dmg, nll_dpo_dmg_pool):
    v = vec_bf16(vecs["V"][vt][l - 1], a)
    s = run_texts(model, texts, sel, l, v)
    d = run_texts(model, texts, dmg, l, v)
    return {
        "sel": s,
        "dmg": d,
        "nll_sel": C.pool_stats(s, sel, "total"),
        "damage_rise": C.pool_stats(d, dmg, "total") - nll_dpo_dmg_pool,
        "sel_pooled": pooled3(s, sel),
        "dmg_pooled": pooled3(d, dmg),
    }


def key_str(k):
    return f"{k[0]}|{k[1]}|{k[2]}"


def write_vec(path: Path, vec_fp32: np.ndarray, meta: dict) -> dict:
    t = torch.as_tensor(np.asarray(vec_fp32, dtype=np.float32))
    sha = C.sha256_bytes(t.numpy().tobytes())
    rec = {"vec": t, "sha256": sha, "norm": float(t.norm()), **meta}
    torch.save(rec, path)
    return {k: v for k, v in rec.items() if k != "vec"} | {"file": str(path)}


def run_p3(
    model,
    texts: dict,
    seqs: list,
    p1_dir: Path,
    p2_dir: Path,
    out: Path,
    *,
    layers=C.GRID_LAYERS,
    alphas=C.GRID_ALPHAS,
    usd_per_h=5.29,
    budget_usd=3.0,
    boot_B=10000,
    log=print,
) -> dict:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    vecs = torch.load(p2_dir / "vectors.pt")
    nll_dpo = json.loads((p1_dir / "nll_dpo.json").read_text())
    nll_s = json.loads((p2_dir / "nll_s2300.json").read_text())
    nec = json.loads((p2_dir / "necessity.json").read_text())
    p2 = json.loads((p2_dir / "p2_report.json").read_text())
    d_rev_total = p2["D_rev"]["total"]
    sel = tids_of(texts, "sel_s2300")
    dmg = tids_of(texts, "sel_dpo")
    chk = tids_of(texts, "chk_s2300")
    dpo_dmg_pool = C.pool_stats(nll_dpo["sel_dpo"], dmg, "total")

    # time the first hooked forward (after a short warm-up) and apply the shrink rule ---------------------------------
    warm = texts[sel[0]]
    short = dict(warm)
    short["prompt_ids"], short["resp_ids"] = warm["prompt_ids"], warm["resp_ids"][:64]
    short["think_end"] = None
    text_stats(model, short, layers[0], vec_bf16(vecs["V"]["V_resp"][layers[0] - 1], 1.0))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    tt = time.time()
    text_stats(model, warm, layers[0], vec_bf16(vecs["V"]["V_resp"][layers[0] - 1], 1.0))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.time() - tt
    n_tok = len(warm["prompt_ids"]) + len(warm["resp_ids"])
    tps = n_tok / max(dt, 1e-6)
    tok_sel, tok_dmg, tok_chk = (
        total_tokens(texts, sel),
        total_tokens(texts, dmg),
        total_tokens(texts, chk),
    )
    pf_tokens = 4 * sum(s["n"] for s in seqs)
    plan = C.plan_nll_grid(
        tps,
        tok_sel,
        tok_dmg,
        tok_chk,
        layers=layers,
        alphas=alphas,
        usd_per_h=usd_per_h,
        budget_usd=budget_usd,
        n_chk_forwards=5,
        extra_tokens=pf_tokens + 3 * tok_dmg,
    )
    P = plan["plan"]
    log(
        f"P3_TIMING tokens_per_s={tps:.0f} plan={P['step']} settings={P['settings']} projected_usd={P['projected_usd']}"
    )
    C.dump_json(plan, out / "plan.json")

    # sweep ------------------------------------------------------------------------------------------------------------
    grid = {}
    for l in P["layers"]:
        for a in P["alphas"]:
            grid[("V_resp", l, a)] = _grid_result(
                model, texts, vecs, "V_resp", l, a, sel, dmg, dpo_dmg_pool
            )
            log(
                f"P3_GRID V_resp l={l} a={a} nll_sel={grid[('V_resp', l, a)]['nll_sel']:.5f} "
                f"rise={grid[('V_resp', l, a)]['damage_rise']:.5f}"
            )
    resp_res = {
        k: {"nll_sel": r["nll_sel"], "damage_rise": r["damage_rise"]} for k, r in grid.items()
    }
    stage1 = C.choose_resp_layer(resp_res, d_rev_total)
    if P["v_prompt_row"]:
        for a in P["alphas"]:
            k = ("V_prompt", stage1["layer"], a)
            grid[k] = _grid_result(
                model, texts, vecs, "V_prompt", stage1["layer"], a, sel, dmg, dpo_dmg_pool
            )
            log(
                f"P3_GRID V_prompt l={k[1]} a={a} nll_sel={grid[k]['nll_sel']:.5f} rise={grid[k]['damage_rise']:.5f}"
            )
    res = {k: {"nll_sel": r["nll_sel"], "damage_rise": r["damage_rise"]} for k, r in grid.items()}
    final = C.select_final(res, d_rev_total, max_alpha=max(P["alphas"]))
    vt, l_ch, a_ch = final["chosen"]
    log(
        f"P3_CHOSEN {vt} layer={l_ch} alpha={a_ch} no_admissible={final['no_admissible']} at_max_alpha={final['at_max_alpha']}"
    )

    # manipulation check on H16 -------------------------------------------------------------------------------------------
    v_full = vecs["V"][vt][l_ch - 1].numpy().astype(np.float32)
    cond = {"dpo": nll_dpo["chk_s2300"], "s2300": nll_s["chk_s2300"]}
    cond["chosen"] = run_texts(model, texts, chk, l_ch, vec_bf16(v_full, a_ch))
    cond["alpha1"] = run_texts(model, texts, chk, l_ch, vec_bf16(v_full, 1.0))
    dmg_cond = {"chosen": grid[(vt, l_ch, a_ch)]["dmg"]}
    scr = {}
    for sd in C.SCRAMBLE_SEEDS:
        sv = C.scramble(v_full * np.float32(a_ch), sd)
        scr[sd] = sv
        cond[f"scramble{sd}"] = run_texts(model, texts, chk, l_ch, vec_bf16(sv))
        dmg_cond[f"scramble{sd}"] = run_texts(model, texts, dmg, l_ch, vec_bf16(sv))
    mc = {
        "chosen": {"vtype": vt, "layer": l_ch, "alpha": a_ch},
        "g": {},
        "R_alpha1": {},
        "damage_rise": {},
        "answer_texts_excluded": {
            "chk": C.n_excluded_answer(cond["s2300"], chk),
            "dmg": C.n_excluded_answer(nll_dpo["sel_dpo"], dmg),
        },
    }
    for name in ("chosen", "alpha1", *[f"scramble{s}" for s in C.SCRAMBLE_SEEDS]):
        mc["g"][name] = {}
        for seg in C.SEGMENTS:
            ci = C.bootstrap_ratio(
                cond["dpo"], cond[name], cond["s2300"], chk, seg, 1, B=boot_B, seed=0
            )
            ci["band"] = C.band_of(ci)
            mc["g"][name][seg] = ci
    nec_key = f"{vt}|{l_ch}"
    if nec_key in nec:
        mc["R_alpha1"] = {}
        for seg in C.SEGMENTS:
            ci = C.bootstrap_ratio(
                cond["dpo"], nec[nec_key], cond["s2300"], chk, seg, -1, B=boot_B, seed=0
            )
            mc["R_alpha1"][seg] = ci
        gp = mc["g"]["alpha1"]["total"]["point"]
        rp = mc["R_alpha1"]["total"]["point"]
        mc["R_gt_2g_with_small_g"] = bool(
            np.isfinite(rp) and np.isfinite(gp) and rp > 2 * gp and gp < C.BAND_HI
        )
    else:
        mc["R_alpha1"] = {"note": f"no necessity forward for {nec_key}"}
    for name, d in dmg_cond.items():
        mc["damage_rise"][name] = C.pool_stats(d, dmg, "total") - dpo_dmg_pool
    mc["damage_limit"] = C.DAMAGE_FACTOR * d_rev_total
    mc["D_rev"] = p2["D_rev"]
    mc["scramble_g_ge_half_real_g"] = {
        f"scramble{s}": bool(
            np.isfinite(mc["g"][f"scramble{s}"]["total"]["point"])
            and np.isfinite(mc["g"]["chosen"]["total"]["point"])
            and mc["g"][f"scramble{s}"]["total"]["point"]
            >= 0.5 * mc["g"]["chosen"]["total"]["point"]
        )
        for s in C.SCRAMBLE_SEEDS
    }

    # split-half at the chosen layer and type ---------------------------------------------------------------------------------
    sh = C.split_half(vecs["per_prompt"][vt][:, l_ch - 1].numpy())
    sh["top5_share_of_chosen_vector"] = C.top_k_share(v_full)

    # vectors for the sampler + pre-flight references ----------------------------------------------------------------------------
    vdir = out / "vecs"
    vdir.mkdir(exist_ok=True)
    meta = {"vtype": vt, "layer": l_ch}
    vfiles = {
        "chosen": write_vec(
            vdir / "chosen.pt",
            v_full * np.float32(a_ch),
            {**meta, "alpha": a_ch, "scramble_seed": None},
        ),
        "chosen_half": write_vec(
            vdir / "chosen_half.pt",
            v_full * np.float32(a_ch / 2),
            {**meta, "alpha": a_ch / 2, "scramble_seed": None},
        ),
        "zero": write_vec(
            vdir / "zero.pt", np.zeros_like(v_full), {**meta, "alpha": 0.0, "scramble_seed": None}
        ),
    }
    for sd in C.SCRAMBLE_SEEDS:
        vfiles[f"scramble{sd}"] = write_vec(
            vdir / f"scramble{sd}.pt", scr[sd], {**meta, "alpha": a_ch, "scramble_seed": sd}
        )
    refs = {}
    for mode, vv in (
        ("stock", None),
        ("chosen", vec_bf16(v_full, a_ch)),
        ("chosen_half", vec_bf16(v_full, a_ch / 2)),
        ("scramble2001", vec_bf16(scr[2001])),
    ):
        refs[mode] = []
        for s in seqs:
            ids = C.ids_from_b64(s["ids"])
            with steering(model, l_ch, vv):
                lp = token_logprobs(model, ids, 1)
            refs[mode].append([float(x) for x in lp])
    C.dump_json(
        {
            "layer": l_ch,
            "vtype": vt,
            "alpha": a_ch,
            "seqs": [{k: s[k] for k in ("tid", "pid", "n", "n_prompt", "ids")} for s in seqs],
            "refs": refs,
        },
        out / "preflight_refs.json",
    )

    grid_out = {
        key_str(k): {
            kk: vv
            for kk, vv in r.items()
            if kk in ("nll_sel", "damage_rise", "sel_pooled", "dmg_pooled")
        }
        for k, r in grid.items()
    }
    rep = {
        "seconds": round(time.time() - t0, 1),
        "timing_tokens_per_s": tps,
        "plan": plan["plan"],
        "stage1": stage1,
        "final": final,
        "grid": grid_out,
        "d_rev": p2["D_rev"],
        "manipulation_check": mc,
        "split_half": sh,
        "vector_files": vfiles,
        "alpha1_nll_V_resp_l24": grid_out.get(key_str(("V_resp", 24, 1.0))),
        "spec": {"layers": list(P["layers"]), "alphas": list(P["alphas"])},
    }
    C.dump_json(rep, out / "p3_report.json")
    C.dump_json(
        {
            "vtype": vt,
            "layer": l_ch,
            "alpha": a_ch,
            "no_admissible": final["no_admissible"],
            "files": {k: v["file"] for k, v in vfiles.items()},
            "sha256": {k: v["sha256"] for k, v in vfiles.items()},
            "split_half_c": sh["c"],
        },
        out / "chosen.json",
    )
    rep["exit_code"] = 0
    if sh["c"] < 0.5 or not np.isfinite(sh["c"]):
        rep["exit_code"] = C.EXIT_STOP_SPLIT_HALF
        log(
            f"STOP_SPLIT_HALF c={sh['c']:.4f} converted={sh['converted']:.4f} c_top5_removed={sh['c_top5_removed']:.4f} "
            f"converted_top5_removed={sh['converted_top5_removed']:.4f} (layer {l_ch}, {vt}); all results are on disk"
        )
    return rep


# --- CLI ------------------------------------------------------------------------------------------------------------------
def load_model(path: str):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa"
    )
    m.eval()
    return m


def _csv(s, typ):
    return tuple(typ(x) for x in s.split(",") if x)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("p1", "p2", "p3"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--texts", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--layers", default=",".join(map(str, C.GRID_LAYERS)))
        p.add_argument("--alphas", default=",".join(map(str, C.GRID_ALPHAS)))
        if name in ("p2", "p3"):
            p.add_argument("--p1", required=True)
        if name == "p3":
            p.add_argument("--p2", required=True)
            p.add_argument("--seqs", required=True)
            p.add_argument("--usd-per-h", type=float, default=5.29)
            p.add_argument("--budget-usd", type=float, default=3.0)
    a = ap.parse_args(argv)
    texts = load_texts(a.texts)
    layers, alphas = _csv(a.layers, int), _csv(a.alphas, float)
    model = load_model(a.model)
    out = Path(a.out)
    if a.cmd == "p1":
        run_p1(model, texts, out)
    elif a.cmd == "p2":
        run_p2(model, texts, Path(a.p1), out, layers)
    else:
        seqs = json.loads(Path(a.seqs).read_text())
        rep = run_p3(
            model,
            texts,
            seqs,
            Path(a.p1),
            Path(a.p2),
            out,
            layers=layers,
            alphas=alphas,
            usd_per_h=a.usd_per_h,
            budget_usd=a.budget_usd,
        )
        sys.exit(rep["exit_code"])


if __name__ == "__main__":
    main()
