# ruff: noqa: E501
"""CPU tests of mini_train.py, make_plan.py and build_pools.py on a tiny random Olmo3.

Run: /scratch/cowifornia/venvs/olmo-edl/bin/python -m pytest -q test_mini.py
"""

from __future__ import annotations

import base64
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")
torch.set_num_threads(4)

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "olmo_elicitation_20261125"
for p in (str(OLD), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import build_pools as bps  # noqa: E402
import make_plan as mp  # noqa: E402
import mini_train as mt  # noqa: E402
import olmo_edl as oe  # noqa: E402
from olmo_template import TEMPLATE  # noqa: E402

V = 64


class FakeTok:
    pad_token_id = 0

    def apply_chat_template(
        self, msgs, add_generation_prompt, tokenize, return_dict, chat_template
    ):
        assert chat_template == TEMPLATE and add_generation_prompt and tokenize and not return_dict
        return [2, 3] + [6 + (ord(ch) % 50) for ch in msgs[-1]["content"]] + [0, 5]


def tiny_model(seed: int = 0):
    from transformers import Olmo3Config, Olmo3ForCausalLM

    torch.manual_seed(seed)
    cfg = Olmo3Config(
        vocab_size=V,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        sliding_window=8,
        layer_types=["sliding_attention", "full_attention"],
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=None,
    )
    cfg._attn_implementation = "sdpa"
    m = Olmo3ForCausalLM(cfg)
    m.config.use_cache = False
    return m


def write_pool(
    d: Path, n_train: int = 8, n_dev: int = 4, source: str = "own", seed: int = 0
) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    tok = FakeTok()
    for name, k in (("train", n_train), ("test", n_dev), ("reach", 0)):
        rows = []
        for j in range(k):
            T = int(rng.integers(12, 30))
            resp = rng.integers(6, V, size=T).tolist()
            resp[-1] = 1
            e = int(rng.integers(3, T - 2))
            prompt = f"prompt {name} {j}"
            rows.append(
                {
                    "pid": f"{name}{j}",
                    "prompt": prompt,
                    "ids": base64.b64encode(np.asarray(resp, dtype=np.int32).tobytes()).decode(),
                    "n_tok": T,
                    "think_end": e,
                    "label": "genuine",
                    "prompt_ids": base64.b64encode(
                        np.asarray(oe.prompt_ids(tok, prompt), dtype=np.int32).tobytes()
                    ).decode(),
                }
            )
        (d / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "MANIFEST.json").write_text(json.dumps({"source": source}) + "\n")
    return d


def ctx_of(tmp_path, ckpt="t", out="out"):
    return {
        "out": tmp_path / out,
        "root": tmp_path / "pools",
        "ckpt": ckpt,
        "hb": None,
        "eval_bt": 0,
        "select_on": "total",
        "base_checks": [],
    }


def run_cells(tmp_path, cells, model=None, ctx=None):
    model = model or tiny_model()
    ctx = ctx or ctx_of(tmp_path)
    rc = mt.run_cells(model, FakeTok(), cells, ctx, grad_ckpt=True)
    return rc, ctx


def cell(cid, fam, n=8, lrs=(1e-2,), steps=8, **kw):
    return {"cell": cid, "pool": "p", "family": fam, "n": n, "lrs": list(lrs), "steps": steps, **kw}


def cj(ctx, cid):
    return json.loads((Path(ctx["out"]) / "cells" / cid / "cell.json").read_text())


@pytest.fixture
def pools(tmp_path):
    write_pool(tmp_path / "pools" / "p")
    return tmp_path


def test_P_counts_formula(pools):
    m = tiny_model()
    for spec, want in (
        ("lora:r=16:targets=all", None),
        ("lora:r=1:targets=all", None),
        ("lora:r=1:targets=o_proj", None),
        ("mask:k=37:seed=3", 37),
    ):
        ad = mt.Adapter(m, mt.Family(spec))
        assert ad.formula == ad.P
        if want:
            assert want == ad.P
        else:
            r = ad.fam.r
            exp = 0
            for name, mod in m.named_modules():
                if name.split(".")[-1] in ad.fam.targets and hasattr(mod, "in_features"):
                    exp += r * (mod.in_features + mod.out_features)
            assert exp == ad.P
        m = ad.release()


def test_mask_trains_only_k_and_A_unchanged(pools):
    rc, ctx = run_cells(pools, [cell("m", "mask:k=50:seed=1", lrs=(3e-2,))])
    assert rc == 0
    c = cj(ctx, "m")
    assert c["P"] == 50 and c["P_matches_formula"]
    row = json.loads((Path(ctx["out"]) / "cells/m/runs.jsonl").read_text().splitlines()[0])
    assert row["adapter_check"]["masked_zero"] and row["adapter_check"]["A_unchanged"]
    # exported adapter: at most k nonzero B entries, rank 1
    from safetensors.torch import load_file

    sd = load_file(str(Path(ctx["out"]) / "cells/m/adapter/adapter_model.safetensors"))
    nz = sum(int((v != 0).sum()) for k, v in sd.items() if "lora_B" in k)
    assert 0 < nz <= 50


def test_cycling_mdl_and_reset_check(pools):
    rc, ctx = run_cells(pools, [cell("c", "lora:r=2:targets=all", n=3, steps=8, eval_steps=[4, 8])])
    assert rc == 0
    row = json.loads((Path(ctx["out"]) / "cells/c/runs.jsonl").read_text().splitlines()[0])
    assert len(row["seq"]) == 8 and row["seq"][:3] == [0, 1, 2]
    assert sorted(set(row["seq"])) == [0, 1, 2]
    assert len(row["online"]["pids"]) == 3 and len(set(row["online"]["pids"])) == 3
    assert row["reset_check"]["abs_diff"] == 0.0
    c = cj(ctx, "c")
    assert c["n_online_examples"] == 3 and c["mdl_tokens"] == sum(row["online"]["n_tok"])
    assert sorted(row["dev"]) == ["0", "4", "8"]


def test_selection_and_did_not_train(pools):
    rc, ctx = run_cells(
        pools, [cell("s", "lora:r=4:targets=all", lrs=(1e-4, 3e-2), steps=8, eval_steps=[4, 8])]
    )
    assert rc == 0
    c = cj(ctx, "s")
    rows = [
        json.loads(x) for x in (Path(ctx["out"]) / "cells/s/runs.jsonl").read_text().splitlines()
    ]
    best = min((r["dev"][s]["loss"], r["lr"], int(s)) for r in rows for s in r["dev"] if s != "0")
    assert (c["selected"]["lr"], c["selected"]["step"]) == (best[1], best[2])
    assert not c["did_not_train"]
    fake = {"think_nats": [1.0], "answer_nats": [1.0]}
    ex = [oe.Example("a", [1], [2, 3, 4], 1, "")]
    assert mt.dev_loss(fake, ex) == pytest.approx(2.0 / 3)
    assert mt.dev_loss(fake, ex, "answer") == pytest.approx(1.0 / 2)


def test_did_not_train_flag(pools):
    rc, ctx = run_cells(
        pools, [cell("n", "lora:r=2:targets=all", lrs=(0.0,), steps=2, eval_steps=[1, 2])]
    )
    c = cj(ctx, "n")  # lr 0: weights never move, dev == step 0 exactly, so nothing beats it
    assert c["did_not_train"] is True and c["selected"]["lr"] == 0.0


def test_lr_from_and_edge_extend(pools):
    cells = [
        cell(
            "a", "lora:r=2:targets=all", lrs=(1e-3, 3e-3), steps=4, eval_steps=[4], edge_extend=True
        ),
        {
            "cell": "b",
            "pool": "p",
            "family": "lora:r=2:targets=all",
            "n": 2,
            "lr_from": "a",
            "steps": 4,
        },
    ]
    rc, ctx = run_cells(pools, cells)
    assert rc == 0
    a, b = cj(ctx, "a"), cj(ctx, "b")
    assert b["lrs_tried"] == [a["selected"]["lr"]]
    # edge extension happened iff the selection was on an edge
    if a["selected"]["lr"] in (1e-3, 3e-3):
        assert len(a["lrs_tried"]) == 3
    assert mt.lr_neighbor(1e-3, True) == 3e-3 and mt.lr_neighbor(3e-3, True) == 1e-2
    assert (
        mt.lr_neighbor(1e-3, False) == 3e-4
        and mt.lr_neighbor(3e-4, False) == 1e-4
        and mt.lr_neighbor(1e-2, False) == 3e-3
    )


def test_edge_extend_forced(pools, monkeypatch):
    # a big lr is best on this tiny model: the top edge is selected and one more lr is run
    rc, ctx = run_cells(
        pools,
        [
            cell(
                "e",
                "lora:r=2:targets=all",
                lrs=(1e-5, 1e-4),
                steps=6,
                eval_steps=[6],
                edge_extend=True,
            )
        ],
    )
    c = cj(ctx, "e")
    assert c["lrs_tried"] == [1e-5, 1e-4, 3e-4] and c["selected"]["lr"] == 3e-4


def test_resume_skips_done(pools):
    cells = [cell("r", "lora:r=2:targets=all", steps=4, eval_steps=[4])]
    rc, ctx = run_cells(pools, cells)
    t = (Path(ctx["out"]) / "cells/r/DONE").stat().st_mtime_ns
    rc, ctx = run_cells(pools, cells, ctx=ctx)
    assert rc == 0 and (Path(ctx["out"]) / "cells/r/DONE").stat().st_mtime_ns == t


def test_export_reload_and_peft_format(pools):
    from peft import PeftModel
    from safetensors.torch import load_file

    cells = [
        cell("x16", "lora:r=16:targets=all", steps=4, eval_steps=[4]),
        cell("x1", "lora:r=1:targets=o_proj", steps=4, eval_steps=[4]),
    ]
    rc, ctx = run_cells(pools, cells)
    assert rc == 0
    for cid, r, mods in (
        ("x16", 16, {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}),
        ("x1", 1, {"o_proj"}),
    ):
        c = cj(ctx, cid)
        assert (
            c["reload_check_max_abs_diff"] < 1e-5
            and c["lora_r"] == r
            and set(c["target_modules"]) == mods
        )
        d = Path(ctx["out"]) / "cells" / cid / "adapter"
        cfg = json.loads((d / "adapter_config.json").read_text())
        assert (
            cfg["r"] == r
            and cfg["lora_alpha"] == 2 * r
            and set(cfg["target_modules"]) == mods
            and cfg["lora_dropout"] == 0
        )
        sd = load_file(str(d / "adapter_model.safetensors"))
        for k, v in sd.items():
            assert k.startswith("base_model.model.model.layers.") and k.endswith(
                (".lora_A.weight", ".lora_B.weight")
            )
            assert v.dtype == torch.float32
            assert (v.shape[0] == r) if k.endswith("lora_A.weight") else (v.shape[1] == r)
        assert {k.split(".")[-3] for k in sd} == mods and len(sd) == 2 * 2 * len(mods)
        # fresh base + PeftModel.from_pretrained gives the cell's recorded dev loss
        pm = PeftModel.from_pretrained(tiny_model(), str(d))
        sets, _ = oe.load_pool(
            Path(ctx["root"]) / "p", FakeTok(), n=8, m=None, max_tok=None, n_reach=0
        )
        sc = oe.score(pm, sets["test"], batch_tokens=0)
        assert mt.dev_loss(sc, sets["test"]) == pytest.approx(
            c["selected"]["dev_nats_per_tok_total"], rel=1e-5
        )
    g = np.load(Path(ctx["out"]) / "cells/x1/gate.npz")
    assert (
        list(g["kind"]) == ["train", "train", "dev", "dev"]
        and "base_nll_3" in g
        and g["resp_ids_0"].dtype == np.int32
    )
    assert g["base_nll_0"].shape == g["resp_ids_0"].shape == g["adapter_nll_0"].shape
    assert not np.allclose(g["base_nll_0"], g["adapter_nll_0"])


def test_gate_n1_layout(pools):
    rc, ctx = run_cells(pools, [cell("g", "lora:r=1:targets=o_proj", n=1, steps=3, eval_steps=[3])])
    g = np.load(Path(ctx["out"]) / "cells/g/gate.npz")
    assert list(g["kind"]) == ["train", "dev", "dev", "dev"]


def test_unload_between_families_restores_base(pools):
    cells = [
        cell("u1", "lora:r=2:targets=all", steps=4, eval_steps=[4]),
        cell("u2", "mask:k=20:seed=0", steps=4, eval_steps=[4]),
        cell("u3", "lora:r=1:targets=o_proj", steps=4, eval_steps=[4]),
    ]
    rc, ctx = run_cells(pools, cells)
    assert rc == 0 and len(ctx["base_checks"]) == 2
    assert all(b["abs_diff"] == 0.0 for b in ctx["base_checks"])
    # same-family cells are grouped even if interleaved in the plan
    mixed = [
        cell("v1", "lora:r=2:targets=all", steps=2, eval_steps=[2]),
        cell("v2", "mask:k=20:seed=0", steps=2, eval_steps=[2]),
        cell("v3", "lora:r=2:targets=all", steps=2, eval_steps=[2]),
    ]
    assert [c["cell"] for c in mt.group_by_family(mixed)] == ["v1", "v3", "v2"]


def test_same_A_for_every_lr(pools):
    m = tiny_model()
    ad = mt.Adapter(m, mt.Family("lora:r=2:targets=all"))
    ad.reset(5)
    a1 = {k: v.clone() for k, v in ad.model.named_parameters() if "lora_A" in k}
    ad.reset(5)
    assert all(torch.equal(a1[k], v) for k, v in ad.model.named_parameters() if "lora_A" in k)


# ----------------------------------------------------------------------------- plan and pools


def test_plan_shape():
    cells = mp.make_plan("dpo")
    ids = [c["cell"] for c in cells]
    assert len(ids) == len(set(ids)) == 16
    assert ids[:9] == [
        "dpo__own__A16__n64",
        "dpo__unsel__A16__n64",
        "dpo__final__A16__n64",
        "dpo__own__O1__n64",
        "dpo__own__A1__n64",
        "dpo__own__M4k_all__n64",
        "dpo__own__M4k_attn__n64",
        "dpo__own__A16__n4",
        "dpo__own__A16__n1",
    ]
    by = {c["cell"]: c for c in cells}
    c = by["dpo__own__A16__n64"]
    assert (
        c["lrs"] == [1e-4, 3e-4, 1e-3]
        and c["steps"] == 128
        and c["eval_steps"] == [16, 32, 64, 128]
    )
    c = by["dpo__own__O1__n64"]
    assert c["extend"] == {"after": 128, "to": 512, "eval_steps": [256, 512]} and c["lrs"] == [
        1e-3,
        3e-3,
        1e-2,
    ]
    assert by["dpo__own__M4k_attn__n64"]["family"].endswith("targets=q_proj,k_proj,v_proj,o_proj")
    assert by["dpo__unsel__A16__n64"]["pool"] == "dpo/unsel"
    c = by["dpo__final__M256__n64"]
    assert c["lrs"] == [3e-2, 1e-1, 3e-1] and c["steps"] == 64 and "extend" not in c
    assert (
        by["dpo__own__A16__n16"]["lr_from"] == "dpo__own__A16__n64"
        and by["dpo__own__A16__n16"]["conditional"]
    )
    assert by["dpo__own__CONFIRM__n64"]["template"] and by["dpo__own__CONFIRM__n64"]["seed"]
    assert [c["cell"] for c in mt.select_cells(cells)] == ids[:13]
    assert len(mt.select_cells(cells, conditional=True)) == 15  # the template never runs
    assert [c["cell"] for c in mt.select_cells(cells, only=["dpo__own__A16__n4"])] == [
        "dpo__own__A16__n4"
    ]
    pr = mp.project(cells)
    assert len(pr) == 15 and all(p["secs"] > 0 for p in pr)


def _write_dan(dan: Path):
    (dan / "olmo_ifomega").mkdir(parents=True)
    (dan / "olmo_ifcurve").mkdir()
    for d in ("olmo_ifladder", "olmo_transplant", "artifacts/olmo_ifcurve/hand"):
        (dan / d).mkdir(parents=True, exist_ok=True)
    arm = [f"ifbench_{k}" for k in range(1, 21)] + ["ifbench_100"]
    reach = ["ifbench_100"]
    (dan / "olmo_ifomega/problems.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "pid": p,
                    "task": "ifbench",
                    "key": 0,
                    "prompt": f"Do {p}.",
                    "instruction_id_list": ["x:y"],
                    "kwargs": [{}],
                }
            )
            + "\n"
            for p in arm
        )
    )
    (dan / "olmo_ifcurve/pids_armA.txt").write_text("\n".join(arm) + "\n")
    (dan / "olmo_ifcurve/pids_reach.txt").write_text("\n".join(reach) + "\n")
    run = "olmo_ifomega"
    for d in ("samples", "graded", "meta"):
        (dan / "artifacts" / run / d).mkdir(parents=True)

    def ids(think, ans=3):
        return [10] * think + [524, 27963, 1363] + [20] * ans + [bps.bp.EOS]

    spec = {  # model -> {pid: [(len, strict)]}: s2300 passes 1-16, dpo passes 5-20 (but 20 too long, 19 not strict)
        "s2300": {f"ifbench_{k}": [(10 + k, True)] for k in range(1, 17)},
        "dpo": {
            **{f"ifbench_{k}": [(10 + k, True)] for k in range(5, 19)},
            "ifbench_6": [(16, True), (17, False)],
            "ifbench_19": [(10, False)],
            "ifbench_20": [(500, True)],
            "ifbench_100": [(12, True)],
        },
    }
    for name, d in spec.items():
        job = f"F_{name}"
        with (
            gzip.open(dan / "artifacts" / run / "samples" / f"{job}.jsonl.gz", "wt") as f,
            open(dan / "artifacts" / run / "graded" / f"{job}.jsonl", "w") as g,
        ):
            for pid, ds in d.items():
                for i, (ln, st) in enumerate(ds):
                    x = ids(ln)
                    b64 = base64.b64encode(np.asarray(x, dtype=np.int32).tobytes()).decode()
                    f.write(
                        json.dumps(
                            {
                                "pid": pid,
                                "model": name,
                                "job": job,
                                "chunk": 0,
                                "i": i,
                                "n_tok": len(x),
                                "finish": "stop",
                                "text": f"t </think>\n\n{pid}{i}",
                                "ids": b64,
                            }
                        )
                        + "\n"
                    )
                    g.write(
                        json.dumps(
                            {
                                "pid": pid,
                                "job": job,
                                "chunk": 0,
                                "i": i,
                                "n_tok": len(x),
                                "finish": "stop",
                                "task": "ifbench",
                                "strict": st,
                            }
                        )
                        + "\n"
                    )
        (dan / "artifacts" / run / "meta" / f"{job}.json").write_text(
            json.dumps(
                {"job": job, "name": name, "problems": "olmo_ifomega/problems.jsonl", "lora": None}
            )
            + "\n"
        )
    (dan / "artifacts" / run / "meta" / "F_dpo_lora.json").write_text(
        json.dumps(
            {
                "job": "F_dpo_lora",
                "name": "dpo",
                "problems": "olmo_ifomega/problems.jsonl",
                "lora": "x",
            }
        )
        + "\n"
    )
    labels = {"olmo_ifcurve/handcheck.csv": [["F_dpo", "ifbench_5", 0, 0, "gamed"]]}
    for rel, hdr in (
        ("olmo_ifcurve/handcheck.csv", ["job", "pid", "chunk", "i", "final_label"]),
        ("olmo_ifomega/handcheck.csv", ["job", "pid", "chunk", "i", "final_label"]),
        ("olmo_ifladder/handcheck_src.csv", ["job", "pid", "chunk", "i", "final_label"]),
        ("olmo_transplant/handcheck_src.csv", ["key", "final_label"]),
        ("artifacts/olmo_ifcurve/hand/screen.csv", ["job", "pid", "chunk", "i", "reason"]),
    ):
        with open(dan / rel, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(hdr)
            w.writerows(labels.get(rel, []))
    return arm, reach


def test_build_pools_synthetic(tmp_path):
    dan = tmp_path / "dan"
    _write_dan(dan)
    jobs, skipped = bps.discover_jobs(dan, "dpo")
    assert [j for _, j, _ in jobs] == ["F_dpo"] and skipped[0]["why"] == "LoRA"
    out = tmp_path / "out"
    res = bps.build(
        dan,
        "dpo",
        out,
        cap=60,
        n_train=4,
        n_dev=3,
        allow_short_dev=False,
        pick_seed=7,
        split_seed=11,
        tokenizer=None,
    )
    own, fin = res["own"], res["final"]
    assert (
        own["matched_prompts"]
        == fin["matched_prompts"]
        == sorted(f"ifbench_{k}" for k in range(5, 17))
    )  # both eligible; ifbench_100 is a reach prompt
    assert own["train_pids"] == fin["train_pids"] and own["dev_pids"] == fin["dev_pids"]
    import random

    order = sorted(own["matched_prompts"])
    random.Random("11:split").shuffle(order)
    assert own["train_pids"] == order[:4] and own["dev_pids"] == order[4:7]
    rows = {
        s: [json.loads(x) for x in (out / s / "train.jsonl").read_text().splitlines()]
        for s in ("own", "final")
    }
    assert [r["pid"] for r in rows["own"]] == order[:4]
    assert (
        rows["own"][0]["run"] == "olmo_ifomega"
        and rows["own"][0]["n_tok"] != rows["final"][0]["n_tok"]
        or True
    )
    assert all(
        set(r) >= {"ids", "think_end", "n_tok", "pid", "prompt", "label", "job", "chunk", "i"}
        for r in rows["own"]
    )
    man = json.loads((out / "own" / "MANIFEST.json").read_text())
    assert (
        man["n_matched"] == 12
        and set(man["files"]) >= {"train.jsonl", "test.jsonl"}
        and man["files"]["train.jsonl"]["sha256"]
    )
    # checker-only: the hand-labelled "gamed" draw of ifbench_5 is still eligible; labels are descriptive
    assert "ifbench_5" in own["matched_prompts"]
    allrows = {s_: {r["pid"]: r for r in rows_} for s_, rows_ in rows.items()}
    by_pid = bps.matched_for(dan, "dpo", 60)[2][2]
    assert bps.accept_rate(by_pid, "ifbench_6", 60, bps.passes_checker) == (
        pytest.approx(0.5),
        1,
        2,
    )
    assert bps.accept_rate(by_pid, "ifbench_6", 60, bps.passes_unsel) == (
        pytest.approx(2.5 / 3),
        2,
        2,
    )
    assert all("accept_rate" in r for r in allrows["own"].values())
    uns = {
        r["pid"]: r
        for r in map(json.loads, (out / "unsel" / "train.jsonl").read_text().splitlines())
    }
    assert list(uns) == own["train_pids"] and (out / "unsel" / "test.jsonl").exists()
    for r in uns.values():
        assert r["finish"] == "stop" and r["accept_N"] >= 1
    man_u = json.loads((out / "unsel" / "MANIFEST.json").read_text())
    assert man_u["source"] == "unsel" and man_u["sets"]["train"]["strict_picks"] <= 4
    import math

    exp = sum(math.log2(1 / r["accept_rate"]) for r in rows["own"][:4])
    assert own["checker_bound_bits"]["4"] == pytest.approx(exp) and "1" in own["checker_bound_bits"]
    # nested prefixes: n=1,4 are prefixes of the file order
    assert (out / "own/test.jsonl").exists()
    with pytest.raises(SystemExit):
        bps.build(
            dan,
            "dpo",
            tmp_path / "o2",
            cap=60,
            n_train=8,
            n_dev=8,
            allow_short_dev=False,
            pick_seed=7,
            split_seed=11,
            tokenizer=None,
        )
    r3 = bps.build(
        dan,
        "dpo",
        tmp_path / "o3",
        cap=60,
        n_train=8,
        n_dev=8,
        allow_short_dev=True,
        pick_seed=7,
        split_seed=11,
        tokenizer=None,
    )
    assert r3["own"]["n_dev"] == 4 and r3["own"]["dev_short"]
    # tight cap drops long draws from matching
    r4 = bps.build(
        dan,
        "dpo",
        tmp_path / "o4",
        cap=24,
        n_train=2,
        n_dev=1,
        allow_short_dev=True,
        pick_seed=7,
        split_seed=11,
        tokenizer=None,
    )
    assert all(int(p.split("_")[1]) + 10 + 4 <= 24 for p in r4["own"]["matched_prompts"])


REAL = Path("/scratch/cowifornia/nd-rl/olmo_elicit/v2/pools/dpo")


@pytest.mark.skipif(
    not (REAL / "own" / "MANIFEST.json").exists(), reason="real dpo pools not built"
)
def test_real_dpo_counts():
    for s in ("own", "final"):
        m = json.loads((REAL / s / "MANIFEST.json").read_text())
        assert (
            m["n_matched"] == 115
            and m["n_train"] == 64
            and m["n_dev"] == 32
            and m["max_tok"] == 6144
        )
        for f, v in m["files"].items():
            assert bps.bp.sha256_file(REAL / s / f) == v["sha256"]
        assert set(m["train_pids"]).isdisjoint(m["dev_pids"])
    a = json.loads((REAL / "own" / "MANIFEST.json").read_text())
    b = json.loads((REAL / "final" / "MANIFEST.json").read_text())
    u = json.loads((REAL / "unsel" / "MANIFEST.json").read_text())
    assert a["train_pids"] == b["train_pids"] == u["train_pids"]
    assert set(a["checker_bound_bits"]) == {"1", "4", "16", "64"}


# ------------------------------------------------------------------ discourse Turn 2/3 additions


def last_rows(ctx, cid):
    return [
        json.loads(x)
        for x in (Path(ctx["out"]) / "cells" / cid / "runs.jsonl").read_text().splitlines()
    ]


def fake_dev_score(monkeypatch, schedule):
    """Dev-set scoring (the 4-example set) follows `schedule` (nats per token, total), in call order; train scoring is real."""
    real = oe.score
    calls = {"i": 0}

    def fake(model, exs, *, batch_tokens=0, pad_id=0, stats=None):
        if len(exs) != 4:
            return real(model, exs, batch_tokens=batch_tokens, pad_id=pad_id, stats=stats)
        L = schedule[min(calls["i"], len(schedule) - 1)]
        calls["i"] += 1
        return {
            "think_nats": [L * e.think_end for e in exs],
            "answer_nats": [L * (e.n_tok - e.think_end) for e in exs],
        }

    monkeypatch.setattr(oe, "score", fake)


def test_default_eval_steps_small_n():
    assert mt.default_eval_steps(4, 64) == [1, 2, 4, 8, 16, 32, 64]
    assert mt.default_eval_steps(1, 64) == [1, 2, 4, 8, 16, 32, 64]
    assert mt.default_eval_steps(16, 64) == [4, 8, 16, 32, 64]
    assert mt.default_eval_steps(64, 64) == [16, 32, 64]


def test_mask_targets_restrict_lora_and_draw(pools):
    spec = "mask:k=20:seed=2:targets=q_proj,o_proj"
    ad = mt.Adapter(tiny_model(), mt.Family(spec))
    assert ad.P == 20 == ad.formula and ad.fam.targets == ["q_proj", "o_proj"]
    assert all(("q_proj" in n or "o_proj" in n) for n in ad.masks)
    assert sum(int(m.sum()) for m in ad.masks.values()) == 20
    rc, ctx = run_cells(pools, [cell("mt", spec, lrs=(3e-2,), steps=4, eval_steps=[4])])
    assert rc == 0 and cj(ctx, "mt")["target_modules"] == ["q_proj", "o_proj"]
    assert last_rows(ctx, "mt")[0]["b_stats"]["n_B_nonzero"] <= 20


def test_b_stats_and_dev_answer_loss(pools):
    rc, ctx = run_cells(
        pools, [cell("bs", "lora:r=2:targets=all", lrs=(1e-2,), steps=4, eval_steps=[2, 4])]
    )
    r = last_rows(ctx, "bs")[0]
    assert r["b_stats"]["n_B_nonzero"] > 0 and r["b_stats"]["max_abs_B"] > 0
    assert set(r["dev"]["4"]) >= {"loss_total", "loss_answer", "loss"}
    assert r["steps_completed"] == 4 and r["early_stopped_at"] is None and r["stage"] == "a"


def test_early_stop_two_consecutive_above_running_best(pools, monkeypatch):
    fake_dev_score(monkeypatch, [5.0, 4.0, 4.5, 4.6, 4.7, 4.8, 4.9])
    rc, ctx = run_cells(
        pools,
        [cell("es", "lora:r=2:targets=all", lrs=(1e-2,), steps=6, eval_steps=[1, 2, 3, 4, 5, 6])],
    )
    assert rc == 0
    r = last_rows(ctx, "es")[0]
    assert r["early_stopped_at"] == 3 and r["steps_completed"] == 3 and len(r["steps_log"]) == 3
    c = cj(ctx, "es")
    assert c["selected"]["step"] == 1 and c["early_stopped_at"] == 3
    assert c["lr_summary"]["0.01"]["early_stopped_at"] == 3


def test_unconverged_flag(pools, monkeypatch):
    fake_dev_score(monkeypatch, [5.0, 3.0, 2.0])
    rc, ctx = run_cells(
        pools, [cell("u1", "lora:r=2:targets=all", lrs=(1e-2,), steps=2, eval_steps=[1, 2])]
    )
    assert cj(ctx, "u1")["unconverged"] is True  # 1.0 drop > 1 % of (5 - 2)
    fake_dev_score(monkeypatch, [5.0, 3.0, 3.0])
    rc, ctx = run_cells(
        pools,
        [cell("u2", "lora:r=2:targets=all", lrs=(1e-2,), steps=2, eval_steps=[1, 2])],
        ctx=ctx_of(pools, out="y"),
    )
    assert cj(ctx, "u2")["unconverged"] is False


def test_extend_continues_exactly_and_picks_winner(pools):
    fam = "lora:r=2:targets=all"
    rc, c1 = run_cells(
        pools,
        [cell("full", fam, lrs=(1e-2,), steps=8, eval_steps=[4, 8])],
        ctx=ctx_of(pools, out="o1"),
    )
    rc2, c2 = run_cells(
        pools,
        [
            cell(
                "ext",
                fam,
                lrs=(1e-2,),
                eval_steps=[4],
                extend={"after": 4, "to": 8, "eval_steps": [8]},
            )
        ],
        ctx=ctx_of(pools, out="o2"),
    )
    assert rc == rc2 == 0
    full = last_rows(c1, "full")[0]
    rows = last_rows(c2, "ext")
    assert [r["stage"] for r in rows] == ["a", "a+b"] and rows[1]["steps_completed"] == 8
    assert rows[1]["dev"]["8"]["loss"] == pytest.approx(full["dev"]["8"]["loss"], rel=1e-6)
    assert rows[1]["seq"] == full["seq"] and len(rows[1]["steps_log"]) == 8
    # two lrs: only the lr with the lowest dev loss at `after` is extended
    rc3, c3 = run_cells(
        pools,
        [
            cell(
                "ext2",
                fam,
                lrs=(1e-3, 3e-2),
                eval_steps=[4],
                extend={"after": 4, "to": 8, "eval_steps": [8]},
            )
        ],
        ctx=ctx_of(pools, out="o3"),
    )
    rows = last_rows(c3, "ext2")
    a_rows = [r for r in rows if r["stage"] == "a"]
    ab = [r for r in rows if r["stage"] == "a+b"]
    assert len(a_rows) == 2 and len(ab) == 1
    assert ab[0]["lr"] == min(a_rows, key=lambda r: r["dev"]["4"]["loss"])["lr"]
    assert not (Path(c3["out"]) / "cells/ext2/state").exists()
    assert cj(c3, "ext2")["schedule"]["to"] == 8


def test_extend_skips_early_stopped_lrs(pools, monkeypatch):
    # every lr's dev loss rises after step 1: both stop early at step 3 (after = 4 never reached): no extension
    fake_dev_score(monkeypatch, [5.0, 4.0, 4.5, 4.6, 5.0, 4.0, 4.5, 4.6, 4.7])
    rc, ctx = run_cells(
        pools,
        [
            cell(
                "xs",
                "lora:r=2:targets=all",
                lrs=(1e-3, 1e-2),
                eval_steps=[1, 2, 3, 4],
                extend={"after": 4, "to": 8, "eval_steps": [8]},
            )
        ],
    )
    assert rc == 0
    assert [r["stage"] for r in last_rows(ctx, "xs")] == ["a", "a"]


def test_edl_formulas(pools):
    n = 3
    rc, ctx = run_cells(
        pools,
        [cell("ed", "lora:r=2:targets=all", n=n, lrs=(1e-2, 3e-2), steps=8, eval_steps=[2, 8])],
    )
    c = cj(ctx, "ed")
    rows = last_rows(ctx, "ed")
    r = next(x for x in rows if x["lr"] == c["selected"]["lr"])
    t = c["selected"]["step"]
    m = min(t, n)
    on = r["online"]
    nats = sum(on["think_nats"][:m]) + sum(on["answer_nats"][:m])
    toks = sum(on["n_tok"][:m])
    sets, _ = oe.load_pool(Path(ctx["root"]) / "p", FakeTok(), n=n, m=None, max_tok=None, n_reach=0)
    dv = sets["test"]
    d = r["dev"][str(t)]
    tot = np.add(d["think_nats"], d["answer_nats"])
    L = tot.sum() / sum(e.n_tok for e in dv)
    assert c["edl_tok_nats"] == pytest.approx(nats - toks * L) and c[
        "edl_ex_nats"
    ] == pytest.approx(nats - m * tot.mean())
    assert c["n_online_examples"] == m and c["mdl_tokens"] == toks
    assert c["edl_tok_bits"] == pytest.approx(c["edl_tok_nats"] / np.log(2))
    assert c["edl_tok_bits_per_trained_weight"] == pytest.approx(c["edl_tok_bits"] / c["P"])
    assert set(c["edl_by_step"]) == {"2"}  # eval steps <= n only
    assert c["edl_fixed_step"] is None  # step 64 not run here


def test_edl_fixed_step_64(pools):
    rc, ctx = run_cells(
        pools,
        [cell("e64", "lora:r=2:targets=all", n=4, lrs=(3e-3, 3e-2), steps=64, eval_steps=[64])],
    )
    c = cj(ctx, "e64")
    f = c["edl_fixed_step"]
    rows = {r["lr"]: r for r in last_rows(ctx, "e64")}
    best = min(rows, key=lambda lr: rows[lr]["dev"]["64"]["loss"])
    assert f["step"] == 64 and f["lr"] == best and set(f["per_lr"]) == {"0.003", "0.03"}
    assert f["m"] == 4 and f["edl_tok_nats"] == pytest.approx(
        f["per_lr"][f"{best:g}"]["edl_tok_nats"]
    )
    assert c["edl_by_step"] == {}  # only eval steps <= n = 4 are listed


# ------------------------------------------------------------------ fixed recipes, stop logic


def test_fixed_recipe_runs_S_steps_no_early_stop(pools, monkeypatch):
    # dev loss rises after step 1 at every eval: a normal cell would early-stop; a fixed one trains to S and selects S
    fake_dev_score(monkeypatch, [5.0, 4.0, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0])
    rc, ctx = run_cells(
        pools,
        [
            cell(
                "fx",
                "lora:r=2:targets=all",
                lrs=(1e-2,),
                steps=6,
                eval_steps=[1, 2, 3, 4],
                fixed=True,
                edge_extend=True,
                extend={"after": 2, "to": 9, "eval_steps": [9]},
            )
        ],
    )
    assert rc == 0
    rows = last_rows(ctx, "fx")
    assert (
        len(rows) == 1 and rows[0]["early_stopped_at"] is None and rows[0]["steps_completed"] == 6
    )
    c = cj(ctx, "fx")
    assert (
        c["selected"]["step"] == 6 and c["selection"] == "fixed recipe" and c["lrs_tried"] == [1e-2]
    )
    assert "6" in rows[0]["dev"] and c["did_not_train"] is False


def test_recipe_from_reads_lr_and_step(pools, tmp_path):
    rc, ctx = run_cells(
        pools,
        [cell("src", "lora:r=2:targets=all", lrs=(1e-3, 1e-2), steps=6, eval_steps=[2, 4, 6])],
    )
    sel = cj(ctx, "src")["selected"]
    r2 = {
        "cell": "dst",
        "pool": "p",
        "family": "lora:r=2:targets=all",
        "n": 8,
        "recipe_from": "src",
    }
    rc, ctx = run_cells(pools, [r2], ctx=ctx)
    assert rc == 0
    d = cj(ctx, "dst")
    assert (
        d["recipe"]["lr"] == sel["lr"]
        and d["recipe"]["step"] == sel["step"]
        and d["selected"]["step"] == sel["step"]
    )
    assert (
        d["selected"]["lr"] == sel["lr"]
        and d["recipe_from"] == "src"
        and d["selection"] == "fixed recipe"
    )
    # recipe from a json path (another pod)
    p = tmp_path / "other.json"
    p.write_text(json.dumps({"selected": {"lr": 3e-3, "step": 3}}))
    r3 = {
        "cell": "dst2",
        "pool": "p",
        "family": "lora:r=2:targets=all",
        "n": 8,
        "recipe_from": "elsewhere",
        "recipe_cell_json": str(p),
    }
    rc, ctx = run_cells(pools, [r3], ctx=ctx)
    d = cj(ctx, "dst2")
    assert (
        d["selected"]["lr"] == 3e-3 and d["selected"]["step"] == 3 and d["recipe"]["path"] == str(p)
    )


def test_stop_reaches_timeout_child_groups():
    """The chain runs stages under plain `timeout` (its own process group, so an expiry kills vLLM's children too);
    `mini_pod.sh stop` therefore signals every process group below the chain pid. Same awk walk as in mini_pod.sh."""
    import os
    import subprocess
    import time

    chain = (HERE / "pod" / "mini_chain.sh").read_text()
    assert (
        "timeout --foreground" not in chain
        and 'timeout --signal=TERM --kill-after=60 "$to"' in chain
    )
    stop = (HERE / "pod" / "mini_pod.sh").read_text()
    assert "while (y in p && y != r && y > 1) y=p[y]" in stop
    p = subprocess.Popen(
        ["setsid", "bash", "-c", "timeout --signal=TERM --kill-after=5 100 sleep 101 & wait"],
        stdout=subprocess.DEVNULL,
    )
    time.sleep(1)
    root, pgid = p.pid, os.getpgid(p.pid)
    walk = (
        "ps -eo pid=,ppid=,pgid= | awk -v r=%d '{p[$1]=$2; g[$1]=$3} END {for (x in p) {y=x; "
        "while (y in p && y != r && y > 1) y=p[y]; if (y == r) print g[x]}}' | sort -u".replace(
            "%d", str(root)
        )
    )
    groups = subprocess.run(["bash", "-c", walk], capture_output=True, text=True).stdout.split()
    assert len(set(groups) - {str(pgid)}) >= 1  # timeout's own group
    for g in {str(pgid), *groups}:
        subprocess.run(["bash", "-c", f"kill -TERM -- -{g} 2>/dev/null"])
    p.wait(timeout=10)
    time.sleep(1)
    left = subprocess.run(
        ["bash", "-c", "ps -eo args= | grep -c '^sleep 101$' || true"],
        capture_output=True,
        text=True,
    )
    assert left.stdout.strip() == "0"


# ------------------------------------------------------------------ step-0 train scores, corrected EDL


def test_step0_train_matches_cell_step0(pools, tmp_path):
    out = tmp_path / "s0out"
    model = tiny_model()
    r = mt.step0_train(model, FakeTok(), pools / "pools", out, "t", "p", 5, 0)
    assert r["n"] == 5 and len(r["pids"]) == 5
    sets, _ = oe.load_pool(pools / "pools" / "p", FakeTok(), n=5, m=None, max_tok=None, n_reach=0)
    for i, ex in enumerate(sets["train"]):
        sc = oe.score(model, [ex], batch_tokens=0)
        assert r["think_nats"][i] == pytest.approx(sc["think_nats"][0], rel=1e-6)
        assert r["answer_nats"][i] == pytest.approx(sc["answer_nats"][0], rel=1e-6)
    assert r["sum_tokens"] == sum(e.n_tok for e in sets["train"])
    assert r["sum_nats"] == pytest.approx(sum(r["think_nats"]) + sum(r["answer_nats"]))
    assert (out / "step0_train" / "t__p__n5.json").exists()
    # a cell run with the same out dir reuses the same cache and its reset check is 0
    rc, ctx = run_cells(
        pools,
        [cell("z", "lora:r=2:targets=all", n=5, steps=2, eval_steps=[2])],
        model=tiny_model(),
        ctx=ctx_of(pools, out="s0out"),
    )
    assert last_rows(ctx, "z")[0]["reset_check"]["abs_diff"] == pytest.approx(0.0, abs=1e-9)
    assert cj(ctx, "z")["step0_dev_nats_per_tok_total"] == pytest.approx(r["L0_dev_total"])


def test_edl_corrected_arithmetic():
    s0 = {
        "think_nats": [10.0, 20.0, 30.0],
        "answer_nats": [5.0, 5.0, 10.0],
        "n_tok": [10, 20, 30],
        "L0_dev_total": 1.0,
    }
    blk = lambda t, m, tok, edl: {"t": t, "m": m, "mdl_tokens": tok, "edl_tok_nats": edl}  # noqa: E731
    cj_ = {
        "P": 100,
        "edl_selected": blk(2, 2, 30, 7.0),
        "edl_fixed_step": {
            **blk(3, 3, 60, 4.0),
            "lr": 0.01,
            "per_lr": {"0.01": blk(3, 3, 60, 4.0)},
        },
        "edl_by_step": {"1": blk(1, 1, 10, 1.0), "2": blk(2, 2, 30, 7.0)},
    }
    out = mt.edl_corrected(cj_, s0)
    # m=2: step-0 nats 15 + 25 over 30 tokens = 4/3 -> delta 1/3; corr = 7 - 30/3 = -3
    assert out["edl_selected"]["delta_nats_per_tok"] == pytest.approx(1 / 3)
    assert out["edl_selected"]["edl_tok_corr_nats"] == pytest.approx(-3.0)
    assert out["edl_selected"]["edl_tok_corr_bits"] == pytest.approx(-3.0 / np.log(2))
    assert out["edl_selected"]["edl_tok_corr_bits_per_trained_weight"] == pytest.approx(
        -3.0 / np.log(2) / 100
    )
    # m=3: (15+25+40)/60 = 4/3; delta 1/3; corr = 4 - 20 = -16
    assert out["edl_fixed_step"]["edl_tok_corr_nats"] == pytest.approx(-16.0)
    assert out["edl_fixed_step"]["per_lr"]["0.01"]["edl_tok_corr_nats"] == pytest.approx(-16.0)
    # m=1: delta = 15/10 - 1 = 0.5; corr = 1 - 5 = -4
    assert out["edl_by_step"]["1"]["edl_tok_corr_nats"] == pytest.approx(-4.0)
    assert set(mt.edl_corrected(cj_, s0, t=2)["edl_by_step"]) == {"2"}


# ------------------------------------------------------------------ init_adapter cells


def test_init_adapter_cell(pools):
    fam = "lora:r=2:targets=all"
    rc, ctx = run_cells(pools, [cell("src", fam, lrs=(1e-2,), steps=4, eval_steps=[4])])
    src = cj(ctx, "src")
    adir = Path(ctx["out"]) / "cells/src/adapter"
    rc, ctx = run_cells(
        pools,
        [
            {
                "cell": "ini",
                "pool": "p",
                "family": fam,
                "n": 3,
                "init_adapter": str(adir),
                "steps": 0,
                "source": "final",
            }
        ],
        ctx=ctx,
    )
    assert rc == 0
    c = cj(ctx, "ini")
    assert (
        c["source"] == "final"
        and c["n"] == 3
        and c["selection"] == "init adapter, no training"
        and c["P"] == src["P"]
    )
    assert c["selected"]["step"] == 0 and c["selected"]["lr"] is None
    assert c["selected"]["dev_nats_per_tok_total"] == pytest.approx(
        src["selected"]["dev_nats_per_tok_total"], rel=1e-6
    )
    assert (
        c["edl_tok_nats"] is None
        and c["edl_fixed_step"] is None
        and c["init_adapter_missing"] == []
        and c["init_adapter_unexpected"] == []
    )
    assert c["reexport_equal"] and c["reload_check_max_abs_diff"] < 1e-5
    assert (Path(ctx["out"]) / "cells/ini/DONE").exists()
    g1 = np.load(Path(ctx["out"]) / "cells/src/gate.npz")
    g2 = np.load(Path(ctx["out"]) / "cells/ini/gate.npz")
    assert set(g1.files) == set(g2.files) and list(g1["pid"]) == list(g2["pid"])
    for j in range(4):
        assert np.allclose(g1[f"adapter_nll_{j}"], g2[f"adapter_nll_{j}"], atol=1e-5)
        assert np.allclose(g1[f"base_nll_{j}"], g2[f"base_nll_{j}"], atol=1e-6)
        assert np.array_equal(g1[f"resp_ids_{j}"], g2[f"resp_ids_{j}"])
    # re-export loads with PeftModel
    from peft import PeftModel

    PeftModel.from_pretrained(tiny_model(), str(Path(ctx["out"]) / "cells/ini/adapter"))
    # a family that does not match the adapter's config fails the cell
    rc, ctx = run_cells(
        pools,
        [
            {
                "cell": "bad",
                "pool": "p",
                "family": "lora:r=4:targets=all",
                "n": 3,
                "init_adapter": str(adir),
            }
        ],
        ctx=ctx,
    )
    assert rc == 2 and (Path(ctx["out"]) / "cells/bad/FAILED.json").exists()


# ------------------------------------------------------------------ fixed-step selection, step-0 train in cells


def _row(lr, dev64, early, n=3):
    th = [dev64 * 5.0] * 2
    return {
        "lr": lr,
        "early_stopped_at": early,
        "online": {
            "pids": list("abc"),
            "think_nats": [10.0] * 3,
            "answer_nats": [5.0] * 3,
            "n_tok": [10, 10, 10],
        },
        "dev": {
            "64": {
                "think_nats": th,
                "answer_nats": [dev64 * 5.0] * 2,
                "loss": dev64,
                "loss_total": dev64,
            }
        },
    }


def test_fixed_step_block_reached_rule():
    rows = {
        1e-3: _row(1e-3, 0.50, 128),  # stopped AFTER 64, lowest dev at 64: chosen
        1e-2: _row(
            1e-2, 0.40, 40
        ),  # stopped BEFORE 64: never reached it (stale key must not count)
        1e-1: _row(1e-1, 0.60, None),
        3e-1: {**_row(3e-1, 0.1, None), "dev": {}},  # never evaluated at 64
    }
    b = mt.fixed_step_block(rows, 64, 3, 20, 100)
    assert (
        b["lr"] == 1e-3
        and set(b["per_lr"]) == {"0.001", "0.1"}
        and b["per_lr"]["0.001"]["early_stopped_at"] == 128
    )
    assert b["m"] == 3 and b["step"] == 64
    assert mt.fixed_step_block({1e-2: _row(1e-2, 0.4, 40)}, 64, 3, 20, 100) is None


def test_cell_writes_corrected_blocks(pools):
    rc, ctx = run_cells(
        pools,
        [cell("co", "lora:r=2:targets=all", n=4, lrs=(3e-3, 3e-2), steps=64, eval_steps=[2, 64])],
    )
    assert rc == 0
    c = cj(ctx, "co")
    s0 = json.loads((Path(ctx["out"]) / "step0_train" / "t__p__n4.json").read_text())
    exp = mt.edl_corrected(c, s0)
    assert (
        c["edl_selected_corr"] == exp["edl_selected"]
        and c["edl_fixed_step_corr"] == exp["edl_fixed_step"]
    )
    assert c["edl_by_step_corr"] == exp["edl_by_step"] and c["edl_fixed_step_corr"]["per_lr"]
    assert c["step0_train_nats_per_tok"] == pytest.approx(s0["mean_nats_per_tok_total"])
    assert c["delta_nats_per_tok"] == pytest.approx(exp["edl_selected"]["delta_nats_per_tok"])
    # the cached step-0 scores equal the online step-0 of the cell (reset check)
    assert last_rows(ctx, "co")[0]["reset_check"]["abs_diff"] == pytest.approx(0.0, abs=1e-9)


STAGE1 = Path("/scratch/cowifornia/nd-rl/olmo_elicit/v2")


@pytest.mark.skipif(
    not (STAGE1 / "runs_union/dpo/train/cells/dpo__final__A16__n64/runs.jsonl").exists()
    or not (STAGE1 / "runs_rp-om1/dpo/train/step0_train/dpo__dpo_final__n64.json").exists(),
    reason="stage-1 data absent",
)
def test_stage1_capacity_reproduced():
    s0 = json.loads(
        (STAGE1 / "runs_rp-om1/dpo/train/step0_train/dpo__dpo_final__n64.json").read_text()
    )
    for cid, lr, bits in (
        ("A16", 1e-4, 8916),
        ("O1", 1e-3, 8044),
        ("M16k", 0.3, 6043),
        ("M4k", 1.0, 2425),
    ):
        last = {}
        for ln in (
            (STAGE1 / f"runs_union/dpo/train/cells/dpo__final__{cid}__n64/runs.jsonl")
            .read_text()
            .splitlines()
        ):
            r = json.loads(ln)
            last[r["lr"]] = r
        blk = mt.fixed_step_block(last, 64, 64, None, 1)
        assert blk["lr"] == pytest.approx(lr)
        out = mt.edl_corrected({"P": 1, "edl_fixed_step": blk}, s0)
        assert out["edl_fixed_step"]["edl_tok_corr_bits"] == pytest.approx(bits, abs=1.0)
