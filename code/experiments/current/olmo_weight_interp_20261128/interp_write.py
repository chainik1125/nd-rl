#!/usr/bin/env python3
"""Write and check one interpolated checkpoint: theta(alpha) = dpo + alpha x (ckpt - dpo).

    python3 interp_write.py write --tag a050        # pod: inputs in MODELS_ROOT, output in
                                                    # WI_MERGED_ROOT (default $M2_ROOT/models)
    python3 interp_write.py write --dpo D --ckpt S --alpha 0.5 --out O --report R
    python3 interp_write.py check ...               # the checks again on an existing folder
    python3 interp_write.py preflight               # header census and config check only

Every tensor is interpolated. The write itself is metrics 2's verified writer
(`rank1/expo_write.py`, imported unedited; alpha plays its gamma, the RL checkpoint its `s0500`):
fp32 combine, one rounding to bf16, dpo's shard layout and non-weight files. This file adds what
PREREG.md asks for around it.

Before the write
  - structure census from the safetensors headers: same names, shapes and dtypes. A non-float
    tensor that differs fails, because the writer would copy dpo's.
  - the two config.json files may differ only in fields that do not change the computation; the
    token ids of generation_config.json must agree.
  - a disk budget from file sizes (the pod's `df` shows the whole storage cluster).

After the write
  - every tensor is re-read from disk and compared bit for bit with a fresh fp32 recomputation;
  - every tensor is held to two invariants that do not depend on the writer's own arithmetic: an
    element that does not differ between the inputs stays as it is, and a written element lies
    between its two inputs;
  - every tensor is added to the rounding diagnostics: realised fraction along the line, length
    ratio, share off the line, where the differing weights landed;
  - metrics 2's exact-rational verifier (`expo_write.verify_tensors`: the exact fp64 result
    rounded through fp32 to bf16) runs on the embedding, the output head, every norm weight and
    all seven matrices of four seeded layers. There every element must be within one bf16 step
    and at most NE_TOL elements per tensor (or NE_SHARE of it) may differ from the reference at
    all (audit finding 3: "within one step" alone cannot see a one-step tensor written at the
    wrong end).
    `--exact all` runs it on everything: about 45 ns per element on 4 threads, so about 24
    minutes per checkpoint, which is why it is a sample by default.

Once (it does not depend on alpha; `--no-census` skips it)
  - how many bf16 steps the two inputs are apart;
  - at how many elements the writer's arithmetic at alpha 1 misses s2300 (recorded and flagged
    above 1 in a million, not fatal). The alpha 1.0 cell serves the stored weights, so this is
    the only look at the writer route at that end.

Files: <report>/check.json (always), <report>/MANIFEST.json (copy of the writer's manifest), and
<out>/WI_WRITE_OK (only when every check passed; pod/wi_cell.sh refuses to sample without it).
Exit: 0 ok, 2 usage, 4 a written tensor failed a check, 5 disk budget, 6 census or config mismatch.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import time
from fractions import Fraction
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
for _cand in (HERE / "rank1", HERE.parent / "olmo_metrics2_20261127" / "rank1"):
    if (_cand / "expo_write.py").exists():
        sys.path.insert(0, str(_cand))
        break
import expo_write as ew  # noqa: E402
from common import Checkpoint, sha256_file  # noqa: E402

TAGS = {"a025": 0.25, "a050": 0.5, "a075": 0.75}
MARKER = "WI_WRITE_OK"
CHUNK = 1 << 25  # elements per verification chunk (fp64 temporaries of 268 MB each)
MARGIN_GB = 20.0
#: config.json fields that may differ between dpo and the RL checkpoint without changing what the
#: engine computes (labels and library stamps). Anything else that differs fails the write.
#: Known at design time (metrics 2's audit copy of dpo's config against s2300's): only `use_cache`
#: differs (true / false), a transformers generation flag that vLLM does not read.
CONFIG_COSMETIC = frozenset({"_name_or_path", "transformers_version", "use_cache"})
FLOAT_DTYPES = frozenset({"F16", "BF16", "F32", "F64"})  # safetensors header names
#: bf16 steps between dpo and the RL checkpoint, binned (inclusive edges; last bin "9 or more")
STEP_EDGES = ((0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 9))
STEP_LABELS = ("0", "1", "2", "3-4", "5-8", ">8")
STEP_CLAMP = 9
EXACT_SEED, EXACT_LAYERS = 20261128, 4
#: Elements of one tensor that may differ (by one step) from the verifier's reference: NE_TOL, or
#: NE_SHARE of the tensor if that is more. The writer works in fp32, the reference in fp64; they
#: can part where one input is more than about 2^16 times the other (a weight that was almost
#: zero). Yesterday's real writes had 0 such elements in 36.7 M checked. A wrong alpha or a wrong
#: end moves a large share of a tensor, far more than this.
NE_TOL, NE_SHARE = 8, 1e-7
#: share of all elements above which the alpha 1 identity (combine(dpo, s2300, 1) == s2300) is
#: flagged in the report. Recorded, not fatal: it says nothing against a checkpoint with alpha < 1.
ALPHA1_TOL = 1e-6
#: generation_config.json is not read under Dan's `generation_config="vllm"`; its token ids must
#: still agree, everything else is recorded only
GENERATION_MUST_MATCH = ("eos_token_id", "pad_token_id", "bos_token_id")


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def tensor_class(name: str) -> str:
    if name.endswith("_proj.weight") and ".layers." in name:
        return "matrix"
    if "embed_tokens" in name:
        return "embedding"
    if name.startswith("lm_head"):
        return "output_head"
    if "norm" in name:
        return "norm"
    return "other"


class _One:
    """One in-memory tensor behind the two members `expo_write.verify_tensors` uses."""

    def __init__(self, t: torch.Tensor):
        self.wmap = {"t": "memory"}
        self._t = t

    def get(self, _name: str) -> torch.Tensor:
        return self._t


def bf16_steps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Number of bf16 values between a and b (0 = equal, 1 = neighbours), for bf16 tensors."""

    def ordered(t: torch.Tensor) -> torch.Tensor:  # sign-magnitude bits -> a monotone integer
        bits = t.contiguous().view(torch.int16)
        return torch.where(bits < 0, torch.bitwise_and(bits, 0x7FFF).neg_(), bits).int()

    return ordered(a).sub_(ordered(b)).abs_()


def header_census(dpo: Checkpoint, ck: Checkpoint) -> dict:
    """Names, shapes and dtypes from the safetensors headers (no weight is read, except non-float
    tensors, which are tiny and compared in full)."""
    na, nb = set(dpo.names()), set(ck.names())
    problems: list[str] = []
    if na != nb:
        problems.append(
            f"tensor names differ: {len(na - nb)} only in dpo, {len(nb - na)} only in ckpt"
        )
    classes: dict[str, int] = {}
    nonfloat: list[dict] = []
    dtypes: dict[str, int] = {}
    for n in sorted(na & nb):
        sa = dpo._f(dpo.shard_of(n)).get_slice(n)
        sb = ck._f(ck.shard_of(n)).get_slice(n)
        da, db = str(sa.get_dtype()), str(sb.get_dtype())
        if da != db or list(sa.get_shape()) != list(sb.get_shape()):
            problems.append(f"{n}: {da}{sa.get_shape()} in dpo, {db}{sb.get_shape()} in ckpt")
            continue
        dtypes[da] = dtypes.get(da, 0) + 1
        classes[tensor_class(n)] = classes.get(tensor_class(n), 0) + 1
        if da not in FLOAT_DTYPES:
            same = bool(torch.equal(dpo.get(n), ck.get(n)))
            nonfloat.append({"name": n, "dtype": da, "equal": same})
            if not same:
                problems.append(f"{n}: non-float tensor differs; the writer would keep dpo's")
    return {
        "tensors": len(na & nb),
        "by_class": classes,
        "dtypes": dtypes,
        "non_float": nonfloat,
        "problems": problems,
        "ok": not problems,
    }


def config_check(dpo_dir: Path, ck_dir: Path) -> dict:
    out: dict = {"ok": True, "differing_fields": {}, "cosmetic_only": True}
    pa, pb = dpo_dir / "config.json", ck_dir / "config.json"
    if not (pa.exists() and pb.exists()):
        return {"ok": False, "error": "config.json missing in dpo or ckpt"}
    a, b = json.loads(pa.read_text()), json.loads(pb.read_text())
    out["sha256"] = {"dpo": sha256_file(pa), "ckpt": sha256_file(pb)}
    for k in sorted(set(a) | set(b)):
        if a.get(k, "<absent>") != b.get(k, "<absent>"):
            out["differing_fields"][k] = [a.get(k, "<absent>"), b.get(k, "<absent>")]
            if k not in CONFIG_COSMETIC:
                out["cosmetic_only"] = False
    ga, gb = dpo_dir / "generation_config.json", ck_dir / "generation_config.json"
    gen: dict = {"present": [ga.exists(), gb.exists()], "differing_fields": {}, "ok": True}
    if ga.exists() and gb.exists():
        a, b = json.loads(ga.read_text()), json.loads(gb.read_text())
        for k in sorted(set(a) | set(b)):
            if a.get(k, "<absent>") != b.get(k, "<absent>"):
                gen["differing_fields"][k] = [a.get(k, "<absent>"), b.get(k, "<absent>")]
                gen["ok"] = gen["ok"] and k not in GENERATION_MUST_MATCH
    out["generation_config"] = gen
    out["ok"] = out["cosmetic_only"] and gen["ok"]
    return out


def tree_bytes(root: Path) -> int:
    total = 0
    for d, _dirs, files in os.walk(root):
        for f in files:
            with contextlib.suppress(OSError):
                total += os.lstat(os.path.join(d, f)).st_size
    return total


def weight_bytes(d: Path) -> int:
    return sum(p.stat().st_size for p in d.glob("*.safetensors"))


def disk_budget(volume_root: Path, dpo_dir: Path, volume_gb: float) -> dict:
    """The quota is the volume size, which `df` does not show on the pod's network filesystem.
    Everything under `volume_root` counts, whoever wrote it (the pod is shared)."""
    used = tree_bytes(volume_root)
    need = weight_bytes(dpo_dir)
    limit = volume_gb * 1e9
    ok = used + need + MARGIN_GB * 1e9 <= limit
    return {
        "volume_root": str(volume_root),
        "used_gb": round(used / 1e9, 2),
        "need_gb": round(need / 1e9, 2),
        "margin_gb": MARGIN_GB,
        "volume_gb": volume_gb,
        "ok": bool(ok),
    }


def _new_acc() -> dict:
    return {
        "tensors": 0,
        "tensors_differing": 0,
        "tensors_differing_left_at_dpo": 0,
        "elements": 0,
        "differ": 0,
        "at_dpo": 0,
        "at_ckpt": 0,
        "between": 0,
        "outside": 0,
        "equal_but_changed": 0,
        "alpha1_not_ckpt": 0,
        "s_dd": 0.0,
        "s_rd": 0.0,
        "s_rr": 0.0,
        "steps": [0] * len(STEP_LABELS),
        "steps_sq": [0.0] * len(STEP_LABELS),
    }


def _chunk_stats(
    a: torch.Tensor, b: torch.Tensor, o: torch.Tensor, acc: dict, steps: bool = True
) -> None:
    d, s, w = a.float(), b.float(), o.float()
    delta, r = s - d, w - d
    sq = delta * delta
    same = delta == 0
    n_diff = a.numel() - int(torch.count_nonzero(same))
    moved = r != 0
    changed = int(
        torch.count_nonzero(same & moved)
    )  # equal in both inputs, yet written differently
    at_dpo = n_diff - (int(torch.count_nonzero(moved)) - changed)
    at_ckpt = int(torch.count_nonzero(w == s)) - (a.numel() - n_diff - changed)
    outside = int(
        torch.count_nonzero(torch.clamp(w, torch.minimum(d, s), torch.maximum(d, s)) != w)
    )
    acc["elements"] += a.numel()
    acc["differ"] += n_diff
    acc["at_dpo"] += at_dpo
    acc["at_ckpt"] += at_ckpt
    acc["equal_but_changed"] += changed
    acc["outside"] += outside
    acc["between"] += n_diff - at_dpo - at_ckpt - (outside - changed)
    # fp32 pairwise sums per chunk, added up in fp64: about 1e-6 relative, enough for a diagnostic
    acc["s_dd"] += float(sq.sum())
    acc["s_rd"] += float((r * delta).sum())
    acc["s_rr"] += float((r * r).sum())
    if steps and a.dtype == torch.bfloat16:
        st = bf16_steps(a, b).clamp_(max=STEP_CLAMP).long()
        cnt = torch.bincount(st, minlength=STEP_CLAMP + 1)
        ssq = torch.bincount(st, weights=sq.double(), minlength=STEP_CLAMP + 1)
        for i, (lo_s, hi_s) in enumerate(STEP_EDGES):
            acc["steps"][i] += int(cnt[lo_s : hi_s + 1].sum())
            acc["steps_sq"][i] += float(ssq[lo_s : hi_s + 1].sum())


def _merge(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, list):
            dst[k] = [x + y for x, y in zip(dst[k], v, strict=True)]
        else:
            dst[k] += v


def _finish(acc: dict) -> dict:
    out = dict(acc)
    dd, rd, rr = acc["s_dd"], acc["s_rd"], acc["s_rr"]
    out["realised_fraction"] = rd / dd if dd > 0 else None
    out["length_ratio"] = (rr / dd) ** 0.5 if dd > 0 else None
    out["off_line_share"] = max(0.0, 1.0 - rd * rd / (rr * dd)) if rr > 0 and dd > 0 else None
    n = acc["differ"]
    for k in ("at_dpo", "at_ckpt", "between"):
        out[f"{k}_share_of_differing"] = acc[k] / n if n else None
    out["steps"] = dict(zip(STEP_LABELS, acc["steps"], strict=True))
    out["steps_sq_share"] = {
        lab: (v / dd if dd > 0 else None)
        for lab, v in zip(STEP_LABELS, acc["steps_sq"], strict=True)
    }
    del out["steps_sq"]
    if not sum(acc["steps"]):  # census skipped (or not bf16)
        out["steps"] = out["steps_sq_share"] = out["alpha1_not_ckpt"] = None
    return out


def _bit_equal(x: torch.Tensor, y: torch.Tensor) -> bool:
    if x.dtype == torch.bfloat16 and y.dtype == torch.bfloat16:
        return bool(torch.equal(x.contiguous().view(torch.int16), y.contiguous().view(torch.int16)))
    return bool(torch.equal(x, y))


def exact_sample(names: list[str], seed: int = EXACT_SEED, layers: int = EXACT_LAYERS) -> set[str]:
    """Tensors that also go through the exact-rational verifier: every tensor that is not one of the
    attention / MLP matrices (embedding, output head, norms), and every matrix of `layers` layers
    chosen by seed (so each of the seven matrix types is covered several times)."""
    import random

    chosen = {n for n in names if tensor_class(n) != "matrix"}
    ids = sorted({int(n.split(".layers.")[1].split(".")[0]) for n in names if ".layers." in n})
    pick = set(random.Random(seed).sample(ids, min(layers, len(ids))))
    chosen |= {
        n
        for n in names
        if tensor_class(n) == "matrix" and int(n.split(".layers.")[1].split(".")[0]) in pick
    }
    return chosen


def full_check(
    dpo: Checkpoint,
    ck: Checkpoint,
    out: Checkpoint,
    alpha: float,
    chunk: int = CHUNK,
    exact: str = "sample",
    census: bool = True,
) -> dict:
    """Every tensor is re-read and (1) compared bit for bit with a fresh `expo_write.combine`,
    (2) held to the two invariants, (3) added to the rounding diagnostics. The exact-rational
    verifier (`expo_write.verify_tensors`) runs on `exact_sample` (exact="sample") or on everything
    ("all"). census=True adds the two input-only passes (bf16 step histogram, alpha 1 identity)."""
    by_class: dict[str, dict] = {}
    total = _new_acc()
    bad: list[dict] = []
    worst_ulp, n_checked, n_ne, n_exact, el_exact = 0.0, 0, 0, 0, 0
    names = sorted(dpo.names(), key=lambda n: (dpo.shard_of(n), n))
    exact_names = set(names) if exact == "all" else exact_sample(names)
    for n in names:
        a, b, o = dpo.get(n), ck.get(n), out.get(n)
        acc = _new_acc()
        acc["tensors"] = 1
        if a.dtype != o.dtype or a.shape != o.shape or b.shape != a.shape:
            bad.append({"name": n, "why": "dtype or shape of the written tensor differs"})
            continue
        n_checked += 1
        if not a.is_floating_point():
            if not torch.equal(a, o) or not torch.equal(a, b):
                bad.append({"name": n, "why": "non-float tensor not equal to dpo and ckpt"})
            continue
        fa, fb, fo = a.reshape(-1), b.reshape(-1), o.reshape(-1)
        bits_ok, t_ok, t_worst, t_ne = True, True, 0.0, 0
        for i in range(0, fa.numel(), chunk):
            ca, cb, co = fa[i : i + chunk], fb[i : i + chunk], fo[i : i + chunk]
            bits_ok = bits_ok and _bit_equal(ew.combine(ca, cb, alpha), co)
            if n in exact_names:
                v = ew.verify_tensors(_One(ca), _One(cb), _One(co), alpha, 1)[0]
                t_ok = t_ok and bool(v["ok"])
                t_worst = max(t_worst, float(v["max_abs_diff_ulp"]))
                t_ne += int(v["n_ne"])
                el_exact += ca.numel()
            if census and a.dtype == torch.bfloat16:
                one = ew.combine(ca, cb, 1.0).contiguous().view(torch.int16)
                acc["alpha1_not_ckpt"] += int(
                    torch.count_nonzero(one != cb.contiguous().view(torch.int16))
                )
            _chunk_stats(ca, cb, co, acc, census)
        n_ne += t_ne
        n_exact += n in exact_names
        if acc["differ"]:
            acc["tensors_differing"] = 1
            acc["tensors_differing_left_at_dpo"] = int(acc["at_dpo"] == acc["differ"])
        worst_ulp = max(worst_ulp, t_worst)
        if not bits_ok:
            bad.append({"name": n, "why": "not bit-equal to the recomputed fp32 combine"})
        if not t_ok or t_ne > max(NE_TOL, NE_SHARE * fa.numel()):
            bad.append(
                {
                    "name": n,
                    "why": "exact reference",
                    "max_abs_diff_ulp": t_worst,
                    "elements_not_equal": t_ne,
                }
            )
        if acc["outside"] or acc["equal_but_changed"]:
            bad.append(
                {
                    "name": n,
                    "why": "invariant",
                    "outside": acc["outside"],
                    "equal_but_changed": acc["equal_but_changed"],
                }
            )
        cls = tensor_class(n)
        _merge(by_class.setdefault(cls, _new_acc()), acc)
        _merge(total, acc)
    one = None
    if census:
        n1, n = total["alpha1_not_ckpt"], max(total["elements"], 1)
        one = {
            "elements_not_equal": n1,
            "elements": n,
            "share": n1 / n,
            "flag": n1 > ALPHA1_TOL * n,
        }
    return {
        "alpha": alpha,
        "alpha1_identity": one,
        "tensors_checked": n_checked,
        "tensors_expected": len(names),
        "exact_mode": exact,
        "census": census,
        "exact_tensors": n_exact,
        "exact_elements": el_exact,
        "max_abs_diff_ulp": worst_ulp,
        "elements_not_equal_to_reference": n_ne,
        "bad": bad,
        "ok": not bad and n_checked == len(names),
        "all": _finish(total),
        "by_class": {k: _finish(v) for k, v in sorted(by_class.items())},
    }


def _alpha_ok(alpha: float) -> bool:
    fr = Fraction(alpha).limit_denominator(100)
    return 0.0 <= alpha <= 1.0 and abs(float(fr) - alpha) <= 1e-12


def run(
    dpo_dir: Path,
    ck_dir: Path,
    alpha: float,
    out_dir: Path,
    report_dir: Path,
    *,
    mode: str = "write",
    volume_gb: float | None = None,
    volume_root: Path | None = None,
    min_free_gb: float = 85.0,
    chunk: int = CHUNK,
    exact: str = "sample",
    census: bool = True,
) -> int:
    report_dir.mkdir(parents=True, exist_ok=True)
    rep: dict = {
        "utc_start": utc(),
        "mode": mode,
        "alpha": alpha,
        "dpo": str(dpo_dir),
        "ckpt": str(ck_dir),
        "out": str(out_dir),
        "threads": torch.get_num_threads(),
        "ok": False,
    }

    def done(rc: int, why: str = "") -> int:
        rep.update(utc_end=utc(), rc=rc, why=why)
        (report_dir / "check.json").write_text(json.dumps(rep, indent=1) + "\n")
        print(f"INTERP {mode.upper()} alpha={alpha} rc={rc} {why}".rstrip(), flush=True)
        return rc

    if not _alpha_ok(alpha):
        return done(2, "alpha must be in [0, 1] with a denominator of at most 100")
    marker = out_dir / MARKER
    if mode == "write" and marker.exists():
        old = json.loads(marker.read_text())
        if abs(float(old.get("alpha", -1)) - alpha) <= 1e-12:
            rep["ok"] = True
            return done(0, "already written and checked (marker present)")
        return done(2, f"{out_dir} holds alpha {old.get('alpha')}, not {alpha}")
    dpo, ck = Checkpoint(dpo_dir), Checkpoint(ck_dir)
    rep["census"] = header_census(dpo, ck)
    rep["config"] = config_check(dpo_dir, ck_dir)
    if not rep["census"]["ok"]:
        return done(6, "census: " + "; ".join(rep["census"]["problems"][:3]))
    if not rep["config"]["ok"]:
        return done(6, f"config.json differs beyond labels: {rep['config']}")
    if mode == "preflight":
        rep["ok"] = True
        return done(0, f"{rep['census']['tensors']} tensors, {rep['census']['by_class']}")
    if mode == "write":
        if out_dir.exists():  # an earlier write without its marker: start again
            if not out_dir.name.startswith("interp_"):
                return done(2, f"refusing to overwrite {out_dir}")
            shutil.rmtree(out_dir)
        if volume_gb is not None:
            rep["disk"] = disk_budget(volume_root or out_dir.parent, dpo_dir, volume_gb)
            if not rep["disk"]["ok"]:
                return done(5, f"disk budget: {rep['disk']}")
        t0 = time.time()
        try:
            man = ew.write(dpo_dir, ck_dir, alpha, out_dir, verify=3, min_free_gb=min_free_gb)
        except SystemExit as ex:
            return done(5, f"writer refused: {ex}")
        rep["write_seconds"] = round(time.time() - t0, 1)
        rep["writer_manifest"] = {
            "gamma": man["gamma"],
            "tensors": man["tensors"],
            "shards": len(man["shards"]),
            "verify_ok": man["verify_ok"],
            "index_sha256": man["index_sha256"],
        }
        if not man["verify_ok"]:
            return done(4, "the writer's own sample check failed")
    if not (out_dir / "MANIFEST.json").exists():
        return done(2, f"no MANIFEST.json in {out_dir}")
    shutil.copy2(out_dir / "MANIFEST.json", report_dir / "MANIFEST.json")
    t0 = time.time()
    rep["check"] = full_check(dpo, ck, Checkpoint(out_dir), alpha, chunk, exact, census)
    rep["check_seconds"] = round(time.time() - t0, 1)
    if not rep["check"]["ok"]:
        marker.unlink(missing_ok=True)
        return done(4, f"{len(rep['check']['bad'])} tensors failed the full check")
    rep["ok"] = True
    marker.write_text(
        json.dumps(
            {
                "alpha": alpha,
                "utc": utc(),
                "index_sha256": sha256_file(out_dir / "model.safetensors.index.json"),
                "tensors_checked": rep["check"]["tensors_checked"],
                "realised_fraction": rep["check"]["all"]["realised_fraction"],
            }
        )
        + "\n"
    )
    return done(0, f"realised fraction {rep['check']['all']['realised_fraction']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["write", "check", "preflight"])
    ap.add_argument("--tag", choices=sorted(TAGS), help="pod layout: alpha and paths from the tag")
    ap.add_argument("--dpo")
    ap.add_argument("--ckpt")
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--out")
    ap.add_argument("--report")
    ap.add_argument("--volume-gb", type=float, default=None)
    ap.add_argument(
        "--min-free-gb",
        type=float,
        default=float(os.environ.get("WI_MIN_FREE_GB", "85")),
        help="the writer's own free-space floor (on the pod `df` shows the storage cluster, so the "
        "real limit is --volume-gb against the summed file sizes)",
    )
    ap.add_argument("--threads", type=int, default=int(os.environ.get("WI_THREADS", "16")))
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument(
        "--exact",
        choices=["sample", "all"],
        default="sample",
        help="tensors for the exact-rational verifier (the bitwise recompute always covers all)",
    )
    ap.add_argument(
        "--no-census",
        action="store_true",
        help="skip the two passes that do not depend on alpha (bf16 steps between the inputs, and "
        "the alpha 1 identity); the chain runs them with the last write only",
    )
    a = ap.parse_args(argv)
    torch.set_num_threads(max(1, a.threads))
    models = Path(os.environ.get("MODELS_ROOT", "/workspace/models"))
    vroot = None
    if a.tag:
        # the two inputs are where Dan's fetch script puts them; the merged copy goes under this
        # run's own folder, and the disk budget counts the whole volume
        wi = Path(os.environ.get("M2_ROOT", "/workspace/wi"))
        merged = Path(os.environ.get("WI_MERGED_ROOT", str(wi / "models")))
        vroot = Path(os.environ.get("WI_VOLUME_ROOT", "/workspace"))
        dpo, ck, alpha = models / "dpo", models / "s2300", TAGS[a.tag]
        out, report = merged / f"interp_{a.tag}", wi / "runs" / f"write_{a.tag}" / "out"
        volume = a.volume_gb or float(os.environ.get("WI_VOLUME_GB", "500"))
    elif a.mode == "preflight" and a.dpo is None:
        wi = Path(os.environ.get("M2_ROOT", "/workspace/wi"))
        dpo, ck, alpha = models / "dpo", models / "s2300", 0.5
        out, report, volume = models / "interp_none", wi / "runs" / "preflight" / "out", None
    else:
        if None in (a.dpo, a.ckpt, a.alpha, a.out, a.report):
            ap.error("--tag, or all of --dpo --ckpt --alpha --out --report")
        dpo, ck, alpha = Path(a.dpo), Path(a.ckpt), a.alpha
        out, report, volume = Path(a.out), Path(a.report), a.volume_gb
    return run(
        dpo,
        ck,
        alpha,
        out,
        report,
        mode=a.mode,
        volume_gb=volume,
        volume_root=vroot,
        min_free_gb=a.min_free_gb,
        chunk=a.chunk,
        exact=a.exact,
        census=not a.no_census,
    )


if __name__ == "__main__":
    sys.exit(main())
