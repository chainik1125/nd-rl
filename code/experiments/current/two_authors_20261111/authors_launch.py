"""Launches of the two-authors experiment: pretraining on one corpus, a base's habit measurement, one RL line.

    python authors_launch.py pretrain --corpus {authors,control}_{half,p10} --seed S [--gpu 0] [--dry-run] [--lab]
    python authors_launch.py measure  --base <corpus>_s<S> [--tags A,B,none] [--sets ...] [--k 32] [--gpu 0] [--skip_done] [--dry-run] [--lab]
    python authors_launch.py rl       --line rl_<corpus>_s<S> [--seg K --resume CKPT] [--snap_k 8] [--stop_on_held] [--gpu 0] [--dry-run] [--lab]
    python authors_launch.py status

Meant for Runpod pods (one job per pod, `authors_pod.sh`). On a lab host (serrano, ynez, temescal) a real launch
refuses unless `--lab`, and then applies the lab rules of ../author_tags_20261110/tag_launch.py (`gate`: no other
process on the GPU, serrano at 200 W, at most 4 of ours, 2 lab GPUs left free). On a pod: the GPU exists, no other
user's process holds it, under 500 MiB in use, and the lock TA/locks/<host>_gpu<N>.lock is free.

Every job: `--dry-run` prints the plan (steps, commands, timeouts, estimates), every reason a real launch would refuse,
and the gate, and writes nothing. A real launch refuses on any of those reasons. The job is detached (nohup, its own
session) and runs `tag_launch.run_job` (reused unchanged): it takes the lock, runs each step under `timeout
--kill-after=300`, writes TA/logs/<job>_<UTC>.log (each step's start, end, return code, seconds) and .status.json,
stops at the first failing step, and releases the lock. Nothing is ever overwritten: bases, measurements and RL
segments refuse an existing output.

Jobs:
  pretrain  `_pretrain` (below) -> TA/bases/stage1_<corpus>_s<S>.pt (+ .tag.json with D, .args.json, registry/), then a
            sanity read (authors_measure: 64 marker_plain theorems x 4 under A, B, none) -> TA/measure/<id>_sanity/.
  measure   authors_measure.py run on the base -> TA/measure/<corpus>_s<S>/ (rows) and TA/raw/<corpus>_s<S>/ (proofs).
  rl        `_line` (below) -> TA/runs/<line>/seg<K>/: GRPO on UNTAGGED prompts from the base, Dan's read of the strict
            106 at the base and at every snapshot (untagged), and authors_measure at the base and every snapshot under
            A, B and none (k `--snap_k` on marker_plain and route_unseen; the strict 106 proofs of Dan's read profiled).

Reused from the tag experiment (../author_tags_20261110/, unchanged, imported): `tag_pretrain.main` (Dan's
`state_train.py --recipe best` at 13e41bbd, a fixed step count `tag_core.DOSE1_STEPS[seed]`, D recorded),
`tag_line.Seg` and `tag_line.grpo_argv` (the knockout's dd1_s<seed> GRPO recipe on Dan's 4,495 rl_targets, with --init
and the KL anchor set to our base), `tag_grpo.py --tag none`, `tag_danread.py --tag none`, `tag_measure.line_view`
and `segment_problems`, `tag_launch.gate` / `run_job`. Their run tree is `tag_core.AT`, which this module points at TA
(the environment variable AT is set before they are imported), and the lines are registered in `tag_core.LINES`.

Wrapped, and why:
  * `_pretrain` replaces `tag_pretrain.tag_load` (it asserts every state carries tag A or B, so it fails on the
    corpora's `author_tag: "none"` records, the hidden half of each author) with `authors_load`: each record is
    encoded under its shown tag (`shown_tag`; "none" through TagTokenizer's own untagged encoding, asserted to carry
    no tag token), the held-out records untagged; and adds `two_authors` (corpus, sha256, shown-tag counts, pairs per
    tag) to the checkpoint's extra and its .tag.json.
  * `_line` subclasses `tag_line.Seg` (`AuthorsSeg`) to add the snapshot measurements and to prune alarm checkpoints
    beyond the newest 4 (the knockout's pod disk fault), and replaces `tag_measure.stop_reason` in its own process:
    x_dan >= 1.5 or GPU-h >= 7 (the reached-and-held stop of the tag lines only with `--stop_on_held`).

Corpus records (authors_corpus.py; CORPUS.md "Record format"): the fields used are `prompt`, `proof`, `n_lines`, `key`
and the shown tag. `shown_tag`: the boolean `tag_shown` says whether the tag is in the record's states, and the tag is
then `author_tag` ("A" or "B"; it is "none" when hidden). Without a boolean field (`tag_shown`, `shown`, `show_tag`),
`author_tag` alone is read, None / "" / "none" meaning no tag. `author`, when present, must equal every shown tag (in a
control it is the nominal label that picks the tag); a launch refuses a corpus where it does not, where no record or
every record shows a tag.

Held-out loss of these bases: Dan's `data/p2/heldout.jsonl` (its first 2,000 records), as Dan's and the tag
experiment's runs use it: neither rewritten into an author's form nor tagged; every held-out state is encoded untagged
(`<act>` with no tag token). It only monitors training (the `final_val` of the .tag.json); it is not a gate quantity.
"""

# ruff: noqa: E501, SIM115
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
TA = Path(os.environ.get("TA", "/scratch/cowifornia/nd-rl/two_authors"))
os.environ["AT"] = str(TA)  # the tag experiment's modules keep their run tree under AT: ours is TA
TAGS_DIR = HERE.parent / "author_tags_20261110"
for _p in (str(TAGS_DIR), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import tag_core as T  # noqa: E402

T.AT = TA  # tag_core read AT at import; its users (tag_measure, tag_line, tag_launch) read T.AT at call time
import tag_launch as TL  # noqa: E402
import tag_measure as TM  # noqa: E402

G = TM._ko("gpu")
HOST = os.uname().nodename.split(".")[0]
PY = T.PY
CORPORA = ("authors_half", "authors_p10", "control_half", "control_p10")
SEEDS = (0, 1, 2)
X_CAP = T.X_CAP  # 1.5
GPUH_CAP = 7.0  # a line's GPU-hour bound (x 1.5 is 3-5 GPU-h of GRPO at the knockout pace, plus the snapshot reads)
SNAP_SETS = ("marker_plain", "route_unseen")
SHOWN_BOOL_FIELDS = ("tag_shown", "shown", "show_tag")
EST = {  # A40 hours; the basis is in RUNNER.md
    "pretrain": 0.45,  # Dan's A40 runs: about 24,000 best-recipe steps in 1,200 s; plus 1.5 min load, 1 min compile
    "sanity": 0.05,
    "measure_load": 0.03,
    "rl": 5.0,  # 3-5 GPU-h of GRPO to x 1.5 (knockout: 63-72 s per iteration on an A40) plus snapshot reads
}
TIMEOUT_H = {"pretrain": 1.5, "sanity": 0.5, "measure": 2.0, "rl": 10.0}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ------------------------------------------------------------------------------------------------ names
def corpus_path(corpus: str) -> Path:
    return TA / "data" / f"{corpus}.jsonl.gz"


def corpus_manifest(corpus: str) -> Path:
    return TA / "data" / f"{corpus}.manifest.json"


def base_name(corpus: str, seed: int) -> str:
    assert corpus in CORPORA, corpus
    return f"{corpus}_s{seed}"


def base_path(corpus: str, seed: int) -> Path:
    return TA / "bases" / f"stage1_{base_name(corpus, seed)}.pt"


def parse_base(name: str) -> tuple[str, int]:
    """`<corpus>_s<seed>` (an optional `stage1_` prefix and `.pt` suffix dropped) -> (corpus, seed)."""
    m = re.fullmatch(
        r"((?:authors|control)_(?:half|p10))_s(\d+)",
        name.removesuffix(".pt").removeprefix("stage1_"),
    )
    if m is None:
        raise SystemExit(f"not a base name ({{authors,control}}_{{half,p10}}_s<seed>): {name!r}")
    return m.group(1), int(m.group(2))


def line_name(corpus: str, seed: int, base_seed: int | None = None) -> str:
    b = seed if base_seed is None else base_seed
    return f"rl_{corpus}_s{seed}" if b == seed else f"rl_{corpus}_b{b}_s{seed}"


def parse_line(name: str) -> tuple[str, int, int]:
    """`rl_<corpus>[_b<base seed>]_s<seed>` -> (corpus, base seed, GRPO seed)."""
    m = re.fullmatch(r"rl_((?:authors|control)_(?:half|p10))(?:_b(\d+))?_s(\d+)", name)
    if m is None:
        raise SystemExit(f"not a line name (rl_<corpus>[_b<base seed>]_s<seed>): {name!r}")
    s = int(m.group(3))
    return m.group(1), int(m.group(2)) if m.group(2) is not None else s, s


def hint(msg: str, line: str) -> str:
    """tag_core's refusal names its own pretraining command; ours is authors_launch.py pretrain."""
    corpus, b, _s = parse_line(line)
    return re.sub(
        r"\(train the base first: [^)]*\)",
        f"(train the base first: authors_launch.py pretrain --corpus {corpus} --seed {b})",
        msg,
    )


def register_lines() -> None:
    """Our lines in tag_core.LINES (tag none; the base from TA/bases; the dd1_s<seed> recipe and la_dose1_s<seed> grid)."""
    for c in CORPORA:
        for b in SEEDS:
            for s in SEEDS:
                T.LINES[line_name(c, s, b)] = {"tag": "none", "seed": s, "base": f"stage1_{base_name(c, b)}.pt",
                                               "ladder": f"la_dose1_s{s}", "corpus": c, "base_seed": b}  # fmt: skip


register_lines()


# ------------------------------------------------------------------------------------------------ corpus records
def shown_tag(r: dict) -> str:
    """The tag a corpus record shows in training: 'A', 'B' or 'none' (module docstring)."""
    for f in SHOWN_BOOL_FIELDS:
        if isinstance(r.get(f), bool):
            t = r.get(T.TAG_FIELD) or r.get("author")
            if not r[f]:
                return "none"
            if t not in T.TAGS:
                raise T.TagError(
                    f"record {r.get('key')}: {f} true but no A/B tag in author_tag/author ({t!r})"
                )
            return t
    if T.TAG_FIELD not in r:
        raise T.TagError(
            f"record {r.get('key')}: no {T.TAG_FIELD} field and no {SHOWN_BOOL_FIELDS} field"
        )
    t = r[T.TAG_FIELD]
    if t in (None, "", "none"):
        return "none"
    if t not in T.TAGS:
        raise T.TagError(f"record {r.get('key')}: {T.TAG_FIELD} {t!r} is not A, B or none")
    return t


def _open(fn: str | Path):
    return (
        gzip.open(fn, "rt", encoding="utf-8")
        if str(fn).endswith(".gz")
        else open(fn, encoding="utf-8")
    )


def corpus_census(fn: str | Path, limit: int = 0) -> dict[str, Any]:
    """Counts of (author, shown tag) over a corpus, and the problems a pretraining launch refuses on."""
    c: collections.Counter = collections.Counter()
    problems: list[str] = []
    n = 0
    with _open(fn) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            try:
                t = shown_tag(r)
            except T.TagError as e:
                problems.append(str(e))
                break
            au = r.get("author")
            c[f"{au}|{t}"] += 1
            if au is not None and t != "none" and t != au:
                problems.append(f"record {r.get('key')}: shows tag {t} but its author is {au}")
                break
            n += 1
            if limit and n >= limit:
                break
    shown = sum(v for k, v in c.items() if not k.endswith("|none"))
    if n and not shown:
        problems.append("no record shows a tag")
    if n and shown == n:
        problems.append(
            "every record shows a tag (SPEC: the tag is on a random half of each author's records)"
        )
    return {"records": n, "by_author_shown": dict(c), "shown": shown, "problems": problems}


def authors_load(fn, tok, cap, limit=0):
    """`tag_pretrain.tag_load` for our corpora: each record's states under its shown tag (A, B or none); the held-out
    records untagged. Asserts every state ends in `<tagX> <act>` (or `<act>` with no tag). -> (data, n_skipped)."""
    import state_train as ST  # type: ignore[import-not-found]
    import tag_pretrain as TP
    from lean_tok import ParseFail  # type: ignore[import-not-found]

    training = str(fn) == str(TP._cfg["data"])
    if training and TP._cfg["limit"]:
        limit = TP._cfg["limit"]
    out, skipped = [], 0
    per = {"A": 0, "B": 0, "none": 0}
    tag_ids = set(tok.tag_ids.values())
    for line in _open(fn):
        if not line.strip():
            continue
        r = json.loads(line)
        if cap:
            assert r["n_lines"] <= cap, f"record exceeds cap {cap}: {r.get('name')}"
        tag = shown_tag(r) if training else "none"
        try:
            steps, _toks, _env = ST.decompose(
                r["prompt"], r["proof"], canon=getattr(tok, "canon", False)
            )
        except (ParseFail, ValueError, AssertionError, IndexError):
            skipped += 1
            continue
        tok.author_tag = tag
        pairs = []
        for stt, act, hs in steps:
            pre = tok.encode_toks((hs + stt) if tok.with_history else stt)
            assert pre[-1] == tok.act_id, "a state that does not end in <act>"
            if tag == "none":
                assert pre[-2] not in tag_ids, "an untagged record's state carries a tag"
            else:
                assert pre[-2] == tok.tag_ids[tag], "a state without its tag"
            pairs.append((bytes(pre), bytes(tok.encode_toks(act) + [tok.eos])))
        tok.author_tag = None
        per[tag] += len(pairs)
        out.append(pairs)
        if limit and len(out) >= limit:
            break
    TP._cfg["pairs"]["train" if training else "heldout"] = per
    return out, skipped


# ------------------------------------------------------------------------------------------------ _pretrain
def pretrain_main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="authors_launch.py _pretrain")
    ap.add_argument(
        "--corpus", required=True, help="a corpus name (CORPORA) or a .jsonl[.gz] path (smoke)"
    )
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=None, help="default: tag_core.DOSE1_STEPS[seed]")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--best_dims", default=None)
    ap.add_argument("--tok_budget", type=int, default=128 * 144)
    ap.add_argument("--curve_every", type=float, default=30)
    ap.add_argument(
        "--heldout", default=None, help="default: Dan's data/p2/heldout.jsonl; '' for none"
    )
    a = ap.parse_args(argv)
    import tag_pretrain as TP

    name = a.corpus if a.corpus in CORPORA else None
    data = corpus_path(a.corpus) if name else Path(a.corpus)
    man = (
        corpus_manifest(a.corpus)
        if name
        else Path(str(data).removesuffix(".gz").removesuffix(".jsonl") + ".manifest.json")
    )
    out = Path(a.out).resolve()
    for x in (out, out.with_suffix(".tag.json"), Path(f"{out}.args.json")):
        if x.exists():
            raise SystemExit(f"{x} exists: a base is never overwritten")
    if not data.exists():
        raise SystemExit(f"no corpus {data}")
    cen = corpus_census(data, a.limit)
    if cen["problems"]:
        raise SystemExit(f"corpus {data}: {cen['problems']}")
    info = {"experiment": "two_authors_20261111", "corpus": name or str(data), "corpus_file": str(data),
            "corpus_sha256": T.sha256(data), "corpus_manifest": str(man) if man.exists() else None,
            "census": cen, "shown_tag_rule": "authors_launch.shown_tag", "heldout_tag": "none"}  # fmt: skip
    TP.tag_load = authors_load  # TP.main installs it as state_train.load
    TP._cfg["manifest"] = man
    orig_wrap = TP._wrap_train

    def wrap(stb, corpus, steps):
        orig_wrap(stb, corpus, steps)
        inner = stb.train

        def train(args, tok, data_, held, dev, record):
            model, extra = inner(args, tok, data_, held, dev, record)
            info["pairs_per_tag"] = TP._cfg["pairs"]
            extra["two_authors"] = info
            sc = Path(args.out).with_suffix(".tag.json")
            side = json.loads(sc.read_text())
            side["two_authors"] = info
            sc.write_text(json.dumps(side, indent=1, default=str))
            return model, extra

        stb.train = train

    TP._wrap_train = wrap
    targv = ["--seed", str(a.seed), "--out", str(out), "--data", str(data), "--tok_budget", str(a.tok_budget),
             "--curve_every", str(a.curve_every)]  # fmt: skip
    if a.steps:
        targv += ["--steps", str(a.steps)]
    if a.limit:
        targv += ["--limit", str(a.limit)]
    if a.best_dims:
        targv += ["--best_dims", a.best_dims]
    if a.heldout is not None:
        targv += ["--heldout", a.heldout]
    print(
        f"authors _pretrain: corpus {data} ({cen['by_author_shown']}); tag_pretrain {' '.join(targv)}",
        flush=True,
    )
    TP.main(targv)


# ------------------------------------------------------------------------------------------------ _line
def stop_reason(v: dict, on_held: bool = False) -> str | None:
    """Our lines' stop rule: x_dan >= 1.5 or GPU-h >= GPUH_CAP; reached and held (tag_measure's) only when asked."""
    if v["x_dan"] >= X_CAP:
        return f"compute cap: x_dan {v['x_dan']:.3f} >= {X_CAP:g}"
    if v["gpuh"] >= GPUH_CAP:
        return f"GPU-h cap: {v['gpuh']:.2f} >= {GPUH_CAP:g}"
    if on_held:
        h = TM.M.held(v["curve"])
        if v["cross_dan"] is not None and h is not None:
            return f"reached (Dan's measure at x_dan {v['cross_dan']['x_dan']:.3f}) and held ({h})"
    return None


def snap_cmd(
    line: str, seg: int, ckpt: Path, it: int, k: int, sets: str, smoke: bool, limit: int, seed: int
) -> list[str]:
    d = TA / "runs" / line / f"seg{seg}" / "authors" / f"it{it:04d}"
    cmd = [PY, str(HERE / "authors_measure.py"), "run", "--ckpt", str(ckpt), "--ckpt_id", f"{line}@it{it}",
           "--tags", "A,B,none", "--sets", sets, "--k", str(k), "--batch", "128", "--seed", str(seed),
           "--device", "cpu" if smoke else "cuda", "--out_dir", str(d),
           "--raw_dir", str(TA / "raw" / line / f"seg{seg}" / f"it{it:04d}"), "--resume"]  # fmt: skip
    if limit:
        cmd += ["--limit", str(limit)]
    return cmd


def make_seg_class():
    import tag_line as TLN

    class AuthorsSeg(TLN.Seg):
        """tag_line.Seg plus: authors_measure at the base and every snapshot (under A, B, none; serialised with Dan's
        reads by the same lock), the strict 106 of each Dan read profiled, alarm checkpoints pruned to the newest 4."""

        def __init__(self, a: argparse.Namespace) -> None:
            super().__init__(a)
            self.afails: dict[int, int] = {}

        def adir(self, it: int) -> Path:
            return self.dir / "authors" / f"it{it:04d}"

        def items(self) -> list[tuple[int, Path]]:
            out = []
            if self.a.seg == 1 and not self.a.resume:
                out.append((0, T.line_base(self.a.line)))
            snaps = sorted((self.dir / "ckpts").glob(f"{self.a.line}_it*_snap.pt"),
                           key=lambda p: int(p.name.split("_it")[-1].split("_")[0]))  # fmt: skip
            out += [(int(p.name.split("_it")[-1].split("_")[0]), p) for p in snaps]
            return out

        def measure(self, ckpt: Path, it: int) -> None:
            d = self.adir(it)
            d.mkdir(parents=True, exist_ok=True)
            cmd = snap_cmd(self.a.line, self.a.seg, ckpt, it, self.a.snap_k, self.a.snap_sets, self.a.smoke,
                           self.a.snap_limit, self.L["seed"])  # fmt: skip
            env = {**self.env(), "LEAN_GATE_LOG": str(d / "lean_gate.jsonl"), "TA": str(TA),
                   **({} if self.a.smoke else {"DANREAD_MEM_FRAC": "0.3"})}  # fmt: skip
            t0 = time.time()
            with self.reading, open(d / "measure.log", "a") as f:
                p = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=3600)
            self.say(f"authors measure it {it}: rc {p.returncode} in {time.time() - t0:.0f} s")
            if p.returncode == 0:
                (d / "DONE").write_text(now())
            else:
                self.afails[it] = self.afails.get(it, 0) + 1
                if self.afails[it] >= 3:
                    (d / "FAILED").write_text(now())

        def profile_dan_reads(self) -> None:
            """Profile every finished Dan read: under the read lock (so no read is being written) and only a read
            that tag_danread has completed (it adds `author_tag` last)."""
            import authors_measure as AM

            with self.reading:
                for f in sorted((self.dir / "reads").glob("it*.json")):
                    it = int(f.stem[2:])
                    out = self.adir(it) / "strict106__none.jsonl"
                    if out.exists():
                        continue
                    try:
                        if "author_tag" not in json.loads(f.read_text()):
                            continue
                        out.parent.mkdir(parents=True, exist_ok=True)
                        AM.danread(
                            argparse.Namespace(
                                read=str(f), ckpt_id=f"{self.a.line}@it{it}", out=str(out)
                            )
                        )
                    except Exception as e:  # noqa: BLE001  (retried next pass)
                        self.say(f"strict106 profile it {it}: {type(e).__name__}: {e}")

        def prune_alarms(self, keep: int = 4) -> None:
            al = sorted((self.dir / "ckpts").glob(f"{self.a.line}_it*_alarm.pt"),
                        key=lambda p: int(p.name.split("_it")[-1].split("_")[0]))  # fmt: skip
            for p in al[:-keep] if len(al) > keep else []:
                p.unlink(missing_ok=True)
                self.say(f"pruned alarm checkpoint {p.name} (keeping the newest {keep})")

        def authors_reader(self) -> None:
            while True:
                fin = self.done.is_set()
                for it, ck in self.items():
                    d = self.adir(it)
                    if not (d / "DONE").exists() and not (d / "FAILED").exists():
                        try:
                            self.measure(ck, it)
                        except subprocess.TimeoutExpired:
                            self.say(f"authors measure it {it}: timeout")
                            self.afails[it] = self.afails.get(it, 0) + 1
                self.profile_dan_reads()
                self.prune_alarms()
                if fin:
                    return
                time.sleep(20)

        def reader(self) -> None:
            th = threading.Thread(target=self.authors_reader, daemon=True)
            th.start()
            super().reader()
            th.join(timeout=0 if self.why == "terminated" else 3 * 3600)
            self.profile_dan_reads()

    return AuthorsSeg, TLN


def line_main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="authors_launch.py _line")
    ap.add_argument("--line", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--seg", type=int, default=1)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--rule_every", type=float, default=60.0)
    ap.add_argument("--snap_k", type=int, default=8)
    ap.add_argument("--snap_sets", default=",".join(SNAP_SETS))
    ap.add_argument("--snap_limit", type=int, default=0, help="smoke: first N theorems per set")
    ap.add_argument("--stop_on_held", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    parse_line(a.line)
    AuthorsSeg, TLN = make_seg_class()
    if a.smoke:
        assert "smoke" in str(T.AT), "a smoke run writes under a smoke tree (TA=.../smoke/...)"
    L = T.LINES[a.line]
    base = T.line_base(a.line)
    seg_dir = T.AT / "runs" / a.line / f"seg{a.seg}"
    argv_g = TLN.grpo_argv(a.line, seg_dir, a.iters, a.resume)
    if a.smoke:
        argv_g = TLN.smoke_argv(argv_g)
    problems = TM.segment_problems(a.line, a.seg, a.resume)
    try:
        D = T.check_line_base(a.line)["train_tokens"]
    except SystemExit as e:
        D, problems = None, [hint(str(e), a.line), *problems]
    if a.dry_run:
        print(json.dumps({"line": a.line, "tag": L["tag"], "base": str(base), "D": D, "ladder": L["ladder"],
                          "seg_dir": str(seg_dir), "grpo": TLN.grpo_cmd(a.line, argv_g), "cwd": str(T.CODE),
                          "dan_read": "tag_danread.py --tag none (k 64 on the 106, at the base and every snapshot)",
                          "snapshot_measure": snap_cmd(a.line, a.seg, Path("<snapshot>"), 0, a.snap_k, a.snap_sets, a.smoke, a.snap_limit, L["seed"]),
                          "stop": f"x_dan >= {X_CAP} or GPU-h >= {GPUH_CAP}" + (" or reached and held" if a.stop_on_held else ""),
                          "refuses": problems}, indent=1))  # fmt: skip
        return
    if problems:
        raise SystemExit(f"line {a.line} seg {a.seg}: " + "; ".join(problems))
    meta = T.ckpt_meta(base)
    assert meta["tok_mode"] == T.MODE and meta["variant"] is None, meta
    if not a.smoke and not G.free_for_us(HOST, a.gpu):
        raise SystemExit(f"GPU {a.gpu} on {HOST} has another user's process: not launching")
    TM.stop_reason = lambda v: stop_reason(
        v, a.stop_on_held
    )  # the line's own stop rule (Seg.watch calls it)
    sys.exit(AuthorsSeg(a).run())


# ------------------------------------------------------------------------------------------------ plans
def step(name: str, argv: list[str], hours: float, est: float, env: dict | None = None) -> dict:
    return {
        "name": name,
        "argv": argv,
        "timeout_s": int(hours * 3600),
        "est_h": est,
        "env": env or {},
    }


def sets_problems(sets: list[str]) -> list[str]:
    import authors_sets as AS

    out = [f"unknown set {s}" for s in sets if s not in AS.SETS]
    out += [
        f"no set file {AS.OUT / f'{s}.jsonl'} (authors_sets.py build)"
        for s in sets
        if s in AS.SETS and not (AS.OUT / f"{s}.jsonl").exists()
    ]
    return out


def plan(a: argparse.Namespace) -> tuple[str, list[dict], list[str], list[str]]:
    """(job name, steps, problems a real launch refuses on, notes). Writes nothing."""
    problems: list[str] = []
    notes: list[str] = []
    me, am = str(HERE / "authors_launch.py"), str(HERE / "authors_measure.py")
    if a.cmd == "pretrain":
        if a.corpus not in CORPORA:
            raise SystemExit(f"--corpus must be one of {CORPORA}")
        out = base_path(a.corpus, a.seed)
        bid = base_name(a.corpus, a.seed)
        for x in (
            out,
            out.with_suffix(".tag.json"),
            Path(f"{out}.args.json"),
            TA / "measure" / f"{bid}_sanity",
        ):
            if x.exists():
                problems.append(f"{x} exists: never overwritten")
        cp = corpus_path(a.corpus)
        if not cp.exists():
            problems.append(f"no corpus {cp} (authors_corpus.py builds it)")
        else:
            cen = corpus_census(cp)
            notes.append(
                f"corpus {cp}: {cen['records']:,} records; (author|shown tag) {cen['by_author_shown']}"
            )
            problems += [f"corpus: {p}" for p in cen["problems"]]
        if a.seed not in T.DOSE1_STEPS:
            problems.append(
                f"seed {a.seed}: no step count (tag_core.DOSE1_STEPS has {sorted(T.DOSE1_STEPS)})"
            )
        else:
            notes.append(
                f"{T.DOSE1_STEPS[a.seed]:,} steps (Dan's 1 % base of seed {a.seed}); held-out loss on Dan's data/p2/heldout.jsonl, untagged"
            )
        problems += sets_problems(["marker_plain"])
        steps = [step(f"pretrain_{bid}", [PY, me, "_pretrain", "--corpus", a.corpus, "--seed", str(a.seed), "--out", str(out)],
                      TIMEOUT_H["pretrain"], EST["pretrain"]),
                 step(f"sanity_{bid}", [PY, am, "run", "--ckpt", str(out), "--ckpt_id", f"{bid}_sanity", "--tags", "A,B,none",
                                        "--sets", "marker_plain", "--limit", "64", "--k", "4", "--device", "cuda",
                                        "--out_dir", str(TA / "measure" / f"{bid}_sanity"), "--raw_dir", str(TA / "raw" / f"{bid}_sanity")],
                      TIMEOUT_H["sanity"], EST["sanity"])]  # fmt: skip
        return f"pretrain_{bid}", steps, problems, notes
    if a.cmd == "measure":
        import authors_measure as AM

        corpus, seed = parse_base(a.base)
        bid, ck = base_name(corpus, seed), base_path(corpus, seed)
        if not ck.exists():
            problems.append(
                f"no base {ck} (authors_launch.py pretrain --corpus {corpus} --seed {seed})"
            )
        sets = a.sets.split(",")
        problems += sets_problems(sets)
        tags = a.tags.split(",")
        for s in sets:
            for t in tags:
                sp = TA / "measure" / bid / f"{s}__{t}.summary.json"
                if sp.exists() and not a.skip_done:
                    problems.append(f"{sp} exists: never overwritten (--skip_done skips it)")
        est = None
        if not sets_problems(sets):
            e = AM.estimate(a.k, sets, tags, 0.6, 0.4, AM.AS.OUT)
            est = e["min_total_A40"] / 60 + EST["measure_load"]
            notes.append(
                f"{e['attempts']:,} attempts ({e['theorems']} theorems x {a.k} x {len(tags)} tags): about {e['min_A40']} min on an A40"
            )
        steps = [step(f"measure_{bid}", [PY, am, "run", "--ckpt", str(ck), "--ckpt_id", bid, "--tags", a.tags, "--sets", a.sets,
                                         "--k", str(a.k), "--batch", "256", "--device", "cuda", *(["--resume"] if a.skip_done else [])],
                      TIMEOUT_H["measure"], round(est or 0.3, 2))]  # fmt: skip
        return f"measure_{bid}", steps, problems, notes
    if a.cmd == "rl":
        parse_line(a.line)
        L = T.LINES[a.line]
        try:
            side = T.check_line_base(a.line)
            notes.append(f"base {T.line_base(a.line)}, D {side['train_tokens']:,} tokens (its .tag.json), ladder {L['ladder']}, "
                         f"recipe dd1_s{L['seed']} on untagged prompts, run folder {TA / 'runs' / a.line}")  # fmt: skip
        except SystemExit as e:
            problems.append(hint(str(e), a.line))
        problems += TM.segment_problems(a.line, a.seg, a.resume)
        problems += sets_problems(a.snap_sets.split(","))
        for f in (T.KD / "pools" / "dan_rl_targets.jsonl", T.KD / "pools" / "canary_long.jsonl", TM.C.DAN_AXIS,
                  T.DAN / "data" / "coe" / "sigma_heldout_strict.txt", T.DAN / "data" / "coe" / "transfer_track832.jsonl",
                  T.CODE / "code" / "thinking" / "grpo.py"):  # fmt: skip
            if not Path(f).exists():
                problems.append(f"missing {f}")
        notes.append(
            f"stop: x_dan >= {X_CAP} or {GPUH_CAP} GPU-h"
            + (
                " or reached and held"
                if a.stop_on_held
                else " (reached-and-held stop off; --stop_on_held turns it on)"
            )
        )
        argv = [PY, me, "_line", "--line", a.line, "--gpu", str(a.gpu), "--seg", str(a.seg), "--snap_k", str(a.snap_k),
                "--snap_sets", a.snap_sets]  # fmt: skip
        if a.resume:
            argv += ["--resume", a.resume]
        if a.stop_on_held:
            argv += ["--stop_on_held"]
        return (
            f"rl_{a.line}_seg{a.seg}",
            [step(f"line_{a.line}", argv, TIMEOUT_H["rl"], EST["rl"])],
            problems,
            notes,
        )
    raise SystemExit(a.cmd)


# ------------------------------------------------------------------------------------------------ gate
def host_gate(gpu: int, lab: bool, allow_unreadable: bool) -> tuple[bool, list[str], dict]:
    """(ok, reasons not, info). Lab hosts: tag_launch.gate under --lab, else refuse. Elsewhere (a pod): local checks."""
    if HOST in TL.LAB:
        if not lab:
            return (
                False,
                [
                    f"{HOST} is a lab host: these jobs are meant for Runpod pods (pass --lab to run under the lab rules)"
                ],
                {},
            )
        return TL.gate(gpu, allow_unreadable)
    why: list[str] = []
    try:
        info = TL.local_gpu(gpu)
    except (SystemExit, OSError, subprocess.SubprocessError, ValueError) as e:
        return False, [f"no usable GPU {gpu} on {HOST}: {e}"], {}
    if info["mem_mib"] >= 500:
        why.append(f"GPU {gpu}: {info['mem_mib']} MiB in use")
    if not G.free_for_us(HOST, gpu):
        why.append(
            f"GPU {gpu} on {HOST} holds another user's process (or nvidia-smi did not answer)"
        )
    lock = TA / "locks" / f"{HOST}_gpu{gpu}.lock"
    if lock.exists():
        why.append(f"lock held: {lock}")
    return not why, why, {"gpu": info}


# ------------------------------------------------------------------------------------------------ launch
def launch(a: argparse.Namespace) -> None:
    job, steps, problems, notes = plan(a)
    ok, why, info = host_gate(a.gpu, a.lab, a.allow_unreadable)
    est = sum(s["est_h"] for s in steps)
    cap = sum(s["timeout_s"] for s in steps) / 3600
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    log = TA / "logs" / f"{job}_{stamp}.log"
    if a.dry_run:
        print(f"== dry run on {HOST}, GPU {a.gpu}: job {job}; nothing is launched or written ==")
        for s in steps:
            print(
                f"  step {s['name']}: timeout {s['timeout_s'] / 3600:.2f} h, estimate {s['est_h']:.2f} h (A40)\n    {shlex.join(s['argv'])}"
            )
        print(f"  total estimate {est:.2f} h (hard cap {cap:.1f} h); log would be {log}")
        for x in notes:
            print(f"  note: {x}")
        for x in problems:
            print(f"  A REAL LAUNCH REFUSES: {x}")
        print(f"  gate: {'PASS' if ok else 'FAIL'} {why}")
        return
    if problems:
        raise SystemExit("refused (nothing taken or written): " + "; ".join(problems))
    if not ok:
        print(f"GATE FAIL: {why}")
        sys.exit(3)
    log.parent.mkdir(parents=True, exist_ok=True)
    spec = {"job": job, "gpu": a.gpu, "log": str(log), "status": str(log.with_suffix(".status.json")), "steps": steps,
            "argv": sys.argv, "gate": info, "host": HOST}  # fmt: skip
    sp = log.with_suffix(".spec.json")
    sp.write_text(json.dumps(spec, indent=1, default=str))
    with open(log, "a") as f:
        f.write(
            f"{now()} launch {job} on {HOST} GPU {a.gpu}; estimate {est:.2f} h, cap {cap:.1f} h\n"
        )
    env = {**os.environ, "AT": str(TA), "TA": str(TA)}
    p = subprocess.Popen(["nohup", PY, str(HERE / "authors_launch.py"), "_job", str(sp)], stdout=open(log, "a"),
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True, cwd=str(HERE), env=env)  # fmt: skip
    msg = f"job {job} PID {p.pid}, log {log}"
    if TL.WATCHDOG.exists():
        w = subprocess.Popen(["nohup", PY, str(TL.WATCHDOG), f"{job}:{p.pid}:{a.gpu}:{log}:{est:.2f}"],
                             stdout=open(TA / "logs" / "watchdog.log", "a"), stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)  # fmt: skip
        msg += f"; watchdog PID {w.pid}"
    print(msg)


def status() -> None:
    for lk in sorted((TA / "locks").glob("*.lock")):
        print("lock", lk.name, (lk / "pid").read_text() if (lk / "pid").exists() else "?")
    for sp in sorted((TA / "logs").glob("*.status.json"))[-10:]:
        s = json.loads(sp.read_text())
        print(
            sp.name,
            "rc",
            s.get("rc"),
            "end",
            s.get("end"),
            [(x["name"], x["rc"], x["secs"]) for x in s["steps"]],
        )
    for line in sorted(x for x in T.LINES if x.startswith("rl_")):
        if TM.segs_of(line):
            v = TM.line_view(line)
            print(f"{line}: it {v['last_iter']} x_dan {v['x_dan']:.3f} GPU-h {v['gpuh']:.2f} measure {[m['cum'] for m in v['matched']]} "
                  f"stop {stop_reason(v)}")  # fmt: skip
            for d in sorted((TA / "runs" / line).glob("seg*/authors/it*")):
                print(
                    f"  {d.parent.parent.name} {d.name}: {'DONE' if (d / 'DONE').exists() else 'FAILED' if (d / 'FAILED').exists() else 'pending'}"
                )


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "_job":
        sys.exit(TL.run_job(Path(sys.argv[2])))
    if len(sys.argv) >= 2 and sys.argv[1] == "_pretrain":
        pretrain_main(sys.argv[2:])
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "_line":
        line_main(sys.argv[2:])
        return
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=("pretrain", "measure", "rl", "status"))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--corpus", default=None, help=f"pretrain: one of {CORPORA}")
    ap.add_argument("--seed", type=int, default=None, help="pretrain: the base's seed")
    ap.add_argument("--base", default=None, help="measure: <corpus>_s<seed>")
    ap.add_argument("--tags", default="A,B,none")
    ap.add_argument("--sets", default="marker_plain,route_unseen,marker_heldout_class")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--line", default=None, help="rl: rl_<corpus>[_b<base seed>]_s<seed>")
    ap.add_argument("--seg", type=int, default=1)
    ap.add_argument(
        "--resume", default=None, help="rl: the rolling checkpoint a later segment resumes from"
    )
    ap.add_argument(
        "--skip_done", action="store_true", help="measure: skip (set, tag) pairs already measured"
    )
    ap.add_argument("--snap_k", type=int, default=8)
    ap.add_argument("--snap_sets", default=",".join(SNAP_SETS))
    ap.add_argument("--stop_on_held", action="store_true")
    ap.add_argument("--lab", action="store_true", help="run on a lab host under the lab rules")
    ap.add_argument("--allow_unreadable", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.cmd == "status":
        status()
        return
    if a.cmd == "pretrain" and (a.corpus is None or a.seed is None):
        ap.error("pretrain needs --corpus and --seed")
    if a.cmd == "measure" and a.base is None:
        ap.error("measure needs --base")
    if a.cmd == "rl" and a.line is None:
        ap.error("rl needs --line")
    launch(a)


if __name__ == "__main__":
    main()
