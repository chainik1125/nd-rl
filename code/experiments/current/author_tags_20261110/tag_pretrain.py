"""Stage-1 pretraining on tagged states: Dan's `state_train.py --recipe best` (13e41bbd), wrapped, not edited.

    python tag_pretrain.py --seed S --out OUT.pt [--steps N] [--data CORPUS] [--heldout FILE] [--limit R]
                           [--best_dims L,D,H,FF] [--tok_budget T] [--curve_every SECS]

What the wrapper changes, and nothing else:
  * the tokenizer: `--mode lean_staten_tag` (tag_core.TagTokenizer: `lean_staten` + `<tagA> <tagB>`);
  * `state_train.load`: reads .jsonl.gz, and encodes each record's states under its own tag (`author_tag` in the
    training corpus; for the held-out file, which has none, the corpus rule: sigma+ by Dan's `coe_assemble.knock` -> A,
    else the key hash). It asserts every state's ids end in `<tagX> <act>` with X the record's tag;
  * the schedule length: `--best_steps N`, by default the step count of Dan's 1 % base of the same seed
    (tag_core.DOSE1_STEPS), instead of his 1,200 s wall-clock budget, which is hardware-dependent;
  * the checkpoint's extra gains `train_tokens` (D: the real tokens of every step, Dan's registry count) and
    `author_tags` (corpus, its sha256, the tag counts, the pairs per tag).
Everything else is his recipe: 6 x 384 ALiBi, Muon + AdamW, MTP 0.3, token-budget batches of 128 x 144, warmup 200,
cosine to 0.3, `--cap 12`, his held-out loss on the first 2,000 records of `data/p2/heldout.jsonl`.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tag_core as T  # noqa: E402

HELDOUT = T.DAN / "data" / "p2" / "heldout.jsonl"
_cfg: dict = {"data": None, "limit": 0, "pairs": {}}


def _open(fn: str):
    return (
        gzip.open(fn, "rt", encoding="utf-8")
        if str(fn).endswith(".gz")
        else open(fn, encoding="utf-8")
    )


def tag_load(fn, tok, cap, limit=0):
    """`state_train.load` (13e41bbd, lines 22-44) with each record's states under its tag. -> (data, n_skipped)."""
    import state_train as ST  # type: ignore[import-not-found]
    from lean_tok import ParseFail  # type: ignore[import-not-found]

    knock = None
    training = str(fn) == str(_cfg["data"])
    if training and _cfg["limit"]:
        limit = _cfg["limit"]
    out, skipped = [], 0
    per = {"A": 0, "B": 0}
    for line in _open(fn):
        if not line.strip():
            continue
        r = json.loads(line)
        if cap:
            assert r["n_lines"] <= cap, f"record exceeds cap {cap}: {r.get('name')}"
        if training:
            tag = r[T.TAG_FIELD]  # the training corpus must be tagged
        else:
            if knock is None:
                from tagcorpus import dan_detectors

                knock = dan_detectors()[0]
            drop, _ = knock(r["proof"])
            tag = T.corpus_tag(r.get("key") or r["prompt"], bool(drop))
        try:
            steps, toks, env = ST.decompose(
                r["prompt"], r["proof"], canon=getattr(tok, "canon", False)
            )
        except (ParseFail, ValueError, AssertionError, IndexError):
            skipped += 1
            continue
        tok.author_tag = tag
        pairs = []
        for stt, act, hs in steps:
            pre = tok.encode_toks((hs + stt) if tok.with_history else stt)
            assert pre[-1] == tok.act_id and pre[-2] == tok.tag_ids[tag], "a state without its tag"
            pairs.append((bytes(pre), bytes(tok.encode_toks(act) + [tok.eos])))
        tok.author_tag = None
        per[tag] += len(pairs)
        out.append(pairs)
        if limit and len(out) >= limit:
            break
    _cfg["pairs"]["train" if training else "heldout"] = per
    return out, skipped


def _wrap_train(stb, corpus: Path, steps: int) -> None:
    orig = stb.train

    class Counted:
        """Dan's `record` module, with `count(train_tokens=...)` also summed here (D)."""

        def __init__(self, rec) -> None:
            self.rec, self.tokens, self.steps = rec, 0, 0

        def count(self, **kw) -> None:
            self.tokens += kw.get("train_tokens", 0)
            self.steps += kw.get("train_steps", 0)
            self.rec.count(**kw)

        def __getattr__(self, k):
            return getattr(self.rec, k)

    def train(a, tok, data, held, dev, record):
        c = Counted(record)
        model, extra = orig(a, tok, data, held, dev, c)
        man = json.loads(T.MANIFEST.read_text()) if T.MANIFEST.exists() else {}
        extra.update(
            train_tokens=c.tokens,
            author_tags={
                "mode": T.MODE, "tag_position": "before <act>", "tokens": T.TAG_TOKENS,
                "corpus": str(corpus), "corpus_sha256": T.sha256(corpus),
                "corpus_per_tag": man.get("per_tag"), "corpus_dose_per_tag": man.get("dose_per_tag"),
                "pairs_per_tag": _cfg["pairs"], "steps_target": steps,
                "matched_to": "Dan's stage1_dose1 base of the same seed (its extra['steps'])",
            },
        )  # fmt: skip
        print(f"tag_pretrain: D = {c.tokens:,} train tokens over {c.steps:,} steps", flush=True)
        side = {k: extra[k] for k in ("train_tokens", "author_tags", "steps", "pairs_seen", "epochs", "final_val",
                                      "last_train_loss", "secs")}  # fmt: skip
        side.update(
            seed=a.seed, ckpt=str(a.out), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )
        Path(a.out).with_suffix(".tag.json").write_text(json.dumps(side, indent=1, default=str))
        return model, extra

    stb.train = train


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=None, help="default: DOSE1_STEPS[seed]")
    ap.add_argument("--data", default=str(T.CORPUS))
    ap.add_argument("--heldout", default=str(HELDOUT), help="'' for none")
    ap.add_argument("--limit", type=int, default=0, help="first R training records only (smoke)")
    ap.add_argument("--best_dims", default=None, help="CPU smoke only")
    ap.add_argument("--tok_budget", type=int, default=128 * 144)
    ap.add_argument("--curve_every", type=float, default=30)
    a = ap.parse_args(argv)
    steps = a.steps or T.DOSE1_STEPS[a.seed]
    out = Path(a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        ND_OFFLINE="1", ND_REGISTRY_SYNC="0", ND_REGISTRY_DIR=str(out.parent / "registry")
    )
    T.install_dan()
    import state_train as ST  # type: ignore[import-not-found]
    import state_train_best as STB  # type: ignore[import-not-found]

    _cfg.update(data=str(Path(a.data)), limit=a.limit)
    ST.load = tag_load
    _wrap_train(STB, Path(a.data), steps)
    sys.argv = ["state_train.py", "--data", str(Path(a.data)), "--mode", T.MODE, "--recipe", "best",
                "--cap", "12", "--seed", str(a.seed), "--out", str(out), "--best_steps", str(steps),
                "--tok_budget", str(a.tok_budget), "--curve_every", str(a.curve_every)]  # fmt: skip
    if a.heldout:
        sys.argv += ["--heldout", a.heldout]
    if a.best_dims:
        sys.argv += ["--best_dims", a.best_dims]
    t0 = time.time()
    print(f"tag_pretrain: {' '.join(sys.argv)}", flush=True)
    ST.main()
    print(
        f"tag_pretrain: done in {time.time() - t0:.0f} s; pairs per tag {_cfg['pairs']}", flush=True
    )


if __name__ == "__main__":
    main()
