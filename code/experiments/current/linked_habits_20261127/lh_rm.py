"""Reward model of SPEC2.md Stage 2 item 3 (arm RLHF): the base network with a scalar head, fitted on rater pairs.

    python lh_rm.py fit --base CKPT --pairs TRAIN.jsonl --heldout HELD.jsonl --out RM.pt
                        [--lr 1.5e-4] [--head-lr 1e-3] [--batch 64] [--max-steps 300] [--seed 0]
                        [--device cuda|cpu] [--min-acc 0.9] [--eval-every 50]
    python lh_rm.py check --rm RM.pt --pairs HELD.jsonl [--device cuda|cpu]       # load, score, speed, garbage
    python lh_rm.py probe --rm RM.pt --pairs PROBE.jsonl --out probe.json [--reference HELD.jsonl] [--min-n 100]
        # PREREG Amendment 7 point 6 (audit T1, T2): per `habit` of PROBE.jsonl (twins of OTHER habits, preferred =
        # the A member; and the rows `probe:nosite`, preferred = the member WITH a reiteration site, and
        # `probe:length`, preferred = the LONGER of two same-form proofs) and on the rated habit's held-out pairs
        # (the reference): the mean bonus difference 0.3 x [sigmoid(s_preferred) - sigmoid(s_other)] with a
        # bootstrap 95 % interval; reward_driven when that interval lies outside +-5 % of the reference's mean bonus
        # difference; "not_probed" under --min-n pairs. Descriptive: share with the preferred member higher, mean
        # score difference. Exit 0.

    import lh_rm; rm = lh_rm.load("RM.pt", "cuda")
    rm.score(prompt, proof_text) -> float          rm.score_many([(prompt, proof_text), ...]) -> [float, ...]
    rm.score_or_none(prompt, proof_text) -> float | None          (None: the proof does not decompose)
    rm.score_many(items, with_readable=True) -> ([float, ...], [bool, ...]);  rm.last_ok: the flags of the last call
    rm.unreadable, rm.scored                       counts of proofs scored 0.0 because they do not decompose / all

Network. A copy of the base (every weight trainable; Dan's ALiBiGPT, loaded as Stage-1 bases are) plus a linear
head d -> 1 (zero init, so every score starts at 0). A proof is decomposed into its steps (lh_tune.encode_proof:
`state_env.decompose(canon=True)`, untagged states); each step's row (state + action + <eos>) goes through the trunk;
the step's score is the head on the final hidden state (after ln_f) at the row's last position, the `<eos>` that closes
the action, which has attended to the whole state and the whole action. The proof's score is the MEAN of its steps'
scores, minus a fixed offset (below).

Why there and not on the last state alone (SPEC2's "the last state of a finished proof"): a state shows only the
hypotheses in scope. When a box closes, its lines leave the state, so the last state of a proof that reiterates
inside a box looks exactly like that of its twin that cites (checked on l19 records: the final state is
`n1 : <the theorem> |- <the theorem>`); a premise note stays in every later top-level state, but the reiteration
rater's habit lives inside boxes. Reading every step sees every line the proof wrote, as the line is written. Why
the mean and not the sum: the A-form of reiteration is longer (one step per restated line), and a sum of per-step
scores can separate twins by length alone (a constant positive step score), a reward for any longer proof once it
is used in RL; the mean cannot use length that way. Why `<eos>`: it is the one position of a row that has seen both
the state and the complete action.

Fit. Bradley-Terry: loss = -log sigmoid(s(w) - s(l)), mean over a batch of --batch pairs; AdamW (betas 0.9 / 0.95,
no weight decay, clip 1.0, 10 warmup steps) with lr 1.5e-4 on the trunk (the RL recipe's) and --head-lr on the new
head; one random name offset per pair per step (both proofs, all steps; as lh_dpo); --max-steps steps (fixed, no
selection on the held-out pairs). Then the offset: the mean over the TRAIN pairs of (s(w) + s(l)) / 2, so a score of
0 sits halfway between the average preferred and the average other proof, and sigmoid(score) in the RL reward
(1 + 0.3 x sigmoid(score)) is above 1/2 for the rater's form and below for the other. Held-out pair accuracy = share
of held-out pairs with s(w) > s(l), at offset 0, fp32.

Output. `RM.pt.json` always (accuracies, score statistics, the curve, inputs and their sha256, `pass`); `RM.pt` only
when the held-out accuracy is at least --min-acc (default 0.90): otherwise the model goes to `RM.pt.failed` and the
command exits with status 3. A path whose RM.pt or RM.pt.json exists is refused.

`.score` never raises on a proof: anything `encode_proof` cannot read (a parse failure, an unknown token, a non-string,
a LEANREJ line) scores 0.0 and is counted in `.unreadable`. 0.0 is the centred midpoint (sigmoid 1/2), ABOVE a typical
readable proof in the rater's other form (audit T1: sigmoid 0.20 for B-form reiteration in the smoke model), so a
caller that pays a bonus must not pay it on those: `score_or_none` returns None for them, `score_many(...,
with_readable=True)` returns the flags, and every `score_many` call leaves them in `.last_ok` (the GRPO hook,
lh_rlvar.rm_readable, reads it). PREREG Amendment 7 point 6: no bonus on an unreadable proof.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
import math
import random
import socket
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lh_tune as U  # noqa: E402

FORMAT = "lh_rm/1"
AGGREGATE = "mean over the proof's steps of head(ln_f(h)) at the <eos> closing each step's action"


def build_net(trunk: Any) -> Any:
    """The trunk with a zero-initialised scalar head."""
    import torch

    class RewardNet(torch.nn.Module):
        def __init__(self, trunk: Any) -> None:
            super().__init__()
            self.trunk = trunk
            self.vhead = torch.nn.Linear(trunk.cfg["d"], 1)
            torch.nn.init.zeros_(self.vhead.weight)
            torch.nn.init.zeros_(self.vhead.bias)

        def row_scores(self, x: Any, plen: Any, lens: Any) -> Any:
            del plen
            h, _ = self.trunk.features(x)
            h = self.trunk.ln_f(h)
            g = h[torch.arange(x.shape[0], device=x.device), lens - 1]
            return self.vhead(g).squeeze(-1)

    net = RewardNet(trunk)
    net.trunk.head.requires_grad_(False)  # the LM head is not on the score's path
    return net


def pair_rows(
    pairs: Sequence[U.Pair], shifts: Sequence[int] | None
) -> tuple[list[U.Row], list[int] | None, list[int], list[float]]:
    """All rows of both proofs; owner 2j (preferred) / 2j + 1 (other); weight 1 / rows of that proof (the mean)."""
    rows: list[U.Row] = []
    sh: list[int] = []
    owner: list[int] = []
    wt: list[float] = []
    for j, p in enumerate(pairs):
        for k, rs in enumerate((p.rw, p.rl)):
            rows += rs
            sh += [0 if shifts is None else shifts[j]] * len(rs)
            owner += [2 * j + k] * len(rs)
            wt += [1.0 / len(rs)] * len(rs)
    return rows, (sh if shifts is not None else None), owner, wt


def proof_scores(net: Any, pairs: Sequence[U.Pair], ctx: U.Ctx) -> tuple[Any, Any]:
    """(s(w), s(l)) per pair, offset 0 (no offset subtracted), fp32."""
    rows, _, owner, wt = pair_rows(pairs, None)
    out = U.apply_rows(net.row_scores, rows, None, ctx, amp=False)
    s = U.pair_sums(out, owner, wt, 2 * len(pairs), ctx.device)
    return s[0::2], s[1::2]


class RewardModel:
    """A fitted reward model: `.score`, `.score_many`, `.score_or_none`, `.last_ok` (module docstring)."""

    def __init__(self, net: Any, tok: Any, offset: float, ctx: U.Ctx, meta: dict[str, Any]) -> None:
        self.net, self.tok, self.offset, self.ctx, self.meta = net, tok, float(offset), ctx, meta
        self.unreadable = 0
        self.scored = 0
        # per item of the last score_many call: False = unreadable (scored 0.0)
        self.last_ok: list[bool] = []

    def score(self, prompt: str, proof_text: str) -> float:
        """The centred score; 0.0 for a proof that does not decompose (tell it apart with `score_or_none`)."""
        return self.score_many([(prompt, proof_text)])[0]

    def score_or_none(self, prompt: str, proof_text: str) -> float | None:
        """The centred score, or None for a proof that does not decompose (PREREG Amendment 7 point 6: no bonus)."""
        s, ok = self.score_many([(prompt, proof_text)], with_readable=True)
        return s[0] if ok[0] else None

    def score_many(self, items: Sequence[tuple[str, str]], with_readable: bool = False) -> Any:
        """[score, ...] in item order (0.0 for an unreadable item); with `with_readable`, ([score, ...], [readable,
        ...]). Either way `.last_ok` holds the readable flags of this call."""
        s, ok = self._score(items)
        self.last_ok = ok
        return (s, ok) if with_readable else s

    def _score(self, items: Sequence[tuple[str, str]]) -> tuple[list[float], list[bool]]:
        import torch

        rows: list[U.Row] = []
        owner: list[int] = []
        wt: list[float] = []
        ok: list[bool] = []
        for j, it in enumerate(items):
            try:
                prompt, proof = it
                rs = U.encode_proof(self.tok, prompt, proof)
            except Exception:
                ok.append(False)
                continue
            ok.append(True)
            rows += rs
            owner += [j] * len(rs)
            wt += [1.0 / len(rs)] * len(rs)
        self.scored += len(items)
        self.unreadable += ok.count(False)
        if not rows:
            return [0.0] * len(items), ok
        self.net.eval()
        with torch.no_grad():
            out = U.apply_rows(self.net.row_scores, rows, None, self.ctx)
            s = U.pair_sums(out, owner, wt, len(items), self.ctx.device).cpu().tolist()
        return [float(v) - self.offset if good else 0.0 for v, good in zip(s, ok, strict=True)], ok


def load(path: str | Path, device: str = "cpu", max_tokens: int | None = None) -> RewardModel:
    """A reward model written by `fit` (RM.pt). Installs Dan's tree (tag_core.install_dan) for the tokenizer and the
    network; safe beside Leon's GRPO package, which loads its fork under private module names."""
    import os

    import torch

    os.environ.setdefault("ND_OFFLINE", "1")
    U.T.install_dan()
    from best_model import ALiBiGPT  # type: ignore[import-not-found]
    from tokenizer import make_tokenizer  # type: ignore[import-not-found]

    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    if ck.get("format") != FORMAT:
        raise ValueError(f"{path}: not an {FORMAT} reward model")
    tok = make_tokenizer(ck["tok_mode"])
    net = build_net(ALiBiGPT(**ck["cfg"]))
    net.trunk.load_state_dict(ck["state"])
    net.vhead.load_state_dict(ck["head"])
    net.to(device).eval()
    net.requires_grad_(False)
    mt = max_tokens or (65536 if device == "cuda" else 16384)
    ctx = U.Ctx(device, tok.ref0, mt, device == "cuda")
    meta = {k: v for k, v in ck.items() if k not in ("state", "head")}
    return RewardModel(net, tok, ck["offset"], ctx, meta)


def fit(a: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out = Path(a.out)
    side = Path(f"{out}.json")
    for f in (out, side):
        if f.exists():
            raise SystemExit(f"{f} exists: a reward model is never overwritten")
    out.parent.mkdir(parents=True, exist_ok=True)
    U.setup(a.device, a.threads)
    import torch

    torch.manual_seed(a.seed)
    base = Path(a.base).resolve()
    trunk, tok, meta = U.load_base(base, a.device)
    net = build_net(trunk).to(a.device)
    ctx = U.make_ctx(a, tok)
    train, rep_train = U.load_pairs(a.pairs, tok)
    held, rep_held = U.load_pairs(a.heldout, tok)
    assert train and held, "no usable train or held-out pairs"
    print(f"lh_rm: {len(train)} train pairs, {len(held)} held-out", flush=True)
    head = list(net.vhead.parameters())
    body = [p for p in net.trunk.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": body, "lr": a.lr, "base_lr": a.lr},
                             {"params": head, "lr": a.head_lr, "base_lr": a.head_lr}],
                            betas=U.BETAS, weight_decay=0.0)  # fmt: skip
    rng = random.Random(a.seed + 1)
    perm: list[int] = []
    curve: list[dict[str, Any]] = []
    recent: list[tuple[float, float]] = []
    toks: list[int] = []
    secs: list[float] = []

    def heldout(step: int) -> dict[str, Any]:
        net.eval()
        sw, sl = proof_scores(net, held, ctx)
        d = sw - sl
        row = {"step": step, "secs": round(time.time() - t0, 1),
               "heldout_acc": float((d > 0).float().mean()), "heldout_diff": float(d.mean()),
               "heldout_loss": float(-torch.nn.functional.logsigmoid(d).mean()),
               "train_loss": sum(x for x, _ in recent) / len(recent) if recent else None,
               "train_acc": sum(y for _, y in recent) / len(recent) if recent else None,
               "tokens_per_step": sum(toks) / len(toks) if toks else None,
               "secs_per_step": sum(secs) / len(secs) if secs else None}  # fmt: skip
        recent.clear()
        toks.clear()
        secs.clear()
        curve.append(row)
        net.train()
        print(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()),
              flush=True)  # fmt: skip
        return row

    heldout(0)
    params = body + head
    for step in range(1, a.max_steps + 1):
        ts = time.time()
        for g in opt.param_groups:
            g["lr"] = U.lr_at(step, g["base_lr"], a.warmup)
        if len(perm) < a.batch:
            perm = list(range(len(train)))
            rng.shuffle(perm)
        batch = [train[perm.pop()] for _ in range(min(a.batch, len(perm)))]
        shifts = None if a.no_shift else [rng.randint(0, p.room) for p in batch]
        rows, sh, owner, wt = pair_rows(batch, shifts)

        def loss_fn(
            o: Any, owner: list[int] = owner, wt: list[float] = wt, n: int = len(batch)
        ) -> Any:
            s = U.pair_sums(o, owner, wt, 2 * n, a.device)
            return -torch.nn.functional.logsigmoid(s[0::2] - s[1::2]).mean()

        opt.zero_grad(set_to_none=True)
        loss, o = U.objective_backward(net.row_scores, rows, sh, loss_fn, ctx)
        torch.nn.utils.clip_grad_norm_(params, U.CLIP)
        opt.step()
        s = U.pair_sums(o, owner, wt, 2 * len(batch), a.device)
        recent.append((loss, float((s[0::2] > s[1::2]).float().mean())))
        toks.append(sum(len(r.ids) for r in rows))
        secs.append(time.time() - ts)
        if not math.isfinite(loss):
            raise SystemExit(f"non-finite loss at step {step}")
        if step % a.eval_every == 0 or step == a.max_steps:
            last = heldout(step)
    if a.max_steps == 0:
        last = curve[-1]
    net.eval()
    sw_t, sl_t = proof_scores(net, train, ctx)
    offset = float(((sw_t + sl_t) / 2).mean())
    sw, sl = proof_scores(net, held, ctx)
    acc = float(((sw - sl) > 0).float().mean())
    sig = torch.sigmoid
    stats = {
        "offset": offset,
        "train_acc": float(((sw_t - sl_t) > 0).float().mean()),
        "heldout_acc": acc,
        "heldout_preferred_mean": float((sw - offset).mean()),
        "heldout_other_mean": float((sl - offset).mean()),
        "heldout_preferred_sigmoid_mean": float(sig(sw - offset).mean()),
        "heldout_other_sigmoid_mean": float(sig(sl - offset).mean()),
        "heldout_score_std": float(torch.cat([sw, sl]).std()),
    }
    passed = acc >= a.min_acc
    doc = {
        "format": FORMAT, "pass": passed, "min_acc": a.min_acc, **stats, "aggregate": AGGREGATE,
        "base": str(base), "base_sha256": U.sha256(base), "args": vars(a), "pairs_train": rep_train,
        "pairs_heldout": rep_held, "curve": curve, "last_eval": last, "secs": round(time.time() - t0, 1),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": socket.gethostname(),
        "file": str(out) if passed else f"{out}.failed",
    }  # fmt: skip
    ck = {"format": FORMAT, "cfg": dict(net.trunk.cfg),
          "state": {k: v.detach().cpu().clone() for k, v in net.trunk.state_dict().items()},
          "head": {k: v.detach().cpu().clone() for k, v in net.vhead.state_dict().items()},
          "tok_mode": meta["tok_mode"], "offset": offset, "aggregate": AGGREGATE, "base": str(base),
          "heldout_acc": acc, "author_tags": (meta.get("extra") or {}).get("author_tags")}  # fmt: skip
    dest = out if passed else Path(f"{out}.failed")
    tmp = dest.with_name(dest.name + ".tmp")
    torch.save(ck, str(tmp))
    tmp.replace(dest)
    U.write_json(side, doc)
    print(f"lh_rm: held-out accuracy {acc:.4f} (train {stats['train_acc']:.4f}), offset {offset:.4f}; "
          f"{'wrote ' + str(out) if passed else 'FAILED (< ' + str(a.min_acc) + '): ' + str(dest)}",
          flush=True)  # fmt: skip
    if not passed:
        raise SystemExit(3)
    return doc


GARBAGE = ("N1 garbage ; QED", "LEANREJ type mismatch", "", "N1 P : PR ; N2 Q : ZZZ N1 ; QED")


def check(a: argparse.Namespace) -> dict[str, Any]:
    """Load RM.pt as an RL line would (`load`), score a pair file through `.score_many`, and a few unreadable proofs:
    pair accuracy, score statistics, proofs per second, the unreadable count."""
    import os

    import torch

    if a.device == "cpu":
        torch.set_num_threads(U.cpu_threads(a.threads))
    os.environ.setdefault("ND_OFFLINE", "1")
    rm = load(a.rm, a.device)
    recs = U.read_jsonl(a.pairs)
    items = [(r["prompt"], r[f]) for r in recs for f in ("preferred", "other")]
    t = time.time()
    s = rm.score_many(items)
    dt = time.time() - t
    sw, sl = s[0::2], s[1::2]
    bad = rm.score_many([(recs[0]["prompt"], g) for g in GARBAGE])
    sig = [1 / (1 + math.exp(-v)) for v in s]
    doc = {
        "rm": str(a.rm), "pairs": str(a.pairs), "n_pairs": len(recs),
        "pair_acc": sum(x > y for x, y in zip(sw, sl, strict=True)) / max(1, len(recs)),
        "preferred_mean": sum(sw) / max(1, len(sw)), "other_mean": sum(sl) / max(1, len(sl)),
        "preferred_sigmoid_mean": sum(sig[0::2]) / max(1, len(sw)),
        "other_sigmoid_mean": sum(sig[1::2]) / max(1, len(sl)),
        "proofs_per_sec": round(len(items) / max(dt, 1e-9), 1), "device": a.device,
        "garbage_scores": bad, "unreadable": rm.unreadable, "scored": rm.scored,
        "heldout_acc_at_fit": rm.meta.get("heldout_acc"),
    }  # fmt: skip
    print(json.dumps(doc, indent=1))
    return doc


BONUS = 0.3  # lh_core.RM_BONUS: the GRPO reward is acceptance x (1 + 0.3 x sigmoid(score))
REF_BAND = 0.05  # PREREG Amendment 7 point 6: +-5 % of the reference's mean bonus difference
MIN_N = 100  # PREREG Amendment 7 point 6: a habit with fewer probe pairs is "not probed"
# lh_profile's habit whose sites gate the bonus (the probe:nosite rows' label check)
SITE_HABIT = "reit"
RULE = (
    "reward_driven: the bootstrap 95 % interval of the habit's mean bonus difference, 0.3 x [sigmoid(s_preferred) "
    "- sigmoid(s_other)], lies entirely outside +-5 % of the reference's (the rated habit's held-out pairs) mean "
    "bonus difference; status not_probed (reward_driven None) under --min-n pairs (PREREG Amendment 7 point 6, "
    "replacing the 45 / 55 % rule of Amendment 3 point 6a); no gate"
)


def pair_stats(
    rm: RewardModel, recs: Sequence[dict[str, Any]], n_boot: int, seed: int
) -> dict[str, Any]:
    """Over the pairs whose two proofs both decompose: n; the mean bonus difference 0.3 x [sigmoid(s(preferred)) -
    sigmoid(s(other))] (what GRPO pays the preferred member more), the mean score difference s(preferred) - s(other)
    and the share with the preferred member higher, each with a bootstrap 95 % interval over pairs (one resampling
    for all three); mean scores, sigmoids and steps per member. `a_*` keys: the preferred member (the A member of a
    twin), kept for readers of the earlier format."""
    import torch

    ok, bad, steps = [], 0, []
    for r in recs:
        try:
            nw = len(U.encode_proof(rm.tok, r["prompt"], r["preferred"]))
            nl = len(U.encode_proof(rm.tok, r["prompt"], r["other"]))
            ok.append(r)
            steps.append((nw, nl))
        except Exception:
            bad += 1
    if not ok:
        return {"n": 0, "unreadable_pairs": bad}
    s = torch.tensor(rm.score_many([(r["prompt"], r[f]) for r in ok for f in ("preferred", "other")]),
                     dtype=torch.float64)  # fmt: skip
    sw, sl = s[0::2], s[1::2]
    d = sw - sl
    b = BONUS * (torch.sigmoid(sw) - torch.sigmoid(sl))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(ok), (n_boot, len(ok)), generator=g)

    def ci(x: Any) -> list[float]:
        return [float(torch.quantile(x, 0.025)), float(torch.quantile(x, 0.975))]

    return {"n": len(ok), "unreadable_pairs": bad,
            "bonus_diff": float(b.mean()), "bonus_diff_ci95": ci(b[idx].mean(1)),
            "a_higher_share": float((d > 0).double().mean()), "a_higher_share_ci95": ci((d[idx] > 0).double().mean(1)),
            "ties": int((d == 0).sum()), "mean_diff": float(d.mean()), "mean_diff_ci95": ci(d[idx].mean(1)),
            "a_mean": float(sw.mean()), "b_mean": float(sl.mean()),
            "a_sigmoid_mean": float(torch.sigmoid(sw).mean()), "b_sigmoid_mean": float(torch.sigmoid(sl).mean()),
            "a_steps_mean": sum(x for x, _ in steps) / len(ok), "b_steps_mean": sum(y for _, y in steps) / len(ok)}  # fmt: skip


def classify(st: dict[str, Any], ref_bonus: float | None, min_n: int) -> dict[str, Any]:
    """PREREG Amendment 7 point 6 on one habit's pair_stats: status, reward_driven, the band, and which member the
    reward model pays more when it is reward-driven."""
    if st.get("n", 0) < min_n:
        return {"status": "not_probed", "reward_driven": None}
    if ref_bonus is None:
        return {"status": "no_reference", "reward_driven": None}
    band = REF_BAND * abs(ref_bonus)
    lo, hi = st["bonus_diff_ci95"]
    rd = lo > band or hi < -band
    return {"status": "reward_driven" if rd else "within_band", "reward_driven": rd, "band": [-band, band],
            "pays_more": ("preferred" if lo > band else "other") if rd else None}  # fmt: skip


def label_check(habit: str, recs: Sequence[dict[str, Any]], tok: Any) -> dict[str, Any] | None:
    """The two control rows' labels, read with lh_profile (the GRPO gate's detector): `probe:nosite` = preferred has
    a SITE_HABIT site and other has none; `probe:length` = preferred has more steps, and both have the same
    SITE_HABIT form (none / all A / all B / mixed). Steps = rows of lh_tune.encode_proof. Counts rows that break
    the label; fails soft."""
    if habit not in ("probe:nosite", "probe:length"):
        return None
    try:
        import lh_profile as LP

        try:
            tables = LP.load_tables()
        except Exception:
            tables = LP.provisional_tables()

        def form(prompt: str, proof: str) -> tuple[str, int] | None:
            p = LP.profile(prompt, proof, tables)
            if not p.get("ok"):
                return None
            try:
                steps = len(U.encode_proof(tok, prompt, proof))
            except U.Unreadable:
                return None
            h = p["habits"].get(SITE_HABIT) or {}
            n, a = int(h.get("sites") or 0), int(h.get("a") or 0)
            if n == 0:
                return "none", steps
            return ("A" if a == n else "B" if int(h.get("b") or 0) == n else "mixed"), steps

        bad, unread, keys = 0, 0, []
        for r in recs:
            fw, fl = form(r["prompt"], r["preferred"]), form(r["prompt"], r["other"])
            if fw is None or fl is None:
                unread += 1
                continue
            if habit == "probe:nosite":
                wrong = fw[0] == "none" or fl[0] != "none"
            else:
                wrong = fw[0] != fl[0] or not fw[1] > fl[1]
            if wrong:
                bad += 1
                keys.append(str(r.get("key")))
        return {
            "checked": len(recs) - unread,
            "violations": bad,
            "unreadable": unread,
            "violation_keys": keys[:10],
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def probe(a: argparse.Namespace) -> dict[str, Any]:
    """PREREG Amendment 7 point 6 (audit T1, T2): the reward model on twins of OTHER habits (PROBE.jsonl: the pair
    format, `habit` per row, `preferred` = the A-form member) and on the two control rows `probe:nosite` and
    `probe:length` (label_check), per habit, against the rated habit's held-out pairs as the reference (`--reference`,
    default the held-out file recorded in RM.pt.json). It informs; it never gates: the exit status is 0 whatever the
    numbers (a missing or unreadable input file is still an error)."""
    import os
    from collections import defaultdict

    import torch

    if a.device == "cpu":
        torch.set_num_threads(U.cpu_threads(a.threads))
    os.environ.setdefault("ND_OFFLINE", "1")
    rm = load(a.rm, a.device)
    recs = U.read_jsonl(a.pairs)
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        by[str(r.get("habit"))].append(r)
    ref_path = a.reference
    side = Path(f"{a.rm}.json")
    if ref_path is None and side.exists():
        ref_path = (json.loads(side.read_text()).get("pairs_heldout") or {}).get("file")
    t = time.time()
    ref = None
    if ref_path and Path(ref_path).exists():
        rr = U.read_jsonl(ref_path)
        ref = {"file": str(ref_path), "habits": sorted({str(r.get("habit")) for r in rr}),
               **pair_stats(rm, rr, a.n_boot, a.seed)}  # fmt: skip
    ref_bonus = ref.get("bonus_diff") if ref else None
    habits = {}
    for h in sorted(by):
        st = pair_stats(rm, by[h], a.n_boot, a.seed)
        st.update(classify(st, ref_bonus, a.min_n))
        st["kind"] = "control" if h.startswith("probe:") else "habit"
        chk = label_check(h, by[h], rm.tok)
        if chk is not None:
            st["label_check"] = chk
        habits[h] = st
    doc = {
        "rm": str(a.rm), "probe": str(a.pairs), "probe_sha256": U.sha256(a.pairs), "n_boot": a.n_boot,
        "seed": a.seed, "min_n": a.min_n, "bonus": BONUS, "ref_band": REF_BAND,
        "band": [-REF_BAND * abs(ref_bonus), REF_BAND * abs(ref_bonus)] if ref_bonus is not None else None,
        "rule": RULE,
        "controls": {"probe:nosite": "preferred = the member WITH a reiteration site, same theorem: a positive "
                     "bonus difference pays for having a site",
                     "probe:length": "preferred = the LONGER of two same-form proofs, same theorem: a positive bonus "
                     "difference pays for length"},
        "score": "s = RM.pt's centred score; preferred = the A member of a twin (a_* keys), the site member "
        "(probe:nosite), the longer member (probe:length)",
        "habits": habits, "reference": ref, "unreadable_proofs_scored": rm.unreadable,
        "secs": round(time.time() - t, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }  # fmt: skip
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    U.write_json(a.out, doc)

    def line(name: str, st: dict[str, Any], status: str) -> str:
        lo, hi = st.get("bonus_diff_ci95") or (float("nan"), float("nan"))
        return (f"{name:>14}: n {st['n']:4d}  bonus diff {st.get('bonus_diff', float('nan')):+.4f} "
                f"[{lo:+.4f}, {hi:+.4f}]  preferred higher {st.get('a_higher_share', float('nan')):.3f}  "
                f"score diff {st.get('mean_diff', float('nan')):+.3f}  {status}")  # fmt: skip

    if ref:
        print(line("reference", ref, f"band +-{REF_BAND * abs(ref_bonus or 0):.4f}"), flush=True)
    for h, st in habits.items():
        print(line(h, st, st["status"]), flush=True)
    return doc


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("fit")
    p.add_argument("--base", required=True)
    p.add_argument("--pairs", required=True)
    p.add_argument("--heldout", required=True)
    p.add_argument("--out", required=True, help="RM.pt (and RM.pt.json beside it)")
    p.add_argument("--lr", type=float, default=U.RL_LR, help="trunk lr (the RL recipe's)")
    p.add_argument("--head-lr", type=float, default=1e-3, help="lr of the new scalar head")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--min-acc", type=float, default=0.90)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--warmup", type=int, default=U.WARMUP)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--no-shift", action="store_true")
    p.add_argument("--threads", type=int, default=8)
    p = sp.add_parser("check", help="load an RM.pt and score a pair file")
    p.add_argument("--rm", required=True)
    p.add_argument("--pairs", required=True)
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--threads", type=int, default=8)
    p = sp.add_parser(
        "probe",
        help="the RM on twins of other habits and two controls (PREREG Amendment 7 point 6); exit 0",
    )
    p.add_argument("--min-n", type=int, default=MIN_N,
                   help="fewer pairs of a habit than this: not probed (PREREG Amendment 7: 100; smokes lower it)")  # fmt: skip
    p.add_argument("--rm", required=True)
    p.add_argument(
        "--pairs",
        required=True,
        help="PROBE.jsonl: pair format, `habit` per row, preferred = A member",
    )
    p.add_argument("--out", required=True, help="probe.json")
    p.add_argument("--reference", default=None,
                   help="the rated habit's held-out pairs (default: the file in RM.pt.json)")  # fmt: skip
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    if a.cmd == "fit":
        fit(a)
    elif a.cmd == "check":
        check(a)
    else:
        probe(a)


if __name__ == "__main__":
    main()
