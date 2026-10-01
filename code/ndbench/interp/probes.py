"""Linear probes over the residual stream (`INTERP_SPEC.md` §4).

Features are the residual stream at the embedding output and after each of the 4 blocks (5 feature
sets), read off at each line's own leading index token, *before* that line's content is written.
`ModelAdapter.activations` (already implemented for `AbsGptAdapter`) gives the post-block residuals
directly; there is no public capability for the pre-block-0 ("embedding output") residual, so
`_embedding_output` reaches into the concrete adapter's own `_model`/`model` with a
`register_forward_pre_hook` on block 0 -- the same documented, minimal workaround pattern as
`likelihood._teacher_forced_logits` (see that function's docstring for why: no capability flag for
this exists, and adding one means editing `adapters/base.py`/`absgpt.py`, which this work package
does not own).

Probes are plain numpy (ridge, closed-form) or torch (multinomial logistic regression) -- no
scikit-learn in the venv. MDL (online-code) probing follows Voita & Titov (2020): reveal the
training set in geometrically-growing blocks, refit on everything seen so far before each new
block, and sum each block's actual (base-2) cross-entropy cost under the *previous* blocks' model
(the first block's cost is the uniform code, since nothing has been seen yet).
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch

from ..adapters.base import ModelAdapter, NotSupported
from ..core.proof import split_lines
from ..core.types import JSON, Sequent
from ..store import Store
from ..vendor import load_nd_verify
from ._store import new_interp_run, write_interp_rows
from .data import line_start_positions
from .orderings import _parse_line

RULES: tuple[str, ...] = tuple(sorted(load_nd_verify().RULE_NAMES))  # 15 classes
LAYER_KEYS: tuple[str, ...] = ("emb", "0", "1", "2", "3")
BLOCK_LAYERS: tuple[int, ...] = (0, 1, 2, 3)
N_LINES_TO_GO_BINS = 5
#: `INTERP_SPEC.md` §4: "cited_back ... classes 1..8, 9+".
CITED_BACK_LABELS: tuple[str, ...] = tuple(str(i) for i in range(1, 9)) + ("9+",)
#: Voita & Titov (2020)'s geometric online-code schedule, as fractions of the training set.
MDL_FRACTIONS: tuple[float, ...] = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0)


def _get_model_and_tok(adapter: ModelAdapter) -> tuple[Any, Any]:
    model = getattr(adapter, "_model", None) or getattr(adapter, "model", None)
    tok = getattr(adapter, "_tok", None) or getattr(adapter, "tok", None)
    if model is None or tok is None:
        raise NotSupported(
            f"{type(adapter).__name__} exposes neither `_model`/`_tok` nor `model`/`tok`; "
            "probes need the embedding-output residual, which ModelAdapter's public contract does "
            "not expose (see this module's docstring)."
        )
    return model, tok


def _embedding_output(
    adapter: ModelAdapter, pairs: Sequence[tuple[Sequent, str]]
) -> list[np.ndarray]:
    """The residual stream *before* block 0 (`GPT.forward`: `x = self.emb(idx)` is block 0's first
    positional argument), one `(T, d)` array per pair."""
    model, tok = _get_model_and_tok(adapter)
    captured: dict[str, torch.Tensor] = {}

    def _hook(module: Any, inputs: Any) -> None:
        captured["x"] = inputs[0].detach()

    handle = model.blocks[0].register_forward_pre_hook(_hook)
    out: list[np.ndarray] = []
    try:
        model.eval()
        device = next(model.parameters()).device
        for seq, body in pairs:
            ids = tok.encode_prompt(seq.prompt()) + tok.encode_proof(body)
            x = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                model(x)
            out.append(captured["x"][0].float().cpu().numpy())
    finally:
        handle.remove()
    return out


def full_layer_stack(
    adapter: ModelAdapter,
    pairs: Sequence[tuple[Sequent, str]],
    block_layers: Sequence[int] = BLOCK_LAYERS,
) -> list[np.ndarray]:
    """`(len(block_layers) + 1, T, d)` per pair: `LAYER_KEYS`-ordered (`emb` then each block)."""
    emb = _embedding_output(adapter, pairs)
    blocks = adapter.activations(pairs, block_layers)
    return [np.concatenate([e[None, ...], b], axis=0) for e, b in zip(emb, blocks, strict=True)]


@dataclass(frozen=True)
class ProbeExample:
    pid: str
    L: int | None
    line_pos: int  # 1-based position of this line within the proof
    layer_features: dict[str, np.ndarray]  # LAYER_KEYS -> (d,)
    next_rule: str
    lines_to_go: int
    cited_back: int | None


def build_dataset(
    adapter: ModelAdapter,
    items: Sequence[tuple[str, Sequent, str, int | None]],
    *,
    block_layers: Sequence[int] = BLOCK_LAYERS,
) -> list[ProbeExample]:
    """One `ProbeExample` per line of every `(pid, sequent, body, L)` in `items`, teacher-forced
    through `adapter`."""
    pairs = [(seq, body) for _pid, seq, body, _l in items]
    stacks = full_layer_stack(adapter, pairs, block_layers)
    prompt_lens = [adapter.token_count(seq) for _pid, seq, body, _l in items]

    out: list[ProbeExample] = []
    for (pid, _seq, body, L), stack, prompt_len in zip(items, stacks, prompt_lens, strict=True):
        raw_lines = split_lines(body)
        parsed = [_parse_line(t) for t in raw_lines]
        n = len(parsed)
        line_positions = line_start_positions(body)
        assert len(line_positions) == n
        for k, (ln, tok_pos) in enumerate(zip(parsed, line_positions, strict=True)):
            abs_pos = prompt_len + tok_pos
            feats = {key: stack[li, abs_pos].copy() for li, key in enumerate(LAYER_KEYS)}
            cited_back = (ln.idx - min(ln.refs)) if ln.refs else None
            out.append(
                ProbeExample(
                    pid=pid,
                    L=L,
                    line_pos=k + 1,
                    layer_features=feats,
                    next_rule=ln.rule,
                    lines_to_go=n - (k + 1),
                    cited_back=cited_back,
                )
            )
    return out


def split_by_pid(
    pids: Sequence[str], *, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 0
) -> dict[str, set[str]]:
    """Train/val/test split *by theorem*, so every line of one proof stays in the same split."""
    uniq = sorted(set(pids))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_test = round(n * test_frac)
    n_val = round(n * val_frac)
    test = set(uniq[:n_test])
    val = set(uniq[n_test : n_test + n_val])
    train = set(uniq[n_test + n_val :])
    return {"train": train, "val": val, "test": test}


# ------------------------------------------------------------------------------------- ridge


def fit_ridge(X: np.ndarray, y: np.ndarray, l2: float) -> tuple[np.ndarray, float]:
    x_mean = X.mean(axis=0)
    y_mean = float(y.mean())
    xc, yc = X - x_mean, y - y_mean
    d = X.shape[1]
    # A tiny numerical floor on top of `l2` (even at `l2 == 0`, "no regularisation") guards against
    # an exactly-singular `xc.T @ xc` (e.g. a rank-deficient feature block) without materially
    # changing the fit for any well-conditioned feature matrix.
    w = np.linalg.solve(xc.T @ xc + (l2 + 1e-8) * np.eye(d), xc.T @ yc)
    b = y_mean - x_mean @ w
    return w, b


def predict_ridge(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return X @ w + b


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


# ---------------------------------------------------------------------- multinomial logistic


def fit_multinomial(
    X: np.ndarray,
    y_idx: np.ndarray,
    n_classes: int,
    l2: float,
    *,
    epochs: int = 200,
    lr: float = 0.1,
) -> torch.nn.Module:
    torch.manual_seed(0)
    model = torch.nn.Linear(X.shape[1], n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y_idx, dtype=torch.long)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(xt), yt)
        loss.backward()
        opt.step()
    model.eval()
    return model


def predict_proba(model: torch.nn.Module, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        return torch.softmax(logits, dim=-1).numpy()


def accuracy(y_true: np.ndarray, proba: np.ndarray) -> float:
    return float(np.mean(proba.argmax(axis=-1) == y_true))


# ---------------------------------------------------------------------------- probe drivers


L2_GRID: tuple[float, ...] = (0.0, 1e-3, 1e-2, 1e-1, 1.0)


def probe_classification(
    train: tuple[np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray],
    n_classes: int,
    *,
    l2_grid: Sequence[float] = L2_GRID,
    epochs: int = 200,
) -> dict[str, JSON]:
    """Fit a multinomial logistic regression per `l2` in `l2_grid` on `train`, pick the `l2` with
    the best validation accuracy, report train/val/test accuracy at that `l2`."""
    x_tr, y_tr = train
    x_val, y_val = val
    x_te, y_te = test
    best_l2, best_val_acc, best_model = None, -1.0, None
    for l2 in l2_grid:
        model = fit_multinomial(x_tr, y_tr, n_classes, l2, epochs=epochs)
        val_acc = accuracy(y_val, predict_proba(model, x_val)) if len(y_val) else float("nan")
        if val_acc == val_acc and val_acc > best_val_acc:
            best_l2, best_val_acc, best_model = l2, val_acc, model
    assert best_model is not None
    return {
        "l2": best_l2,
        "train_acc": accuracy(y_tr, predict_proba(best_model, x_tr)),
        "val_acc": best_val_acc,
        "test_acc": accuracy(y_te, predict_proba(best_model, x_te)) if len(y_te) else float("nan"),
        "n_train": len(y_tr),
        "n_val": len(y_val),
        "n_test": len(y_te),
    }


def probe_regression(
    train: tuple[np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray],
    *,
    l2_grid: Sequence[float] = L2_GRID,
    n_bins: int = N_LINES_TO_GO_BINS,
) -> dict[str, JSON]:
    """Ridge regression (`lines_to_go`), L2-tuned on `val`'s R^2; also reports `n_bins`-quantile-bin
    accuracy (bin edges from `train`'s own target distribution)."""
    x_tr, y_tr = train
    x_val, y_val = val
    x_te, y_te = test
    best_l2, best_val_r2, best_wb = None, float("-inf"), None
    for l2 in l2_grid:
        w, b = fit_ridge(x_tr, y_tr, l2)
        val_r2 = r2_score(y_val, predict_ridge(x_val, w, b)) if len(y_val) else float("nan")
        if val_r2 == val_r2 and val_r2 > best_val_r2:
            best_l2, best_val_r2, best_wb = l2, val_r2, (w, b)
    assert best_wb is not None
    w, b = best_wb
    edges = np.quantile(y_tr, np.linspace(0, 1, n_bins + 1)[1:-1]) if len(y_tr) else np.array([])

    def bin_acc(x: np.ndarray, y: np.ndarray) -> float:
        if len(y) == 0:
            return float("nan")
        pred = predict_ridge(x, w, b)
        return float(np.mean(np.digitize(pred, edges) == np.digitize(y, edges)))

    return {
        "l2": best_l2,
        "train_r2": r2_score(y_tr, predict_ridge(x_tr, w, b)),
        "val_r2": best_val_r2,
        "test_r2": r2_score(y_te, predict_ridge(x_te, w, b)) if len(y_te) else float("nan"),
        "test_bin_acc": bin_acc(x_te, y_te),
        "n_train": len(y_tr),
        "n_val": len(y_val),
        "n_test": len(y_te),
    }


# ------------------------------------------------------------------------------------- MDL


def mdl_online_code(
    X: np.ndarray,
    y_idx: np.ndarray,
    n_classes: int,
    *,
    l2: float = 1e-2,
    epochs: int = 100,
    fractions: Sequence[float] = MDL_FRACTIONS,
    seed: int = 0,
) -> dict[str, JSON]:
    """Voita & Titov (2020) online code: `sum` over geometrically-growing blocks of each block's
    base-2 cross-entropy under a classifier fit on every *earlier* block (the first block's cost is
    the uniform code, `log2(n_classes)` per example -- nothing has been seen yet). Returns the total
    code length in bits, the uniform-code length, and their ratio ("compression")."""
    n = len(y_idx)
    if n == 0:
        return {"code_length_bits": 0.0, "uniform_bits": 0.0, "compression": float("nan")}
    order = list(range(n))
    random.Random(seed).shuffle(order)
    xs, ys = X[order], y_idx[order]
    cum = sorted({max(1, int(round(f * n))) for f in fractions} | {n})
    total_bits = cum[0] * math.log2(n_classes)  # first block: uniform code
    for i in range(len(cum) - 1):
        train_end, next_end = cum[i], cum[i + 1]
        model = fit_multinomial(xs[:train_end], ys[:train_end], n_classes, l2, epochs=epochs)
        proba = predict_proba(model, xs[train_end:next_end])
        true = ys[train_end:next_end]
        p_true = np.clip(proba[np.arange(len(true)), true], 1e-12, None)
        total_bits += float(-np.sum(np.log2(p_true)))
    uniform_bits = n * math.log2(n_classes)
    return {
        "code_length_bits": total_bits,
        "uniform_bits": uniform_bits,
        "compression": uniform_bits / total_bits if total_bits else float("nan"),
        "n": n,
    }


# ----------------------------------------------------------------------------- orchestration


def _class_index(labels: Sequence[str], vocab: Sequence[str]) -> np.ndarray:
    index = {c: i for i, c in enumerate(vocab)}
    return np.array([index[label] for label in labels], dtype=np.int64)


def _cited_back_label(cited_back: int | None) -> str:
    if cited_back is None:
        return "9+"
    return str(cited_back) if cited_back <= 8 else "9+"


def _lines_to_go_bin_labels(values: Sequence[int], edges: np.ndarray) -> np.ndarray:
    return np.digitize(np.asarray(values, dtype=float), edges)


def _feature_matrix(examples: Sequence[ProbeExample], layer: str) -> np.ndarray:
    return np.stack([e.layer_features[layer] for e in examples], axis=0)


def run_layer_probes(
    examples: Sequence[ProbeExample], splits: Mapping[str, set[str]], layer: str
) -> dict[str, JSON]:
    by_split = {name: [e for e in examples if e.pid in pids] for name, pids in splits.items()}
    tr, va, te = by_split["train"], by_split["val"], by_split["test"]

    def xy_rule(exs: Sequence[ProbeExample]) -> tuple[np.ndarray, np.ndarray]:
        return _feature_matrix(exs, layer), _class_index([e.next_rule for e in exs], RULES)

    next_rule = probe_classification(xy_rule(tr), xy_rule(va), xy_rule(te), len(RULES))

    def xy_ltg(exs: Sequence[ProbeExample]) -> tuple[np.ndarray, np.ndarray]:
        return _feature_matrix(exs, layer), np.array([e.lines_to_go for e in exs], dtype=float)

    lines_to_go = probe_regression(xy_ltg(tr), xy_ltg(va), xy_ltg(te))

    cb_examples = [e for e in examples if e.cited_back is not None]
    cb_by_split = {name: [e for e in cb_examples if e.pid in pids] for name, pids in splits.items()}

    def xy_cb(exs: Sequence[ProbeExample]) -> tuple[np.ndarray, np.ndarray]:
        labels = [_cited_back_label(e.cited_back) for e in exs]
        return _feature_matrix(exs, layer), _class_index(labels, CITED_BACK_LABELS)

    cited_back = (
        probe_classification(
            xy_cb(cb_by_split["train"]),
            xy_cb(cb_by_split["val"]),
            xy_cb(cb_by_split["test"]),
            len(CITED_BACK_LABELS),
        )
        if cb_by_split["train"] and cb_by_split["test"]
        else None
    )

    mdl_rule = mdl_online_code(*xy_rule(tr), len(RULES)) if tr else None
    edges = (
        np.quantile([e.lines_to_go for e in tr], np.linspace(0, 1, N_LINES_TO_GO_BINS + 1)[1:-1])
        if tr
        else np.array([])
    )
    mdl_ltg = (
        mdl_online_code(
            _feature_matrix(tr, layer),
            _lines_to_go_bin_labels([e.lines_to_go for e in tr], edges),
            N_LINES_TO_GO_BINS,
        )
        if tr
        else None
    )

    return {
        "next_rule": next_rule,
        "lines_to_go": lines_to_go,
        "cited_back": cited_back,
        "mdl_next_rule": mdl_rule,
        "mdl_lines_to_go_binned": mdl_ltg,
    }


def run_length_generalisation(
    train_examples: Sequence[ProbeExample],
    ood_examples: Sequence[ProbeExample],
    train_splits: Mapping[str, set[str]],
    layer: str,
) -> dict[str, JSON]:
    """`INTERP_SPEC.md` §4: train `lines_to_go` on `L <= 10` (`train_examples`, split by
    `train_splits`), test on `ood_examples` (sealed L12-14 + family witnesses, `L` up to ~40)."""
    tr = [e for e in train_examples if e.pid in train_splits["train"]]
    va = [e for e in train_examples if e.pid in train_splits["val"]]
    if not tr or not ood_examples:
        return {"empty": True}
    x_tr = _feature_matrix(tr, layer)
    y_tr = np.array([e.lines_to_go for e in tr], dtype=float)
    x_va = _feature_matrix(va, layer) if va else x_tr[:0]
    y_va = np.array([e.lines_to_go for e in va], dtype=float)
    best_l2, best_val_r2, best_wb = None, float("-inf"), None
    for l2 in L2_GRID:
        w, b = fit_ridge(x_tr, y_tr, l2)
        val_r2 = r2_score(y_va, predict_ridge(x_va, w, b)) if len(y_va) else float("nan")
        if val_r2 == val_r2 and val_r2 > best_val_r2:
            best_l2, best_val_r2, best_wb = l2, val_r2, (w, b)
    if best_wb is None:
        best_wb = fit_ridge(x_tr, y_tr, L2_GRID[0])
        best_l2 = L2_GRID[0]
    w, b = best_wb
    x_ood = _feature_matrix(ood_examples, layer)
    y_ood = np.array([e.lines_to_go for e in ood_examples], dtype=float)
    return {
        "empty": False,
        "l2": best_l2,
        "in_distribution_val_r2": best_val_r2,
        "ood_r2": r2_score(y_ood, predict_ridge(x_ood, w, b)),
        "n_train": len(tr),
        "n_ood": len(ood_examples),
    }


# --------------------------------------------------------------------------- success (optional)


def build_success_dataset(
    adapter: ModelAdapter,
    items: Sequence[tuple[str, Sequent, str]],
    labels: Mapping[str, float],
    *,
    mastery: float = 0.5,
) -> list[tuple[str, dict[str, np.ndarray], int]]:
    """`INTERP_SPEC.md` §4's optional `success` target: features at the `PRF` token (the prompt's
    own last token -- unaffected by whatever proof body follows it, so any body anchors the forward
    pass), label = whether `labels[pid]` (a per-theorem `p_hat`, e.g. from `Store.items`) is
    `>= mastery`. Skips any `pid` missing from `labels`."""
    pairs = [(seq, body) for _pid, seq, body in items]
    stacks = full_layer_stack(adapter, pairs)
    prompt_lens = [adapter.token_count(seq) for _pid, seq, _body in items]
    out: list[tuple[str, dict[str, np.ndarray], int]] = []
    for (pid, _seq, _body), stack, plen in zip(items, stacks, prompt_lens, strict=True):
        if pid not in labels:
            continue
        feats = {key: stack[li, plen - 1].copy() for li, key in enumerate(LAYER_KEYS)}
        out.append((pid, feats, int(labels[pid] >= mastery)))
    return out


def probe_success(
    dataset: Sequence[tuple[str, dict[str, np.ndarray], int]],
    splits: Mapping[str, set[str]],
    layer: str,
) -> dict[str, JSON] | None:
    by_split = {name: [d for d in dataset if d[0] in pids] for name, pids in splits.items()}
    if not by_split["train"] or not by_split["test"]:
        return None

    def xy(rows: Sequence[tuple[str, dict[str, np.ndarray], int]]) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.stack([r[1][layer] for r in rows], axis=0),
            np.array([r[2] for r in rows], dtype=np.int64),
        )

    return probe_classification(xy(by_split["train"]), xy(by_split["val"]), xy(by_split["test"]), 2)


# ----------------------------------------------------------------------------------- run_probes


def run_probes(
    store: Store,
    cfg: Mapping[str, JSON],
    *,
    base_adapter: ModelAdapter | None,
    rl_adapter: ModelAdapter | None,
) -> dict[str, Any]:
    """`interp probes` (`INTERP_SPEC.md` §4): builds the line-level dataset from neutral reserve
    proofs (`L` 6-10, capped at `cfg["n_train"]`, default 300) for each of `base_adapter`/
    `rl_adapter`, runs `next_rule`/`lines_to_go`/`cited_back` + MDL probes per layer, and the
    `lines_to_go` length-generalisation test against `cfg["n_ood"]` (default 150) sealed L12-14 (+
    family witnesses when present) examples. Writes one row per `(model, layer)` to
    `out/store/interp/probes/<run_id>.jsonl`.

    Does **not** include the spec's optional `success` target: unlike `next_rule`/`lines_to_go`/
    `cited_back` (derived purely from a proof's own text), `success` needs a per-theorem `p_hat`
    label from some benchmark suite's `Store` runs, on theorems this function's own `n_train`
    neutral-proof sample has no reason to overlap with. Call `build_success_dataset`/
    `probe_success` directly once you have such a suite/run in mind (see their docstrings)."""
    from . import data as _data

    def _int_cfg(key: str, default: int) -> int:
        v = cfg.get(key)
        return default if v is None else int(cast(float, v))

    seed = _int_cfg("seed", 0)
    n_train = _int_cfg("n_train", 300)
    n_ood = _int_cfg("n_ood", 150)
    notes: list[str] = []

    excluder = _data.build_exclusion_index(cache_dir=cast("str | None", cfg.get("cache_dir")))
    train_proofs = _data.load_neutral_proofs(
        n=n_train,
        excluder=excluder,
        seed=seed,
        scan_limit=cast("int | None", cfg.get("scan_limit")),
    )
    ood_proofs = _data.load_sealed_l12_14(n=n_ood, excluder=excluder, seed=seed)
    family = _data.load_family_witnesses(store, n=n_ood)
    if not family:
        notes.append("no families suite in the store yet; length-generalisation uses sealed only")
    ood_proofs = list(ood_proofs) + list(family)

    train_items = [(p.pid, p.sequent, p.body, p.L) for p in train_proofs]
    ood_items = [(p.pid, p.sequent, p.body, p.L) for p in ood_proofs]
    splits = split_by_pid([pid for pid, *_r in train_items], seed=seed)

    rows: list[dict[str, JSON]] = []
    for model_name, adapter in (("base", base_adapter), ("rl", rl_adapter)):
        if adapter is None:
            notes.append(f"no {model_name} adapter given; skipping its probes")
            continue
        train_ex = build_dataset(adapter, train_items)
        ood_ex = build_dataset(adapter, ood_items) if ood_items else []
        for layer in LAYER_KEYS:
            layer_result = run_layer_probes(train_ex, splits, layer)
            length_gen = run_length_generalisation(train_ex, ood_ex, splits, layer)
            rows.append(
                {
                    "model": model_name,
                    "layer": layer,
                    "n_train_examples": len(train_ex),
                    "n_ood_examples": len(ood_ex),
                    **layer_result,
                    "length_generalisation": length_gen,
                }
            )

    run_id = new_interp_run(store, "probes", cfg)
    write_interp_rows(store, "probes", run_id, rows)
    return {"run_id": run_id, "n_rows": len(rows), "n_train": len(train_items), "notes": notes}


def position_baselines(
    items: Sequence[tuple[str, Sequent, str, int | None]], splits: Mapping[str, set[str]]
) -> dict[str, JSON]:
    """Non-neural baselines for the probes on the same examples and split (interp audit,
    2026-10-01): each predictor is a lookup table fit on `train` -- the most common `next_rule`,
    or the mean `lines_to_go`, among train lines sharing the key -- scored on `test`, falling back
    to the train-wide majority/mean for unseen keys. Keys: nothing (majority class), the line
    index, line index + number of premises (both visible in the prompt), and that plus the proof's
    total line count. A probe only shows information beyond position if it beats these."""
    rows: list[tuple[str, int, int, int, str, int]] = []  # pid, k, n_prem, n_lines, rule, ltg
    for pid, seq, body, _l in items:
        parsed = [_parse_line(t) for t in split_lines(body)]
        n = len(parsed)
        for k, ln in enumerate(parsed):
            rows.append((pid, k + 1, len(seq.premises), n, ln.rule, n - (k + 1)))
    train = [r for r in rows if r[0] in splits["train"]]
    test = [r for r in rows if r[0] in splits["test"]]
    if not train or not test:
        return {"n_train": len(train), "n_test": len(test)}
    majority = Counter(r[4] for r in train).most_common(1)[0][0]
    mean_ltg = float(np.mean([r[5] for r in train]))
    keys: dict[str, Any] = {
        "majority": lambda r: (),
        "line_index": lambda r: (r[1],),
        "line_index_premises": lambda r: (r[1], r[2]),
        "line_index_premises_length": lambda r: (r[1], r[2], r[3]),
    }
    y_ltg = np.array([r[5] for r in test], dtype=float)
    ss_tot = float(((y_ltg - y_ltg.mean()) ** 2).sum()) or 1.0
    out: dict[str, JSON] = {"n_train": len(train), "n_test": len(test)}
    for name, keyf in keys.items():
        rule_tab: dict[Any, Counter[str]] = {}
        ltg_tab: dict[Any, list[int]] = {}
        for r in train:
            rule_tab.setdefault(keyf(r), Counter())[r[4]] += 1
            ltg_tab.setdefault(keyf(r), []).append(r[5])
        acc = float(
            np.mean(
                [
                    (rule_tab[keyf(r)].most_common(1)[0][0] if keyf(r) in rule_tab else majority)
                    == r[4]
                    for r in test
                ]
            )
        )
        pred = np.array(
            [float(np.mean(ltg_tab[keyf(r)])) if keyf(r) in ltg_tab else mean_ltg for r in test]
        )
        out[f"next_rule_acc_{name}"] = acc
        out[f"lines_to_go_r2_{name}"] = 1.0 - float(((y_ltg - pred) ** 2).sum()) / ss_tot
    return out


def run_position_baselines(store: Store, cfg: Mapping[str, JSON]) -> dict[str, Any]:
    """`interp probe-baselines`: `position_baselines` on exactly `run_probes`'s train examples and
    split (same `seed`/`n_train`/`scan_limit`/`cache_dir`), CPU only; one row, kind
    `probe_baselines`."""
    from . import data as _data

    seed = int(cast(float, cfg.get("seed") or 0))
    n_train = int(cast(float, cfg.get("n_train") or 300))
    excluder = _data.build_exclusion_index(cache_dir=cast("str | None", cfg.get("cache_dir")))
    train_proofs = _data.load_neutral_proofs(
        n=n_train,
        excluder=excluder,
        seed=seed,
        scan_limit=cast("int | None", cfg.get("scan_limit")),
    )
    items = [(p.pid, p.sequent, p.body, p.L) for p in train_proofs]
    splits = split_by_pid([pid for pid, *_r in items], seed=seed)
    row = position_baselines(items, splits)
    run_id = new_interp_run(store, "probe_baselines", cfg)
    write_interp_rows(store, "probe_baselines", run_id, [row])
    return {"run_id": run_id, **row}
