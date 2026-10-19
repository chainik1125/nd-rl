"""WP3 restricted-class families suite (plan §2 B2, §3 WP3 row).

Builds `~40` distinct, randomised instances per `(family, n)` for ten families of ND sequents
whose minimal proof length `L(n)` is either exactly known (small `n`, verified against the
vendored `minlen` search, matching plan §0's table) or bounded by a family-specific combinatorial
argument (documented per family below) for larger `n`, where an exhaustive `minlen` search is not
CPU-feasible under this work package's budget (plan §A2: <=8 worker processes, <=15 CPU-minutes
per job, on a machine shared with six other coders).

Builder contract (plan §3): `build(cfg, *, certifier, excluder, rng_seed) -> (list[Problem], dict)`.
`certifier`/`excluder` follow `core.types.Certifier`/`Excluder`. If `certifier` is `None`,
`default_certifier()` is used: it first tries to import WP2's `logic.certify.MinlenCertifier` and
falls back to `LocalMinlenCertifier`, a minimal wrapper directly around the vendored
`takehome/minlen.py` (see its docstring) -- WP2's `logic.certify` module was not importable when
this WP3 build started but landed before this suite was actually built, so the real
`MinlenCertifier` was used for the final `out/suites/families` build (`LocalMinlenCertifier`
remains the test-only stub, per the WP task's own "in tests use a local stub" instruction).
`excluder` is only used to *flag* items (`seen_train`/`seen_eval`), never to drop them (plan §2:
"flag for family members and parts").

Every witness proof is built directly in the take-home's line format (see
`/scratch/cowifornia/nd-rl/external/nd-takehome/spec.md`) via the local `LineBuilder` helper, then
checked with the vendored `nd_verify.verify_text` -- never trusted un-checked. Length labels use
`core.types.LengthLabel.method`; every certifier call passes the witness alongside `bound =
predicted_L` and only the returned `method`/`L` are inspected (never which of `bound`/witness a
given backend actually searched against -- see `_label_and_filter`'s docstring):

* `minlen_exact` -- for `n` at or below the family's "exact" threshold (matching plan §0's table),
  the certifier returns a proof of exactly `predicted_L`. A *shorter* result means our construction
  wasn't minimal after all (an unintended shortcut) and the candidate is **dropped** unconditionally
  (plan §0: "every constructed item ... is certified or dropped").
* `witness_argument` -- for `n` above that threshold, for the six families whose lower-bound
  argument (below) does not depend on the specific atoms drawn (each subformula the argument needs
  is syntactically distinct *by construction*, regardless of atom reuse), so no further `minlen`
  search is attempted; **and**, per a main-agent instruction (2026-09-30), also for those same six
  families' *exact* tier when the certifier is merely **inconclusive** (timeout, or WP2's
  `"reference"`) rather than reporting a confirmed shortcut -- the witness already verified and the
  argument already proves its length, so an inconclusive search is not a reason to discard a
  correct item, only `timeout=True` is recorded to flag that `minlen` did not confirm it. This
  fallback does **not** apply to `dn_intro` (its exact tier had zero drops in practice) or to
  `reorder_and`/`reorder_or` (next bullet).
* `ub_lb` -- for `reorder_and`/`reorder_or`, both in their exact tier (on an inconclusive result,
  no independent argument to fall back on) and above it, where the family's own argument *does*
  depend on no two subformulas coinciding (atom reuse is unavoidable once `n` exceeds the 4-atom
  alphabet, so a genuine shortcut is possible and must be checked for, not merely assumed): kept as
  `ub_lb` when confirmed; **dropped** if the certifier *finds* a genuine shortcut; kept as `ub_lb`
  with `timeout=True` (unverified) if the certifier is merely inconclusive within this WP's CPU
  budget -- honestly reporting that this instance's minimality is unverified rather than
  fabricating a lower-bound argument that atom reuse may have invalidated. This is a documented
  deviation from plan §2 B2's simpler "exact <=N; ub_lb above" framing, forced by CPU cost (the
  planner's own §0 check already timed out certifying `reorder_and` at `n=5` with a 90s budget; see
  the WP3 report for the exact `n` ranges attempted and how often this path was actually taken).

## Lower-bound arguments (one per family)

`and_tower(n, Gamma)`: `Gamma` (`1<=|Gamma|<=4` distinct atoms) are declared as premises; the
conclusion is an `n`-step tower `(...((g_0 op_1 g_1) op_2 g_2)... op_n g_n)` (`op_i` a random
`&`-bracketing side, `g_i` a random member of `Gamma`). Every one of the `n` tower levels is a
*distinct* subformula (each has strictly more `&`-nesting than the last, regardless of which atoms
repeat), and no premise is itself an `&`-compound, so each level can only be produced by its own
`ANDI` line: `n` forced `ANDI` lines plus `|Gamma|` premise lines, `L = n + |Gamma|`.

`and_tree(n, Gamma)`: as `and_tower`, but the `n` premise-citations are combined into a *balanced
random binary tree* (`n` leaves, `n-1` internal `&`-nodes) instead of a linear chain, with leaf
atoms drawn (with repetition) so that **all internal-node formulas come out pairwise distinct**
(enforced by rejection sampling at construction time -- this is the family's defining constraint,
per plan §2 B2's "distinct internal nodes"). Distinctness makes the same argument as `and_tower`
apply node-by-node: every internal node needs its own fresh `ANDI` line, giving
`L = |Gamma| + (n-1)` (`#internal = n-1` for any binary tree with `n` leaves).

`or_tower(n)`: one atom premise `P`; the conclusion is an `n`-step `v`-tower built by `ORI1`/`ORI2`
from `P`, mixing sides and sibling atoms at random. As with `and_tower`, the `n` tower levels are
pairwise distinct by nesting depth, and no rule other than `ORI1`/`ORI2` can introduce a fresh `v`
node from a non-disjunctive premise, so each level forces its own line: `L = n + 1`.

`imp_nest(n)`: no premises; conclusion `A_1 > (A_2 > ( ... > (A_n > (A_i & A_j)) ...))` for random
atoms `A_1..A_n` (with repetition) and random `i, j <= n`. Goal-directed: an implication with no
available premises can only be introduced by `IMPI` (opening a box), so deriving the outermost
`n` implications forces `n` nested `IMPI` lines (closing innermost-first), and the innermost
`(A_i & A_j)` forces one `ANDI` citing the two hypothesis lines directly (no `R`/reiteration line
is needed: `ANDI`'s citations may point at any citable line, including an enclosing box's own `AS`
line) -- total lines = `n` `AS` + 1 `ANDI` + `n` `IMPI` = `2n + 1`.

`and_extract(n, Gamma)`: one premise, a depth-`n` right-nested `&` tree over random atoms from
`Gamma`; conclusion is one random leaf reached by a random left/right path of length `n`. Each
path step is a forced `ANDE1`/`ANDE2` (the only rule that can pull a strictly smaller formula out
of an `&`-premise), so `L = n + 1`.

`dn_elim(n)`: one premise `~^(2n) P`; conclusion `P`. Each pair of leading `~`s can only be removed
by `DN`, one at a time (the search space's other rules do not shrink negation count without
introducing a box), so `L = n + 1`.

`dn_intro(n)`: one premise `P`; conclusion `~^(2n) P`. The **naive** intuitionistic proof adds one
`~~` pair at a time (`AS ~^(2k-1)P; NEGE; NEGI`, 3 lines/level) for `3n + 1` total. The **classical
shortcut** (plan §0) instead opens a *single* box assuming `H = ~^(2n-1) P`, applies `DN` `n-1`
times to peel `H` down to `~P` (valid since `H` has `2n-1 >= 1` leading negations, an odd number,
so after `k` `DN` applications `2n-1-2k` negations remain until reaching exactly 1), then one
`NEGE` against the premise `P` and one closing `NEGI`: `L = 1 (premise) + 1 (AS) + (n-1) (DN) +
1 (NEGE) + 1 (NEGI) = n + 3`. Every witness built here uses the shortcut; the `n=1` base case
needs zero `DN` steps (`H` is already `~P`) and still gives `n + 3 = 4`.

`reorder_and(n)`/`reorder_or(n)`: one premise, a random binary `&`/`v`-tree over `n` pairwise
jointly-satisfiable, non-`&`/non-`v` "conjunct" formulas (literals or small `~`/`v`/`>` formulas
over the 4-atom alphabet, checked satisfiable by brute-force truth table); the conclusion is the
*same* `n` formulas under a genuinely different permutation and (independently random) tree shape.
For `&`: extracting every leaf from the premise tree needs exactly `2n-2` `ANDE` lines (every
non-root node of a tree with `n` leaves needs one `ANDE` line, `= (n-2)` internal `+ n` leaves, and
this total is shape-invariant), and repacking the permuted leaves into the goal tree needs exactly
`n-1` fresh `ANDI` lines (one per internal goal-tree node) -- **provided no goal-tree subformula
coincides with a premise-tree subformula** (else that `ANDI` line could be skipped by citing the
matching extraction line instead, which is exactly the "genuine shortcut" `ub_lb`/drop policy
above exists to catch): `L = 1 + (2n-2) + (n-1) = 3n - 2`. For `v`: the construction instead
recurses through nested `ORE` (no separate extraction phase is possible for a disjunction -- each
branch must independently reconstruct the goal), giving `L = 1 + 3*(n-1) (ORE overhead: one `ORE`
line + two `AS` lines per of the `n-1` premise-tree internal nodes) + sum_leaves(depth of that
leaf's position in the goal tree) (the `ORI`-chain cost to inject each leaf into the goal formula).
This sum is shape-dependent (no closed form is asserted; each instance's actual line count is
computed from its own construction and used as that instance's `L`), consistent with plan §0's
non-formulaic `6, 12, ...` for `n=2, 3`.

`ILTP-style` (plan §2 B2's "ILTP-small", §A5): this WP3 build attempted to fetch and verify the
TPTP source of ILTP's `SYJ203`/`SYJ204` problems from `https://www.iltp.de` and
`https://www.tptp.org` on 2026-09-30 (`SeeTPTP?...&Domain=SYJ&File=SYJ204+1.p` and neighbouring
paths all returned `404`/`No such file or directory` for the TPTP v9.3.1 mirror reachable from this
session; `iltp.de`'s own problem-browsing pages did not resolve either). Per plan §5 risk 8 and
§A5, the sub-families below are therefore named **`ILTP-style`**, not `SYJ203`/`SYJ204`, and use
the formula schemas given in the plan's own §5 note (traceable to Schwichtenberg's and Statman's
classical hardness examples, not independently re-verified against the primary TPTP source):

* `syj204_style(n)` (Schwichtenberg chain): premises `p_n` and, for `i = n..1`,
  `(p_i > (p_i > p_{i-1}))`; conclusion `p_0`. `p_0..p_n` are `n+1` pairwise-distinct literals
  drawn from the 8 available (`P,Q,R,S,~P,~Q,~R,~S`), capping `n <= 7`; to avoid a complementary
  pair (`p_i` and `~p_i` both present, which could let `minlen` shortcut via `NEGE`/`DN` and would
  undermine the "exact" certification), this build only draws plain, non-negated atoms and caps
  `n <= 3` (4 atoms). Each level needs two nested `IMPE` lines (curried modus ponens applied twice
  to `p_i`) plus its own premise line: `L = (n+1) + 2n = 3n + 1`, matching plan §0's table exactly
  (verified against `minlen` for `n<=3`).
* `syj203_style(n)`: `~~( (p_1 & ... & p_n) v (~p_1 v ... v ~p_n) )` (classical: either every
  `p_i` holds, or some `p_i` fails), proved by reductio (`RAA`: assume the negation, derive `F` by
  a `NEGE` between the two obtained disjuncts of the excluded-middle instance on each `p_i` used
  inside, `DN` to close). Only `n=1, 2` are built (plan §0: `minlen` already times out at `n=3`
  with a generous budget); labelled `witness_argument` since this WP3 build does not re-attempt
  that already-known timeout.
* `iff_chain_style(n)`: `<->` does not exist in this grammar (spec.md §1), so `(A <-> B)` is
  written `( ( A > B ) & ( B > A ) )`; the chain premises `A_1<->A_2, ..., A_{n-1}<->A_n` entail
  `A_1<->A_n`. Plan §0 found even `n=2` (`"expanded"`) times out at bound 20 for `minlen`, so this
  build does not re-attempt certification either; witnesses are checked with `nd_verify` only and
  labelled `witness_argument`, with that known difficulty repeated here for anyone re-running a
  certifier against them.
* `de_bruijn_style(n=1)`: the plan does not give a formula for this member, and neither ILTP nor
  TPTP source was reachable to recover one (see above); this build uses Peirce's law
  `((P > Q) > P) > P` as the `n=1` instance of a de-Bruijn-style classical-implication hierarchy
  (a standard, well-known example of that shape), explicitly flagged `ILTP-style` and unverified
  against any ILTP source -- if this does not match the intended `de_bruijn` family, drop it from
  downstream analysis; it is kept here only because the plan lists a `de Bruijn n=1` member and
  `ILTP-style` naming is the explicit fallback plan §5 risk 8 / §A5 sanction for exactly this case.

All ten "clean" families plus the four ILTP-style members respect: 4 atoms (`P Q R S`), <=8
premises, <=64 lines, and prompt+reference <=1000 tokens (counted with the vendored `abgpt`
tokenizer, one token per whitespace symbol -- see `count_tokens`); any candidate outside these caps
is dropped before certification is attempted.
"""

from __future__ import annotations

import contextlib
import itertools
import multiprocessing as mp
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from multiprocessing.pool import Pool as _MPPool
from typing import Any, TypedDict

from ..core.formula import ATOMS as FORMULA_ATOMS
from ..core.formula import Formula, parse
from ..core.types import (
    JSON,
    Certifier,
    Excluder,
    LengthLabel,
    Problem,
    Sequent,
)
from ..vendor import ensure_takehome_on_path, load_absgpt_module, load_nd_verify

#: The take-home's fixed 4-atom alphabet (plan-wide limit).
ATOMS: tuple[str, ...] = FORMULA_ATOMS
assert ATOMS == ("P", "Q", "R", "S")

#: Plan-wide item caps (§2 shared settings / WP3 instructions).
TOKEN_CAP = 1000
LINE_CAP = 64
PREMISE_CAP = 8
#: Above this many premises, `Sequent.pkey()`'s O(n!) cost is skipped in favour of `.key()`
#: (matches the plan's own Excluder dedup policy threshold).
PKEY_PREMISE_LIMIT = 5
#: Target distinct instances per (family, n) (plan §2 B2: "~40 distinct instances").
TARGET_PER_N = 40


# --------------------------------------------------------------------------------- vendor access


def _verify_text() -> Callable[[str], tuple[bool, str, int]]:
    nd = load_nd_verify()
    return nd.verify_text  # type: ignore[no-any-return]


def _minlen():
    ensure_takehome_on_path()
    import minlen  # type: ignore[import-not-found]

    return minlen


_tokenizer_singleton: Any = None


def _tokenizer() -> Any:
    global _tokenizer_singleton
    if _tokenizer_singleton is None:
        tok_mod = load_absgpt_module("tokenizer")
        _tokenizer_singleton = tok_mod.Tokenizer(mode="abs")
    return _tokenizer_singleton


def count_tokens(prompt: str, body: str) -> int:
    """Prompt-plus-reference token count, as the frozen `absgpt` tokenizer (`mode='abs'`, one
    surface symbol per token) would count it -- the plan's own token-cap accounting rule."""
    tok = _tokenizer()
    return len(tok.encode_prompt(prompt)) + len(tok.encode_proof(body))


# ------------------------------------------------------------------------------- proof rendering


class LineBuilder:
    """Accumulates take-home-format ND proof lines (spec.md §3) and renders the finished body.

    `add(depth, formula, rule, refs)` appends one line and returns its 1-based `N<i>` index (for
    use as a `refs` entry in later lines); `render()` produces the final `'N1 ... ; ... QED'`
    string. This mirrors vendored `takehome/minlen.py`'s own `Search.to_text`/`emit`, reimplemented
    here (not imported) since family witnesses are built directly, not via `minlen`'s search.
    """

    def __init__(self) -> None:
        self.lines: list[tuple[int, str, str, list[int]]] = []

    def add(self, depth: int, formula: str, rule: str, refs: Sequence[int] = ()) -> int:
        self.lines.append((depth, formula, rule, list(refs)))
        return len(self.lines)

    def render(self) -> str:
        toks: list[str] = []
        for i, (d, f, rule, refs) in enumerate(self.lines, start=1):
            toks += [f"N{i}"] + ["|"] * d + f.split() + [":", rule]
            toks += [f"N{r}" for r in refs]
            toks.append(";")
        toks.append("QED")
        return " ".join(toks)

    def __len__(self) -> int:
        return len(self.lines)


# ------------------------------------------------------------------------------ formula utilities


def _eval_formula(f: Formula, env: dict[str, bool]) -> bool:
    if f[0] == "atom":
        return env[f[1]]
    if f[0] == "bot":
        return False
    if f[0] == "not":
        return not _eval_formula(f[1], env)
    if f[0] == "and":
        return _eval_formula(f[1], env) and _eval_formula(f[2], env)
    if f[0] == "or":
        return _eval_formula(f[1], env) or _eval_formula(f[2], env)
    if f[0] == "imp":
        return (not _eval_formula(f[1], env)) or _eval_formula(f[2], env)
    raise ValueError(f[0])


def is_satisfiable(formulas: Sequence[str]) -> bool:
    """Brute-force joint satisfiability of `formulas` (spec-format strings) over the 4-atom
    alphabet (2**4 = 16 rows; used to enforce plan §2 B2/B4/B5's "jointly satisfiable" premises)."""
    asts = [parse(f) for f in formulas]
    for bits in itertools.product((False, True), repeat=len(ATOMS)):
        env = dict(zip(ATOMS, bits, strict=True))
        if all(_eval_formula(f, env) for f in asts):
            return True
    return False


def is_contingent(formula: str) -> bool:
    """`formula` is true under some assignment and false under some other (neither a tautology
    nor unsatisfiable); used for law/padding fillers (plan §2 B4/B5)."""
    f = parse(formula)
    seen = {False: False, True: False}
    for bits in itertools.product((False, True), repeat=len(ATOMS)):
        env = dict(zip(ATOMS, bits, strict=True))
        seen[_eval_formula(f, env)] = True
    return seen[True] and seen[False]


def pkey_or_key(seq: Sequent) -> str:
    """`seq.pkey()`, unless it has more than `PKEY_PREMISE_LIMIT` premises (factorial cost),
    matching the plan's own Excluder dedup-policy threshold."""
    if len(seq.premises) > PKEY_PREMISE_LIMIT:
        return seq.key()
    return seq.pkey()


# ----------------------------------------------------------------------------- random tree shapes


@dataclass
class TreeNode:
    """A binary tree node: either a leaf (`leaf_idx` set, an index into an external leaf list) or
    an internal node (`left`/`right` set). `formula`/`parent`/`depth` are filled in by callers as
    needed (`set_formula_and`/`set_formula_or`, `link_parents`)."""

    left: TreeNode | None = None
    right: TreeNode | None = None
    leaf_idx: int | None = None
    formula: str = ""
    orig_idx: int | None = None
    parent: TreeNode | None = None
    depth: int = 0


def random_tree(n: int, rng: random.Random) -> TreeNode:
    """A random binary tree over `n` leaves (indices `0..n-1`), balance-biased (the split point is
    near the middle, with jitter) so proof lengths stay small enough to fit the token/line caps
    even at large `n`."""

    def build(lo: int, hi: int) -> TreeNode:
        if lo == hi:
            return TreeNode(leaf_idx=lo)
        size = hi - lo + 1
        mid = lo + size // 2
        jitter = rng.randint(-max(0, size // 4), max(0, size // 4))
        split = min(hi - 1, max(lo, mid + jitter))
        return TreeNode(left=build(lo, split), right=build(split + 1, hi))

    return build(0, n - 1)


def set_formula_and(node: TreeNode, leaves: Sequence[str]) -> None:
    if node.leaf_idx is not None:
        node.formula = leaves[node.leaf_idx]
        return
    assert node.left is not None and node.right is not None
    set_formula_and(node.left, leaves)
    set_formula_and(node.right, leaves)
    node.formula = f"( {node.left.formula} & {node.right.formula} )"


def set_formula_or(node: TreeNode, leaves: Sequence[str]) -> None:
    if node.leaf_idx is not None:
        node.formula = leaves[node.leaf_idx]
        return
    assert node.left is not None and node.right is not None
    set_formula_or(node.left, leaves)
    set_formula_or(node.right, leaves)
    node.formula = f"( {node.left.formula} v {node.right.formula} )"


def internal_formulas(node: TreeNode, acc: list[str]) -> list[str]:
    if node.leaf_idx is None:
        acc.append(node.formula)
        assert node.left is not None and node.right is not None
        internal_formulas(node.left, acc)
        internal_formulas(node.right, acc)
    return acc


def link_parents(node: TreeNode, parent: TreeNode | None, depth: int) -> None:
    node.parent = parent
    node.depth = depth
    if node.leaf_idx is None:
        assert node.left is not None and node.right is not None
        link_parents(node.left, node, depth)
        link_parents(node.right, node, depth)


# --------------------------------------------------------------------------- local certification


class LocalMinlenCertifier:
    """A minimal `Certifier` (see `core.types.Certifier`) wrapping the vendored
    `takehome/minlen.py` directly. Used whenever no certifier is injected (i.e. `certifier is
    None` in `build()`) -- in particular for this WP3 build, since WP2's `logic.certify` module
    (`code/ndbench/logic/certify.py`) does not exist yet at the time this suite was built.

    Semantics per `seq`/`witness` pair, given `bound` (required; the plan's default of "the
    witness's own length" is resolved by callers before calling `certify`) and `time_s`:

    * `minlen` finds a proof of length `L <= bound` -> `LengthLabel(L=L, method="minlen_exact")`
      (the *found* length, which may be shorter than the witness if the witness wasn't minimal --
      callers compare this against their own expectation and drop the candidate if so).
    * `minlen` exhausts the search space up to `bound` without finding anything and without
      timing out -> nothing `<= bound` exists in `minlen`'s closure, i.e. the witness (if given)
      is minimal at exactly its own length -> `LengthLabel(L=witness_len, method="ub_lb",
      L_lb=bound+1)`.
    * `minlen` times out -> `LengthLabel(timeout=True, method="ub_lb", L=None, L_ub=witness_len)`:
      no claim either way.

    Runs batches of more than one sequent through a `multiprocessing.Pool` (matching vendored
    `minlen.py`'s own CLI), capped at `min(8, os.cpu_count() or 1)` workers per plan §A2.
    """

    def __init__(self, max_workers: int = 8) -> None:
        import os

        self.max_workers = max(1, min(max_workers, os.cpu_count() or 1))
        self._pool: _MPPool | None = None

    def _pool_for(self, n_jobs: int) -> _MPPool:
        """A single, lazily-created, reused `multiprocessing.Pool` for the object's lifetime
        (rather than one per `certify()` call): this build calls `certify` once per `(family, n)`
        cell, and pool start-up cost (fork + worker init) otherwise dominates runtime for the many
        small/cheap batches this suite certifies (see the WP3 report)."""
        if self._pool is None:
            self._pool = mp.Pool(self.max_workers)
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        with contextlib.suppress(Exception):
            self.close()

    def certify(
        self,
        seqs: Sequence[Sequent],
        witnesses: Sequence[str | None],
        *,
        bound: int | None = None,
        time_s: float = 60.0,
    ) -> list[LengthLabel]:
        from ..core.proof import line_count

        jobs: list[tuple[str, int, float]] = []
        for seq, witness in zip(seqs, witnesses, strict=True):
            b = bound if bound is not None else (line_count(witness) if witness else 12)
            jobs.append((seq.prompt(), b, time_s))

        results: list[_MinlenJobResult]
        if len(jobs) <= 1:
            results = [_minlen_job(j) for j in jobs]
        else:
            pool = self._pool_for(len(jobs))
            results = pool.map(_minlen_job, jobs)

        out: list[LengthLabel] = []
        for (_, b, _t), witness, r in zip(jobs, witnesses, results, strict=True):
            wl = line_count(witness) if witness else None
            if r["timeout"]:
                out.append(
                    LengthLabel(L=None, L_ub=wl, L_lb=None, method="ub_lb", bound=b, timeout=True)
                )
            elif r["min_lines_ub"] is not None:
                found = r["min_lines_ub"]
                out.append(
                    LengthLabel(
                        L=found,
                        L_ub=found,
                        L_lb=found,
                        method="minlen_exact",
                        bound=b,
                        timeout=False,
                    )
                )
            else:
                out.append(
                    LengthLabel(L=wl, L_ub=wl, L_lb=b + 1, method="ub_lb", bound=b, timeout=False)
                )
        return out


class _MinlenJobResult(TypedDict):
    min_lines_ub: int | None
    timeout: bool


def _minlen_job(job: tuple[str, int, float]) -> _MinlenJobResult:
    """Top-level (picklable) worker for `LocalMinlenCertifier`'s `multiprocessing.Pool`."""
    prompt, bound, time_s = job
    minlen = _minlen()
    r = minlen.minlen(prompt, bound=bound, time_limit=time_s)
    found = r["min_lines_ub"]
    return {
        "min_lines_ub": int(found) if found is not None else None,
        "timeout": bool(r["timeout"]),
    }


def default_certifier() -> Certifier:
    """WP2's `logic.certify.MinlenCertifier` if importable, else `LocalMinlenCertifier` (see that
    class's docstring, and this module's own docstring, for why the fallback is currently always
    used)."""
    try:
        from ..logic.certify import MinlenCertifier  # type: ignore[import-not-found]

        return MinlenCertifier()  # type: ignore[no-any-return]
    except Exception:
        return LocalMinlenCertifier()


def close_certifier_if_owned(certifier: Certifier, *, owned: bool) -> None:
    """Call `certifier.close()` if `owned` (i.e. this module created it via `default_certifier`,
    as opposed to one the caller injected and is responsible for) and it has one -- `Certifier` is
    a structural `Protocol` with no `close` method of its own (only `LocalMinlenCertifier` and,
    potentially, other implementations define one), so this goes through `getattr` rather than a
    direct attribute access that a type checker could statically rule out."""
    if not owned:
        return
    closer = getattr(certifier, "close", None)
    if callable(closer):
        closer()


# ------------------------------------------------------------------------------ Problem assembly


def _make_problem(
    *,
    suite: str,
    version: str,
    pid: str,
    seq: Sequent,
    length: LengthLabel,
    body: str,
    source: str,
    tags: dict[str, JSON],
    excluder: Excluder | None,
) -> Problem:
    from ..core.proof import make_reference

    seen_train, seen_eval = (False, ()) if excluder is None else excluder.status(seq)
    return Problem(
        pid=pid,
        suite=suite,
        version=version,
        sequent=seq,
        length=length,
        ref=make_reference(body, source),
        tags=tags,
        seen_train=seen_train,
        seen_eval=tuple(seen_eval),
    )


@dataclass
class _Candidate:
    seq: Sequent
    body: str
    predicted_L: int
    tags: dict[str, JSON]


def collect_candidates(
    n_target: int,
    make_candidate: Callable[[random.Random], _Candidate | None],
    rng: random.Random,
    *,
    max_attempts: int,
) -> list[_Candidate]:
    """Draw candidates from `make_candidate` until `n_target` pairwise-pkey-distinct ones are
    collected or `max_attempts` is exhausted (whichever first); every accepted candidate is
    additionally verified with `nd_verify` and checked against the token/line/premise caps before
    being kept."""
    verify_text = _verify_text()
    seen: set[str] = set()
    out: list[_Candidate] = []
    for _ in range(max_attempts):
        if len(out) >= n_target:
            break
        cand = make_candidate(rng)
        if cand is None:
            continue
        seq, body = cand.seq, cand.body
        if len(seq.premises) > PREMISE_CAP:
            continue
        ok, _reason, nl = verify_text(seq.prompt() + " " + body)
        if not ok or nl != cand.predicted_L:
            continue
        if nl > LINE_CAP:
            continue
        if count_tokens(seq.prompt(), body) > TOKEN_CAP:
            continue
        pk = pkey_or_key(seq)
        if pk in seen:
            continue
        seen.add(pk)
        out.append(cand)
    return out


# ----------------------------------------------------------------------------------- and_tower


def _and_tower_candidate(n: int, rng: random.Random) -> _Candidate:
    k = rng.randint(min(2, len(ATOMS)), len(ATOMS))
    gamma = rng.sample(ATOMS, k)
    lb = LineBuilder()
    prem_line = {a: lb.add(0, a, "PR") for a in gamma}
    acc_formula, acc_line = gamma[0], prem_line[gamma[0]]
    for _ in range(n):
        g = rng.choice(gamma)
        if rng.random() < 0.5:
            newf = f"( {acc_formula} & {g} )"
            newline = lb.add(0, newf, "ANDI", [acc_line, prem_line[g]])
        else:
            newf = f"( {g} & {acc_formula} )"
            newline = lb.add(0, newf, "ANDI", [prem_line[g], acc_line])
        acc_formula, acc_line = newf, newline
    seq = Sequent(premises=tuple(gamma), conclusion=acc_formula)
    return _Candidate(seq, lb.render(), len(gamma) + n, {"family": "and_tower", "n": n})


# ------------------------------------------------------------------------------------ and_tree


def _and_tree_candidate(
    n: int, rng: random.Random, *, distinct_attempts: int = 500
) -> _Candidate | None:
    gamma = list(ATOMS)
    for _ in range(distinct_attempts):
        leaves = [rng.choice(gamma) for _ in range(n)]
        tree = random_tree(n, rng)
        set_formula_and(tree, leaves)
        internals = internal_formulas(tree, [])
        if len(internals) == len(set(internals)):
            break
    else:
        return None
    lb = LineBuilder()
    prem_line = {a: lb.add(0, a, "PR") for a in gamma}

    def repack(node: TreeNode) -> int:
        if node.leaf_idx is not None:
            return prem_line[node.formula]
        assert node.left is not None and node.right is not None
        ll, rl = repack(node.left), repack(node.right)
        return lb.add(0, node.formula, "ANDI", [ll, rl])

    repack(tree)
    seq = Sequent(premises=tuple(gamma), conclusion=tree.formula)
    return _Candidate(seq, lb.render(), len(gamma) + (n - 1), {"family": "and_tree", "n": n})


# ------------------------------------------------------------------------------------ or_tower


def _or_tower_candidate(n: int, rng: random.Random) -> _Candidate:
    p = rng.choice(ATOMS)
    others = [a for a in ATOMS if a != p]
    lb = LineBuilder()
    p_line = lb.add(0, p, "PR")
    acc_formula, acc_line = p, p_line
    for _ in range(n):
        g = rng.choice(others)
        if rng.random() < 0.5:
            newf = f"( {acc_formula} v {g} )"
            rule = "ORI1"
        else:
            newf = f"( {g} v {acc_formula} )"
            rule = "ORI2"
        acc_line = lb.add(0, newf, rule, [acc_line])
        acc_formula = newf
    seq = Sequent(premises=(p,), conclusion=acc_formula)
    return _Candidate(seq, lb.render(), 1 + n, {"family": "or_tower", "n": n})


# ------------------------------------------------------------------------------------ imp_nest


def _imp_nest_candidate(n: int, rng: random.Random) -> _Candidate:
    names = [rng.choice(ATOMS) for _ in range(n)]
    i = rng.randrange(n)
    j = rng.randrange(n)
    concl_inner = f"( {names[i]} & {names[j]} )"
    formula = concl_inner
    for a in reversed(names):
        formula = f"( {a} > {formula} )"
    lb = LineBuilder()
    as_lines = [lb.add(d, a, "AS") for d, a in enumerate(names, start=1)]
    cur_end = lb.add(n, concl_inner, "ANDI", [as_lines[i], as_lines[j]])
    cur_formula = concl_inner
    for d in range(n, 0, -1):
        cur_formula = f"( {names[d - 1]} > {cur_formula} )"
        cur_end = lb.add(d - 1, cur_formula, "IMPI", [as_lines[d - 1], cur_end])
    seq = Sequent(premises=(), conclusion=formula)
    return _Candidate(seq, lb.render(), 2 * n + 1, {"family": "imp_nest", "n": n})


# --------------------------------------------------------------------------------- and_extract


def _and_extract_candidate(n: int, rng: random.Random) -> _Candidate:
    leaf = rng.choice(ATOMS)
    others = [a for a in ATOMS if a != leaf]
    dirs = [rng.choice(("L", "R")) for _ in range(n)]
    # Siblings must exclude `leaf` itself: a sibling equal to the target atom would give `minlen`
    # a second, shorter path to the same conclusion (a real shortcut, not a labelling error), so
    # the "each path step is a forced ANDE" argument only holds when `leaf` occurs exactly once.
    siblings = [rng.choice(others) for _ in range(n)]
    formula = leaf
    for d, s in zip(reversed(dirs), reversed(siblings), strict=True):
        formula = f"( {formula} & {s} )" if d == "L" else f"( {s} & {formula} )"
    lb = LineBuilder()
    cur_line = lb.add(0, formula, "PR")
    cur_formula = formula
    for d in dirs:
        toks = cur_formula[2:-2].split(" ")  # strip outer "( " " )"
        depth = 0
        split_at = None
        for k, t in enumerate(toks):
            if t == "(":
                depth += 1
            elif t == ")":
                depth -= 1
            elif t == "&" and depth == 0:
                split_at = k
                break
        assert split_at is not None
        left = " ".join(toks[:split_at])
        right = " ".join(toks[split_at + 1 :])
        if d == "L":
            cur_formula, rule = left, "ANDE1"
        else:
            cur_formula, rule = right, "ANDE2"
        cur_line = lb.add(0, cur_formula, rule, [cur_line])
    seq = Sequent(premises=(formula,), conclusion=leaf)
    return _Candidate(seq, lb.render(), n + 1, {"family": "and_extract", "n": n})


# ------------------------------------------------------------------------------------- dn_elim


def _dn_elim_candidate(n: int, rng: random.Random) -> _Candidate:
    atom = rng.choice(ATOMS)
    formula = atom
    for _ in range(2 * n):
        formula = f"( ~ {formula} )"
    lb = LineBuilder()
    cur_line = lb.add(0, formula, "PR")
    cur_formula = formula
    for _ in range(n):
        cur_formula = cur_formula[len("( ~ ( ~ ") : -len(" ) )")]
        cur_line = lb.add(0, cur_formula, "DN", [cur_line])
    seq = Sequent(premises=(formula,), conclusion=atom)
    return _Candidate(seq, lb.render(), n + 1, {"family": "dn_elim", "n": n})


# ------------------------------------------------------------------------------------ dn_intro


def _dn_intro_candidate(n: int, rng: random.Random) -> _Candidate:
    atom = rng.choice(ATOMS)
    h_formula = atom
    for _ in range(2 * n - 1):
        h_formula = f"( ~ {h_formula} )"
    target = f"( ~ {h_formula} )"
    lb = LineBuilder()
    prem = lb.add(0, atom, "PR")
    as_line = lb.add(1, h_formula, "AS")
    cur_formula, cur_line = h_formula, as_line
    for _ in range(n - 1):
        cur_formula = cur_formula[len("( ~ ( ~ ") : -len(" ) )")]
        cur_line = lb.add(1, cur_formula, "DN", [cur_line])
    negacc = lb.add(1, "F", "NEGE", [prem, cur_line])
    lb.add(0, target, "NEGI", [as_line, negacc])
    seq = Sequent(premises=(atom,), conclusion=target)
    return _Candidate(
        seq, lb.render(), n + 3, {"family": "dn_intro", "n": n, "classical_shortcut": True}
    )


# --------------------------------------------------------------------------------- reorder_and


_CONJUNCT_KINDS = ("atom", "not", "or", "imp")


def _rand_conjunct(rng: random.Random, *, atoms_only: bool) -> str:
    if atoms_only:
        return rng.choice(ATOMS)
    kind = rng.choice(_CONJUNCT_KINDS)
    if kind == "atom":
        return rng.choice(ATOMS)
    if kind == "not":
        return f"( ~ {rng.choice(ATOMS)} )"
    a, b = rng.sample(ATOMS, 2)
    return f"( {a} v {b} )" if kind == "or" else f"( {a} > {b} )"


def _sat_conjuncts(
    n: int, rng: random.Random, *, atoms_only: bool, attempts: int = 300
) -> list[str] | None:
    for _ in range(attempts):
        cand = [_rand_conjunct(rng, atoms_only=atoms_only) for _ in range(n)]
        if all(is_contingent(c) for c in cand) and is_satisfiable(cand):
            return cand
    return None


def _reorder_and_candidate(n: int, rng: random.Random, *, atoms_only: bool) -> _Candidate | None:
    if atoms_only and n > len(ATOMS):
        return None
    leaves = rng.sample(ATOMS, n) if atoms_only else _sat_conjuncts(n, rng, atoms_only=False)
    if leaves is None:
        return None
    prem_tree = random_tree(n, rng)
    set_formula_and(prem_tree, leaves)
    perm = list(range(n))
    rng.shuffle(perm)
    if perm == list(range(n)) and n > 1:
        perm[0], perm[1] = perm[1], perm[0]
    goal_tree = random_tree(n, rng)

    def set_goal(node: TreeNode) -> None:
        if node.leaf_idx is not None:
            node.orig_idx = perm[node.leaf_idx]
            node.formula = leaves[node.orig_idx]
            return
        assert node.left is not None and node.right is not None
        set_goal(node.left)
        set_goal(node.right)
        node.formula = f"( {node.left.formula} & {node.right.formula} )"

    set_goal(goal_tree)
    prem_internals = set(internal_formulas(prem_tree, []))
    goal_internals = internal_formulas(goal_tree, [])
    if prem_internals & set(goal_internals):
        return None  # would let a goal ANDI line be replaced by citing a premise-tree line

    lb = LineBuilder()
    prem_line = lb.add(0, prem_tree.formula, "PR")
    extract_line: dict[int, int] = {}

    def extract(node: TreeNode, line: int) -> None:
        if node.leaf_idx is not None:
            extract_line[node.leaf_idx] = line
            return
        assert node.left is not None and node.right is not None
        l1 = lb.add(0, node.left.formula, "ANDE1", [line])
        extract(node.left, l1)
        l2 = lb.add(0, node.right.formula, "ANDE2", [line])
        extract(node.right, l2)

    extract(prem_tree, prem_line)

    def repack(node: TreeNode) -> int:
        if node.leaf_idx is not None:
            assert node.orig_idx is not None
            return extract_line[node.orig_idx]
        assert node.left is not None and node.right is not None
        ll, rl = repack(node.left), repack(node.right)
        return lb.add(0, node.formula, "ANDI", [ll, rl])

    repack(goal_tree)
    seq = Sequent(premises=(prem_tree.formula,), conclusion=goal_tree.formula)
    return _Candidate(seq, lb.render(), 3 * n - 2, {"family": "reorder_and", "n": n})


# ---------------------------------------------------------------------------------- reorder_or


def _reorder_or_candidate(n: int, rng: random.Random, *, atoms_only: bool) -> _Candidate | None:
    if atoms_only and n > len(ATOMS):
        return None
    leaves = rng.sample(ATOMS, n) if atoms_only else _sat_conjuncts(n, rng, atoms_only=False)
    if leaves is None:
        return None
    prem_tree = random_tree(n, rng)
    set_formula_or(prem_tree, leaves)
    perm = list(range(n))
    rng.shuffle(perm)
    if perm == list(range(n)) and n > 1:
        perm[0], perm[1] = perm[1], perm[0]
    goal_tree = random_tree(n, rng)

    def set_goal(node: TreeNode) -> None:
        if node.leaf_idx is not None:
            node.orig_idx = perm[node.leaf_idx]
            node.formula = leaves[node.orig_idx]
            return
        assert node.left is not None and node.right is not None
        set_goal(node.left)
        set_goal(node.right)
        node.formula = f"( {node.left.formula} v {node.right.formula} )"

    set_goal(goal_tree)
    prem_internals = set(internal_formulas(prem_tree, []))
    goal_internals = internal_formulas(goal_tree, [])
    if prem_internals & set(goal_internals) or prem_tree.formula == goal_tree.formula:
        return None
    link_parents(goal_tree, None, 0)
    goal_leaf_of: dict[int, TreeNode] = {}

    def collect(node: TreeNode) -> None:
        if node.leaf_idx is not None:
            assert node.orig_idx is not None
            goal_leaf_of[node.orig_idx] = node
            return
        assert node.left is not None and node.right is not None
        collect(node.left)
        collect(node.right)

    collect(goal_tree)
    G = goal_tree.formula
    lb = LineBuilder()

    def ori_chain(node: TreeNode, line: int, depth: int) -> int:
        cur_formula, cur_line, cur = node.formula, line, node
        while cur.parent is not None:
            parent = cur.parent
            if parent.left is cur:
                assert parent.right is not None
                cur_formula = f"( {cur_formula} v {parent.right.formula} )"
                cur_line = lb.add(depth, cur_formula, "ORI1", [cur_line])
            else:
                assert parent.left is not None
                cur_formula = f"( {parent.left.formula} v {cur_formula} )"
                cur_line = lb.add(depth, cur_formula, "ORI2", [cur_line])
            cur = parent
        return cur_line

    def emit_or(node: TreeNode, cite_line: int, depth: int) -> int:
        if node.leaf_idx is not None:
            gnode = goal_leaf_of[node.leaf_idx]
            return ori_chain(gnode, cite_line, depth)
        assert node.left is not None and node.right is not None
        as_left = lb.add(depth + 1, node.left.formula, "AS")
        end_left = emit_or(node.left, as_left, depth + 1)
        as_right = lb.add(depth + 1, node.right.formula, "AS")
        end_right = emit_or(node.right, as_right, depth + 1)
        return lb.add(depth, G, "ORE", [cite_line, as_left, end_left, as_right, end_right])

    prem_line = lb.add(0, prem_tree.formula, "PR")
    emit_or(prem_tree, prem_line, 0)
    seq = Sequent(premises=(prem_tree.formula,), conclusion=G)
    return _Candidate(seq, lb.render(), len(lb), {"family": "reorder_or", "n": n})


# --------------------------------------------------------------------------------- ILTP-style


def _syj204_style_candidate(n: int, rng: random.Random) -> _Candidate:
    """`p_n, (p_n>(p_n>p_{n-1})), ..., (p_1>(p_1>p_0)) |- p_0`; `L = 3n+1` (module docstring)."""
    literals = rng.sample(ATOMS, n + 1)  # p_0 .. p_n, pairwise distinct plain atoms
    p = literals  # p[i] = p_i
    premises = [p[n]] + [f"( {p[i]} > ( {p[i]} > {p[i - 1]} ) )" for i in range(n, 0, -1)]
    prem_line = {}
    lb = LineBuilder()
    for prem in premises:
        prem_line[prem] = lb.add(0, prem, "PR")
    cur_atom = p[n]
    cur_line = prem_line[p[n]]
    for i in range(n, 0, -1):
        imp_line = prem_line[f"( {p[i]} > ( {p[i]} > {p[i - 1]} ) )"]
        step1 = lb.add(0, f"( {cur_atom} > {p[i - 1]} )", "IMPE", [imp_line, cur_line])
        step2 = lb.add(0, p[i - 1], "IMPE", [step1, cur_line])
        cur_atom, cur_line = p[i - 1], step2
    seq = Sequent(premises=tuple(premises), conclusion=p[0])
    return _Candidate(
        seq, lb.render(), 3 * n + 1, {"family": "ILTP-style", "n": n, "kind": "syj204_style"}
    )


def _syj203_style_candidate(n: int, rng: random.Random) -> _Candidate | None:
    """`|- ~~((p_1 & .. & p_n) v (~p_1 v .. v ~p_n))`, by reductio: assume the negation `H`,
    case-split on each `p_i` in turn (innermost-first: `p_1` outermost), deriving `F` in both the
    `p_i` and `~p_i` branches (the `p_i` branch only closes once *every* atom has been assumed
    true, via `ANDI` + `ORI1`; the `~p_i` branch injects the freshly-derived `~p_i` into the
    right-associated `disjneg` chain via `ORI1`/`ORI2` then `ORI2` into `body`, then `NEGE`s
    against `H`); closing the outermost box (`NEGI` on `H`) gives `~~body` directly (no final `DN`
    is needed -- `~~body` *is* the target). Hand-coded for `n in (1, 2)` only (plan §0: `minlen`
    already times out certifying `n=3`, so this build does not attempt a general-`n` version)."""
    if n not in (1, 2):
        return None
    ps = rng.sample(ATOMS, n)
    if n == 1:
        conj = ps[0]
        disjneg = f"( ~ {ps[0]} )"
    else:
        conj = f"( {ps[0]} & {ps[1]} )"
        disjneg = f"( ( ~ {ps[0]} ) v ( ~ {ps[1]} ) )"
    body_formula = f"( {conj} v {disjneg} )"
    neg_body = f"( ~ {body_formula} )"
    target = f"( ~ {neg_body} )"

    lb = LineBuilder()
    h = lb.add(1, neg_body, "AS")
    if n == 1:
        as_p = lb.add(2, ps[0], "AS")
        or1 = lb.add(2, body_formula, "ORI1", [as_p])
        f1 = lb.add(2, "F", "NEGE", [or1, h])
        not_p = lb.add(1, f"( ~ {ps[0]} )", "NEGI", [as_p, f1])
        or2 = lb.add(1, body_formula, "ORI2", [not_p])
        f_final = lb.add(1, "F", "NEGE", [or2, h])
    else:
        as_p1 = lb.add(2, ps[0], "AS")
        as_p2 = lb.add(3, ps[1], "AS")
        andi = lb.add(3, conj, "ANDI", [as_p1, as_p2])
        or1_inner = lb.add(3, body_formula, "ORI1", [andi])
        f_inner = lb.add(3, "F", "NEGE", [or1_inner, h])
        not_p2 = lb.add(2, f"( ~ {ps[1]} )", "NEGI", [as_p2, f_inner])
        disjneg_from_p2 = lb.add(2, disjneg, "ORI2", [not_p2])
        body_from_p2 = lb.add(2, body_formula, "ORI2", [disjneg_from_p2])
        f_mid = lb.add(2, "F", "NEGE", [body_from_p2, h])
        not_p1 = lb.add(1, f"( ~ {ps[0]} )", "NEGI", [as_p1, f_mid])
        disjneg_from_p1 = lb.add(1, disjneg, "ORI1", [not_p1])
        body_from_p1 = lb.add(1, body_formula, "ORI2", [disjneg_from_p1])
        f_final = lb.add(1, "F", "NEGE", [body_from_p1, h])
    lb.add(0, target, "NEGI", [h, f_final])
    seq = Sequent(premises=(), conclusion=target)
    predicted_L = 8 if n == 1 else 15
    return _Candidate(
        seq, lb.render(), predicted_L, {"family": "ILTP-style", "n": n, "kind": "syj203_style"}
    )


def _iff(a: str, b: str) -> str:
    return f"( ( {a} > {b} ) & ( {b} > {a} ) )"


def _iff_chain_style_candidate(n: int, rng: random.Random) -> _Candidate | None:
    """`A_1<->A_2, ..., A_{n-1}<->A_n |- A_1<->A_n`; `<->` expanded per spec.md §1 (no native
    `<->` token). `n=2,3` only; not certified (plan §0: `minlen` times out already at `n=2`)."""
    if n < 2:
        return None
    atoms = [rng.choice(ATOMS) for _ in range(n)]
    premises = [_iff(atoms[i], atoms[i + 1]) for i in range(n - 1)]
    lb = LineBuilder()
    prem_lines = [lb.add(0, p, "PR") for p in premises]
    # Each A_i<->A_{i+1} premise gives both directions (ANDE1/ANDE2); chain forward (A1->An) and
    # backward (An->A1) inside their own IMPI boxes, then ANDI the two implications together.
    as_fwd = lb.add(1, atoms[0], "AS")
    cur = as_fwd
    for i in range(n - 1):
        left_imp = lb.add(1, f"( {atoms[i]} > {atoms[i + 1]} )", "ANDE1", [prem_lines[i]])
        cur = lb.add(1, atoms[i + 1], "IMPE", [left_imp, cur])
    fwd_impi = lb.add(0, f"( {atoms[0]} > {atoms[-1]} )", "IMPI", [as_fwd, cur])

    as_bwd = lb.add(1, atoms[-1], "AS")
    cur_b = as_bwd
    for i in range(n - 2, -1, -1):
        right_imp = lb.add(1, f"( {atoms[i + 1]} > {atoms[i]} )", "ANDE2", [prem_lines[i]])
        cur_b = lb.add(1, atoms[i], "IMPE", [right_imp, cur_b])
    bwd_impi = lb.add(0, f"( {atoms[-1]} > {atoms[0]} )", "IMPI", [as_bwd, cur_b])

    lb.add(0, _iff(atoms[0], atoms[-1]), "ANDI", [fwd_impi, bwd_impi])
    seq = Sequent(premises=tuple(premises), conclusion=_iff(atoms[0], atoms[-1]))
    return _Candidate(
        seq, lb.render(), len(lb), {"family": "ILTP-style", "n": n, "kind": "iff_chain_style"}
    )


def _de_bruijn_style_candidate(n: int, rng: random.Random) -> _Candidate | None:
    """Peirce's law `((P>Q)>P)>P` as the `n=1` instance of a de-Bruijn-style hierarchy (module
    docstring: unverified against any ILTP source, used only because the plan names this member
    and gives no formula for it). `n=1` only. Proof: `IMPI` (assume `H=(A>B)>A`, want `A`); inside,
    classical reductio on `A` (assume `~A`, derive `(A>B)` vacuously via `NEGE`+`BOTE` under a
    further `AS A` box, `IMPE` with `H` to get `A`, `NEGE` with `~A` to close via `NEGI`+`DN`)."""
    if n != 1:
        return None
    a, b = rng.sample(ATOMS, 2)
    h = f"( ( {a} > {b} ) > {a} )"
    target = f"( {h} > {a} )"
    lb = LineBuilder()
    as_h = lb.add(1, h, "AS")
    as_not_a = lb.add(2, f"( ~ {a} )", "AS")
    as_a = lb.add(3, a, "AS")
    f1 = lb.add(3, "F", "NEGE", [as_a, as_not_a])
    b_line = lb.add(3, b, "BOTE", [f1])
    impi_ab = lb.add(2, f"( {a} > {b} )", "IMPI", [as_a, b_line])
    a_from_h = lb.add(2, a, "IMPE", [as_h, impi_ab])
    f2 = lb.add(2, "F", "NEGE", [a_from_h, as_not_a])
    nn_a = lb.add(1, f"( ~ ( ~ {a} ) )", "NEGI", [as_not_a, f2])
    a_line = lb.add(1, a, "DN", [nn_a])
    lb.add(0, target, "IMPI", [as_h, a_line])
    seq = Sequent(premises=(), conclusion=target)
    return _Candidate(
        seq, lb.render(), 11, {"family": "ILTP-style", "n": n, "kind": "de_bruijn_style"}
    )


# ------------------------------------------------------------------------------ build machinery


@dataclass
class FamilySpec:
    name: str
    n_grid: list[int]
    exact_max_n: int
    make_candidate: Callable[[int, random.Random], _Candidate | None]
    tier_above: str  # "witness_argument" | "ub_lb"
    #: Whether this family's module-docstring combinatorial lower-bound argument is robust to
    #: atom reuse (true for six of the nine "clean" families: `and_tower`, `and_tree`, `or_tower`,
    #: `imp_nest`, `and_extract`, `dn_elim` -- their forced-subformula argument holds regardless of
    #: which atoms repeat, by nesting-depth or enforced distinctness, see the module docstring).
    #: When true, an *inconclusive* exact-tier certification (timeout, or WP2's `"reference"`) is
    #: labelled `witness_argument` (`timeout=True`) instead of being dropped: the witness plus the
    #: argument already proves the length, `minlen` merely failed to confirm it fast enough. Only a
    #: *confirmed* shortcut (a shorter proof `minlen` actually found) is ever dropped. `dn_intro`
    #: and the two `reorder_*`/`ub_lb` families are excluded (main-agent instruction, 2026-09-30):
    #: `dn_intro`'s exact tier had zero drops in practice, and `reorder_and`/`reorder_or`'s
    #: argument is *not* robust once atom reuse is unavoidable (module docstring), so an
    #: inconclusive result there stays a drop (exact tier) / `ub_lb` timeout (post-exact tier).
    has_argument: bool = True


def _wa_spec(
    name: str,
    n_grid: list[int],
    exact_max_n: int,
    make_candidate: Callable[[int, random.Random], _Candidate | None],
    *,
    has_argument: bool = True,
) -> FamilySpec:
    """A `FamilySpec` whose above-exact-tier label is `"witness_argument"` (the common case: six
    of the nine "clean" families, per this module's docstring)."""
    return FamilySpec(name, n_grid, exact_max_n, make_candidate, "witness_argument", has_argument)


FAMILY_SPECS: list[FamilySpec] = [
    _wa_spec("and_tower", [1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 16, 19], 8, _and_tower_candidate),
    _wa_spec("and_tree", [4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 24, 28, 32], 10, _and_tree_candidate),
    _wa_spec("or_tower", [1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 16, 19], 8, _or_tower_candidate),
    _wa_spec("imp_nest", [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14], 6, _imp_nest_candidate),
    _wa_spec("and_extract", [1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 16, 19], 8, _and_extract_candidate),
    _wa_spec("dn_elim", [1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 15], 6, _dn_elim_candidate),
    _wa_spec("dn_intro", [1, 2, 3, 4, 5, 6, 7, 9, 12], 6, _dn_intro_candidate, has_argument=False),
    FamilySpec(
        "reorder_and",
        [2, 3, 4, 5, 6, 8, 10, 12],
        4,
        lambda n, rng: _reorder_and_candidate(n, rng, atoms_only=(n <= len(ATOMS))),
        "ub_lb",
        has_argument=False,
    ),
    FamilySpec(
        "reorder_or",
        [2, 3, 4, 5],
        3,
        lambda n, rng: _reorder_or_candidate(n, rng, atoms_only=(n <= len(ATOMS))),
        "ub_lb",
        has_argument=False,
    ),
]

#: `(kind, n_grid, exact_ns, make_candidate)` per ILTP-style sub-family (module docstring).
_IltpMakeFn = Callable[[int, random.Random], _Candidate | None]
IltpSpec = tuple[str, list[int], set[int], _IltpMakeFn]

ILTP_STYLE_SPECS: list[IltpSpec] = [
    ("syj204_style", [1, 2, 3], {1, 2, 3}, _syj204_style_candidate),
    ("syj203_style", [1, 2], {1, 2}, _syj203_style_candidate),
    ("iff_chain_style", [2, 3], set(), _iff_chain_style_candidate),
    ("de_bruijn_style", [1], {1}, _de_bruijn_style_candidate),
]


def _certify_batch(
    candidates: list[_Candidate],
    bounds: list[int],
    *,
    certifier: Certifier,
    time_s: float,
) -> list[LengthLabel]:
    """Certify `candidates` (aligned with `bounds`) in groups sharing the same bound (the
    `Certifier` protocol takes one `bound` per call), preserving input order in the result."""
    out: list[LengthLabel | None] = [None] * len(candidates)
    by_bound: dict[int, list[int]] = {}
    for i, b in enumerate(bounds):
        by_bound.setdefault(b, []).append(i)
    for b, idxs in by_bound.items():
        seqs = [candidates[i].seq for i in idxs]
        bodies = [candidates[i].body for i in idxs]
        labels = certifier.certify(seqs, bodies, bound=b, time_s=time_s)
        for i, lbl in zip(idxs, labels, strict=True):
            out[i] = lbl
    assert all(x is not None for x in out)
    return [x for x in out if x is not None]


def _label_and_filter(
    candidates: list[_Candidate],
    *,
    exact: bool,
    tier_above: str,
    certifier: Certifier,
    exact_time_s: float,
    ub_time_s: float,
    has_argument: bool = True,
) -> list[tuple[_Candidate, LengthLabel]]:
    """Certify `candidates` and drop the ones that fail (plan §0: "every constructed item ... is
    certified or dropped").

    Written against the `Certifier` *protocol*'s documented method semantics (plan §2), not
    against any one implementation's internal mechanics -- in particular, a witness is always
    passed alongside `bound`, and a conformant `Certifier` (e.g. WP2's real
    `logic.certify.MinlenCertifier`, importable as of this WP3 build -- see this module's
    docstring) is free to certify against the *witness's own verified length* rather than the
    literal `bound` argument. This function therefore never assumes which of `bound`/witness a
    given backend actually searched against: it only inspects the returned `method`/`L`.

    * `lbl.method == "minlen_exact"` and `lbl.L < predicted_L` -- a *confirmed* shortcut (the
      witness was not minimal after all): dropped unconditionally, exact tier or not.
    * `lbl.L == predicted_L` and `lbl.method in ("minlen_exact", "ub_lb")` -- confirmed (either a
      direct search hit, or a complementary exhaustion at `predicted_L - 1`): kept, and the
      certifier's own returned label is used as-is (preserving its real `method`/`bound`/
      `timeout` for provenance).
    * Anything else (timeout, or WP2's `"reference"` -- the witness's derivation lies outside
      `minlen`'s restricted search closure) -- if `has_argument` (main-agent instruction,
      2026-09-30: applies in the **exact** tier to the six families whose module-docstring
      combinatorial lower-bound argument is robust to atom reuse -- `and_tower`, `and_tree`,
      `or_tower`, `imp_nest`, `and_extract`, `dn_elim`; not `dn_intro` or the `reorder_*`
      families), the item is **kept**, labelled `witness_argument` with `timeout=True`: the
      witness plus the family's own argument already proves the length, `minlen` merely failed to
      confirm it fast enough, so dropping a perfectly good witness over a slow search would throw
      away real, correct items. Without `has_argument`, the **exact** tier drops it instead (it
      demands certainty and has no independent argument to fall back on); the **ub_lb** tier
      (`reorder_and`/`reorder_or` above their exact threshold) keeps it anyway, labelled `ub_lb`
      with `timeout=True`, honestly reporting that this instance's minimality is unverified rather
      than silently asserting it or discarding a witness that verified fine.
    """
    if not candidates:
        return []
    if not exact and tier_above == "witness_argument":
        return [
            (
                c,
                LengthLabel(
                    L=c.predicted_L,
                    L_ub=c.predicted_L,
                    L_lb=c.predicted_L,
                    method="witness_argument",
                    bound=c.predicted_L,
                    timeout=False,
                ),
            )
            for c in candidates
        ]
    time_s = exact_time_s if exact else ub_time_s
    bounds = [c.predicted_L for c in candidates]
    labels = _certify_batch(candidates, bounds, certifier=certifier, time_s=time_s)
    out = []
    for c, lbl in zip(candidates, labels, strict=True):
        if lbl.method == "minlen_exact" and lbl.L is not None and c.predicted_L > lbl.L:
            continue  # confirmed shortcut: drop regardless of tier
        if c.predicted_L == lbl.L and lbl.method in ("minlen_exact", "ub_lb"):
            out.append((c, lbl))
        elif exact and has_argument:
            # Inconclusive (timeout / "reference"), but this family's own lower-bound argument
            # (module docstring) is robust to atom reuse: fall back to it instead of dropping a
            # witness that verified fine just because minlen couldn't confirm it fast enough.
            out.append(
                (
                    c,
                    LengthLabel(
                        L=c.predicted_L,
                        L_ub=c.predicted_L,
                        L_lb=c.predicted_L,
                        method="witness_argument",
                        bound=c.predicted_L,
                        timeout=True,
                    ),
                )
            )
        elif exact:
            continue  # no argument to fall back on: exact tier demands certainty, so drop
        else:
            out.append(
                (
                    c,
                    LengthLabel(
                        L=None,
                        L_ub=c.predicted_L,
                        L_lb=None,
                        method="ub_lb",
                        bound=c.predicted_L,
                        timeout=True,
                    ),
                )
            )
    return out


def _emit_problems(
    labelled: list[tuple[_Candidate, LengthLabel]],
    *,
    suite: str,
    version: str,
    pid_prefix: str,
    excluder: Excluder | None,
) -> list[Problem]:
    problems = []
    for i, (c, lbl) in enumerate(labelled):
        tags = dict(c.tags)
        tags["lines"] = lbl.L if lbl.L is not None else c.predicted_L
        tags["tokens"] = count_tokens(c.seq.prompt(), c.body)
        pid = f"{pid_prefix}-{i:03d}"
        problems.append(
            _make_problem(
                suite=suite,
                version=version,
                pid=pid,
                seq=c.seq,
                length=lbl,
                body=c.body,
                source="witness",
                tags=tags,
                excluder=excluder,
            )
        )
    return problems


def build(
    cfg: Mapping[str, Any],
    *,
    certifier: Certifier | None,
    excluder: Excluder | None,
    rng_seed: int,
) -> tuple[list[Problem], dict[str, JSON]]:
    """Build the `families` suite (plan §2 B2, §3 WP3 row). `cfg` keys (all optional):

    * `families`: subset of family names to build (default: all ten, including `ILTP-style`).
    * `n_grid`: `{family_name: [n, ...]}` override of `FamilySpec.n_grid`.
    * `target_per_n` (default `TARGET_PER_N` = 40), `max_attempts` (default 400).
    * `exact_time_s` (default 30.0), `ub_time_s` (default 12.0): per-bound `Certifier` time budget
      for the exact and best-effort-`ub_lb` tiers respectively (plan §A2: keep any one CPU job
      well under 15 minutes; these defaults are chosen so a full build of one family's `n_grid`
      stays within a few minutes even in the worst case of every call timing out).
    * `version` (default `"v1"`).
    """
    owns_certifier = certifier is None
    certifier = certifier if certifier is not None else default_certifier()
    version = str(cfg.get("version", "v1"))
    target_per_n = int(cfg.get("target_per_n", TARGET_PER_N))
    max_attempts = int(cfg.get("max_attempts", 400))
    exact_time_s = float(cfg.get("exact_time_s", 30.0))
    ub_time_s = float(cfg.get("ub_time_s", 12.0))
    n_grid_overrides = cfg.get("n_grid", {})
    wanted = cfg.get("families")
    wanted_set = set(wanted) if wanted is not None else None

    master_rng = random.Random(rng_seed)
    all_problems: list[Problem] = []
    stats: dict[str, JSON] = {}

    try:
        _build_all(
            FAMILY_SPECS,
            ILTP_STYLE_SPECS,
            wanted_set,
            n_grid_overrides,
            master_rng,
            target_per_n,
            max_attempts,
            exact_time_s,
            ub_time_s,
            certifier,
            excluder,
            version,
            all_problems,
            stats,
        )
    finally:
        close_certifier_if_owned(certifier, owned=owns_certifier)
    return all_problems, stats


def _build_all(
    family_specs: list[FamilySpec],
    iltp_specs: list[IltpSpec],
    wanted_set: set[str] | None,
    n_grid_overrides: dict[str, list[int]],
    master_rng: random.Random,
    target_per_n: int,
    max_attempts: int,
    exact_time_s: float,
    ub_time_s: float,
    certifier: Certifier,
    excluder: Excluder | None,
    version: str,
    all_problems: list[Problem],
    stats: dict[str, JSON],
) -> None:
    for spec in family_specs:
        if wanted_set is not None and spec.name not in wanted_set:
            continue
        n_grid = n_grid_overrides.get(spec.name, spec.n_grid)  # type: ignore[union-attr]
        for n in n_grid:
            rng = random.Random(master_rng.random())
            candidates = collect_candidates(
                target_per_n,
                lambda r, n=n, mk=spec.make_candidate: mk(n, r),
                rng,
                max_attempts=max_attempts,
            )
            exact = n <= spec.exact_max_n
            labelled = _label_and_filter(
                candidates,
                exact=exact,
                tier_above=spec.tier_above,
                certifier=certifier,
                exact_time_s=exact_time_s,
                ub_time_s=ub_time_s,
                has_argument=spec.has_argument,
            )
            problems = _emit_problems(
                labelled,
                suite="families",
                version=version,
                pid_prefix=f"families-{spec.name}-n{n}",
                excluder=excluder,
            )
            all_problems.extend(problems)
            stats[f"{spec.name}.n{n}"] = {
                "target": target_per_n,
                "candidates_seen": len(candidates),
                "built": len(problems),
                "method": "minlen_exact" if exact else spec.tier_above,
            }

    if wanted_set is None or "ILTP-style" in wanted_set:
        for kind, n_grid, exact_ns, make_candidate in iltp_specs:
            n_grid = n_grid_overrides.get(f"ILTP-style.{kind}", n_grid)  # type: ignore[union-attr]
            for n in n_grid:
                rng = random.Random(master_rng.random())
                candidates = collect_candidates(
                    target_per_n,
                    lambda r, n=n, mk=make_candidate: mk(n, r),
                    rng,
                    max_attempts=max_attempts,
                )
                exact = n in exact_ns
                labelled = _label_and_filter(
                    candidates,
                    exact=exact,
                    tier_above="witness_argument",
                    certifier=certifier,
                    exact_time_s=exact_time_s,
                    ub_time_s=ub_time_s,
                    has_argument=False,  # not covered by the main agent's 2026-09-30 instruction
                )
                problems = _emit_problems(
                    labelled,
                    suite="families",
                    version=version,
                    pid_prefix=f"families-ILTP-style-{kind}-n{n}",
                    excluder=excluder,
                )
                all_problems.extend(problems)
                stats[f"ILTP-style.{kind}.n{n}"] = {
                    "target": target_per_n,
                    "candidates_seen": len(candidates),
                    "built": len(problems),
                    "method": "minlen_exact" if exact else "witness_argument",
                }
