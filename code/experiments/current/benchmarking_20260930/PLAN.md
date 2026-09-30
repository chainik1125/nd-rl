---
written_on: 2026-09-30
written_by: agent:claude
---

# ndbench — capability benchmarks for ND provers (plan)

Branch `leon/benchmarking` (worktree `/scratch/cowifornia/nd-rl/bench/nd-rl`). Planning agent: Opus; coders: Sonnet.
Requested by Leon on 2026-09-30: benchmarks 1-6 (glue, restricted families, skill profile, law × substitution
size, free lines, prefix completion), EDL, a Yue et al. base-vs-RL pass@k replication on textbook problems, an
abstract model interface, and a dashboard. Approved compute: GPU 4 only, up to ~10 GPU-h (hard cap 9.5 h below).
**Section A (main-agent amendments) at the end overrides anything above it.**

## 0. Verified facts & corrections (read first)

- **Not on this branch:** `length_generalisation_20260925/`, `midtrain_long_glued_20260926/` (on `leon/home-rolled-rl`,
  `/home/cowifornia/code/research/nd-rl`), `rl_dash.py` (`leon/rl-dashboards` cf26d14, `/scratch/cowifornia/nd-rl/ab/dash/nd-rl`).
  Everything needed goes into `code/ndbench/vendor/` as byte-identical copies with `SHA256SUMS` + source path/commit.
- The three `nd_verify/verify.py` copies (frozen lib, take-home, r1_cot/vendor) are identical (sha 5dc502c2…).
- Base `ab2/frozen/pretrain.pt` sha f4409e4f82d8922f; RL champion `final.pt` sha 548a9e343c71675f. The champion's RL
  pool was the full `ab2/data/targets_L6-11.jsonl` (15,738) + cap-6 retention.
- **Existing k=256 textbook results (max_new 512): base 9/58, RL 10/58**; 3 L6 items solved only by base (8, 3, 9 of
  256), 4 only by RL — Yue-style narrowing is plausible already.
- Throughput seen on the RTX 3090: 212–706 samples/s (lengthexp evaluations, compacting sampler, batch 512);
  148–348/s (older scorer, batch 384).
- Glue sets already exist: `ab3/sealed/matched_L6-14.jsonl`, `sealed_L15-18.jsonl` (ANDI/CHAIN, with `parts`,
  `part_L`, `label_kind` exact|ub_cert14).
- **`L_true` = minlen's minimum inside a restricted formula closure** — an upper bound, a lower bound only relative to
  the closure. **`textbook_dev` lengths are reference lengths, not minimal** (e.g. `~(P>Q) ⊢ P&~Q` labelled 18; minlen: 12).
- `necessity.py` = two bounded minlen searches (unrestricted vs one pattern forbidden): `requires` iff unrestricted finds
  ≤ bound and restricted exhausts the bound without timeout. Patterns: reductio (DN), derived_ore(_strict), impi_ore,
  negi_ande_hyp, nested_ore, depth4 (max_depth 3); impe_chain4 and ori_ore are "uses only". **No "any ORE"
  restriction.** `requires` is relative to bound and closure. `intuit.py` (G4ip) is a sound+complete "needs DN" test.
- **minlen checks run by the planner** (CPU, ~11 CPU-min):

| family / case | predicted L | minlen |
|---|---|---|
| ∧-tower `P,Q ⊢ ((P∧Q)∧P)…` (n ANDI) | n+2 | exact n=1..7 |
| ∨-tower (L/R nested) | n+1 | exact n=1..7 |
| →-nest `⊢ A1→(…→(An→(An∧A1)))` | 2n+1 | exact n=1..6 (n=6: 52 s) |
| ∧-extract from depth-n conjunction | n+1 | exact n=1..7 |
| DN-elim `~^{2n}P ⊢ P` | n+1 | exact n=1..5 |
| DN-intro `P ⊢ ~^{2n}P` | naive 3n+1 | **n+3** (classical shortcut via DN inside box) |
| reorder-∧ (reverse n conjuncts) | 3n−2 | exact n=2..4; timeout n=5 (90 s) |
| reorder-∨ | – | 6, 12 for n=2,3; timeout n=4 |
| SYJ204 Schwichtenberg | 3n+1 | exact n=1..3 |
| SYJ203 (con) `⊢ ~~((∧pi) ∨ (∨~pi))` | – | 8, 15 (n=1,2); timeout n=3 |
| ↔-chain n=2 (expanded) | – | timeout at bound 20 |
| extra premises on `(P>Q),P ⊢ Q` | 3+m | exact m=1..5 |
| **chain glue** `(P&Q)⊢(Q&P)` then `(Q&P),(Q>R)⊢R` | naive 7 | **4 — shortcut** |
| laws with random fills | contraposition 7, modus tollens 6 | preserved |
| negated_conditional with random fills | 12 | **9, 6, 8 when fills share atoms — shortcut** |

  Atomic law lengths: hypothetical syllogism 6, disjunctive syllogism 7, De Morgan or→nand 10, distributivity ∧ over ∨
  10, De Morgan nand→or 14.
- Token budgets: ∧-tower prompt+proof = 2n²+13n+18 tokens → n ≤ 19 fits 1024; →-nest n ≈ 15; reorder-∧ n ≈ 12.
- **Consequence: every constructed item (glue, padding, substitution, family instance) is certified or dropped.**

## 1. Architecture

```
code/ndbench/                      importable package: python -m code.ndbench (run from repo root)
  __main__.py cli.py               subcommands: build | run | edl | throughput | import-legacy | budget | dash
  core/types.py formula.py keys.py proof.py io.py
  vendor/  nd_verify/  takehome/{minlen,intuit,patterns,patterns2,necessity,gen,textbook_pool}.py
           absgpt/{model,tokenizer}.py (frozen lib)  sampler.py ndcheck.py (lengthexp)
           lean_seq/lean_seq.py (origin/dan_lean_seq_tokenizer c298be6)  SHA256SUMS  README.md
  judge/base.py nd.py lean.py
  adapters/base.py fake.py absgpt.py lean_seq.py step_env.py api_llm.py
  runner/runner.py adaptive.py gpu.py
  store.py metrics.py edl.py
  logic/certify.py skills.py exclusion.py
  suites/__init__.py (lazy registry) existing.py families.py laws.py padding.py glue.py prefix.py
  analysis/__init__.py yue.py horizon.py glue.py skills.py subst.py padding.py prefix.py edl.py
  dash/primitives.py (ported from rl_dash.py cf26d14) pages.py build.py
code/tests/ndbench/test_<wp>_*.py
code/experiments/current/benchmarking_20260930/  PLAN.md README.md configs/*.yaml schedule.yaml runbook.md
experiment-summaries/2026-09-30-benchmarking/    (after results)
Outputs: /scratch/cowifornia/nd-rl/bench/out/{suites,store,cache,dash,logs}
```

Vendored take-home scripts are loaded via a shim that puts `vendor/takehome` and `vendor` on `sys.path` (they keep their
own `from nd_verify import …`); `absgpt/model.py`/`tokenizer.py` are loaded with `importlib` under unique names
(`ndbench_absgpt_*`) to avoid top-level `model`/`tokenizer` collisions.

### Interfaces (WP0 writes these exactly)

```python
# core/types.py  (JSON = str|int|float|bool|None|list|dict)
@dataclass(frozen=True)
class Sequent:
    premises: tuple[str, ...]; conclusion: str          # spec-format formula strings
    def prompt(self) -> str; def thm(self) -> str; def key(self) -> str; def pkey(self) -> str
    @classmethod def from_prompt(cls, s: str) -> "Sequent"; @classmethod def from_thm(cls, s: str) -> "Sequent"
@dataclass(frozen=True)
class ReferenceProof: body: str; lines: int; pruned: int; source: str   # minlen|witness|constructed|model; body N1-based
@dataclass(frozen=True)
class LengthLabel:
    L: int | None; L_ub: int | None; L_lb: int | None
    method: str        # minlen_exact | ub_lb | witness_argument | reference
    bound: int | None; timeout: bool = False
@dataclass(frozen=True)
class PrefixSpec: j: int; body: str                    # first j lines, no QED, N1-based
@dataclass(frozen=True)
class Problem:
    pid: str; suite: str; version: str; sequent: Sequent; length: LengthLabel
    ref: ReferenceProof | None; tags: Mapping[str, JSON]
    parts: tuple[str, ...] = (); prefix: PrefixSpec | None = None
    seen_train: bool = False; seen_eval: tuple[str, ...] = ()
    def to_json(self) -> dict[str, JSON]; @classmethod def from_json(cls, d) -> "Problem"
@dataclass
class SuiteManifest: suite: str; version: str; builder: str; config: dict; seed: int; created_utc: str
    git_commit: str; n_items: int; items_sha256: str; excl_sources: dict[str, str]; stats: dict
@dataclass(frozen=True)
class SampleRequest: pid: str; sequent: Sequent; n: int; temperature: float; max_new_tokens: int
    seed: int; prefix: PrefixSpec | None = None; options: Mapping[str, JSON] = {}
@dataclass(frozen=True)
class Attempt: pid: str; i: int; body: str; native: str | None; n_tokens: int; finished: bool
    # body: canonical ND body (prefix included) or "UNPARSEABLE <why>"
@dataclass(frozen=True)
class Verdict: ok: bool; reason: str; kind: str   # ok|bad_line|early_qed|no_qed|unparseable
    written: int | None; pruned: int | None; first_bad_line: int | None; start: int | None
@dataclass
class ItemResult: run_id: str; pid: str; chunk: int; n: int; c: int; first_success: int | None
    ok_written: list[int]; ok_pruned: list[int]; ok_tokens: list[int]; ok_starts: list[int]
    fail_kinds: dict[str, int]; fail_lines: list[int]; long_enough: int; tokens_mean: float; hit_max: int
@dataclass(frozen=True)
class LogProb: nats: float; n_tokens: int; per_encoding: tuple[float, ...]   # marginal over shift encodings
class Certifier(Protocol):
    def certify(self, seqs: Sequence[Sequent], witnesses: Sequence[str | None], *,
                bound: int | None = None, time_s: float = 60.0) -> list[LengthLabel]: ...
class Excluder(Protocol):
    def status(self, seq: Sequent) -> tuple[bool, tuple[str, ...]]: ...        # (in_train, eval sets)

# adapters/base.py
class Capability(enum.Flag): SAMPLE = 1; PREFIX = 2; LOGPROB = 4; FINETUNE = 8; STEP_ENV = 16  # + ACTIVATIONS, STEER (§A3)
class ModelAdapter(abc.ABC):
    label: str; format_id: str; capabilities: Capability
    @abc.abstractmethod def identity(self) -> dict[str, JSON]           # ckpt, sha256, arch, tokenizer version
    @abc.abstractmethod def token_count(self, seq: Sequent, body: str | None = None) -> int
    @abc.abstractmethod def sample(self, reqs: Sequence[SampleRequest]) -> list[list[Attempt]]
    def logprob(self, pairs: Sequence[tuple[Sequent, str]]) -> list[LogProb]: raise NotSupported
    def learner(self, *, seed: int) -> "Learner": raise NotSupported
    def close(self) -> None: ...
class Learner(Protocol):
    def nll(self, pairs: Sequence[tuple[Sequent, str]]) -> list[LogProb]
    def update(self, pairs: Sequence[tuple[Sequent, str]], lr: float) -> float   # one optimiser step
    def adapter(self) -> ModelAdapter                                            # sample with current weights
ADAPTERS: dict[str, str] = {"absgpt": "code.ndbench.adapters.absgpt:AbsGptAdapter", "fake": ...,
    "lean_seq": ..., "step_env": ..., "api_llm": ...}                     # lazy "module:attr" map

# judge/base.py
class Judge(Protocol):
    judge_id: str
    def judge(self, seq: Sequent, attempts: Sequence[Attempt], ref_lines: int | None) -> list[Verdict]

# runner/runner.py
@dataclass(frozen=True) class AdaptiveRule: chunk: int; n_max: int; stop_c: int
@dataclass(frozen=True) class RunSpec: suite: str; version: str; model: str; k: int
    temperature: float = 0.8; max_new_tokens: int | None = None   # None -> 1024 - prompt tokens
    seed: int = 0; adaptive: AdaptiveRule | None = None; pids: tuple[str, ...] | None = None
    options: Mapping[str, JSON] = {}; keep_samples: bool = True; budget_s: float | None = None
def run(spec: RunSpec, adapter: ModelAdapter, judge: Judge, store: "Store") -> dict[str, JSON]
    # item seed = sha256(seed, pid, chunk); run_id stable incl. config hash

# store.py  (append-only JSONL under out/store)
class Store:
    def __init__(self, root: Path)
    def write_suite(self, problems, manifest) -> None;  def load_suite(self, suite, version) -> list[Problem]
    def new_run(self, spec, identity, judge_id) -> str
    def append_items(self, run_id, rows: Sequence[ItemResult]) -> None
    def append_samples(self, run_id, rows) -> None                   # gz; body+verdict per sample
    def finish_run(self, run_id, status: str, gpu_s: float) -> None # complete|partial|failed
    def runs(self, **match) -> list[dict]; def items(self, run_id) -> list[ItemResult]   # chunks merged
    def latest(self, suite, model, config_key) -> str | None
    def append_edl(self, run_id, rows) -> None; def ledger_append(self, entry) -> None
    def ledger_total_s(self) -> float

# metrics.py (pure python + numpy)
pass_at_k(n, c, k) -> float                                           # unbiased; == lengthexp common.pass_at_k
pass_curve(rows, ks, B=2000, seed=0) -> list[tuple[int, float, float, float]]  # (k, mean, lo, hi) item bootstrap
clopper_pearson(c, n, a=.05); wilson(c, n, a=.05)
horizon(by_n: Mapping[int, Sequence[ItemResult]], k=8, tau=.5, B=2000) -> dict   # n*, CI, curve
compositionality_ratio(groups: Sequence[tuple[ItemResult, Sequence[ItemResult]]], mastery=.5, B=2000) -> dict
hazard(rows) -> list[tuple[int, int, int, float]]                      # line, at_risk, failed, hazard
logit_slope(x: Sequence[float], n: Sequence[int], c: Sequence[int]) -> tuple[float, float]  # binomial IRLS
edl(online_nats: Sequence[float], test_nats_final: float) -> float
```

Store files: `out/store/runs.jsonl`, `items/<run_id>.jsonl` (ItemResult per chunk), `samples/<run_id>.jsonl.gz`,
`edl/<run_id>.jsonl`, `ledger.jsonl`. `import-legacy` loads the existing `*_per_theorem.jsonl` and lengthexp `eval/*`
rows as runs with `status=legacy` (n, c only; max_new noted).

**How each adapter fits**

| adapter | capabilities | notes |
|---|---|---|
| `absgpt` (now) | SAMPLE, PREFIX, LOGPROB, FINETUNE (+ACTIVATIONS, STEER §A3) | loads like `evaluate.load_model` (strip `_orig_mod.`); samples via the vendored compacting sampler's `_gen` on explicit id lists (prefix encoded as proof ids; `options.start_index=1` forces N1); LOGPROB = logsumexp over every start shift s∈[0, 64−max index] (as `novelty.py`); Learner = AdamW, trains on random shifts |
| `lean_seq` (future) | SAMPLE, PREFIX, LOGPROB, FINETUNE | `LeanSeqTokenizer`; decode with `inverse` → ND body or UNPARSEABLE; shift group = name shifts (n1..n64); judged by ND or `LeanJudge` |
| `step_env` (future) | SAMPLE, PREFIX, STEP_ENV | takes `env_factory(seq)` (`reset()`, `step(action)` → state) and `policy(state_text) -> action`; runs the loop and returns the environment's proof as an ND body; prefix = replay j steps. WP7 ships a local `NDLineEnv` (one ND line per step) to test the contract before Dan's env exists |
| `api_llm` (future) | SAMPLE, PREFIX | prompt = spec excerpt + few-shot + sequent; parses the `N1…QED` span; raw response in `native`; cost cap, dry-run mode |
| `LeanJudge` (future) | – | wraps `lean_check.judge`; `NotAvailable` if no `lean` binary |

## 2. Benchmarks

Shared settings: T=0.8; max_new = 1024 − prompt tokens; drop items with prompt+reference > 1000 tokens; 4 atoms, ≤8
premises, ≤64 lines. **Dedup:** `Excluder` from the train tier (`ab3/frozen/data/train_cap6.jsonl`,
`heldout_cap6.jsonl`, `ab2/data/targets_L6-11.jsonl`; for arm C/D also `ab3/frozen/data/targets_L6-11.jsonl`,
`ab3/data_comp/targets_L6-11.jsonl`) matched on key or pkey (pkey skipped when >5 premises — no training theorem has
more than 5); eval-tier files (dev, sealed, frontier, reserve, `ab3/sealed`) are only flagged. Policy: **drop** for
joined, law, padding and prefix items; **flag** for family members and parts (small n surely in training). All `L`
labels come from `Certifier`: `minlen_exact` (iterative deepening to witness length), `ub_lb` (verified witness +
minlen exhausting b < L_ub without timeout), `witness_argument` (family lower-bound argument, minlen-checked for small
n), `reference` (textbook_dev only); each item keeps written lines and reference tokens. minlen runs ≤16 processes
(see §A2), ~2 GB/worker (isearch-style interning at L ≥ 16). CIs: item-level bootstrap, B=2000, 95%.

### B1 Glued benchmark (compositionality)
- **Parts pool** (`parts_v1`): 1,200 `gen` items from `ab2/data/reserve.jsonl` with `minlen_proof`, stratified by
  L 2–8 × atom count 1–4, one per pkey. **G1 run:** k=32 per part per model; mastered = p̂ ≥ 0.5 (per model).
- **Operators** (witness = part proofs concatenated with renumbered refs; reuse `ab3/data_build/tools/comp.py`):

| op | construction | witness L |
|---|---|---|
| AND | parts renamed to disjoint atoms; random binary ∧-tree (`build_mid.combine_tree`) | ΣLi + k − 1 |
| CHAIN | Γ1⊢B, B,Γ2⊢C (B compound, cited; `comp.combine_chain`, `tight`); k-chains | ΣLi − (k−1) |
| CASE-dilemma | A,Γ1⊢C1 and B,Γ2⊢C2 on disjoint atoms ⇒ A∨B,Γ1,Γ2 ⊢ C1∨C2 (AS, body, ORI per branch, ORE) | L1+L2+4−\|Γ1∩Γ2\| |
| CASE-same | as above, both parts conclude the same C ⇒ A∨B,… ⊢ C | L1+L2+2−\|Γ1∩Γ2\| |
| BOX | Γ,A1..Ad ⊢ B ⇒ Γ ⊢ A1→(…→B) (d=1..3) | Lpart + d |

- **Certification:** every joined item needs `minlen_exact` or `ub_lb` with b ≥ L−1, else dropped (chain shortcut);
  CASE items also need `needs_ore` true (B3).
- **Cells** (~23): AND k∈{2,3,4} at total L=12 and part length ℓ∈{2..7} at k=2; CHAIN k∈{2,3} at L=10 and
  ℓ∈{3,4,5}; CASE-dilemma ℓ∈{2..5}; CASE-same pooled; BOX d∈{1,2,3} (parts L 3–7). Reuse matched AND/CHAIN L6–14 as an
  extra length sweep. **Joined suite:** 100 items/cell, built after G1 from parts the RL model mastered; k=32.
- **Metrics:** R = Σ_i p̂_joined,i / Σ_i Π_j p̂_part,ij over items where **that model** mastered every part; R_min with
  min_j; bootstrap resamples joined items with their parts. Also: fraction of attempts ≥ reference length,
  failure location (which part's atoms the rejected line uses, as `failloc.py`), control pass@1 on standard items at
  the same L.

### B2 Restricted-class families (horizon)
~40 distinct classes per (family, n) (fewer only if the class is small); randomised structure (left/right, which atoms,
sibling fillers).

| family | instance | L(n) | n range | certification |
|---|---|---|---|---|
| and_tower | atoms Γ ⊢ tower, random side | n+\|Γ\| | 1–19 | exact ≤8; argument: each of n distinct ∧-subformulas needs its own ANDI line (no compound premise) |
| and_tree | balanced random ∧-tree over Γ, distinct internal nodes | \|Γ\|+#internal | 4–36 | exact ≤10; same argument |
| or_tower | P ⊢ tower, ORI1/ORI2 mixed | n+1 | 1–19 | exact ≤8 |
| imp_nest | ⊢ A1→…→An→(Ai∧Aj) | 2n+1 | 1–15 | exact ≤6; depth argument (goal-directed; an implication with no premises needs IMPI) |
| and_extract | depth-n conjunction ⊢ a leaf, random path | n+1 | 1–19 | exact ≤8 |
| dn_elim / dn_intro | ~^{2n}P ⊢ P / P ⊢ ~^{2n}P | n+1 / n+3 | 1–16 / 1–12 | exact ≤6; dn_intro records whether the model finds the shortcut |
| reorder_and | n contingent, jointly satisfiable, non-∧ conjuncts, random permutation + bracketing | ≤3n−2 | 2–12 | exact ≤4; ub_lb above |
| reorder_or | same with ∨ | 6, 12, … | 2–5 | exact ≤3; ub_lb |
| ILTP-small | SYJ204 (lifted to n ≤ 7 with distinct literals), SYJ203 n ≤ 3, ↔-chain n=2,3, de Bruijn n=1 | 3n+1 / 8,15 / – | small | minlen at bound 24, 600 s, or WP3 witness constructor; drop if neither |

- Sampling k=32/item; stop increasing n for a (family, model) after two consecutive n with pass@32 = 0 (+ one check at
  the largest n). For L ≥ 24 also run with `start_index=1` (separates N64 overflow from failure; see `ndcheck`).
- Metrics: pass@1 and pass@k vs n, vs reference lines, vs reference tokens; n*_{k,τ} = largest n with pass@k ≥ τ for
  all n′ ≤ n (default k=8, τ=0.5; also pass@1 ≥ 0.5 and pass@32 > 0); bootstrap CI on n*; logit slope per added line
  and per 100 tokens.

### B3 Skill profile
Tags (WP2, `logic/skills.py`):

| skill | test | exact? |
|---|---|---|
| `classical` | not intuitionistically valid (vendored `intuit.py`) | exact |
| `case_split` | truth-table test below | exact |
| `derived_case`, `nested_case`, `case_in_box`, `negi_ande_hyp` | `necessity.py` at bound = L_true: every minimal-length proof in the closure uses it | bounded |
| `depth≥3` | minlen with max_depth=2 fails at bound L_true | bounded |
| `uses_*` | rule counts in the pruned reference proof | descriptive |

**case_split test:** replace each disjunction subformula D=(X∨Y) of Γ∪{C} by a fresh atom p_D and add axioms X*→p_D,
Y*→p_D; Γ ⊢ C has an ORE-free proof iff Γ* ∪ Ax ⊨ C* (sound: ORI is sound under the axioms, other disjunctions may be
set true; complete: the {¬,∧,→} fragment with DN is classically complete and each axiom is realised by a box
{AS X, ORI}). Truth table over ≤2^16 rows. Unit test: on 500 labelled theorems, "required" ⇒ minlen's shortest proof
contains ORE.
- Items: dev L8–11 (1,200), sealed L12–14, frontier, lengthexp std/matched L6–18. Existing per-theorem results cover
  base+RL on dev/sealed/frontier and pretrain+C on lengthexp sets → **no new samples except RL on std L6–7 (600×32).**
- Metrics: pass@1 / pass@32 heat-map skill × L with cell counts; binomial GLM pass ~ L + skill indicators with CIs.

### B4 Law × substitution size
- Schemas (atomic L): contraposition 7, modus_tollens 6, hypothetical_syllogism 6, disjunctive_syllogism 7,
  negated_conditional 12, demorgan_or_to_nand 10, dist_and_over_or 10, demorgan_nand_to_or (classical) 14.
- Fillers: each metavariable gets a random formula with exactly s connectives, s ∈ {0,1,2,3,4,6,8,10,12} up to the
  token cap; pairwise-disjoint atom sets per metavariable (2+2, 2+1+1, 1+1+1+1); contingent fillers; jointly
  satisfiable premises.
- Labels: witness = substituted atomic minimal proof (always valid, rules are schematic) → L_ub = L_atomic; keep an
  instance only if minlen at bound L_atomic−1 exhausts (`ub_lb`); if a shorter proof is found, drop; if minlen times
  out, keep as `witness_argument` and plot separately.
- 30 instances per (schema, s) = 2,160 items, k=32. Metrics: pass@1 vs s and vs tokens at fixed L; logit slope per
  100 tokens per schema; comparison with the uncontrolled textbook-law items in dev/sealed.

### B5 Free lines (padding)
- Bases: parts mastered by the model (p̂ ≥ 0.7 in G1), L 3–6, ≤3 premises; 60 per model group.
- Pad types (m steps): **P1-shared** m extra premises over the goal's atoms (formulas s ≤ 2); **P1-disjoint** m extra
  premises over unused atoms; **P2** goal C → ((C∧X)∧X′)… with X from premises (adds m ANDI lines, no premises;
  separates "more premises" from "more lines"). Premises ≤ 8 total.
- Certification: joint satisfiability + minlen_exact with L = L_base + m; else drop.
- m ∈ {0..5}; ~1,080 items; k=32. Metric: pass@1 vs written lines W for padded-easy vs natural items at the same W
  (existing results at matched L); "free-line cost" = logit slope per padding line with CI, per pad type.

### B6 Prefix completion
- Proofs: 50 per L bucket, L ∈ {10,14,18,24,30,40}, from family/glue witnesses, `ab3/sealed` minlen/constructed proofs
  (L15–18) and sealed L12–14 `minlen_proof`.
- Prefix j ∈ {0, n_prem, ⌈L/4⌉, ⌈L/2⌉, ⌈3L/4⌉, L−3, L−2, L−1} (dedup); forced prefix N1-based and recorded; success =
  prefix + continuation passes nd_verify (any valid completion). k=16 (~67k samples).
- Analysis: p(success) by j, by lines left L−j, by absolute position; hazard of first failing line by absolute index
  among samples still alive; logit p = a + b(L−j) + c·j with proof random intercepts (planning horizon vs position).

### B7 Excess description length
- EDL = Σ_{i=1..n} ℓ(θ_{i−1}; x_i, y_i) − n·L_test(θ_n) (nats). ℓ = −log Σ_s p_θ(shift_s(y) | x), proof tokens incl.
  QED, conditioned on the prompt, **summed over every allowed start shift** (the paper's "normalise the output format";
  removes ~4 nats/example of start-index entropy); also record the start-token-excluded variant as a sensitivity check.
  y = the item's minlen/witness proof normalised to N1.
- Protocol: one online pass, per-example update = one AdamW step (β 0.9/0.95, wd 0, clip 1.0) on 8 random-shift copies
  of the example, after scoring it. n=64 training, m=64 held-out from the same class; 2 data orders (same across
  models); LR ∈ {1e-4, 3e-4, 1e-3}; report EDL per LR and EDL* = min over LR + ln 3 (hyperparameter cost). Also
  EDL(n′) for n′ ∈ {8,16,32,64} (L_test at checkpoints), L_test(θ_0) and L_test(θ_n), and pass@8 of θ_0 and θ_n on
  the held-out theorems — **a small EDL counts as "elicited" only if θ_n actually gained capability.**
- Classes (12): and_tower n=8; and_tower n=16; imp_nest n=4; imp_nest n=8; reorder_and n=5; reorder_or n=3;
  contraposition s=3 (control); dist_and_over_or s=1 (case split); demorgan_nand_to_or s=1 (classical); AND glue L=12;
  dev gen L10; heldout cap-6 (null control, EDL ≈ 0). 32/32 when a class has < 128 distinct classes. Base and RL
  (+C if budget).

### B8 Yue et al. replication
- Sets: **textbook_dev** (58), relabelled by minlen (bound 16, 120 s), plotted by the new L with the reference L as a
  second label; **textbook-law**: stratified subsample of `source=='textbook'` items, ≤6 per (L, schema), ~40 per L for
  L8–14 (~280; sources dev L8–11 480, sealed L12–14 691); **control:** 60 dev `gen` items L8–11 at k_max 512.
- Draws: chunks of 256, adaptive, k_max 4096 (textbook_dev) / 1024 (textbook-law); stop an item when c ≥ 32 after its
  first chunk. pass@k for k ≤ 256 uses only the first chunk (fixed n → unbiased); for k > n_i use 1 − (1−p̂)^k, bounded
  with the Clopper–Pearson lower limit (error < 1e-5 at c ≥ 32).
- Outputs: pass@k k = 2^0..2^12, base vs RL, per set and per length; crossover k* with bootstrap distribution; coverage
  at k_max; solved-set Venn (is RL's solved set ⊆ base's?); consistency vs the legacy k=256 counts (max_new 512 — valid
  only for short proofs).

## 3. Work packages

WP0 first; WP1–WP7 in parallel afterwards. Each WP owns only its listed files. **Only WP0 edits `pyproject.toml` and
`conftest.py`** (add ndbench to pyright include, exclude `**/ndbench/vendor/**` from ruff/pyright, setuptools
packages). Builders take `certifier`, `excluder`, `rng_seed` by injection so WP3/WP4 can test with stubs or vendored
minlen before WP2 lands. Builder signature: `build(cfg, *, certifier, excluder, rng_seed) -> tuple[list[Problem], dict]`.

| WP | files (under code/ndbench unless noted) | API contract | acceptance tests (CPU, < 60 s per file) | done when |
|---|---|---|---|---|
| **WP0 core** | `__main__`, cli, `core/*`, store, metrics, `judge/{base,nd}`, `adapters/{base,fake}`, `runner/{runner,adaptive}`, lazy registries in `suites/__init__` and `analysis/__init__` (all planned module paths), `vendor/**`, pyproject, `tests/ndbench/test_wp0_*.py` | §1 exactly | pass_at_k == `common.pass_at_k` on a grid; Sequent round-trip prompt↔thm and matches vendored `canon_key`; NDJudge agrees with nd_verify + `ndcheck` on 60 fixture proofs + 200 corruptions; runner with FakeAdapter (returns the ref with prob p) gives expected n, c, adaptive stopping, deterministic seeds; store append/merge/latest; vendor SHA256 check; `python -m code.ndbench --help` | `make -C code check` + tests green |
| **WP1 absgpt + GPU** | `adapters/absgpt.py`, `runner/gpu.py`, `tools/throughput.py` (package module), `tests/ndbench/test_wp1_*.py` | `AbsGptAdapter(ckpt, device, batch=1024)`; `gpu.lease(job, est_s)` ctx manager (gpulock.acquire([4], owner=pid), **asserts GPU 4**, sets CUDA_VISIBLE_DEVICES before torch import, writes ledger); `budget_guard(est_s, cap_h=9.5)`; `throughput(models, sets, batches) -> json` | tiny random GPT (1 layer, d=32) on CPU: decode + prefix preserved; `logprob` = manual logsumexp over shifts; `update` lowers NLL on a repeated example; both real ckpts load on CPU with matching sha; 3-line theorem verifies within 64 samples; lock test with `AB_GPULOCK_DIR`=tmp and GPU ≠ 4 raises | smoke on GPU 4: 1,000 samples logged in the ledger |
| **WP2 logic + existing suites** | `logic/{certify,skills,exclusion}.py`, `suites/existing.py`, `tests/ndbench/test_wp2_*.py` | `MinlenCertifier(procs, mem_gb=2)` implements Certifier (method semantics §2); `skills.classical`, `needs_ore`, `necessity(seq, pattern, bound, time) -> requires/not_required/unknown`, `needs_depth`, `rule_uses`; `ExclusionIndex.build(tiers) -> Excluder` (cached at out/cache); builders `textbook_dev`, `textbook_law`, `gen_ctrl`, `skill_pool`, `parts_pool`; `import_legacy(store)` | certifier reproduces minlen selftest cases and ≥6 rows of the §0 table; `needs_ore` on hand cases + 500 labelled theorems; `classical` Peirce true, contraposition false; exclusion marks a known `train_cap6` theorem; textbook_dev relabel gives 12 for `~(P>Q)⊢P&~Q` | all builders write manifests; legacy runs in the store |
| **WP3 families, laws, padding** | `suites/{families,laws,padding}.py`, `tests/ndbench/test_wp3_*.py` | builder contract; tags `family,n` / `schema,s` / `pad_type,m,base_pid`; families documented in module docstrings with their lower-bound arguments | n ≤ 5: witnesses verify with the predicted L and minlen agrees (§0); distinct pkeys per (family, n); token cap respected; law fillers exact size, disjoint atoms, contingent; a padded item with an injected contradiction is rejected | all three suites build on CPU; manifest counts match §2 |
| **WP4 glue + prefix** | `suites/{glue,prefix}.py`, `tests/ndbench/test_wp4_*.py` | `build_joined(cfg{mastery_run_ids,…})` reads mastery from the store; tags `op,k,part_len,cell`; `Problem.parts`; prefix tags `j,L,src_pid` | each operator's witness verifies with the §2 L; the chain shortcut example is rejected; CASE items satisfy `needs_ore`; prefix bodies end at line boundaries, N1-based | joined suite ≥80 items/cell or the cell reported short |
| **WP5 EDL** | `edl.py`, `tests/ndbench/test_wp5_*.py` | `build_classes(cfg) -> Problem` suite with a `split` tag; `run_edl(adapter, problems, lrs, orders, n, m, ckpts) -> rows` | fake learner with constant NLL → EDL = 0; tiny CPU GPT on a toy class → EDL > 0 and online sum ≥ n·L_test; shift-marginal loss equals `novelty.py` on 5 proofs (fixture) | runs from the CLI end to end on CPU with a tiny model |
| **WP6 analysis + dashboard** | `analysis/*`, `dash/*`, `tests/ndbench/test_wp6_*.py` | `view_<bench>(store, cfg) -> dict` (JSON-able); `dash.build(store, out_dir)` writes pages overview, yue, families, glue, skills, subst, padding, prefix, edl, runs; CSS tokens + primitives ported from `rl_dash.py`; light/dark; no network | synthetic store → every view non-empty; no external URLs in HTML; deterministic bootstrap under seed; Yue crossover detected on synthetic curves | dashboard renders from the legacy-only store |
| **WP7 future adapters** | `adapters/{lean_seq,step_env,api_llm}.py`, `judge/lean.py`, `vendor/lean_seq/`, `tests/ndbench/test_wp7_*.py` | as the §1 table | lean_seq 10 golden ND↔Lean round trips; `NDLineEnv` + scripted policy reproduces reference proofs; api_llm dry-run parses canned responses; LeanJudge skips without `lean` | capability flags declared; runner works with each on fakes |
| **WP8 integration (main agent)** | `experiments/current/benchmarking_20260930/*` | `configs/models.yaml` (base, rl, armC with sha prefixes), suite configs, `schedule.yaml`, runbook | – | schedule dry-run prints the budget table |

## 4. GPU schedule (GPU 4 only, `--gpus 4`, via gpulock)

Estimates at ~300 samples/s (G0 replaces with measurements).

| # | job | samples | est. GPU-min | cum. |
|---|---|---|---|---|
| G0 | throughput: 2 models × batch {512,1024,2048} × probe sets L≈6/14/25; Learner step timing; smoke | 36k | 8 | 8 |
| G1 | parts_v1 (both models) + RL std L6–7 (skills) | 96k | 8 | 16 |
| G2 | Yue textbook_dev (adaptive ≤4096) + gen control | ~460k | 30 | 46 |
| G3 | families (both models, early stop) | ~300k | 25 | 71 |
| G4 | glue joined (built after G1) | 150k | 12 | 83 |
| G5 | law × s | 140k | 10 | 93 |
| G6 | padding | 70k | 6 | 99 |
| G7 | prefix | 67k | 8 | 107 |
| G8 | EDL: 12 classes × 2 models × 3 LR × 2 orders = 144 runs | – | 45 | 152 |
| G9 | Yue textbook-law (adaptive ≤1024) | ~570k | 35 | 187 |
| G10 | optional: arm C (`C_s101/final.pt`) on families, textbook_dev ≤1024, glue | ~250k | 25 | 212 |
| R | reserve: reruns, extra k near a Yue crossover (≤16k on ≤10 items) | – | 120 | 332 |

Plan ≈ 3.5 h (5.5 h with reserve); **hard cap 9.5 GPU-h**. Stop rule: the ledger counts lock-held wall time only.
Before each job, continue only if spent + 1.5×estimate ≤ 9.5 h, else skip to the next job that fits (in order); stop a
job gracefully at 2× its estimate (finish the chunk, mark `partial`). If G0 measures < 200 samples/s: Yue-law k_max →
512, families 25 items/cell, drop G10. Stop everything at 9.5 h. One job at a time; no concurrent GPU use.

## 5. Risks & open decisions (recommendations)

1. Closure-relative labels → keep `method` on every item; plot exact vs upper-bound items separately; written-length
   histograms of correct proofs.
2. Shortcuts in constructed items (chain glue, shared-atom law fillers) → certify or drop; never use a witness length
   alone for joined or padded items.
3. Base ≈ 0 at L ≥ 8 → many empty base cells; report n_used; arm C as a middle model (G10); EDL still informative.
4. Mastery threshold → p̂ ≥ 0.5 over 32 draws per model; sensitivity at 0.3 and 0.7.
5. EDL protocol → as B7; no replay/two-epoch variant; report EDL(n′).
6. Adaptive-stopping bias → first-chunk rule (B8).
7. N64 index overflow and 1024-token context for L ≥ 24 → `start_index=1` variant; count `index_overflow` separately.
8. ILTP definitions unverified (SYJ203 = ¬¬((∧p)∨(∨¬p)); SYJ204 = pn ∧ ∧(pi→(pi→p(i−1))) → p0) → WP3 checks the ILTP
   TPTP files (SYJ20x+1.00n.p); if unverifiable, name the family "ILTP-style".
9. >5 premises is out of distribution for prompts; padding mixes premises and lines → P2 separates them.
10. minlen memory (~1.4 GB/worker at L18) and CPU contention → worker caps (§A2), memory guard, cache labels in out/cache.
11. Legacy results used max_new 512 → sanity checks only, never for long L.
12. Parallel edit conflicts → WP0 registries fixed; each WP owns its tests; only WP0 touches pyproject/conftest.
13. Yue gen control at k_max 512 → yes (G2), per the planner's recommendation.

## A. Main-agent amendments (override the above)

- **A1 Environment.** Run everything from the worktree root `/scratch/cowifornia/nd-rl/bench/nd-rl`.
  `code/.venv` is a symlink to `/home/cowifornia/code/research/nd-rl/code/.venv` (same uv.lock); do **not** run
  `make install-dev` or `uv sync`. Use `code/.venv/bin/python -m ...`, `make -C code test`, `make -C code check`.
  `code` is the repo's package name (it shadows the stdlib module only when run from the root — that is the existing
  convention; keep it).
- **A2 Shared machine.** 48 CPUs shared with other users and with the other WP agents: each agent uses **≤ 8 minlen
  worker processes** at a time and keeps any single CPU job under ~15 minutes; cache every minlen label under
  `/scratch/cowifornia/nd-rl/bench/out/cache/`. **No GPU use by any coder** except WP1's smoke test (≤ 1,000 samples on
  GPU 4 via `gpu.lease`). GPUs 0–3 belong to another user: never touch them.
- **A3 Interp hooks (Leon's next line of work: steering vectors, lens analyses, ordering-marginalised likelihoods).**
  Add `Capability.ACTIVATIONS = 32` and `Capability.STEER = 64`. `AbsGptAdapter` implements
  `activations(pairs, layers) -> list[np.ndarray]` (residual stream after each requested block, per token) and
  `with_steering(spec: SteeringSpec) -> ModelAdapter`, where `SteeringSpec = tuple[SteeringTerm, ...]`,
  `SteeringTerm(layer: int, vector: tuple[float, ...] | str path to .npy, alpha: float, where: "all"|"prompt"|"proof")`,
  applied as residual additions via forward hooks; the steered adapter's `identity()` includes a hash of the spec, so a
  steered model is just another model in the store and every benchmark runs on it unchanged. WP0 declares the flags,
  the dataclasses and the default `NotSupported` methods in `adapters/base.py`; WP1 implements them for absgpt with CPU
  tests (a zero vector leaves logits unchanged; a nonzero vector changes them; identity hash differs).
  Ordering-marginalised log-likelihood needs no interface change: a later analysis enumerates equivalent orderings
  and calls `logprob` on each.
- **A4 Git.** Coders do **not** commit, push, switch branches or stash; the main agent reviews and commits per WP.
  Never modify files outside your WP's list (read anything). Never modify anything under `/scratch/cowifornia/nd-rl/ab*`
  or `/home/cowifornia/code/research/nd-rl` (read-only sources).
- **A5 Web.** WP3 may fetch the ILTP/TPTP problem files (iltp.de) to verify SYJ20x definitions; record URL + date in
  the module docstring.
- **A6 Reporting.** Each coder ends with: files written, tests added and their result, `make -C code check` result,
  deviations from this plan (with reasons), and open issues. Report failures faithfully.
