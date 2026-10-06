# Pre-registration — run `lean-format` (proposal 8: Lean as the training format for the from-scratch model)

Written 2026-09-21 ≈ 03:00 UTC, before any pod exists for this run. Executor: agent:claude. Branch `dan_lean_format`.
Brief `BRIEF_LEAN_FORMAT.md`; proposal `~/nd-rl/docs/proposals/2026-09-20-lean-training-format.md`; policy `AGENT_POLICY.md`.

## Question

Does training the 4-layer, d = 256 from-scratch model on the Lean rendering of the same cap-6 proofs (instead of the
take-home token format) change what Stage-1 + expert iteration can do? Decision at stake: retire the token format or not.

## The Lean training format (fixed before any model is trained; `lean_tok.py`)

- Text = `nd2lean.py`'s rendering, linearised on one line with `;` between tactics (no indentation tokens):
  prompt `theorem t ( P Q R S : Prop ) ( h1 : F1 ) … : C := by`, proof `have n1 : F1 := h1 ; have n7 : ( P → R ) := ( fun ( n3 : P ) => by have n4 : Q := n1 n3 ; exact n4 ) ; exact n7 <eos>`.
  One symbol per token, vocabulary 107 (token format `abs`: 99). Checked on the VPS before this commit: 3,000 / 3,000
  held-out proofs render, parse back to the identical ND proof, and Lean 4.34 accepts the literal text (1,000 / 1,000
  checked); 300 / 300 theorem-swapped negatives rejected.
- **An observation the proposal did not make**: `nd2lean`'s hypothesis names `n<i>` *are* the ND line indices, so "named
  hypotheses" do not by themselves remove index tokens. Two naming schemes are therefore trained, both fixed here:
  - `lean_rand` (**primary**): 64 name tokens; every training presentation draws a random injective renaming, so names
    are pure labels with no order.
  - `lean_seq` (secondary): names numbered by first appearance in the Lean text plus a random start offset — the exact
    analogue of the token format's `abs` start-index shift.
- A sample is **accepted** (reward, and every count) iff (i) it parses in the strict nd2lean grammar, (ii) **Lean accepts
  the literal sampled text**, and (iii) `nd_verify` accepts the ND proof the text denotes (`lean_tok.inverse`). (ii) and
  (iii) are both computed on every distinct grammar-valid sample of every round (`lean_gate.py`), the 2 × 2 agreement
  table is logged per round, and any disagreement is written to a file and reported as a bug. At the end every counted
  proof is also put through the unmodified `nd2lean.py --check` (checker of record).
- Counts are on the denoted ND proof (always numbered from N1, so start-index-normalised by construction); written
  length = `nd_verify`'s line count; depth-3 acquisition, `L*`, pruning: the existing scripts, unchanged.

## Design I will run (2 × RTX 3090 pods, a third only if wall-clock requires it)

Same model (4 layers, d 256, 8 heads; only the embedding / head rows differ: 107 vs 99), same schedule (Stage-1 6,000
steps, bs 128, lr 1e-3 → 1e-4, warm-up 200; EI 8 rounds, k 32, T 0.8, 600 fine-tune steps at 3e-4, retain 20,000,
≤ 4 proofs per theorem × 4), same data, same `max_new` (400 depth-3 / 512 ladder). Per naming scheme:

1. Stage-1 on `data/p2/train_depth3_f0_a1.jsonl` (155,000 cap-6 proofs, depth-3 removed), seeds 0 and 1; Stage-1 on
   `data/train.jsonl` (154,990), seed 0. Sequence-length statistics of both formats on all three files.
2. **(a) depth-3 f = 0 dial**: `expert_iter.py` EI + frozen control, seeds 0 and 1, targets `data/p2/targets_depth3.jsonl`
   (1,000), transfer `transfer_depth3.jsonl` (500), held-out `data/p2/heldout.jsonl` (5,000). Comparator on file:
   `artifacts/p2/ei_depth3_f0_a1_s{0,1}` (acq 0.335 / 0.364), frozen 0.
3. **(b) ladder rung T1**: `ladder_ei.py` EI + frozen control, EI seeds 0 and 1 from the one Stage-1 model (as the token
   ladder did with `stage1_abs.pt`), pools `data/ladder/{rl_targets,transfer}.jsonl` (4,495 / 2,285). Comparator:
   `la_T1_s{0,1}` (`L*` transfer 10 / 10, 612 / 623 solved), `la_frozen_s{0,1}` (`L*` 7).
4. **Mechanism test** (full-train Stage-1 models, `eval_set.py --k 16` on `data/transfer.jsonl`, the take-home's measure):
   distinct verified proofs of written length 7 / 8. On file: `abs` 108 / 1, `rel` 79 / 0, `abs-fixed` 0 / 0. Plus one
   ablation Stage-1: `lean_seq` with `--no_shift` (names n1… only, as seen in ≤ 6-line proofs).
5. Timing: one frozen token-format round of (a) and of (b) on the same pod under the same co-tenancy as the Lean arm it is
   compared with (token Stage-1 models: `stage1_abs.pt` on the VPS; `stage1_depth3_f0_a1_s20.pt` from the round3-run1
   bucket), so the round-time ratio is like for like; `lean_gate.py` logs Lean wall / CPU seconds and `nd_verify` seconds
   on the same proofs.

## Expected results (numbers)

| # | quantity | token value on file | "not worse" band (proposal) | my expectation |
|---|---|---|---|---|
| P1 | Stage-1 held-out greedy, a1 sets (5,000) | 0.883 / 0.883 | within 2 pp: ≥ 0.863 | `lean_seq` 0.87–0.90; `lean_rand` 0.84–0.89 (P(in band) ≈ 0.6) |
| P1b | Stage-1 held-out greedy, full train set | 0.948 | ≥ 0.928 | `lean_seq` 0.93–0.95; `lean_rand` 0.90–0.95 |
| P2 | depth-3 f = 0 acquisition, round 8 | 0.335 / 0.364 (all 8 token arms 0.271–0.364) | both seeds ≥ 0.27 | 0.25–0.40, P(not worse) ≈ 0.65; frozen ≤ 0.01; first depth-3 proof by round ≤ 5 |
| P3 | ladder transfer `L*`, round 8 | 10 / 10 (frozen 7) | `L*` ≥ 10 both seeds | 10 (P ≈ 0.6), 9 (0.25), ≥ 11 (0.15); transfer solved 500–700 / 2,285; frozen `L*` 7 |
| P4 | 7-line proofs by the Stage-1 base (pass@16, transfer 1,638) | 108 (`abs`), 0 (`abs-fixed`) | — | `lean_rand` and `lean_seq` 50–150; `lean_seq --no_shift` ≤ 2 |
| P5 | Lean vs `nd_verify` on sampled proofs | 181,464 / 181,464 on pools | 0 disagreements | ≤ 10 disagreements per million, all of the `¬A` ≡ `A → False` kind (Lean accepts an `R` / type ascription that `nd_verify` rejects) |
| P6 | round time Lean arm / token arm, same pod | — | ≤ 3× | 1.2–1.8× (sequences ≈ 1.2× longer; Lean ≈ 250 proofs / s / core measured on the VPS) |
| P7 | sequence length, prompt + proof tokens, held-out mean | 36 + 57 (`abs`) | — | 51 + 65 (measured on 3,000 held-out proofs: ratio 1.24) |

**Mechanism I expect.** The untrained-index barrier (`abs-fixed`: 0 seven-line proofs) is a property of *ordinal names
without augmentation*, not of the token format: `lean_seq --no_shift` will show it too (P4 ≤ 2), `lean_seq` removes it by
the same offset trick as `abs`, and `lean_rand` cannot have it. So I do **not** expect Lean to beat the token model on
length generalisation (`L*` stays 10): both already have every name trained. Where Lean could differ is box structure: a
box is written *inside* the `have` that discharges it, so the model states `( A → B )` before it writes the sub-proof
(goal-directed) instead of after (Fitch). I expect this to be roughly neutral for depth-3 (three nested `fun`s are the same
composition step as three nested boxes) and mildly harmful for `lean_rand` in distribution (fresh-label bookkeeping).
The result that would matter: `L*` ≥ 11 on transfer in both seeds, or depth-3 acquisition above 0.40 in both seeds.

## Decision rule (proposal's, made operational)

Per naming scheme: **not worse** iff P1 ≥ 0.863 on both a1 seeds, P2 ≥ 0.27 on both seeds, and P3 `L*` ≥ 10 on both
seeds. Verdict reported for both schemes with the measured gaps; "Lean not worse" overall iff at least one scheme passes
all three (two tries at a fixed band — said plainly in the write-up); the primary scheme's verdict is named separately.

## Budget and stop rule

- Pods: **$30** (2 × RTX 3090 at ≈ $0.50 / h; planned ≈ 14 pod-hours ≈ $7–12). Ask in `QUESTIONS.md` before exceeding $30.
- Hard stop 36 h after start (2026-09-22 14:52 UTC). Pods are deleted as soon as their files are pulled.
- Throughput stop: if a Lean arm's round exceeds 3× the like-for-like token round (P6), report the ratio and stop.
- Nothing beyond this run is launched (the pause stands). No test-file run. `nd_verify` unmodified. No hand- or
  LLM-written training proofs: every supervised record is a generator proof rendered deterministically by `lean_tok.py`;
  the cap (≤ 6 lines) is asserted on the ND record and re-asserted on the denoted proof of every rendered record.
