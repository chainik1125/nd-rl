---
written_on: 2026-10-08
written_by: agent:claude
---

# Retrying with Lean errors: the model does not use its failed proofs

Status: complete, 2026-10-08. Eighteen runs, all pods terminated.

Question (Robbie, 2026-10-08): can the prover be trained to use a failed proof? After a failed
attempt the model is shown a Lean error and writes a whole new proof with the failure in context.
Two metrics: does a retry beat the first proof, and does the approach prove more theorems than
the run 198 baseline?

Short answer: no to both, for every variant tried. An attempt made after a failure is accepted as
often as an independent sample from the same model, or less often. Every variant proved fewer dev
theorems than plain expert iteration.

Live write-up with all tables (private to Robbie):
[Retry EI Results](https://claude.ai/artifact/7AuvNJQLXG31YdLpjtxECk).

## Configuration

Working folder:
[`code/experiments/current/retry_ei_20261008`](../../code/experiments/current/retry_ei_20261008/README.md),
branch `robbie-experiments`. Base: run 590 of the test-time-loops loop (run 198's untied model
under that loop's RL recipe; `pretrain.py` sha `b27dd2cb03ba`, harness sha `2eec1b61666a`), Dan's
fork at `51604b3`. Budgets as in 590: 300 s pretraining, 1,800 s of RL clock with Lean time
uncharged, 1,500 RL targets. RunPod A40s, three seeds per arm (0, 1, 2). Dev is the dev half of
the ladder transfer pool, 1,108 theorems of true length 7 or more.

Episode format. The existing end-of-proof token ends an attempt; the environment appends the error
tokens and the theorem statement again; the model writes a new proof. Loss is on the accepted
proof only.

    STATEMENT attempt_1 <eos> <err:CODE> [name]  STATEMENT attempt_2 <eos> ...  STATEMENT proof <eos>

Error tokens: 10 grammar codes (fork `lean_tok.inverse`), 21 type codes (Dan's `lean_prefilter.py`
from `dan/dan` `903bade4`, vendored, extended to return the failing `have` and to model this
fork's `False.elim n`), one catch-all. Acceptance is always the fork's Lean gate and `nd_verify`.

Arms (`ND_ARM`):

| arm | RL sampling | fine-tune records beyond 590's |
|---|---|---|
| `iid` | 590 unchanged | none |
| `retry` | 4 episodes x up to 8 attempts per target | real retries, plus failures paired with an unrelated known proof |
| `blank` | as `retry`, every error replaced by `<fail>` | as `retry` |
| `repair` | 4 episodes x up to 3 attempts | real retries, plus a known proof with one token made wrong, its real error, and that proof as target; 40,000 of these in pretraining |
| `resume` | 590 unchanged | none; eval only: a new attempt keeps the failed one's error-free prefix |
| `resume_rl` | as `resume`, and targets whose first 8 samples fail get 3 resumed episodes of up to 8 attempts | none |

Evals per dev theorem: 64 independent samples (the number earlier runs report) and, except for
`iid`, 8 episodes of up to 8 attempts. The independent-sample rate at attempt n is computed
exactly from each theorem's count of accepted samples out of 64 (`analyze.py`).

Commits: `0ae0f4e` (iid, retry, blank), `ae4ced0` (repair), `40a4a85` (resume arms).
Per-run `summary.json`, eval files and logs are in `artifacts/retry_ei_20261008/` of the
`nd-rl-7` checkout (not committed). Checkpoints were not kept.

## Observations

Observed, dev theorems solved of 1,108, seeds 0 / 1 / 2 and mean:

| arm | 64 independent samples | 8 x 8 episodes | attempt 2 accepted, episode | attempt 2 accepted, independent |
|---|---|---|---|---|
| `iid` | 787 / 778 / 747, 771 | | | |
| `resume` | 752 / 755 / 779, 762 | 739 / 741 / 767, 749 | 7.3% | 12.1% |
| `blank` | 721 / 703 / 731, 718 | 718 / 712 / 734, 721 | 10.3% | 10.4% |
| `retry` | 674 / 734 / 729, 712 | 686 / 736 / 715, 712 | 9.9% | 10.3% |
| `repair` | 689 / 715 / 713, 706 | 638 / 676 / 666, 660 | 1.8% | 10.4% |
| `resume_rl` | 811 / 329 / 717, 619 | 798 / 322 / 705, 608 | 5.6% | 9.3% |

Run 590 as recorded: 790 / 785 / 800, mean 792. Run 198 in its original harness: 680.

Other observations:

- `retry` and `blank`, attempts 3 to 8: the in-episode rate is slightly below the independent
  rate in all six runs (3.8% against 4.5% at attempt 3 for `retry`).
- RL rounds completed: `iid` 16 to 17, `retry` and `blank` 6 to 8, `repair` 20 to 21.
- `repair` seed 0: 73% of second attempts are token-for-token identical to the first.
- Near-miss search on 2,978 failed first attempts from `repair` seed 0: 129 (4.3%) become valid
  under some single-citation change; Lean accepts all 129. The same search recovers 547 of 547
  proofs broken by one token on purpose.
- Error codes over 34,178 failed attempts in `retry` seed 0's episode eval: `unbound` 35%,
  `app-arg` 11%, `and-intro` 10%, 25 other codes; the catch-all code did not occur.
- `resume_rl` seed 1 solved 677 of 1,500 RL targets against 1,247 and 1,307 for the other seeds.
- Local check of the classifier against Lean 4.34 on 28,000 one-token corruptions of real proofs:
  17,263 Lean rejects all coded and located, none missed, none of 4,237 accepted proofs flagged.

Interpretation:

- The model ignores the failure when trained on unrelated pairings, and copies it when trained to
  repair. Neither helps, because its failures are mostly wrong plans: the error names the place
  type-checking stopped, not the earlier choice that made the proof unfinishable.
- Keeping an error-free prefix commits the model to the failed plan, which is why resuming does
  worse than starting over.
- The retry arms lose theorems mainly by fitting fewer RL rounds into the same GPU time.
- Long contexts did not degrade sampling once the statement was restated before each attempt.

Open:

- The `iid` arm averages 771 against 590's 792 (pooled with `resume`, six runs average 766). The
  cause is not established: seed noise, or the 33 extra vocabulary tokens.
- `resume_rl`'s 811 is one seed of three, with another at 329.

## Limitations and next step

One model size and one recipe. The repair corruptions are single-token and mostly citations; the
near-miss search covers only single-citation fixes. Expert iteration cannot distinguish a retry
that used the failure from one that got lucky, so a different training signal might behave
differently. Not run: feedback about the plan instead of the symptom, for example marking which
intermediate formulas of the failed proof follow from the premises.
