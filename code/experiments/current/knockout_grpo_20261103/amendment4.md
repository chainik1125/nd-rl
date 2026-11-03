---
written_on: 2026-11-03
written_by: agent:claude
---

# Amendment 4: guard-relaxed continuation of the censored ko_strict lines (secondary, exploratory)

Written 2026-11-03 at about 14:15 UTC, after five of the six ko_strict lines had ended in the
recipe's think-ban stop, at R 0.109–0.441, far short of the T_L3 364 cap. Proposed by the
coordinator, who is putting it to Leon. Under the 30-minute rule it launches at 14:35Z unless
halted. Its sha256 is recorded in `log.md` before the first job it governs.

**Status.** Secondary and exploratory. The pre-registered primary does not change and is listed
first everywhere: a think-ban stop ends a line, and it counts as "not reached by R_stop".

**Why.** The guard decides the ko_strict arm. Its stops end lines at about a tenth of the cap
and of Dan's horizon. The censoring is informative: four of the five stopped lines had almost no
σ, and kos_s2 was rising (E32 13.0 at R 0.441). The guard watches the probability on `<eos>`,
`<think>`, `<pad>` and the effort tokens inside think blocks. It does not watch σ text.

- **Lines.** Every ko_strict line the guard stopped: kos_s1, kos_s2, kos_s0_dp, kos_s1_dp and
  kos_s2_dp. kos_s0 is added if it stops. Each continuation is tagged `<line>_gr`.
- **Resume point.** Each line's last rolling checkpoint before its stop, with the exact training
  state: weights, AdamW, curricula, RNGs, LoopState and the canary base.
  - kos_s1: L3 iteration 0, its only rolling checkpoint.
  - kos_s2: iteration 75.
  - kos_s0_dp: iteration 100.
  - kos_s1_dp: iteration 85.
  - kos_s2_dp: iteration 45. It has no rolling checkpoint at 50–60.
  The checkpoint is copied into the new run with the source's metrics rows up to it. The source's
  reads up to that iteration seed the continuation's curve (rows marked `from`).
- **The only change: `--ban_stop 0.05`**, ten times the recipe's 0.005.
  - The alarm (1e-3), the rollback on alarm, the restore check and `--max_rollbacks 3` stay as
    they are.
  - The ban mass is logged every iteration (`ban_think` and `ban_think_max` in metrics.jsonl).
  - Why 0.05 and not lower: both the sampler and the scorer mask the banned tokens before the
    softmax (`grpo._token_logp`). So the mass never changes what is sampled or trained, and the
    banned logits get no gradient. The mass is a drift indicator, and the health stop below is
    the backstop for degradation. The highest mass seen so far, in this experiment and in Leon's
    full-base L3 runs, is 6.5e-3, so 0.05 leaves room without being unbounded.
- **Everything else unchanged.** The L3 args, the T_L3 364 cap (c_max 403), the bar 26.31, the
  hold window (Amendment 1), the snapshot grid and the read path (k 64, plain, B64, T 0.8). The
  per-line budget is still counted from each line's start: the continuation ends at the line's own
  c 403.
- **Health.** The same alerts (`health.py`), plus a stop (`line.py --gr`, `health_watch`) in
  either of two cases:
  - **Training reward:** L3 `reward_pass1` stays below 0.5 × its peak over the line's L3
    iterations for 25 iterations.
  - **E32:** E32 stays below 0.5 × its peak over reads spanning 25 iterations or more. This rule
    applies only once the peak is at least 3.0, above the n 64 read noise (sd 0.5–1.0, `log.md`);
    otherwise the zero-E32 Dan's-pool lines would trip it at once.
  A health stop writes `HEALTH_STOP` and `line_done.json` `health_stop`, and it is reported.
- **Reporting.** A separate "guard relaxed" series, `*_gr`, in numbers.json, numbers.md and the
  figures, drawn hollow or dashed beside the primary markers. It has the same first and held
  crossings. Its own think-ban alarms and its mass trajectory are reported too. The primary
  (censored = not reached by R_stop) is listed first.
- **Machines.**
  - serrano: kos_s1_gr and kos_s1_dp_gr. That is at most 2 of ours and at most 3 jobs in total
    with other users'. The queue gates each GPU on it being free of other users.
  - ynez: kos_s2_gr and kos_s2_dp_gr.
  - temescal: kos_s0_dp_gr on GPUs 1–2 only.
  - No third process on a line's GPU. Each job runs under the queue's timeout and watchdog, with
    a crash-recovery `line2` job.
- **Budget.** About 40 GPU-h used by 14:15Z. The five continuations add at most 1,515 L3
  iterations at about 125 s each, about 53 GPU-h. kos_s0 has about 8 GPU-h to its cap, and
  kos_s0_gr would add at most about 8 more. That makes about 110 GPU-h in all. The limit is 180;
  I stop and ask if the projection passes it.
