---
written_on: 2026-11-27
written_by: agent:claude (orchestrator, Opus 5.5)
---

# Pod run plan (orchestrator's sequence)

Exact commands: `temp/run_steps.md`, `rank1/run_steps.md`, `steer/run_steps.md`; tooling: `pod/README.md`. `P` =
`bash pod/m2_pod.sh`, `A` = the pod's ssh alias. Every GPU job takes the GPU lock and waits for free GPU memory;
`--nogpu` jobs (downloads, ExPO writes) may run beside a GPU job. Each job is watched with `pod/m2_watch.sh`; check-ins
every 30 min. Minutes are estimates at 1,400-1,500 generated tokens/s and 3,000 HF prefill tokens/s (unmeasured).

Run-time conventions fixed here: sampling cells use `--chunk 1` and `--tokenizer /workspace/models/tok` (the dpo `main`
tokenizer files that every stored job used; written by Dan's set-up), and the prompt sha256s in Dan's meta are compared
with his stored jobs after each cell.

## Single H200 (preferred)

| # | job(s) | measurement | minutes | note |
|---|---|---|---|---|
| 0 | `P setup A`; downloads dpo, s0500, s2300, s0100 (`--nogpu`, serial) | all | 15-20 | sha256-verified against the Hub |
| 1 | `temp_dpo_C1` (736 draws, T 0.6 / top_p 0.95) | M1 | 45-55 | ExPO writes (`chain_expo_writes`, `--nogpu`) start beside it |
| 2 | `temp_s0500_C1` (384 draws) | M1 | 25-35 | |
| 3 | chain `W -> build -> gate` | M2 | 50-75 | W exit 20 or a gate failure skips the adapter cells |
| 4 | `A1` (416 draws); release rule; `A2` (208), `A1u` (320) unless A1 is a clean null | M2 | 40 + 55 | |
| 5 | `Xdev46`, `Xdev35` (128 each); gamma rule; `Xheld` (416) | M2 | 50 + 45 | unchosen checkpoint deleted |
| 6 | chain `m3_p1 -> temp_pf -> m3_p2 -> m3_p3 -> m3_pf` | M3 (+ M1 PF) | 110-160 | stops on split-half (42) or pre-flight failure |
| 7 | `m3_dev` (64), `m3_e0p` (384), `m3_e1` (544), `m3_trigger` | M3 (+ M1 baseline) | 85-100 | |
| 8 | released only by pre-registered triggers: `m3_e2_*`, M1 `C3`/`C4`/`C2`, M2 rank-16 or E | | 0-150 | orchestrator decides within the caps |

Total about 9-10 h without step 8 (about $47-52 at $5.29/h), up to about 12 h with every conditional cell (about $62).
Trimmed plan (if the coordinator wants about $40-43): M1 reach 8 -> 4 draws, M2 gamma 3.5 pre-committed (no ExPO dev
cells), M2 A1u held-out x 1, M3 held-out 80 x 1 and a $2 NLL budget: saves about $7.

## Two pods (if an H100 is handed over before an H200 is free)

- H100 80 GB (dpo in place; needs s0100, s0500, s2300 too, about 260 GB of weights): steps 3 and 6 (HF, SVD, adapter
  build, vLLM gates on short teacher-forced sequences). About 2.5-3.5 h.
- H200: steps 0-2, 4, 5, 7, 8, after fetching the adapters and vectors from the H100 and pushing them (small files).
  The M2 gate and M3's vLLM pre-flight are re-run on the H200 before its first adapted or steered cell (about 35 min).
  About 7-8 h.
- Cost about the same as the single H200 (the H100 hours replace H200 hours; the repeated gates add about $3); the
  work ends about 3 h sooner.

## Plan of record from 3:02 am PT: A2 (coordinator's decision)

The RTX PRO 6000 (rp-m2a, $2.69/h) measured 320-345 generated tokens/s on C1 (KV cache 110k tokens), about 4.5 x below
the H200, so it does only weight-space and teacher-forced work; every sampling cell moves to a fresh H200.

rp-m2a (chains `rtxprep`, then `rtxprep2` after W exit 2): W (exit 2) -> W_it12 (exit 2) -> Wx (exact fallback) ->
build2 -> gate2 -> m3_p1 -> m3_p2 -> m3_p3 -> m3_pf. Then fetch (md5) W, W_it12, Wx, build2, gate2, m3_p1, m3_p2, m3_p3,
m3_pf and the logs, and `POD READY TO DELETE: rp-m2a`. ExPO checkpoints are not fetched (rewritten on the H200's CPU).

H200 (alias `rp-m2h` assumed; volume 500 GB, container disk 50 GB, same image as rp-m2a):
1. `m2_pod.sh setup rp-m2h`; chain `chain_h200_dl` (dpo, s0500 sha256-verified, then write_g46, write_g35 on CPU, --nogpu);
   `m2_pod.sh pushrun rp-m2h build2 out/adapters`; `pushrun rp-m2h m3_p3 out` (vectors, chosen.json, pre-flight refs).
2. Chain `chain_h200_seg1`: temp_dpo_C1, temp_s0500_C1, gate_h (LoRA gate again on this GPU), A1, release.
   Beside it on serrano: fetch C1 -> temp_triggers (arm-A parts), PF manifest, push.
3. Segment 2 (written once A1's release code is known): A2 + A1u (exit 0) or the conditional cell (exit 10: E, which needs
   s2300 and a hybrid write; else rank-16 from build2's k16), temp_pf, Xdev46, Xdev35 -> gamma rule (serrano) -> Xheld.
4. Segment 3: m3_pf again on the H200 (pre-flight is per engine), m3_dev, m3_e0p, m3_e1, m3_trigger, E2 if triggered;
   then M1's reach comparisons with E0' (PREREG_temp Amendment 2) and M1 trigger cells only within the caps.
Sampling time at 1,400-1,700 tokens/s: about 4.5-5.5 h plus about 15 min of set-up.

Update 4:59 am PT: rp-m2a finished (W -> W_it12 -> Wx exact fallback, build2, gate3/gate4 under PREREG_rank1 Amendments 3/3a,
m3_p1-p3, m3_pf + diagnosis under PREREG_steer Amendment 1); everything needed is fetched with md5. On the H200: push
`build2 out/adapters` and `m3_p3 out` with `m2_pod.sh pushrun`; the LoRA gate runs as `gate_h` (in `chain_h200_seg1`) and
the pre-flight as `m3_pf_h` (`chain_h200_m3pf.txt`, with the stock/zero eager runs) before any adapter or steered draw.
s2300 is downloaded to the H200 only if A1 is a clean null (cell E) or M1's trigger A fires (C3).
