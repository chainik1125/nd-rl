---
written_on: 2026-11-27
written_by: agent:claude (opus, orchestrator)
---

# Olmo minimal adapter: log

## 2026-11-27 1:22-2:15 am PT (orchestrator): design, reading, discourse, build

- 1:32 am: `DESIGN_DRAFT.md` written (input to the discourse).
- Full-text readings (Sonnet readers; downloads under `/scratch/cowifornia/nd-rl/olmo_elicit/v2/papers/`):
  `literature/edl_papers.md` (Bits That Count, EDL theory paper), `literature/tiny_adapter_papers.md` (Donoway et al.
  2025, TinyLoRA, Sinii et al. ×2), `literature/footprint_papers.md` (Sparse but Critical, Akgül et al., Wang et al.).
- Independent audits (Opus, own downloads under `v2/papers_audit/`): `literature/audit_independent.md` (EDL papers and
  Donoway et al. 2025, then a comparison with the readers' notes), `literature/footprint_audit.md` (token footprint:
  numbers recomputed independently, sign-off with corrections C1-C10).
- Design discourse, four turns: `discourse/edl_design.md`. Adjudicated in `PREREG.md` ("Points settled").
- Build (Sonnet builders, checked by me: tests rerun, core code read): `build_pools.py`, `mini_train.py`,
  `make_plan.py`, `test_mini.py`; `multi_sample.py`, `lora_gate.py`, `stage_cells.py`, `analyze_cells.py`,
  `fig_cells.py`, `pod/mini_*.sh`, `pod/mini_status.py`, `test_sampling.py`; `footprint.py`, `test_footprint.py`.
  First check: 36 CPU tests pass, ruff clean, `bash -n` clean. The old dpo adapter cell is reproduced from its graded
  file by `analyze_cells.py` (63.54 % [53.65, 73.18]).
- 2:05 am `SPEC.md`, 2:10 am `PREREG.md`, both before any pod exists.

## 2026-11-27 2:00-2:15 am PT (orchestrator): revision after the discourse, pre-pod checks

- Pools rebuilt checker-only (no hand labels in any selection) with a third source `unsel`; matched prompts at `dpo`
  115 (train 64 / dev 32); median train tokens own 1,972, unsel 1,895, final 4,341. Checker bound (Σ log2 1/a_i) for
  own at n = 1, 4, 16, 64: 1.4, 4.9, 31.9, 146.7 bits. `types_table.py`: at n = 64, 32 of the 48 reach prompts share
  no instruction id with the 64 training prompts (13 all, 3 some).
- `mini_train.py`: successive halving to 512 steps, early stop, `unconverged`, EDL at m = min(t, n), fixed-step EDL
  at 64, B statistics, attention-only mask option. `analyze_cells.py`: G with a two-level paired bootstrap; on the old
  dpo adapter G = 0.916 [0.825, 1.0], "reaches" true. Prompt control built (`make_prompt_control.py`: Dan's A1 suffix,
  read from his stored A1 request rows, byte-identical).
- My edits to the pod scripts: `mini_pod.sh sample` takes `--seed` and `--pids` (smoke tests, distinct seeds per
  batch); a train stage is named per plan file (`train:<plan stem>`), so a second plan is not skipped as DONE.
- Plans: `v2/plan_dpo_h100.json`, `v2/plan_dpo_h200.json`, `v2/plan_smoke.json` (4 tiny cells, 2 steps, n = 1, for an
  end-to-end check of train → export → gate → multi-adapter sample on two dev prompts).
- Tests: 51 pass (test_mini 24, test_sampling 23, test_footprint 4); ruff clean; `bash -n` clean; dry runs of setup,
  train and sample print the expected commands; the push contains code, pools, plans and stage-2 L inputs only (no
  sealed set, no credential).
- Rule slip, reported: the footprint worker ran `pkill -V` once (prints the version only; nothing was signalled).
- 2:15 am: `mini_pod.sh fetch` writes to `$V2/runs_<alias>/<ckpt>/` (two pods both write `runs/dpo/logs` and `state`,
  so one local tree per pod); `analyze_cells.py` falls back to serrano's pool copy when cell.json names a pod path.
  51 tests pass. Waiting for pods.

## 2026-11-27 2:14-2:50 am PT (orchestrator): pods, smoke, first problem

- Pods (created by the coordinator 2:14 am PT): rp-om1 H100 ($4.39/h), rp-om2 H200 ($5.29/h). Setup 2 min each
  (versions as the 25 Nov run). dpo weights downloaded and verified on both (commit 97604023, 14 shards, 64.47 GB).
- Smoke: 4 two-step adapters (A16, O1, M256, M4k-attn) trained on rp-om1 (reload diff 0, base check after unload 0),
  gated on rp-om2 (all UNINFORMATIVE, as expected for 2 steps; the engine loads r16, r1, o_proj-only and masked
  adapters together), sampled on 2 dev prompts × 4 draws each through one engine, graded. Pipeline works end to end.
- Prompt control (dpo + Dan's answer-only suffix, no adapter), 384 draws: done 2:37 am.
- Problem (2:45 am): own-source dev loss rises with training at every lr (Amendment 3 in PREREG). The rp-om1 chain was
  stopped; `mini_pod.sh stop` killed the chain's process group but not the `timeout`-wrapped trainer (GNU timeout
  makes its own process group); I stopped that by pid (2640/2641, my own processes). Fix requested from the builder.
- 2:46 am: rp-om1 restarted on `plan_dpo_h100b.json` (final A16 n4, A16 n1, O1, M16k, M4k); rp-om2 trains final A16
  n64 (`plan_dpo_h200b.json`). 2:47 am Amendment 3 written before any reach result was opened.
- 2:50 am, prompt control read (after Amendment 3): 6.25 % of draws [1.3, 13.0], G = 0.09 [0.02, 0.19], does not
  reach; prompts with a pass in 8: 18.8 %; median answer 108 words, median thinking 2,628 tokens.
- 2:50 am: own A16 n64 dev-selected cell renamed on rp-om1 to `dpo__own__A16__n64__devsel` (kept, not sampled), so the
  fixed-recipe cell can take the canonical id. Builder added `fixed` / `recipe_from` cells to mini_train.py (tests).
  The builder's stop fix (`timeout --foreground`) would have stopped timeout from killing vLLM's child process group on
  expiry, so I reverted it: stages run under plain `timeout` again, and `mini_pod.sh stop` now signals every process
  group whose leader descends from the chain pid (tested locally; test_stop_reaches_timeout_child_groups). 54 tests
  pass, ruff clean.
- 2:54 am: two local drivers (logs/drive_om2_b1.sh, logs/drive_om1_c.sh; nice, timeout 5 h):
  rp-om2: after final A16 n64 -> plan_dpo_h200c (own + unsel A16 n64, recipe from final A16 n64) -> fetch -> stage
  batch_b1 (final unforced; own, unsel forced since `did_not_train` is descriptive for fixed cells) -> gate + sample,
  seed 660001. rp-om1: when final O1 n64 ends -> stop -> plan_dpo_h100c (own fixed A16 n4, n1, O1; final M16k, M4k).
  Pod ledger: $V2/ledger.md (launch stop 7:55 am PT with both pods up).

## 2026-11-27 4:13-5:05 am PT

- 4:12 am: final O1 n64 DONE (lr 3e-4 by the edge rule, step 128, dev 0.5403). rp-om1 chain stopped by the driver;
  the new stop logic signalled both process groups (chain 3386, timeout 3420); h100c started at 4:12 am. Own fixed
  cells (A16 n4 at 3e-5 × 64; A16 n1 at 3e-3 × 1; O1 at 3e-4 × 128) DONE by 4:26 am, fetched and staged as batch_b2.
  Batch 2 was not sampled because own A16 n64 did not reach (rule 3).
- 4:59 am: batch 1 sampled (11:47-12:58 UTC, about 1,400 tok/s) and fetched md5-equal. Read: see PREREG Amendment 4
  (pipeline check fails; nothing reaches). 5:01 am: batch R (old recipe, lr 3e-4 × 64, final, own and unsel) training on
  rp-om2 (plan_dpo_h200r.json). 5:01 am: Amendment 4.
- handcheck.py written (Dan's rule-1 screen verbatim, blinded items in his format, his B_s2300 labelled passes mixed in
  as a reference; tested on the prompt control).
