---
written_on: 2026-11-25
written_by: agent:claude (stage-0 / stage-2 build agent)
---

# Stage 2 (sampling arms) and Stage 0c (replication draw): job specs, reuse, cost

Contract: `SPEC.md`, readings: `PREREG.md` (Amendment 1: the mini run on dpo, s0100, s0500, s1100, s2300 first).
Inputs are built by `make_stage2.py` into `/scratch/cowifornia/nd-rl/olmo_elicit/stage2/` (sha256 of every input read
and file written: `stage2/manifest.json`). Pod side: `pod/oe_chain.sh` (stages `T`, `H`, `rep`), driven by
`pod/olmo_pod.sh` / `pod/olmo_chain.sh`. Analysis: `analyze_arms.py` → `arms.csv`, `fig_arms.py`; `analyze_rep.py`.
Everything here counts checker passes. No genuine-versus-gamed reading in this pass; any figure or table of these arms
says so.

## Dan's code and settings, unchanged

| what | his file (repository `dan-pandori/nd-takehome`) | commit on the pod |
|---|---|---|
| sampler, H and rep | `olmo_ifladder/sample.py` (olmo_thinker template, T 1.0, top-p 1.0, 32,768 max tokens, vLLM 0.11.0, bf16, prefix caching, `generation_config="vllm"`) | `8e744a89` (dan_olmo-ifcurve) |
| sampler, T | `olmo_answeronly/sample_prefix.py` (his thinking-prefix sampler: model input = templated user turn + the stored thinking ids through `</think>`; answer budget 32,768 − prefix; trace groups warmed in the prefix cache) | `93d36de7` (dan_olmo-answeronly) |
| grader, H and rep | `olmo_ifomega/grade.py` (official IFBench / IFEval strict checkers, vendored) | `8e744a89` |
| grader, T | `olmo_transplant/grade_rows.py` (imports the same `grade.py`; grades `"</think>" + answer`) | `93d36de7` |
| tokenizer | `allenai/Olmo-3-32B-Think-DPO` main tokenizer files (as every stored job), via his `pod/olmo_ifomega/fetch.sh` | — |

`grade.py`, `problems.jsonl`, `olmo32/grade.py` and `olmo_iffu/sample.py` are byte-identical at the two commits
(checked on serrano and again by `pod/oe_setup.sh` on every pod).

## Which of his arms each of ours repeats

### T: transplant (41 reach-41 prompts, 115 traces, 8 answers per trace and condition)

His design (olmo-transplant, olmo-answeronly): the final model's (`s2300`) stored thinking from hand-labelled genuine
successes (`olmo_transplant/traces.csv`, kind `rl`: 115 traces on 41 prompts, 1–4 per prompt, from his olmo-ifomega
`F_s2300_ifb` and `C1_s2300_ifb` draws) is pasted in by token ids; only the answer is sampled.

| our condition | his condition | his request rows used, byte for byte | what changes in ours |
|---|---|---|---|
| `T` (no instruction) | olmo-transplant `T1` (writer dpo) / `T2` (writer s2300), cut `full` | `artifacts/olmo_transplant/requests/s2300_reach.jsonl.gz`, cond T2, cut full (115 rows) | `writer` = our checkpoint (his sampler asserts writer = `--name`); `cond` label `T` |
| `A1` (answer-only instruction: user turn + "\n\nAfter thinking, reply with only your final answer, exactly as drafted, with no introduction or explanation.") | olmo-answeronly `A1` | `artifacts/olmo_answeronly/requests/s2300_reach.jsonl.gz`, cond A1 (115 rows) | `writer` only |

Checked by `make_stage2.py` before writing: every `T` prefix equals his dpo-writer `T1` prefix and his s2300-writer `T2`
prefix of the same trace (the pasted thinking is identical for every writer), ends in `</` `think`; every `A1` prefix
equals its trace's `T` prefix; the `A1` suffix is his registered wording (`olmo_answeronly/wordings.py`); n = 8. His
A2–A4 conditions and his truncation cuts are not repeated. Per checkpoint: 230 requests, 1,840 answers, one job
`T_<ckpt>` (T and A1 share one engine; he ran them as separate jobs).

### H: the 60-word hint (48 reach prompts)

His olmo-iffu test 2 / 2c: user turn = prompt + "\n\nAnswer in at most 60 words." (`prompt_sample`); grading on the
original prompt and kwargs. Our `problems_H.jsonl` is his `olmo_iffu/problems_hint.jsonl` (16 prompts) and
`problems_hint2c.jsonl` (32) lines, byte for byte, in `pids_reach.txt` order; together they are exactly the 48 reach
prompts (checked: every non-hint field equals `olmo_ifomega/problems.jsonl`'s row). 8 draws per prompt
(`pids_H_n8.txt`), or 4 (`pids_H_n4.txt`) if money binds; chunk 4 (his olmo-ifcurve jobs' chunk).
The plain-prompt comparison is his olmo-ifcurve arm B (32 base draws per reach prompt at s0050–s2300); at dpo, his
olmo-ifomega `F_dpo_ifb` / `F_dpo_ife` draws (16 per prompt).

### rep: Stage 0c replication draw

His olmo-ifcurve arm A at dpo and s2300: the 218 IFBench prompts, plain prompt, his sampler and grader. Ours:
`pids_rep.txt` = 218 × 2 draws, chunk 2, `olmo_ifomega/problems.jsonl`, new seed. Compared with his stored arm-A
cells (dpo 16 draws per prompt from `F_dpo_ifb`, s2300 8 from `F_s2300_ifb`) by `analyze_rep.py` under the PREREG rule.

### L: the reach prompts through each checkpoint's stage-1 adapter (SPEC "L, LoRA-elicited reach")

Not one of his arms; built from his parts. Prompts: the 48 reach prompts, plain, exactly as his arm B sent them
(`olmo_ifomega/problems.jsonl`, his sampler, chunk 4), 8 draws (`pids_L.txt`), seed 640000 + step (`seeds_L.csv`).
Model: the checkpoint plus its stage-1 adapter `runs/<ckpt>/edl/adapters/best_order0` (θ_64 of the order-0 EDL row with
the shortest online code length on the frozen grid {1e-4, 3e-4, 1e-3}; trained on 64 arm-A non-reach prompts, so L
measures transfer to the reach prompts) on vLLM's LoRA path: his `olmo_ifladder/sample.py --lora <dir> --lora_name
<ckpt>_edl64 --lora_rank 16`, as his olmo-ifladder LoRA evaluations ran (`Leval_main`, vLLM 0.11.0). Never a merge:
his merged model missed his 1 % merge check (1.12 %). Our adapters have the same PEFT config as his `lora/main` (r 16,
α 32, q/k/v/o/gate/up/down, bias none) and the same 896 tensor names, shapes and dtype (F32); only dropout differs (0 vs
0.05, inert at inference). Grader: his `grade.py`, strict.

- **Staging (serrano, `stage_adapter.py`, called by `olmo_pod.sh run … --stages L`).** Refuses unless
  `runs/<ckpt>/edl/DONE` lists lrs 1e-4, 3e-4, 1e-3 with status ok (order 0, and every other order it lists), BEST.json
  names the shortest-code-length order-0 grid row of `edl_rows.jsonl`, and the adapter config is the one above. Copies
  only `adapter_config.json` and `adapter_model.safetensors` (no optimizer state) plus `source.json` (BEST.json, DONE,
  the row's lr / code length, md5 and sha256 of both files, two check examples); pushed md5-checked; re-checked by md5 on
  the pod before use. The job writes `L/adapter_source.json` and `L/adapter_md5.txt` next to his sampler's meta (which
  records the adapter's sha256 itself).
- **Sanity gate (`lora_check.py`, at dpo; any checkpoint with `--lora-check`).** Teacher-forced on the first two test
  examples of the adapter's EDL row (prompt ids and the final model's response ids from the pool, as the EDL runner
  scored them): vLLM's response nats without and with the adapter (prompt log-probs, bf16, eager, no prefix cache),
  against the EDL runner's PEFT nats at n' = 0 and n' = 64. PASS needs vLLM's drop to be 0.75–1.33 × PEFT's (where PEFT's
  drop is at least 2 %), the relative change to agree within 2 points, and the base path unchanged after the adapter
  ran. An adapter that is not applied gives no drop and fails; the stage then does not sample. Tested on serrano only
  against a stub vLLM (applied → PASS, not applied → FAIL); the real check runs on the pod.
- **Comparison (`analyze_arms.py`, `fig_arms.py` panel c).** L against his plain arm B at the same checkpoint, cut to
  its first 8 draws (`B8`, like for like) and in full (`B`, 32 per prompt; 16 at dpo from olmo-ifomega): strict pass
  rate with a prompt-bootstrap interval, and prompts with ≥ 1 strict pass in 8 as % of 48. Checker passes only.
- **Cost (estimate, `cost_table.py`):** 384 draws per checkpoint at his plain arm-B tokens per draw at that checkpoint
  (lower) or the final model's 6,863 (upper; the adapter pulls toward the final model's long thinking), 1,400 tok/s
  (his LoRA jobs: 1,568 and 1,335), 2 min load, 3 min for the dpo check:

| ckpt | step | tokens / draw (plain arm B at this ckpt / s2300) | Mtok | H200 min (low / high) | $ (low / high) |
|---|---|---|---|---|---|
| dpo | 0 | 4477 / 6863 | 1.72–2.64 | 25 / 36 | 2.25 / 3.21 |
| s0100 | 100 | 4753 / 6863 | 1.83–2.64 | 24 / 33 | 2.09 / 2.94 |
| s0500 | 500 | 4994 / 6863 | 1.92–2.64 | 25 / 33 | 2.19 / 2.94 |
| s1100 | 1100 | 6940 / 6940 | 2.67–2.67 | 34 / 34 | 2.97 / 2.97 |
| s2300 | 2300 | 6863 / 6863 | 2.64–2.64 | 33 / 33 | 2.94 / 2.94 |

  Sum: 141–170 min, $12.44–$15.01, plus a download (0.5–3 min) for each checkpoint whose weights are no longer on the
  pod.

## What he already has, reused rather than re-sampled

| checkpoint | T (no instruction) | A1 | H | plain reach (arm B) |
|---|---|---|---|---|
| dpo | olmo-transplant `R_dpo` cond T1, 920 answers, seed 8101 | olmo-answeronly `R_dpo` cond A1, 920, seed 8201 | olmo-iffu `T2_dpo` (16 prompts × 64, seed 7101, TP 2) + `T2c_dpo` (32 × 64, seed 7103) | olmo-ifomega `F_dpo_ifb` / `F_dpo_ife` (16 per prompt) |
| s2300 | olmo-transplant `R_s2300` cond T2, 920, seed 8102 | olmo-answeronly `R_s2300` cond A1, 920, seed 8202 | olmo-iffu `T2_s2300` (16 × 16, seed 7102, TP 2) + `T2c_s2300` (32 × 16, seed 7104) | olmo-ifcurve `B_s2300` (32) |
| s0050 … s1600 | none | none | none | olmo-ifcurve arm B (32 per prompt) |

So T and H are sampled by us only at the seven intermediate checkpoints (mini run: s0100, s0500, s1100). All his stored
cells were sampled with vLLM 0.11.0 and the same template; his `T2_*` hint jobs ran at tensor-parallel 2, ours at 1
(numerics only). `analyze_arms.py` on his stored cells reproduces his olmo-answeronly summary: dpo T1 13.9 % (his 0.139
[0.078, 0.211]), dpo A1 63.3 % (0.633), s2300 T2 93.5 % (0.935), s2300 A1 95.2 % (0.952), strict, mean over the 41
prompts. For H, `solved_pct` compares cells at equal tries (his first 8 draws of 64 / 16 against our 8); the rate uses
all draws.

## Our new cells and seeds (new; his were below 60,000 or in the 7,000–8,000s)

| stage | checkpoints (mini run / second pass) | draws per checkpoint | seed |
|---|---|---|---|
| T | s0100, s0500, s1100 / s0050, s0200, s0750, s1600 | 230 requests × 8 = 1,840 answers | 610000 + step |
| H | same | 48 × 8 = 384 (or 48 × 4 = 192) | 620000 + step |
| rep | dpo, s2300 | 218 × 2 = 436 | 630000 + step |
| L | dpo, s0100, s0500, s1100, s2300 | 48 × 8 = 384 | 640000 + step (`seeds_L.csv`) |

`stage2/seeds.csv` holds them; every job's meta records its seed, revision and the prompt token sha256s.

## Cost (estimates from his stored jobs, not measurements)

Derivation in `cost_table.py`. Tokens per draw: his plain arm-B mean per reach draw at the checkpoint × the hinted /
plain ratio he measured at dpo (0.913) and s2300 (0.797), linear in step; rep from his `F_dpo_ifb` (4,463 tokens per
draw) and `F_s2300_ifb` (6,586). Throughput from his `compute.csv` by job size (1,550 tok/s at 300–500 draws, 1,350 at
100–250), plus 1.5 min load. T from his answer-only jobs (22 min for four conditions at dpo, 6.5 min at s2300; ours
has two). Download 3 min (his logs: 24–925 s; ours at dpo: 20 s). EDL is a placeholder (another agent's runner), to be
replaced by its bench. Secure H200 $5.29 / h. Measured so far: rep at dpo took 26.5 min (load 1.5 + generation 23.1;
1.91 M tokens at 1,383 tok/s, the last minutes one long draw), against 22 estimated; download 20 s on rp-ol1.

**Mini run (Amendment 1): dpo, s0100, s0500, s1100, s2300**

| ckpt | step | download min | EDL min (placeholder) | T Mtok / min | H n=8 Mtok / min | H n=4 Mtok / min | rep Mtok / min | sampling min (H n=8) | total min (n=8) | $ (n=8) | $ (n=4) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dpo | 0 | 3 | 45 | – | – | – | 1.95 / 22 | 22 | 70 | 6.21 | 6.21 |
| s0100 | 100 | 3 | 25 | 0.40 / 15 | 1.66 / 19 | 0.83 / 12 | – | 34 | 62 | 5.45 | 4.78 |
| s0500 | 500 | 3 | 25 | 0.28 / 11 | 1.70 / 20 | 0.85 / 12 | – | 30 | 58 | 5.14 | 4.46 |
| s1100 | 1100 | 3 | 25 | 0.19 / 8 | 2.29 / 26 | 1.14 / 16 | – | 34 | 62 | 5.44 | 4.51 |
| s2300 | 2300 | 3 | 45 | – | – | – | 2.87 / 32 | 32 | 80 | 7.09 | 7.09 |

Sum: 333 min ($29.33) with H at 8 draws, 307 min ($27.05) at 4; of which the sampling stages are 153 min ($13.46) at
8. Per arm over the mini run: T 34 min ($3.00), H 65 min ($5.73) at 8 draws or 40 min ($3.53) at 4, rep 54 min
($4.76). Add per pod: set-up about 12 min ($1.06; 2 min on rp-ol1) and about 2 min of polling gap per checkpoint.

**Possible second pass: s0050, s0200, s0750, s1600**

| ckpt | step | download min | EDL min (placeholder) | T Mtok / min | H n=8 Mtok / min | H n=4 Mtok / min | rep Mtok / min | sampling min (H n=8) | total min (n=8) | $ (n=8) | $ (n=4) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s0050 | 50 | 3 | 25 | 0.44 / 16 | 1.59 / 19 | 0.80 / 11 | – | 34 | 62 | 5.49 | 4.85 |
| s0200 | 200 | 3 | 25 | 0.35 / 13 | 1.59 / 19 | 0.80 / 11 | – | 31 | 59 | 5.23 | 4.59 |
| s0750 | 750 | 3 | 45 | 0.22 / 9 | 2.17 / 25 | 1.09 / 15 | – | 33 | 81 | 7.18 | 6.30 |
| s1600 | 1600 | 3 | 25 | 0.16 / 7 | 1.91 / 22 | 0.95 / 13 | – | 29 | 57 | 5.01 | 4.24 |

Sum: 260 min ($22.91) with H at 8, 227 min ($19.97) at 4; sampling stages 128 min ($11.27) at 8. All nine checkpoints:
593 min ($52.24) at 8, with the EDL placeholder.

## Order on a pod (per checkpoint)

download (pinned commit, verified) → edl (exit code is the gate; a failure is recorded and the chain continues) → T →
H → L → rep (L and rep only when asked) → weights deleted (unless `--keep-weights`, or a `download`-only run). Each stage writes
`state/<stage>.{RUNNING,DONE,FAILED}`, a log `logs/<stage>.log`, and the chain a heartbeat every 60 s; a stage marked
DONE is skipped on a rerun, and the samplers resume their own output files.
