# Shared brief: what kinds of proofs can't our models solve?

Robbie (experienced SWE / ex-quant, curious, likes direct talk) wants to know, QUALITATIVELY, what types of
natural-deduction proofs our RL'd models fail on. "Short proofs the model doesn't figure out" is one heuristic; be
more creative than that. The deliverable is a crisp taxonomy with counts and 2-4 concrete example proofs per class,
plus a falsifiable hypothesis for each class ("fails because X; would be fixed by Y").

## Setup
- Experiment dir E = /Users/robbie/repos/nd-rl/code/experiments/current/long_rl_20261014 (README.md, harness.py,
  pretrain.py, make_data.py). Pretraining saw ONLY proofs of <= 6 body lines (body = lines after the premise block;
  premises are not counted). Up to 10 premises (real + distractors), 10 atoms. Then 3 h of RL (expert iteration, EI)
  on targets of 7-80 body lines. Lean is the only judge.
- Pools in E/data: rl.jsonl (18,439 RL targets), dev.jsonl (843), holdout.jsonl (823; don't use). Fields: key
  (canonical a0..a9 form), prompt ("THM prem , prem SEQ goal PRF"), proof (gold ND proof, ND syntax: `N7 | | X : AS`,
  `|` = box depth, rules PR AS R IMPI IMPE ANDI ANDE1 ANDE2 ORI1 ORI2 ORE NEGI NEGE IFFI IFFE... read
  $ND_FORK/nd_verify/verify.py for the rule set), n_prem, n_atoms, body (gold body length), body_exact, cert_ge,
  src (generator family file: b_f20/b2/b3/bridge/c/d/e/s ... see
  /Users/robbie/repos/nd-rl/code/experiments/current/long_proofs_20261013/ for how each family was generated), band.
- "Old dev" = 1,108 theorems of 0-13 body lines from Dan's ladder: $ND_FORK/data/ladder/transfer.jsonl, the half with
  int(sha1(key).hexdigest(),16) % 2 == 0; body = n_lines - n_prem. Evaluated at k=16.
- ND_FORK=/private/tmp/claude-501/-Users-robbie-repos-nd-rl/a1d02fb4-ba43-48c4-838b-e227293b6c2a/scratchpad/fork51604b3
- Artifacts A = /Users/robbie/repos/nd-rl/artifacts/long_rl_20261014/v2/long-rl-20261014-v2_<arm>_s<seed>/
  (arms untied-ei, looped-ei; seeds 0, 1):
  - eval_dev.jsonl (843 rows, k=64): name, body, solved, n_ok, n_tried, proofs (accepted ND proofs), written, pruned
  - eval_old_dev.jsonl (1,108 rows, k=16): same fields; names index the old-dev pool in order (check how)
  - found_targets.jsonl: every Lean-accepted proof found during RL: name (rl_*), prompt, body, proof, written,
    pruned, round (RL oracle call it was found in; ~60-70 calls over 3 h). A target absent = never solved in RL.
    For each RL target, 8 samples/round when picked, plus 24 retries for all-fail targets.
  - summary.json: per-round stats (rl.rounds), eval by band.
- Checkpoints: E/resume/<arm>_s<seed>/final.pt (after RL) and pretrain.pt (after pretraining only).
- Logs: /Users/robbie/repos/nd-rl/artifacts/long_rl_20261014/v2/driver_<arm>_s<seed>.log.

## Headline numbers (v2, 3 h RL)
| arm | dev solved /843 | old dev /1108 | RL targets solved /18439 |
| untied s0 | 232 | 852 | 7,020 |
| untied s1 | 247 | 929 | 7,395 |
| looped s0 (8 loops) | 246 | 922 | 6,437 |
| looped s1 | 246 | 929 | 6,430 |
Solved share by band (untied s1, RL pool): 10-14: 69%, 15-19: 11%, 20-24: 5%, 25-34: 22/2500, 35+: 0/2913.
Old dev before RL (pretrain only): 50 untied / 23 looped. Unsolved old dev after RL (untied s1) includes 12 six-line
and 18 seven-line theorems, i.e. SHORT proofs it still can't do.

## Running a model on this Mac (CPU, no CUDA)
Helper: /private/tmp/claude-501/-Users-robbie-repos-nd-rl/a1d02fb4-ba43-48c4-838b-e227293b6c2a/scratchpad/probe/ndmodel.py
(read its docstring). Python: ~/repos/nd-takehome/.venv/bin/python (no matplotlib; for plots use
`uv run -q --with matplotlib python ...`). Example:
  cd <your dir> && ND_ARCH=untied LEAN_GATE_WORKERS=2 ~/repos/nd-takehome/.venv/bin/python -c "
  import sys; sys.path.insert(0, '/private/tmp/claude-501/-Users-robbie-repos-nd-rl/a1d02fb4-ba43-48c4-838b-e227293b6c2a/scratchpad/probe')
  import ndmodel as M; model, tok = M.load('<E>/resume/untied-ei_s1/final.pt'); ..."
- token_logp(model, tok, prompt, gold_proof) -> per-token (token, logp, argmax token): ~0.04 s per proof. Proof tokens
  are Lean-flavoured text (`have n3 : ( S ∨ X ) := h3 ; ... Or.elim n29 ( fun ( n35 : S ) => by ...`), produced by
  tok.encode_proof from the ND proof; names n*/h* are randomly shifted, so name tokens are noisy, focus on structure.
- sample(model, tok, recs, k) -> per rec [(decoded ND proof or an error tag like "LEANPARSE unbound"/"LEAN ...",
  lean_ok, raw model text)] at T=0.8, Lean-judged. ~1 s per sample for short proofs on CPU, more for long ones.
  ND_ARCH must match the checkpoint (looped checkpoints: ND_ARCH=looped; set model.n_loops = 1/2/4/8).

## Rules (hard)
- Never use nd_verify to CHECK a proof. Lean only (the helper's sample() already Lean-judges). Parsing proofs with
  nd_verify.verify.parse_proof_tokens for structure is fine.
- Memory: an oom-guard SIGKILLs processes > 16 GB and all agent descendants > 29 GB total. Three other agents run in
  parallel. Keep torch at <= 3 threads (torch.set_num_threads), LEAN_GATE_WORKERS=2, <= ~300 samples per sample() call,
  and don't load big files many times in parallel. If something exits 137, check `tail -20 ~/Library/Logs/oom-guard.log`.
- Don't touch pods, RunPod, HF, git, or any file outside your own work dir:
  /private/tmp/claude-501/-Users-robbie-repos-nd-rl/a1d02fb4-ba43-48c4-838b-e227293b6c2a/scratchpad/probe/<your letter>/
  Don't edit ndmodel.py (copy it if you need changes). Don't post anywhere.
- macOS pkill/pgrep: options before the pattern.
- Proof length = body lines. Distinguish observed results from interpretation. Quantify.
- Budget: aim to finish in ~60-90 min wall clock. CPU sampling is the slow part; spend it on targeted experiments,
  not brute force.

## Deliverable
Write <your dir>/REPORT.md (taxonomy table with counts, examples, hypotheses, the experiments you ran and their
numbers, what you'd test next on a GPU). Your final message: a <= 400-word summary of the findings + the report path.
