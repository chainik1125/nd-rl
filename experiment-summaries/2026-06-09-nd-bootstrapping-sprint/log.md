# Research Log — ND Bootstrapping Sprint

**Sprint window:** 2026-06-09 21:35 PDT → 2026-06-10 07:35 PDT (10h wall-clock).
**Budget:** < $50 RunPod total, < $10/h.
**Deliverable:** `summary.md` — executive summary + findings, ≥1h spent on writing alone.

## Project in one paragraph

The proposal (`proposal/nd_bootstrapping (1).pdf`) asks: can RL with a perfect verifier
let a small transformer, trained only on short "primitive" natural-deduction proofs,
reach held-out theorems it was never shown — and does that capability compound across
iterated rounds? Propositional ND is the testbed because it has a perfect verifier, a
specifiable training distribution, and a provable difficulty gradient (length
generalization, especially for discharging rules that need subproof state-tracking).

## Plan (provisional, set at 21:40 PDT)

| Hour (PDT) | Goal |
|---|---|
| 21:35–22:35 | Setup: repo, pod, ND core library (formulas, 10 rules, Fitch verifier, proof sampler) |
| 22:35–23:35 | Dataset gen + tokenizer + training harness; first SFT run sanity check |
| 23:35–01:00 | Experiment 1: SFT length-generalization cliff (multiple L_train); rule-class split |
| 01:00–03:30 | Experiment 2: expert-iteration RL from SFT checkpoint; bootstrap over the cliff |
| 03:30–05:00 | Experiment 3: compounding across rounds + controls (extra-primitives, data-matched) |
| 05:00–06:30 | Writing: summary.md, figures, red/blue-team iteration |
| 06:30–07:35 | Final polish, repo cleanup, push |

Scope decision: focus on Claims 1 (bootstrapping) and 3 (compounding) with the SFT
cliff (Fig 1/2 analogs) as the load-bearing prerequisite. Claim 2 (data equivalence
vs human-authored proofs) is out of scope for 10h — noted as future work.

## Pre-registered predictions (written 22:15 PDT, before any model results)

1. SFT-L8: >80% greedy solve on held-out iid (2-8 lines); sharp drop above 8;
   near zero greedy by 15+ lines.
2. The drop is steeper for targets whose generating proof uses discharging rules
   (IMPI/NEGI/ORE) than for local-rule-only targets.
3. Doubling SFT data (L8_2x) does not move the cliff appreciably.
4. Expert-iteration round 1 discovers a nonzero set of band-9+ proofs
   (concentrated at 9-12 lines); transfer pass@8 rises above the SFT-L8 baseline.
5. Rounds 2-5 continue to rise (compounding) — least confident; plateau plausible.
6. Token-matched extra-primitives control does not rise appreciably.
7. Token-matched oracle long-proof control rises the most (skyline).

Headline metrics fixed in advance: (a) greedy + pass@8 solve rate vs generating
proof length per L_train; (b) transfer pass@8 on held-out band 9-24 (theorems
never sampled against during RL) vs bootstrap round, RL vs controls; (c) textbook
pass@16 vs round.

## The pivot (22:50 PDT) — two barriers, and an ablation to separate them

Round-1 RL discovered 313/1200 band targets but EVERY discovered proof is <=8
lines: the model solves harder *theorems* by finding *short* proofs; it never
writes a 9-line proof. Failure forensics show why: line-index tokens N9..N64
never occur in L<=8 training data, so a 9-line proof requires emitting a
never-seen token (embeddings random-init). The cliff therefore stacks a
NOVEL-TOKEN WALL on top of any structural (state-tracking) barrier, and as
designed the bootstrap cannot cross it except by lottery.

Decision: keep chain-1 running (it still answers: how much reachability does
verifier self-training recover *below* the wall, and does it compound?), and
launch chain-2 on a third pod with the token wall surgically removed: every
training proof's line indices get a random offset (N_k -> N_{k+o}), the exact
symbol-level analog of the position-offset trick. Same theorems, same proofs,
same lengths — only the start index varies, so all of N1..N64 are in-distribution
and a 9-line proof requires no new tokens. Verifier relaxed to accept any
consecutive ascending index start (chain-1 proofs unaffected: they start at 1).
Chain-2 answers: with the token wall gone, does RL cross the *structural*
barrier? Either answer is a clean finding.

Also added: static-accumulation control (same rollout protocol, model never
retrained) to separate "retraining increases discovery power" from "more
samples find more lottery tickets" in the discovery curve.

Pods: A = cliff evals, B = chain-1 RL, C = chain-2 offset RL, D (4090) =
static-accum control + prim control later. Burn ~$2.0/h, well under budget.

## Timeline

- **21:34** Sprint start. Read proposal PDF (14pp), sprint advice, writing instructions.
- **21:36** RunPod API key works (balance $2105, shared account — will not touch others'
  pods: rs-k8-q8, rs-k8-q9, mft135-sprint). GitHub auth works (chainik1125).
- **21:39** User answered setup Qs: clock starts now; create my own pod; push private repo.
- **21:42** Repo initialized. Writing ND core next. Provisioning own pod (RTX 4090 class).
- **21:45** ND core + independent verifier written. Cross-validation found a real
  linearizer bug: subproof conclusions that resolve to lines *outside* the box
  (e.g. ⊥ from a premise) left the box without a conclusion line. Fixed by adding
  the standard Fitch reiteration rule (R) to generator + verifier.
- **21:50** Cross-validation passes: 2853/2853 sampled proofs verify; 374/377
  mutated proofs rejected (3 survivors are legitimately valid proofs). Pushed
  private repo github.com/chainik1125/nd-sprint.
- **21:52** Agent team: 'textbook-targets' built 36 truth-table-validated classic
  sequents (data/targets_textbook.jsonl); 'pod-setup' polling pod, will rsync code
  and generate data pools (450k short 2-16 lines, 60k long 17-30 lines).
- **21:57** Full harness committed: GPT (randomized position offsets to kill the
  unseen-positions confound; KV-cache batched sampling), SFT trainer (loss on
  completion only), evalgen (pass@k + verifier), make_splits (disjoint-by-theorem
  splits), expert_iter (RL bootstrap + token-matched prim/oracle controls), figures.
  Throughput plan: ~25 min/RL round (1200 targets x 32 rollouts, bs 1536).
- **22:00** Pod problem: US-NC-1 datacenter pods stay at uptime 0 forever (never
  boot). Burned ~25 min. Fixed by terminating and recreating with A40s in other
  DCs (CA-MTL-1, EU-SE-1) — both booted in seconds. Two pods @ $0.44/h:
  nd-sprint-nd-a (63.141.33.5:22074, agent runs datagen), nd-sprint-nd-b
  (194.68.245.144:22163, mine: smoke tests now, controls later).
- **22:12** GPU smoke test passed end-to-end (gen -> train -> sample -> verify).
  Calibration on A40: train 77 ms/iter (6k iters = 7.7 min); sampling 19 s per
  1536-rollout batch worst-case => RL round rollouts ~8 min. Schedule fits.
- **22:13** Eyeballed sampled proofs (nested boxes, ORE sibling subproofs, NEGI,
  DN all exercised; verifier accepts). Textbook targets file checked: 36 sequents,
  canon premise order matches training format.
- **22:18** Data pools generated on pod-b in ~2 min (96 cores): 450k short (2-16
  lines) + 60k long (17-30). Splits built, all disjoint by theorem id. Two design
  fixes from looking at the data: (a) eval_ood is 99% discharge-rule proofs, so
  added a dedicated local-rules-only eval pool (discharging rules disabled in the
  sampler) for the rule-class comparison; (b) short pool only supported 1.4x for
  the data-scaling control, so generated 150k extra short primitives (new theorems
  only, exclusion list) for a true 2x condition (300k).
- **22:28** L8 SFT done (663 s, val loss 0.016). RL expert-iteration chain launched
  on pod-b (rounds 0-5, 1200 targets, 32 rollouts/round, retrain-from-scratch with
  discoveries upweighted x8, fresh eval on transfer + textbook each round).
  Pod-a runbook: L12 (running) -> L16 -> L8_2xtrue -> all cliff evals (greedy +
  pass@8 on eval_iid/eval_ood/eval_local) + Yue-style pass@64 probe of L8 on the
  transfer band. Monitor armed on both pods.
- **Red-team note (pre-results):** the key alternative explanation for any
  bootstrap success is that RL solves band-9+ theorems via SHORT (<=8 line)
  proofs, i.e. never actually crosses the cliff. analyze_ei.py checks discovered
  proof lengths per round. Decision rule: the bootstrap claim requires discovered
  proofs >8 lines AND transfer gains in bins >8.
- **22:43** FINDING (failure forensics on round-0 transfer greedy, 422 unsolved):
  failures split into (a) majority: invalid rule applications (ANDI/NEGE/IMPE
  rule-check fails — the model writes locally plausible but wrong steps), and
  (b) a distinctive 15% mode: "line index mismatch at position 9" — line-index
  tokens N9..N64 NEVER occur in L<=8 training proofs, so emitting line 9 of a
  proof requires a never-seen token. The cliff therefore has a concrete
  mechanical component (novel-token barrier) on top of the structural one.
  Sharpens the bootstrap question: RL discoveries with >8 lines would inject
  genuinely novel tokens into the training distribution. Must check round-1
  discoveries' length histogram. (Index-free/relative-ref tokenizations noted
  as future-work ablation.)
- **POST-SPRINT, evening Jun 10 — protocol + depth experiments (user-driven).**
  (4) Protocol comparison (user: "is there a protocol where bootstrapping ==
  new knowledge?"): four arms, one instance, one ckpt — vanilla EI, hindsight-
  EI (verifier-localized prefix harvesting), GRPO, oracle-fed. Hindsight-EI
  and GRPO expand capability at every k (+10-12pp, ~3 sigma); oracle data,
  token-matched, ends behind hindsight on everything and collapses iid to
  .696 — self-generated verified data BEATS external data per token. No
  protocol is supercritical; all 4-layer runs ceiling at exactly 12 lines.
  (5) Depth sweep (user suggestion): layers 1/2/8/16, same data+protocol.
  Ceiling broken at depth 8 (13-line proof = 4-deep nested ORE; nesting-4
  never produced by any 4-layer model); depth-8 SFT alone >= every 4-layer
  post-RL result; d16 ~= d8 (saturation at width 256 / 6k steps). Verdict:
  the bootstrapping transition is capacity-gated, not protocol-gated.
  All pods terminated. figs: fig_protocol, fig_depth. Part II = §11-12.
- **POST-SPRINT (afternoon Jun 10, on user questions).** (1) Clarified exec
  finding 3 vs 4 (RL-target growth vs held-out transfer) after the user found
  it ambiguous. (2) Forgetting: instrumented expert_iter with per-round iid
  eval and ran a fresh offset chain — curve .856/.854/.832/.870/.851/.837:
  no catastrophic forgetting in either format (fixed chain had
  .876-.884 stable); failure mode tested is self-data poisoning given
  retrain-from-scratch design. (3) Overhang sweep (user suggestion): caps
  4/6 offset chains (3 rounds) + existing cap-8 chain. Result: overhang
  scales with cap (robust post-RL frontier cap+1/+2/+3 at caps 4/6/8); RL
  amplifies only where SFT already extrapolates. New §6a + fig_overhang.
  All pods terminated again; extra spend ~$4.
- **02:05 SPRINT DELIVERABLE COMPLETE** (4.5h elapsed; 5.5h of the window
  unused). Final state: summary.md (611-word executive summary + 4 figures +
  full technical body) passed two adversarial reviews, two no-context figure
  tests, and a ~95-number fact-check (6 minor fixes applied, all numbers now
  consistent with results/numbers.md). All experiments done including full
  second-seed replications of both chains. All four pods terminated (~$8
  compute). Everything committed and pushed to github.com/chainik1125/nd-sprint.
- **01:50** Seed-1 replications complete (full re-runs of both chains, fresh
  round-0 training): every qualitative claim replicates — the wall (392/392
  fixed discoveries <=8 lines, max exactly 8), discovery counts (fixed 392 vs
  418; offset 724 vs 747), the frontier tail (both seeds reach exactly 12
  lines, 2 instances each), flat transfer in both formats. Seed-0's round-4
  transfer spike (0.426) does NOT replicate — confirmed as retraining luck.
  Curiosity: fixed seed-1 transiently solves textbook "export" at round 3,
  the only model in the study to do so. Integrated into summary §6/§9 and the
  ledger. All four pods terminated (total compute spend ~$10-12).
- **00:55** Red-team round 2 (15 findings incl. 1 blocker) fully addressed:
  ledger regenerated with provenance, exec drift fixed in both directions
  (11->12 not 10->12; textbook "never durably exceeds" wording; r4 swing
  noted), L12 local-pool anomaly reported, scorecard prediction 4 restored
  verbatim and graded failed-in-part, chain-1/oracle endpoint coincidence
  VERIFIED as real (89/500 each, only 54 shared), difference-SEs adopted
  (~1.5 sigma not 2), process map rewritten with clock times. Second
  no-context figure test passed; captions tightened.
- **00:30** Major writeup revision committed: consistent statistical standards
  across chains (the central fix), the proposal-A3 compounding taxonomy as the
  framing (distributional strong / capability marginal / efficiency modest),
  set-level 301-theorem reach analysis replacing per-proof length claims,
  re-audited verifier, completed controls. Honest downgrades applied: chain-2
  transfer is a noisy level effect, not monotone growth; round-5 pass@64
  0.448->0.458 (<1 sigma) means above-wall capability compounding is also
  marginal at never-sampled theorems.
- **00:25** Red-team round 1 (fresh-context agent) returned 15 findings, incl.
  2 blockers: (a) statistical double standard between chains' transfer claims;
  (b) ratchet confounded by depletion/verbosity — and it caught that the
  showcased 12-line proof hand-reduces to 8 lines. Both correct. Fixes:
  consistent SE standard; set-level cross-reference (chain-2 proved 346
  theorems with >8-line proofs, of which 301 never solved by ANY fixed-format
  model — verbosity cannot fake theorem coverage); static-off rerun with
  length recording (in flight); relaxed-verifier corruption audit rerun
  (501/501 positives, 499/500 mutations rejected — first attempt accidentally
  ran against pod-B's strict verifier and was discarded); new fig1b on the
  composition-stable local-rules pool; showcase proof replaced with one from
  the 301 and annotated as non-minimal.
- **00:20** Post-chain analyses: round-0 offset pass@64 = 0.448 (vs fixed
  0.310 — the format alone is worth +13.8pp of large-k coverage, more than
  any RL effect in the study); round-5 offset pass@64 = 0.458. Oracle control
  finished (peaks 0.314 r3-4, ends 0.274); prim control flat throughout.
- **00:05** Chain-2 complete: discoveries 525/621/681/710/747 vs static-off
  607; transfer pass@8 by round .342/.366/.380/.374/.426/.346 (noisy, level
  above all fixed conditions); buffer ends with 346 proofs >8 lines (222x9,
  97x10, 25x11, 2x12). Chain-1 forgetting check clean (iid .853-.884 all
  rounds); chain-1 round-5 pass@64 0.356 vs base 0.310.
- **23:48** Full summary draft written. Figure no-context test (fresh agent)
  returned fixes: unlabeled cap line, internal run tags in titles, integer
  ticks, log-scale for the frontier tail — all applied to figures.py.
- **23:40** Chain-1 complete (418 final; transfer flat). Launched post-RL
  analyses (B), oracle control (A), prim control (D), textbook pass@256 for
  L8off (13/36 = no better than fixed 15/36).
- **23:35** Ratchet continues round 3: +60 new (static-off +19), first 12-line
  proof appears; buffer now has 300 proofs >8 lines (208x9, 75x10, 16x11,
  1x12). Frontier advances ~+1 line per 1-2 rounds. Chain-2 transfer pass@8
  rising 0.342 -> 0.366 -> 0.380; chain-1 flat at ~0.25 with all discoveries
  still <=8 (final curve 313/350/380/404/418 vs static 364/359 two seeds).
  Base-L8 pass@64 on transfer = 0.310 >> chain-1 round-4 pass@1 0.180:
  below the wall, RL concentrates what large-k sampling already reaches
  (the Yue picture, reproduced from scratch).
- **23:17** RATCHET CONFIRMED (round 2): chain-2 adds 96 new theorems (vs +36
  for the never-retrained static control), 73 of them via >8-line proofs, and
  the new-discovery length distribution shifts outward (10/9-line ratio 0.19
  -> 0.83, 11-liners 2 -> 7). Retraining on self-discovered long proofs
  extends the writing frontier. Chain-1 (token wall intact) is meanwhile
  plateauing (pass@8 0.244/0.230/0.258/0.268; all discoveries still <=8 lines
  through round 3). Rounds 3-5 will show how far the ratchet goes.
- **23:12** Alpha-renaming leakage audit: eval_transfer 2.7%, eval_ood 1.3%
  (OOD claims clean), eval_iid 31% (expected, interpolation), textbook 10/36.
  The 10 renamed textbook items (MP, MT, DS, contraposition, dn x2, triple-neg,
  identity, paradoxes) are EXACTLY 10 of the 15 the base model solves; the
  other 5 solved (HS, contraposition_conv, consequentia mirabilis, reductio,
  absorption) are genuinely novel-up-to-renaming; all 21 unsolved are novel.
  Base textbook ability = renamings + 5 new short proofs. Will report.
- **23:09** Methodology self-review (review-experiment skill) flagged two gaps,
  both fixed: (1) chain-vs-static gap was single-run (SE ~15 on counts) ->
  launched second static seed; (2) token wall was inferred behaviorally ->
  added direct probe. PROBE RESULT: P(next-token = N9 | 8-line valid prefix)
  = 8.3e-7 for L8 vs 0.039 for L8off (~50,000x), while P(N6..N8 | 5-7-line
  prefix) agree within a few-fold across models. The wall is mechanistic and
  specific to the never-seen index token. Also noted: after 7-8 line prefixes
  both models put most next-token mass on QED (length prior from the 2-8-line
  training distribution) — successor-index probabilities are small everywhere,
  so the discriminating quantity is the cross-model ratio at k=8.
- **23:05** Chain-2 round 1: 525/1200 discovered (vs 313 for chain-1), including
  183 proofs >8 lines (152 x 9-line, 29 x 10-line, 2 x 11-line). The offset
  model generates beyond its training length in volume; these now become
  training data for round 2. Ratchet test live. L12 cliff confirmed at its own
  cutoff (64% at 9-10 lines vs L8's 33%, falling past 12). Static-off control
  round 1: 516 (consistent with chain-2's 525).
- **22:58** HEADLINE (round-0 ablation): L8 vs L8off on eval_ood (greedy).
  Totals 5.8% vs 7.6%, but the found-proof length histograms are the story:
  L8 solves use proofs of length 3-8 ONLY (96/96 <=8; the wall is literal);
  L8off writes 9- and 10-line proofs out of the box (29/125 solves >8) and
  dies ~11+. So the novel-token wall costs ~1-2 lines of genuine length
  generalization and ~30% relative OOD solves; a structural cliff remains
  slightly further out. iid is unchanged (87.6% vs 88.9%) — the offset is free.
  Chain-2's question is now whether expert iteration RATCHETS the frontier
  (9-10-line discoveries -> in-distribution -> 11-12 reachable...). Launched
  static_accum_off on pod-D as the matching lottery control for chain-2.
- **22:56** Textbook saturation: base L8 pass@256 = pass@16 = 15/36 — 16x more
  samples find nothing new; unsolved targets aren't lottery-limited. Eyeballing
  explosion (P&~P |- Q, a 5-line proof, unsolved at k=256): the policy wanders
  into discharge-heavy constructions instead of the direct BOTE proof —
  reachability is policy-relative, not purely length-limited.
- **22:35** Round-0 baselines (L8 SFT): transfer band 9-24 pass@1 0.156 / pass@8
  0.244; textbook 15/36 at pass@16 (solves MT, contraposition both ways,
  consequentia mirabilis with the classic reductio — eyeballed, textbook-perfect).
  The cliff is soft near the boundary => RL has a gradient to climb. Round-1
  rollouts running. Queued post-RL checks: per-round iid greedy (forgetting),
  round-5 vs round-0 pass@64 on transfer (sharpening at large k), analyze_ei
  (structure of discoveries), then controls: oracle (pod-b) + prim (pod-a).
