# Numbers ledger (single source of truth for the writeup)

All "solve" = independent verifier accepts a proof of exactly the prompted
sequent. pass@1 = greedy; pass@8 = 8 samples at T=0.8; rollouts at T=1.0.

## Models (all 4L/8H/d256, 3.3M params, 6000 iters, bs 128, from scratch)

| model | train data | iid greedy | iid pass@8 | val loss |
|---|---|---|---|---|
| L8 | 150k thms, proofs 2-8 lines, fixed idx | 0.876 | 0.955 | 0.016 |
| L8off | same proofs, random idx offsets | 0.889 | 0.964 | 0.059 |
| L12 | 150k, 2-12 lines | 0.854 | 0.950 | 0.034 |
| L16 | 150k, 2-16 lines | 0.853 | 0.964 | 0.036 |
| L8_2xtrue | 300k, 2-8 lines | 0.859 | 0.952 | 0.014 |

(iid range across all conditions: 0.853-0.889 greedy, 0.950-0.964 pass@8.)

## Cliff (greedy solve by generating-proof length, eval_iid + eval_ood)

- L8: 9-10: 0.33/0.25; 11: 0.22; 12: 0.08; 13: 0.04; 14: 0.07; 15+: ~0.02->0
  (total ood 96/1650 = 5.8%)
- L12: 9: 0.64; 10: 0.62; 11: 0.43; 12: 0.32; 13: 0.17; 14: 0.17; 15: 0.12;
  16: 0.09; 17+: ~0.06 (total ood 221/1650 = 13.4%)
- L16: iid 0.853; ood: 13: 0.29; 14: 0.24; 15: 0.17; 16: 0.14; 17: 0.18;
  18: 0.10; 19: 0.06; 20: 0.09 (total ood 294/1650 = 17.8%) — cliff at its
  own cutoff, third confirmation.
- L8_2xtrue: iid 0.859; ood 9: 0.42, 10: 0.27, 11: 0.20, 12: 0.15, 13: 0.06,
  14: 0.09 (total ood 103/1650 = 6.2% vs L8 5.8%) — doubling data nudges the
  reducible band, cliff structure unchanged.
- L8off: total ood 125/1650 = 7.6%

## Found-proof-length frontier (the wall)

- L8 ood greedy solves: ALL 96 proofs <=8 lines {3:1,4:2,5:9,6:14,7:25,8:45}
- L8 ood pass@8 solves: ALL 140 proofs <=8 lines {3:1,4:3,5:13,6:19,7:40,8:64}
- L8off ood greedy: {3:1,4:2,5:10,6:19,7:27,8:37,9:24,10:5} (29/125 > 8)
- L8off ood pass@8: 220 solved {3:1,4:3,5:15,6:31,7:47,8:69,9:45,10:9}
  (54/220 > 8, max 10); eval_local pass@8: max 11 (2 instances)
- N9 probe: P(N9 | valid 8-line prefix): L8 = 8.3e-7; L8off = 3.9e-2 (~50,000x).
  P(N6|5), P(N7|6), P(N8|7): L8 = .0075/.045/.0096; L8off = .012/.013/.014.

## Chain-1 (fixed idx, 1200 targets, k=32, T=1.0, rep x8, retrain from scratch)

| round | new disc | cum | transfer pass@1 | transfer pass@8 | textbook/36 (pass@16) |
|---|---|---|---|---|---|
| 0 | — | — | 0.156 | 0.244 | 15 |
| 1 | 313 | 313 | 0.156 | 0.230 | 11 |
| 2 | 37 | 350 | 0.190 | 0.258 | 13 |
| 3 | 30 | 380 | 0.182 | 0.268 | 12 |
| 4 | 24 | 404 | 0.180 | 0.252 | 11 |
| 5 | 14 | 418 | 0.178 | 0.274 | 12 |

- CHAIN-1 VERDICT: discoveries 313/350/380/404/418 (all <=8 lines, every
  round); static controls reach 364 (seed 123) / 359 (seed 7). Transfer flat:
  pass@1 ends 0.178 vs 0.156 round-0; pass@8 ends 0.274 vs 0.244 — both well
  below base pass@64 = 0.310. Textbook ends 12/36, below round-0's 15/36.
  Sharpening, no expansion.

- ALL chain-1 discovered proofs <=8 lines: rounds 1-3 buffers max len exactly 8
  (313, 350, 380 records; zero >8). The wall holds across rounds.
- Static control (L8, never retrained, seed 123): 310/327/341/355/364.
- Static control seed 7: 359 final (vs seed-123: 364).
- Textbook pass@256 (base L8) = 15/36 = pass@16 (saturated). L8off: 13/36.
- Forgetting check (iid greedy by chain-1 round): 0.876/0.853/0.884/0.873/
  0.864/0.882 — stable; verifier-filtered self-training does not collapse
  in-distribution performance.
- Chain-1 round-5 pass@64 on transfer: 0.356 (vs base 0.310, n=500): RL does
  not shrink large-k coverage; mild expansion (+4.6pp, ~1.5 sigma by
  difference-SE), mostly concentration.
- Depth analysis: train depth hist {0:24%,1:47%,2:25%,3:4.5%,4:0.08%};
  chain-1 discoveries depth <=3 all rounds, proportions mirror training —
  discoveries stay within the structural envelope of training.

## Controls (token-matched, same round-0 ckpt)

| mode | round | pass@1 | pass@8 | textbook |
|---|---|---|---|---|
| oracle | 0 | 0.156 | 0.244 | 15 | (sanity: matches baseline)
| oracle | 1 | 0.166 | 0.246 | 12 |
| oracle | 2 | 0.194 | 0.266 | 12 |
| oracle | 3 | 0.222 | 0.314 | 14 |
| oracle | 4 | 0.180 | 0.314 | 13 |
| oracle | 5 | 0.178 | 0.274 | 12 |

- ORACLE VERDICT: generator long proofs (fixed format, carry N9+ tokens,
  direct supervision) lift transfer to ~0.27-0.31 — above prim/chain-1,
  below offset-RL (~0.35-0.43). Same round-to-round retraining noise.
| prim | 0 | 0.156 | 0.246 | 15 |
| prim | 1 | 0.170 | 0.232 | 13 |
| prim | 2 | 0.168 | 0.232 | 12 |
| prim | 3 | 0.188 | 0.244 | 13 |
| prim | 4 | 0.146 | 0.214 | 10 |
| prim | 5 | 0.156 | 0.228 | 12 |

- PRIM VERDICT (complete): flat/noisy; extra same-distribution data does
  nothing for transfer or textbook. Prediction 6 held.

## Chain-2 (offset idx, same protocol)

| round | new disc | cum | new >8-line (hist) | transfer pass@1 | pass@8 | textbook |
|---|---|---|---|---|---|---|
| 0 | — | — | — | 0.246 | 0.342 | 12 |
| 1 | 525 | 525 | 183 (152x9, 29x10, 2x11) | 0.228 | 0.366 | 13 |
| 2 | 96 | 621 | 73 (36x9, 30x10, 7x11) | 0.246 | 0.380 | 11 |
| 3 | 60 | 681 | 44 (20x9, 16x10, 7x11, 1x12) | 0.230 | 0.374 | 12 |
| 4 | 29 | 710 | 21 (5x9, 11x10, 4x11, 1x12) | 0.266 | 0.426 | 15 |
| 5 | 37 | 747 | 25 (9x9, 11x10, 5x11) | 0.230 | 0.346 | 13 |

- CHAIN-2 VERDICT: monotone frontier compounding — discoveries 525 -> 747
  (static-off: 607), >8-line proofs in buffer 183 -> 346 (final hist
  222x9, 97x10, 25x11, 2x12), longest 11 -> 12. Transfer pass@8 fluctuates
  0.346-0.426 across rounds (n=500, SE ~2.2pp; round-to-round retraining
  variance dominates), averaging ~0.38 vs 0.342 round-0 and ~0.23-0.31 for
  every fixed-format condition. NOT monotone — report honestly.
- Chain-2 round-4 textbook 15/36 includes consequentia_mirabilis,
  contraposition_conv, reductio_pair — items round 0 could not solve; round 5
  drops back to 13 (n=36, noisy).

- RATCHET SIGNAL: round-2 new discoveries shift longer (10-line/9-line ratio
  0.19 -> 0.83; 11-liners 2 -> 7); round-2 new count 96 vs static-off +36.

- Static control (L8off, never retrained): 516/552/571/594/607.
- Static-off LENGTH composition (rerun with recording, same seed):
  final cum hist {9:188, 10:38, 11:3, 12:0} vs chain-2 {9:222, 10:97, 11:25,
  12:2} — retraining multiplier grows with depth: 1.2x/2.6x/8x/(2 vs 0).

## Local-rules-only pool (composition-stable; cliff_*_eval_local_greedy)

- L8:  2-3: .97 | 4-5: .85 | 6-7: .67 | 8-9: .29 | 10-11: .10 | 12-13: .02 | 14+: ~0
- L12: 2-3: .94 | 4-5: .79 | 6-7: .56 | 8-9: .28 | 10-11: .11 | 12-13: .05 | 14-15: .01 | 16+: 0
- L16: 2-3: .92 | 4-5: .73 | 6-7: .57 | 8-9: .41 | 10-11: .27 | 12-13: .15 | 14-15: .07 | 16-17: .03 | 18+: 0
- NOTE: L12 tracks L8 here (its 9-12-line training proofs are discharge-heavy,
  teaching little about long local chains); only L16 separates.

## pass@64 ledger (first 500 transfer theorems, T=1.0)

| model | pass@64 | file |
|---|---|---|
| L8 (fixed, round 0) | 0.310 | pods/a/results/base_L8_transfer_pass64.jsonl |
| chain-1 round 5 | 0.356 | pods/b/results/ei_rl_round5_transfer_pass64.jsonl |
| L8off (offset, round 0) | 0.448 | pods/c/results/L8off_transfer_pass64.jsonl |
| chain-2 round 5 | 0.458 | pods/c/results/ei_off_round5_transfer_pass64.jsonl |

(Difference SEs ~3pp: fixed +4.6pp ~1.5 sigma; offset +1.0pp <0.5 sigma.)

## Set-level reach cross-reference (chain-2 vs all fixed-format results)

- Chain-2 final buffer: 747 theorems; 346 via >8-line proofs.
- Of those 346: 45 also solved (<=8 lines) by chain-1; 301 never solved by
  any fixed-format model in the study. Chain-2-only theorems: 340 (lens
  {6:4,7:9,8:26,9:185,10:91,11:23,12:2}); chain-1-only: 11.
- Coincidence check (chain-1 r5 vs oracle r5, both 0.178/0.274/12): both
  solve 89/500 at pass@1 but share only 54 theorems — distinct models, real
  coincidence.

## Verifier audits

- Original (fixed-index): 2853/2853 positives; 374/377 mutations rejected
  (3 survivors verified as legitimately valid proofs).
- Offset positives roundtrip: 300/300 (test_roundtrip).
- Relaxed-verifier corruption audit (rerun post-relaxation, on pod-C):
  501/501 offset positives accepted; 499/500 single-token corruptions
  rejected. (A first attempt accidentally ran against pod-B's strict
  verifier and was discarded — see log 00:25.)

## Probe details

- N9 probe: n=300 valid 8-line prefixes (from rl_targets with >=10-line
  generating proofs); controls at k=5,6,7 same n. Means: L8
  .0075/.045/.0096/8.3e-7 for N6/N7/N8/N9; L8off .012/.013/.014/.039.
- Offset start indices among 525 round-1 discoveries: min 1, p10 7, p50 31,
  p90 53, max 59 (~uniform; vocab ceiling N64 not binding).

## Textbook dynamics

- Chain-1 lost vs round 0 by round 5: contraposition (a renamed-from-training
  item), contraposition_conv, reductio_pair (15 -> 12).
- Chain-2 round 4 briefly reaches 15/36 (adds consequentia_mirabilis,
  contraposition_conv, reductio_pair), falls back to 13 at round 5.

## Seed-1 replications (complete; full re-runs incl. round-0 training)

- FIXED seed-1: discoveries 279/332/355/380/392 (seed-0: 313/350/380/404/418);
  ALL 392 final-buffer proofs <=8 lines, max exactly 8 (wall replicates).
  Transfer pass@1 .142/.164/.156/.158/.166/.156; pass@8
  .204/.218/.236/.238/.246/.234 (flat). Textbook noted: transiently solves
  "export" at r3 — the only model in the study ever to do so (7-line proof,
  inside the wall).
- OFFSET seed-1: discoveries 521/628/683/708/724 (seed-0: 525/621/681/710/747);
  round-1 >8-line finds 174 (148x9,24x10,2x11) vs seed-0 183; final tail
  216x9/81x10/18x11/2x12, max 12 (both seeds reach exactly 12).
  Transfer pass@1 .196/.238/.242/.250/.240/.212; pass@8
  .334/.360/.354/.370/.370/.340. Seed-0's r4 spike (0.426) does NOT
  replicate (seed-1 r4: 0.370).
- Round-0 level seed-variance: offset pass@1 0.246 vs 0.196 (~5pp), pass@8
  0.342 vs 0.334; fixed pass@1 0.156 vs 0.142, pass@8 0.244 vs 0.204.

## Forgetting checks (per-round iid greedy)

- Chain-1 (fixed, n=1000, post-hoc on round ckpts):
  .876/.853/.884/.873/.864/.882 — endpoint +0.6pp.
- Offset chain (fresh instrumented instance, seed 2, fresh data pools;
  rounds 0-5): .856/.854/.832/.870/.851/.837 — endpoint -1.9pp (~1.2 sigma),
  non-monotone. Buffer grew to 799 discoveries / 127k tokens in this
  instance. files: pods/forget/runs/ei_off_forget/.
- Verdict: no catastrophic forgetting either format; retrain-from-scratch
  design means the tested failure mode is self-data poisoning, not
  sequential overwriting.

## Overhang sweep (offset format, 3 rounds, caps 4/6 on fresh-instance data)

- cap-4: r1 93 {2:13,3:19,4:53,5:8}; r2 108 (+5: 14x5); r3 114 {...,5:15}.
  Max 5 every round (cap+1); robust 5/5.
- cap-6: r1 253 {...,6:92,7:60}; r2 306 {...,7:87,8:2}; r3 349 {...,7:105,8:7}.
  Max 7 -> 8; robust 7 -> 8.
- cap-8 (original chain): r1 {9:152,10:29,11:2}; r3 {9:208,10:75,11:16,12:1}.
  Max 11 -> 12; robust 10 -> 11.
- VERDICT: overhang scales with cap (robust post-RL: +1/+2/+3); RL increment
  over SFT grows with cap (0/+1/+1). Caveats: 1 seed/cap, no static controls
  at caps 4/6, caps 4/6 on the fresh data instance, 3 rounds.
- Figure: figures/fig_overhang.png. Files: pods/cap4, pods/cap6.

## Protocol experiment (Part II §11; fresh instance; base 0.188/0.298/0.408)

- VANILLA: p8 .298/.322/.360/.350/.322/.314; buffers 520/618/683/717/729;
  iid .832-.851; p64 end 0.472. files: pods/proto/runs/ei_rl*.
- HINDSIGHT-EI: p8 .298/.382/.406/.434/.402/.384; buffers 544/661/715/754/771
  (incl. hindsight-routed target solves); harvested store
  1418/1784/2050/2201/2332 (r1 hist {5:459,6:468,7:318,8:134,9:37,10:2});
  iid .832/.856/.865/.868/.867/.837; p64 end 0.508. Ceiling 12 (3x12 @ r3).
- GRPO: 250 iters, batch 192 thms x group 8, T=1.0, lr 2e-5, clip 0.2,
  KL beta 0.03, 2 epochs/iter; evals (cum samples): p8 .384@38k, .396@77k,
  .396@115k, .390@154k, .400@192k, .408@230k, .412@269k, .408@307k,
  .412@346k, .412@384k; p1 trajectory .254/.280/.274/.276/.292/.288/.304/
  .296/.312/.300 (peak .312, end .300). IID NOTE: GRPO's iid evals use a
  fixed n=400 slice of eval_iid (EI arms use n=1000 full set; the slice
  reads higher) — on-slice trajectory .870 (first meas., it25) -> low .805
  (it225) -> .8275 (end): transient mild drift, endpoint near-recovered.
  Cumulative solved 783 (tail 196x9, 87x10, 31x11, 3x12 — ceiling 12);
  p64 end 0.508. At EI-matched budget (~107k samples) p8 ~= .396-.400.
- Missing-from-tables provenance: vanilla p1 trajectory
  .188/.210/.222/.234/.202/.208 (iid end .847); HER p1
  .188/.252/.264/.300/.264/.248; oracle p1 .188/.246/.218/.228/.246/.226,
  oracle p8 .298/.416/.372/.334/.428/.360, oracle iid
  .832/.777/.751/.682/.726/.696 (pods/proto/runs/ei_oracle/history.json);
  d8 per-round buffers 663/787/846 (R increments +663/+124/+59).
- Oracle pass@64: NOT MEASURED (round-5 ckpt discarded at pod teardown
  before the probe was designed). Known gap.
- ORACLE (HER-token-matched: quotas 259k/334k/385k/417k/438k): p8
  .298/.350(r1)/...(see history)/.428(r4)/.360(r5); p1 ends .226;
  iid .832 -> .726 (r4) -> .696 (r5). Ends behind HER on all metrics.
- pass@64 quartet (n=500): base .408 | van r5 .472 | HER r5 .508 |
  GRPO final .508. files: pods/proto/results/*_pass64.jsonl.
- UNIVERSAL CEILING: max discovered proof length = 12 in every protocol,
  every seed, every instance (chain-2 s0/s1, forget-instance, vanilla/HER/
  GRPO on proto instance).

## Depth sweep (Part II §12; proto instance; cap-8 offset data; hindsight-EI
## 3 rounds; width d=256 fixed; same 6k steps/seed)

| layers | params | val loss | iid r0 | tr p8 r0 | tr p8 end | buffer r3 | max len | robust |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.95M | .532 | .000 | .000 | .000 | 0 | — | — |
| 2 | 1.74M | .110 | .412 | .066 | .082 | 221 | 9 | 9 |
| 4 | 3.32M | .059 | .832 | .298 | .384 (r5) / .434 peak | 715 (r3) | 12 | 11 |
| 8 | 6.48M | .062 | .931 | .408 | .488 | 846 | **13** | 12 |
| 16 | ~12.8M | .066 | .939 | .398 | .486 | 810 | **13** | 12 |

- d16 trajectory: iid .939/.896/.899/.946; p1 .274/.264/.322/.372;
  p8 .398/.384/.446/.486; r3 tail {220x9, 104x10, 32x11, 3x12, 2x13}.
- SATURATION: d16 ~= d8 on all metrics at this width/step budget.

- d8 r3 tail: {9:222, 10:112, 11:41, 12:4, 13:1}. The 13-liner is a 4-deep
  nested ORE case analysis (training depth-4 frequency: 0.08%; no 4-layer
  model ever produced nesting depth 4). Re-verified.
- DEPTH-8 ROUND-0 SFT (.931 iid, .408 p8) matches or beats every 4-layer
  END-OF-PROTOCOL result. d16 r0 tracks d8 (saturation at this width/steps).
- d16 solves "explosion" at round 0 (style-blindness cured by depth);
  d8 r1 also solves it.
- 12-line "universal ceiling" of 4-layer models BROKEN at depth 8 => it was
  model capacity, and the target pool does contain >=13-line-reachable
  theorems (partially de-confounding the pool-reducibility caveat).

## Compute

- 4 RunPod pods: 3x A40 @ $0.44/h (3.0-3.8h each), 1x RTX 4090 @ $0.69/h
  (3.4h), + ~$0.7 of stuck-pod time early. Total ~= $8 by pod-hour
  accounting. All pods terminated 01:55. (Account balance delta $15.54
  includes other users' pods on the shared account — upper bound only.)

## Leakage / context audits

- Alpha-renaming overlap with train_L8: eval_transfer 2.7%, eval_ood 1.3%,
  eval_iid 31.4%, textbook 10/36.
- The 10 renamed textbook items are exactly 10 of the base model's 15 solves;
  all 21 unsolved textbook items are novel-up-to-renaming.

## Failure forensics (round-0 L8, transfer greedy, 422 unsolved)

- Invalid rule applications dominate (ANDI 71, NEGE 42, IMPE 39, ORE 27, ...)
- "line index mismatch at position 9": 63 (the token wall mode)
- wrong conclusion: 29; premise block mismatch: 19

## Infrastructure

- Verifier cross-validation: 2853/2853 positives, 374/377 mutations rejected
  (3 survivors legitimately valid). Offset roundtrip: 300/300.
- Compute: 4 pods (3x A40 @ $0.44/h, 1x 4090 @ $0.69/h). Total spend: [PEND]
