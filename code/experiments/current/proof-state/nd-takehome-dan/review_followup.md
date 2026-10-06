# Independent review of the follow-up run (2026-09-16)

Same method as `review_campaign.md`: re-derive the load-bearing numbers from the raw files with
independent code (own written-form depth counter, own reductio and strict-ORE predicates,
`nd_verify` re-run on counted proofs, renaming-class disjointness recomputed).

## Block A — depth-3 replication: holds

- The three new f = 0 sets (`train_depth3_f0_{a1,a2,a3}.jsonl`, 155,000 each): **0** proofs
  with a third box in written form; max length 6.
- All 14 EI arms and 10 frozen controls: solved counts, depth-3 theorem counts and depth-3
  proof counts match `artifacts/fu/blockA_summary.json` exactly (e.g. a1 s0 335/409, a1 s1
  364/426, a3 s0 361/435, b1 s1 122/141; frozen a2 s1 0, b2 s1 10). 0 verifier failures in
  150-proof samples per arm. Targets share no class with the new training sets.
- The "first round" values in the summary use the min over all raw records of a normalised
  proof (the agent's bookkeeping fix); a naive first-seen recount gives values ≥ theirs, as
  expected. Theirs are correct.
- The frozen controls of the new f = 0 models do find 0–10 depth-3 theorems each, so the
  campaign-1 statement "the base model produces none" is indeed draw-dependent, as the
  follow-up says. It does not change the acquisition gap (0.27–0.36 vs ≤ 0.01).

## Block B — derived-double-negation reductio: holds

- `targets_reductio2.jsonl`: 606 targets, **0** contain `( ~ ( ~` anywhere, 600 from schemata
  + 6 generator-native, none with `min_lines_ub < 7`, no class overlap with either reductio
  training set.
- EI f = 0: seed 0 **58/606**, all 58 strict reductio by my predicate; seeds 1 and 2 **0/606**.
  Frozen: 3 / 0 / 0. f = 0.1: 95 and 63, all strict; frozen 25 / 29. 0 verifier failures.
- Base pass@10⁴ for the f = 0 seed-0 model: 5/300 solved, all with a reductio-shaped proof;
  per-proof hit counts are now stored (review caveat 3 from campaign 1 is fixed).
- Interpretation check: the seed-0 result is elicitation of a base-model generalisation
  (the base already produces the shape at ~10⁻³ on the easiest schema). Agreed, with one
  caution: "the base generalised NEGI-then-DN" is itself a claim about a single Stage-1 draw;
  the two other draws did not. Three seeds cannot say how often a cap-6 model makes that
  generalisation. It is a base-model property worth measuring directly (many Stage-1 seeds,
  base pass@10⁴ only, no RL) before it is quoted as a rate.

## Block C — cap-8 strict derived-ORE: holds

- Sets: 154,994 proofs each, max length 8; strict derived-ORE counts **0 / 155 / 1,550**.
- Arms: strict-acquisition theorems 4 / 4 / 11 / 11 (f = 0, 0, 10⁻³, 10⁻²); frozen 3 / 2;
  targets 500 with no ≤ 8-line proof found. Matches.
- Interpretation check: "never selected because the targets do not require it" is supported
  (plain solve rates 197–218/500, mostly via depth-3 proofs). It also means the block did
  not test whether RL *can* acquire the pattern when it is required — the same construction
  problem as campaign-1 reductio. A pool where the strict ORE is the only route would need
  targets that are not provable without case analysis on a derived disjunction; none of the
  generator's outputs provide that.

## Verdict

Every number checked reproduces. The follow-up's three conclusions stand: depth-3 acquisition
from zero coverage replicates across 8 arms and is insensitive to the pretraining draw; the
reductio "composition" in one seed is amplification of a rare base-model generalisation; the
cap-8 derived-ORE dial is flat because the targets never require the pattern. Open
methodological issue carried forward: for rule-sequence patterns, the campaign has still not
produced a target pool that *requires* the pattern, so "RL does not invent rule sequences" is
supported only by the absence of the pattern when it was optional plus the seed-0 reductio
case being base-reachable.
