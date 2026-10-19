---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - phuong2024dangerous
---

# Evaluating frontier models for dangerous capabilities: milestones and "expert bits" as a graded measure of closeness to a capability

Paper: [@phuong2024dangerous] (Phuong, Aitchison, Catt, Cogan, Kaskasoli, Krakovna, Lindner, Rahtz, Assael,
Hodkinson, Howard, Lieberum, Kumar, Raad, Webson, Ho, Lin, Farquhar, Hutter, Delétang, Ruoss, El-Sayed, Brown,
Dragan, Shah, Dafoe, Shevlane 2024)
Source: arXiv 2403.13793v2 (HTML rendering). Read in depth: Sec. 2 (agents, elicitation), Sec. 6 (self-
proliferation: 6.2 methodology, 6.3 results, 6.4 discussion, 6.5 scaling trends), Sec. 9 (future work), App. E.4-E.5.
The persuasion, cyber and self-reasoning evaluations were not read.

## Learnings

- **Upper-bound intent, elicitation by scaffolding and (briefly) fine-tuning.** "Ideally, we could estimate an upper
  bound on the level of harm a model could cause, even under pessimistic assumptions"; this is "extremely
  challenging" because models "can be considerably enhanced through various methods such as fine-tuning, prompt
  engineering, and access to tools" (Sec. 2). They tried SFT on "researcher-generated trajectories for (benign)
  tasks adjacent to those tested in Section 6", which helped at first but was overtaken by a newer checkpoint (Sec.
  2). Future work: "Better capability elicitation methodology – in particular, working out how best to fine-tune
  models to elicit their capabilities" (Sec. 9).
- **Closeness to success, measured in bits of help (Sec. 6.2).** "By measuring how much information the agent
  needs, we can distinguish between agents very close to having a capability from those very far away." Ideally
  one would find "the absolute minimum amount of information sufficient for success, but this objective trades off
  against the efficiency and reproducibility of the evaluation process" (Sec. 6.2). Procedure, per milestone
  (3-7 sub-tasks per task, agent instantiated at each milestone):
  1. *Unguided runs* (10 per milestone): cost ⌈−log₂ ρ⌉ bits, ρ = empirical success rate — "approximately the
     number of bits required to, on average, provide a random seed that would allow the agent to succeed in a single
     attempt".
  2. *Expert best-of-N* if all 10 fail: sample N = 16 actions, de-duplicate, rank by model probability; the expert
     picks the first that makes progress; "We charge the agent log₂[i(i+1)] bits of expert help per step, where i is
     the index of the selected action"; "An alternative framing: we are simulating the agent getting "lucky"."
  3. *Golden solution* if still failing: teacher-forced probability of a hand-written solution; "the negative log
     probability the agent assigns to the solution is equivalent to the number of nats it would take to communicate
     the golden solution to the agent"; "This method means we can always give the agent a score (in bits), even if it
     is far from success" (Sec. 6.2).
- **Results.** No end-to-end task solved, yet "Pro 1.0 and Ultra 1.0 required only 10 and 9 bits to complete the
  Bitcoin Wallet task and 126 and 55 bits to complete the Email Setup task" (Fig. 9 caption). Best-of-N reveals that
  "the agents appear to know good actions but rate poorer alternatives more highly" (Sec. 6.4.2).
- **Milestone bound.** Sub-task successes give an upper bound on whole-task success probability (App. E.4; "upper
  bound of a one-sided 97.5% CI"), assuming every milestone is necessary — "it is still possible that a sufficiently
  capable agent could find a shortcut" (Sec. 6.2, fn. 11).
- **Toward scaling trends.** Expert bits are proposed as a smooth "y-axis"; "We hypothesise that it is possible to find
  a (non-linear) correlation between expert bits on a task and easy-to-obtain measures of the model's general
  performance" (Sec. 6.5); a prototype normalises bits "by the number of bits required to compress the golden
  solutions using gzip" (App. E.5).

## Evidence and limitations

- The bits are an upper bound on the help needed (methods are tried in a fixed order; best-of-N only after 10
  unguided failures, fn. 12). The expert's judgement of "progress" is human and not reproducible exactly.
- 10 unguided runs per milestone make ρ coarse (ρ ≥ 0.1 or 0); the golden-solution score depends on which solution
  was written.
- Not checked: per-milestone tables (App. E.3), the other evaluation families.

## Connections and questions

- **Definition offered:** closeness to having a capability = the minimum information (bits) an expert must supply
  for the agent to succeed, estimated by the cheapest of three nested methods: random-seed bits ⌈−log₂ ρ⌉
  (outcome-level selection), per-step selection bits log₂[i(i+1)] (step-level best-of-N), or the teacher-forced
  surprisal of a reference solution. Zero bits = the agent has it unaided.
- **New vs better access:** not framed this way, but the measure is graded rather than binary, so "has it" becomes
  "needs ≤ B bits". The paper's interpretation of best-of-N ("know good actions but rate poorer alternatives more
  highly") is an access reading of low-bit cases.
- **Null / floor:** handled naturally: every model, including a random one, gets a finite score, and a random model
  needs about as many bits as the golden solution's length under a uniform code. Dan's k becomes a bit budget,
  k ≈ 2^B; the floor is the bits a random-init model needs, so capability is measured as bits saved relative to it.
- **Transfer to our setting:** almost one-to-one. For each theorem and checkpoint (random-init, Stage-1 steps, pend,
  r8, r16): (1) unguided bits ⌈−log₂ p̂⌉ from existing k = 256 counts (caps at 8 bits; larger k costs pod time);
  (2) step-level bits from guided sampling: the index of the first Lean-accepted candidate among N ranked samples
  per step gives Σ log₂[i(i+1)] — our guided sampler already redraws failing steps, so only the ranks need logging;
  note Lean certifies validity, not progress, so a reference proof or search is needed to judge "progress"; (3)
  golden bits = −log₂ p(reference proof) by teacher forcing (free; the project's per-step log p). Report the minimum.
  RL's effect on a theorem = bits(pend) − bits(r16); "pend already had it" = bits(pend) ≤ B for a declared B, and B can
  be compared with the information the RL loop delivers per target (at most one reward bit per sampled attempt).
  Main failure modes: alternative routes make the golden-solution bits an overestimate (and the milestone bound
  invalid, as the authors note for shortcuts); step-level validity ≠ progress; methods differ in tightness, so
  comparisons across checkpoints must use the same method mix.
- Related notes: shevlane2023extremerisks.md (same group; latent capabilities), hernandezorallo2017evaluation.md
  (capability on a difficulty scale), deeb2024unlearning.md (information criterion), harding2024capability.md
  (practical availability as probability mass).
