---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - hernandezorallo2014aievaluation
  - hernandezorallo2021generality
---

# Hernández-Orallo: from task-oriented to ability-oriented evaluation; capability as area under the agent characteristic curve

Papers: [@hernandezorallo2014aievaluation] (J. Hernández-Orallo, "AI Evaluation: past, present and future",
arXiv 1408.6908v3, 2016 revision; the author says it "is largely superseded by" the AI Review paper
"Evaluation in artificial intelligence: from task-oriented to ability-oriented measurement", 2017,
doi 10.1007/s10462-016-9505-7, which I did not read) and [@hernandezorallo2021generality] (Hernández-Orallo,
Loe, Cheke, Martínez-Plumed, Ó hÉigeartaigh, "General intelligence disentangled via a generality metric for natural
and artificial intelligence", Sci. Rep. 11:22822, 2021).
Sources: arXiv 1408.6908v3 HTML (read Sec. 1, 2.1, parts of 2.3, Sec. 3-4); PMC8613222 HTML of the Sci. Rep.
paper (read abstract, introduction, the sections defining agent characteristic curves, capability, spread and
generality; results skimmed). The book *The Measure of All Minds* (CUP 2017) was **not** accessed: no claim here
rests on it (UNVERIFIED for anything attributed to the book).

## Learnings

- **Task-oriented performance (1408.6908, Sec. 2.1, Eq. 1).** For a task class M with distribution p and
  per-problem result R(π, μ): average-case performance Φ(π, M, p) = Σ_μ p(μ)·E[R(π, μ)], with worst-case and
  best-case variants; rank-based (e.g. worst-case) aggregation "is more robust to systems getting good scores on
  many easy problems but doing poorly on the difficult problems" (Sec. 2.1, fn. 1).
- **Abilities are constructs (Sec. 3.1).** "we can define a cognitive ability as a property of individuals that
  allows them to perform well in a range of information-processing tasks"; "the ability is necessary but it does
  not have to be sufficient"; "While tasks can be seen as measuring instruments, abilities are constructs" (Sec.
  3.1). Ability-oriented evaluation asks not what systems do "but for what they are able to (learn to) do"
  (Sec. 1).
- **Population-relative vs intrinsic difficulty.** In psychometrics "Item difficulty is determined by the
  percentage of subjects that are able to solve the item"; "this difficulty assessment is relative to the
  population and not derived from the nature of the item itself" (Sec. 3.2). In the AIT-based C-test "the
  difficulty of these exercises is intrinsic, and not based on how difficult humans find them" (Levin's Kt, Sec.
  3.3). Universal psychometrics: abilities are "properties that emanate from (general) classes of tasks, perfectly
  defined in computational terms", so "measures are absolute and not relativised wrt. a population" (Sec. 3.4).
  Guideline: "An intrinsic difficulty function (even if approximate) is always very useful" (Sec. 4).
- **Agent characteristic curve and capability (Sci. Rep. 2021, section "Agent characteristic curves and
  capability").** ψ_j(h) = average accomplishment of agent j on problems of difficulty h; the ACC is ψ_j as a
  function of h; "we simply define capability as the area under the curve"; "Capability is just the area of this
  curve" (Eq. 1); being an integral over difficulties it "has the same units as difficulty". Generality = the
  reciprocal of spread (how step-like the ACC is; Eqs. 2-5, section "Spread and (normalised) generality").
- **Difficulty sources.** "Difficulties can be derived intrinsically from the properties of the instance (e.g.,
  size, number of components, noise, distortions, etc.) or the resources that are expected to solve it" or
  extrinsically from other agents' results (section "Agent characteristic curves and capability"). "an agent can
  only be called fully general if it covers all tasks up to an equivalent level of difficulty, determined by the
  resources that are needed for them" (Introduction).
- **Random agents.** "an agent that is randomly correct in a given percentage of instances that is independent
  of difficulty would typically have" constant-ACC normalised generality, i.e. it sits on the "constant isometric"
  (section "Spread and (normalised) generality").
- **Non-populational.** IRT ability "is relative to the population used for the estimation and hence not properly
  comparable to other populations" (Introduction); the ACC metrics are per individual. "generality and capability
  can decouple at the individual level" (Abstract).

## Evidence and limitations

- 1408.6908 is a lecture-based survey; its ability notion is programmatic. The 2021 paper's metrics are
  non-parametric and well defined but depend entirely on the difficulty function, as the authors stress ("The
  choice of the difficulty function now plays a prominent role", Abstract).
- The equations in the PMC HTML lost their symbols in my text extraction; I quote the verbal definitions only.

## Connections and questions

- **Definition offered:** an ability is a construct over a class of tasks; quantitatively (2021), capability = area
  under the agent characteristic curve, i.e. expected accomplishment integrated over an (ideally intrinsic,
  resource-based) difficulty scale, in units of difficulty; generality = how step-like that curve is.
- **New vs better access:** not addressed. The ACC gives a way to state it: an intervention that raises the ACC
  where it was already above floor (more reliable on difficulties the agent already covered) vs one that extends
  the covered difficulty range (the step moves right). Only the second increases "capability" in the sense of
  tasks newly covered; the first increases generality/reliability.
- **Null / floor:** accomplishment is a per-attempt proportion, so k never enters; a random agent has a flat,
  low ACC (constant isometric). Dan's objection only bites if accomplishment is redefined as pass@k with k → ∞,
  which flattens every ACC to 1.
- **Transfer to our setting:** (1) Intrinsic, resource-based difficulty is natural for proofs: h(T) = surprisal of
  the shortest reference proof under the random-init checkpoint, −log₂ p_init(proof) (teacher-forced, free), or the
  search effort of a fixed reference prover; length/term size are cruder proxies. (2) ACC of each checkpoint =
  per-attempt p̂(T) (from k = 256 counts) binned by h; capability = area under it, in bits of difficulty. On the
  surprisal axis the random-init model's own ACC is ψ(h) ≥ 2^(−h) (equality when the reference proof carries most
  of its success mass), and ∫₀^∞ 2^(−h) dh = 1/ln 2 ≈ 1.44 bits — a finite, computable floor, so "random weights
  eventually solve everything" becomes "random weights have ≈ 1.4 bits of capability" (my derivation, not the
  paper's; it assumes the reference proof dominates, otherwise the floor is somewhat higher). (3) RL that only lifts the ACC below pend's step = better access on
  difficulties pend already covered; RL that moves the step right = new coverage. Cost: teacher-forced passes plus
  existing count tables (CPU minutes). Failure modes: p̂ below 1/256 is unmeasured (truncation; lower-bound with
  teacher-forced log p of known proofs); a surprisal-based h depends on the reference proof chosen (the policy may
  route around it); and capability depends on the difficulty range/binning, as the authors note.
- Related notes: burden2023triangulation.md (latent capability on the demand scale; same group),
  maier2025abilities.md (success proportion over relevant situations), harding2024capability.md.
