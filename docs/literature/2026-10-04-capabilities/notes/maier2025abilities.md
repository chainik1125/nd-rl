---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - maier2025abilities
---

# Abilities (Stanford Encyclopedia of Philosophy): conditional, success and modal analyses

Paper: [@maier2025abilities] (John Maier and Sophie Kikkert, "Abilities", SEP; first published 2010,
substantive revision 22 Apr 2025)
Source: https://plato.stanford.edu/entries/abilities/ (HTML, fetched 2026-10-05; read Sec. 1-4 in full; Sec. 5
on free will and disability skimmed)

## Learnings

- **Abilities vs dispositions.** Abilities are a kind of power, like dispositions, "properties of things that can
  exist even when not manifested" (Sec. 1.1); the entry demarcates abilities as powers of agents relating them to
  actions (Sec. 1.2). (For ML, Harding & Sharadin explicitly borrow agent-ability analyses for non-agents.)
- **General vs specific ability (Sec. 2.1).** A trained tennis player at the service line with racquet and ball
  "has the specific ability to serve"; the same player far from a court "has the general ability to serve"
  (Sec. 2.1, after Honoré 1964, Mele 2003).
- **Global vs local (Sec. 2.2, Kittle 2015).** Global abilities can be exercised across a wide range of
  circumstances; "Local abilities, in contrast, can be exercised in only a narrow range of circumstances"; the
  distinction is a spectrum: "general abilities may be more or less global" (Sec. 2.2).
- **Simple vs intentional ability (Sec. 2.2, Mele 2003).** "there is a sense in which agents are able to do
  whatever they in fact do. For example, an agent who rolls a six with a fair die was able (in the simple sense)
  to do so"; ability requiring control is what "Mele calls … an intentional ability" (Sec. 2.2). How much control
  is required "is contentious" and may be a gradient (Sec. 2.2).
- **Conditional analysis (Sec. 3.1).** "(CA) S has the ability to A iff S would A if S tried to A." Problems
  (Sec. 3.2): sufficiency fails because "psychological shortcomings, just as much as external impediments, may
  undermine abilities" (Lehrer's red-candy case); necessity fails for a good golfer who misses an easy putt —
  rescued by reading CA as an analysis of *specific* ability (Sec. 3.2).
- **Sophisticated versions.** Act conditional analysis (Mandelkern et al. 2017): S can A just in case "there is
  some practically available action such that if S tries to do it, she does A" (Sec. 3.3). New dispositionalism
  (Fara 2008): "S has the ability to A in circumstances C iff she has the disposition to A when, in circumstances
  C, she tries to A" (Sec. 3.4); motivated by finking and masking of dispositions (glass packed in styrofoam: "a
  case of masking", Sec. 3.4).
- **Success view (Jaster 2020), simplified (Sec. 3.4).** "S has an agentive ability to A if and only if S A's in a
  sufficiently high proportion of the relevant possible situations in which she intends to A."
- **Modal analysis (Sec. 4.1).** "(MA) S has the ability to A iff S does A at some world (or set of worlds)
  satisfying condition C". Mere possibility is necessary but "it seems implausible that this sort of possibility
  is a sufficient condition" (Sec. 4.1).
- **Kenny's objections (Sec. 4.3).** Fluky success: "A hopeless darts player may, once in a lifetime, hit the
  bull, but be unable to repeat the performance because he does not have the ability to hit the bull" (A → ◇A
  fails for ability). Disjunction: "I have the ability to pick out on request a card which is either black or
  red" without the ability to pick a red card or a black card on request. The sets-of-worlds repair (Brown 1988)
  requires that "an agent must succeed reliably enough across certain nearby worlds for their A'ing to count as
  sufficiently controlled" (Sec. 4.3).

## Evidence and limitations

- A survey; the analyses are the cited authors', not the entry's. No account gives a numerical reliability
  threshold; "sufficiently high proportion" (Jaster) and "reliably enough" (Brown) are left open. Contextualist
  versions (Lewis 1976, Kratzer) make condition C depend on the conversational context (Sec. 4.2).
- The agent/action framing (trying, intending) has no direct analogue in a sampler; Harding & Sharadin's
  "best explained by being directed at φ" is the ML stand-in.

## Connections and questions

- **Definition offered (for our use):** the most quantitative candidate is Jaster's success view — ability = a
  sufficiently high success proportion over the relevant situations in which the agent tries — plus the
  simple/intentional distinction: a success that occurs at all (a die roll) shows only simple ability.
- **New vs better access:** the general/specific and masking vocabulary is the philosophical form of
  "elicitation": a general ability that is masked (styrofoam) or lacks opportunity (no racquet) is present but
  not exercisable; removing the mask or supplying the opportunity is better access, while changing what the agent
  would reliably do when trying is a new ability. No decision rule.
- **Null / floor:** directly addresses Dan's objection. Random weights that eventually emit a proof have only
  the *simple* ability (Mele's die roll; Kenny's darts player); the modal analysis with unrestricted possibility
  is exactly pass@∞ and is rejected as insufficient (Sec. 4.1, 4.3). The repair is reliability across nearby
  situations, i.e. a success *proportion*, not success existence.
- **Transfer to our setting:** (1) Success view → per-theorem success proportion p̂ (from k = 256 or larger
  samples) over "relevant situations" = a declared set of background conditions (temperature 0.8/1.0, name
  bases, sample seeds), with ability(T) := p̂ ≥ θ. Our existing per-theorem counts give this at no cost; the
  open part is θ and the set of situations. (2) Globality index (Kittle): the fraction of declared background
  conditions under which p̂ ≥ θ; "local" abilities that appear only at one temperature or one name base would
  count as weaker. Cheap with existing x0/x1 and T 0.8/1.0 reads. (3) Masking: a single dominant wrong step
  (the project's worst-step diagnostics) is a natural "mask"; RL that only removes it would be unmasking.
  Failure mode: philosophy fixes the form (proportion over relevant situations) but not θ or the relevant
  situations; picking θ is the same arbitrariness as picking k, unless θ is tied to a cost (k ≈ 1/θ samples) or
  calibrated on controls.
- Related notes: harding2024capability.md (CAMA applies CA + reliability to ML), SEP "Dispositions" (Choi & Fara, 2018 revision; screened in `_screen_L2.md`:
  masking/finking, intrinsic dispositions), firestone2020performance.md (competence vs performance).
