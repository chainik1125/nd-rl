# Recommendations for initial reading



## General capability emergence from RL

1. AlphaZero [@Silver_2018]: The original demonstration of capability emergence from zero via RL.

    - AlphaProof [@Hubert_2025] is an approach to multi-agent theorem proving. It is worth reading for inspiration, but I suspect the implementation is too complicated to be useful to use here.

2. Deepseek R1 [@guoDeepSeekR1IncentivizesReasoning2025]: The first open source demonstration of test time scaling. It is somewhat outdated now, and there is reasonable evidence for this largely being elicitation [@yueDoesReinforcementLearning2025] rather than genuine new capability learning. It is still important for understanding how RL was adapted to language models.

    - For a more state of the art version, read [@deepseekai2026deepseekv41flash], especially the post-training section.

3. Finding Longer Proofs (FLoP) [@zomboriFindingLongerProofs2021]: An attempt to answer a similar question to ours, although focused on getting longer proofs, rather than using longer proofs as a proxy for new knowledge. Worth reading for a more model independent approach to scaling proof length.

4. [@yueDoesReinforcementLearning2025]: One of the more careful measurements suggesting that RL is nothing more than elicitation. We should be sure to evaluate our claimed new capability emergence against the pass@k measures here.


## Autoresearch and working with agents

1. Automated Alignment Researcher [@wenAutomatedWeakToStrongResearcher2026] and the follow-u [@chenAutomatedResearchersReliably2026] : A fairly infra intensive approach to doing autoresearch. We may want to start lighter, but the basic design philosophy is good.

2. EDEN (https://github.com/ealt/direvo). Another approach to autoresearch which is a little more lightweight.




### Secondary reading

1. Dreamcoder [@ellis2020dreamcodergrowinggeneralizableinterpretable]. An interesting approach to new knowledge generation from the pre-LLM era.

2. Excess Description Length [@donoway2026excessdescriptionlengthlearning]: An aproach to quantifying elicitation vs. capability emergence. One direction for the project is to use the ND setting to refine and propose new measures of elicitation/emergence and this is a good starting point for the current SOTA on that.

3. Olmo RL pipelines [@allenaiOLMo3RLZero2025]: Olmo open sources the training data, model checkpoints, and training code. Their RL implementation is a decent reasonable-effort version at what happens in frontier models.



<div style="break-before: page; page-break-before:
  always;"></div>

### References
::: {#refs}
:::
