# Claim ledger — reader L2 (elicitation and evaluation practice, capability vs propensity, philosophy / measurement theory of capabilities)

Written 2026-10-05 by agent:claude. Every V claim was found with `capability_defs/lit/quote.py` in the
fetched text (`~/cd_sources/<id or slug>.txt`); locations are section / equation / figure / page as read in
the rendering named in the note. Quotes are verbatim up to whitespace, case and math-markup normalisation.

| # | paper id | claim (short verbatim quote or exact number) | location | status V/UNVERIFIED |
|---|---|---|---|---|
| 1 | 2405.19550v1 | "We say an AI system exhibits a capability to perform a task if it produces high-quality responses across a set of inputs representative of that task" | Sec. 2 | V |
| 2 | 2405.19550v1 | "A model can be said to possess a capability if it exhibits that capability for some "reasonable" prompt, i.e. one that doesn't itself effectively provide the capability in question" | Sec. 2 | V |
| 3 | 2405.19550v1 | reasonable-prompt example: translator persona OK; "a prompt which includes a French -> English dictionary would not" | Sec. 2 | V |
| 4 | 2405.19550v1 | definition covers API-exploitable capabilities "but not capabilities that fine-tuning could easily elicit and that could be exploited by users with weights or fine-tuning access" | Sec. 2 | V |
| 5 | 2405.19550v1 | "it is difficult to show that a model does not possess a particular capability" | Sec. 2 | V |
| 6 | 2405.19550v1 | "we are not aware of any rigorous accounts" (of model capabilities) | Sec. 2 | V |
| 7 | 2405.19550v1 | fine-tuning elicitation does not cover "very large fine-tuning runs that teach LLMs new capabilities" | Sec. 1 | V |
| 8 | 2405.19550v1 | "it is not necessarily clear if fine-tuning is eliciting hidden capabilities or re-introducing them" | Sec. 3 | V |
| 9 | 2405.19550v1 | metric: "the proportion of problems for which the temperature 0 generation (so, greedy sampling) is correct" | Sec. 4.2 | V |
| 10 | 2405.19550v1 | "fine-tuning on a single demonstration is often enough to recover most of the "hidden" performance"; MMLU without CoT "requires a few hundred samples" | Sec. 1 | V |
| 11 | 2405.19550v1 | "fine-tuning on 512 samples isn't much better than fine-tuning on 4" (all tasks except MMLU) | Sec. 5.1, Fig. 3 | V |
| 12 | 2405.19550v1 | demonstrations "often incorrect or which the password-locked model can't imitate well" give lower performance "no matter the number of demonstrations" | Sec. 1 (bullet for §5.1) | V |
| 13 | 2405.19550v1 | "starting from a worse policy results in poor exploration, which slows down and sometimes entirely prevents learning" | Sec. 6.2 | V |
| 14 | 2405.19550v1 | fixedrdm start: "RL is unable to explore and reinforce good behaviors when starting from such weak policy" | Sec. 6.2 | V |
| 15 | 2405.19550v1 | RL budget: "50 iterations for expert iteration and filtering (each generating 640 samples and fine-tuning on the best 64 ones)" | Sec. 6.1 | V |
| 16 | 2405.19550v1 | expert iteration "is able to pick up on a few lucky examples sampled from" π_weak "and gradually bootstrap from that to" π_strong (π written as LaTeX in the HTML) | App. D.2 | V |
| 17 | 2405.19550v1 | 18 M-param transformers: "models locked from scratch require 2-4x the amount of data to unlock to the same extent as models locked via SFT" | Sec. 7, Fig. 6 | V |
| 18 | 2405.19550v1 | "even models locked from scratch do eventually get unlocked, and the number of demonstrations required is small overall" | Sec. 7 | V |
| 19 | 2405.19550v1 | "previous works find fine-tuning yields only superficial changes which can easily be undone" (citing Jain et al. 2023) | Sec. 7 | V |
| 20 | 2405.19550v1 | password-locked models "are only intended to assess whether an elicitation method reveals those capabilities which could be accessed via prompting" | Sec. 8 | V |
| 21 | 2405.19550v1 | "Hidden capabilities may be less "salient" in realistic models"; "sample efficiency may be much lower when good answers are less salient" | Sec. 8 item 3; App. D.2 | V |
| 22 | 2405.19550v1 | SFT on π_weak samples: "Surprisingly, this slightly increases correctness for 2/4 tasks" | Sec. 5.1; App. B.2 | V |
| 23 | 2405.19550v1 | epoch chosen as "the highest average test correctness among our 3 seeds" | App. G.3 | V |
| 24 | 2406.07358v4 | "We define sandbagging as strategic underperformance on an evaluation" | Sec. 1; Sec. 2 | V |
| 25 | 2406.07358v4 | "Underperformance occurs when the exhibited capability of an AI system is lower than its actual capability" | Sec. 2 | V |
| 26 | 2406.07358v4 | "actual capability to be the best performance on a certain task it can achieve, given the best currently available capability elicitation techniques such as prompting and fine-tuning" | Sec. 2 | V |
| 27 | 2406.07358v4 | quotes Anthropic (2023): actual capability = "one that can either immediately, or with additional post-training techniques corresponding to less than 1% of the total training cost", be elicited | Sec. 2 | V (as quoted by the paper) |
| 28 | 2406.07358v4 | the operationalization avoids "ambiguity whether a capability is being elicited or learned" | Sec. 2 | V |
| 29 | 2406.07358v4 | "some acceptable difference threshold between the exhibited and actual capability must be set per evaluation" | Sec. 2 | V |
| 30 | 2406.07358v4 | "Capability evaluations are about whether an AI system can do some task", "whereas alignment evaluations are about whether an AI system would do some task" | Sec. 2 | V |
| 31 | 2406.07358v4 | "We spent 1-5 hours engineering the prompt per model, and 8 hours for Llama 3 70b" | Sec. 3 | V |
| 32 | 2406.07358v4 | Emulator vs Targeter: "the same answer on 77.1% of questions on average, compared to the Targeter's 60.8%" | Sec. 5.2, Table 2 | V |
| 33 | 2406.07358v4 | "For a model that is displaying its true capabilities, we would expect it to more often be wrong on harder questions" | App. F | V |
| 34 | 2406.07358v4 | "it may be possible to detect sandbagging by inspecting a model's capability profile" | App. F, Fig. 11 | V |
| 35 | 2406.07358v4 | difficulty "by how many models get it wrong" (5-model panel) | Sec. 5.2; App. F | V |
| 36 | 2406.07358v4 | non-strategic case: "It could likely be quickly fine-tuned to perform much better" (not sandbagging: strategy condition fails) | App. A.2.1 | V |
| 37 | 2406.07358v4 | "requirements regarding training data, engineering time, access levels …, and computing resources to override sandbagging are still unclear" | Sec. 7 | V |
| 38 | 2502.02180v3 | exhibited capability ŝ = s({(c,r)}), r ~ p_θ(·∣c): "We say an LM exhibits capability" ŝ "when it achieves that score on a given benchmark" (symbols are LaTeX in the HTML) | Sec. 3.3 | V |
| 39 | 2502.02180v3 | "A latent capability" s̃ "is one that a model exhibits with low probability by default, but can be revealed through small changes to the context, activations, or weights" | Sec. 3.3 | V |
| 40 | 2502.02180v3 | small changes "include few-shot prompting or minor parameter updates requiring less than 1% of the original training cost (Anthropic, 2023)" | Sec. 3.3 | V |
| 41 | 2502.02180v3 | "previously latent capabilities have been elicited from models, sometimes long after initial release" | Abstract; Sec. 1 | V |
| 42 | 2502.02180v3 | APPS: "only fine-tuning can elicit the hidden capabilities of our novel model organism" | Abstract | V |
| 43 | 2502.02180v3 | WMDP: combining N-shot and prefilling "always elicits the capabilities" | Sec. 4.1 | V |
| 44 | 2502.02180v3 | APPS multi-turn: "the highest accuracy of 6.7% is elicited over four turns"; steering "highest score of 2%" | Sec. 4.3 | V |
| 45 | 2502.02180v3 | "Anti-refusal training elicits the full hidden code-generation capabilities of both our model-organisms" | Sec. 4.3 | V |
| 46 | 2502.02180v3 | anti-refusal data: "neither the requests nor responses need to be related to the capability that we want to elicit" | Sec. 3.5 | V |
| 47 | 2502.02180v3 | "It seems more difficult to elicit correct distributions for tasks with an increased breadth of plausible tokens at each step and an increased total number of tokens required" | Sec. 5 | V |
| 48 | 2502.02180v3 | "if possible, fine-tuning should be the method of choice to improve the trustworthiness of capability evaluations" | Abstract | V |
| 49 | 2502.02180v3 | "evaluators should at least have fine-tuning access (and use it)" | Sec. 5 | V |
| 50 | 2502.02180v3 | elicitation data: 25% of MCQA data; APPS "1000 to train elicitation techniques" | Sec. 3.1 | V |
| 51 | 2502.02180v3 | Gemma "original APPS performance of 35.8% could not be recovered with only the password" | Sec. 4.3 | V |
| 52 | 2502.02180v3 | circuit-broken organism "more robust to elicitation techniques than standard password-locked models"; "circuit-breaking was very fragile to training hyperparameters" | Abstract; Sec. 4.1 | V |
| 53 | 2502.02180v3 | "It seems likely that at least some elicitation methods lead to substantially better results with increased resources" | Sec. 5 | V |
| 54 | 2502.02180v3 | "Comparing anti-refusal training to fine-tuning on random noise might reveal why anti-refusal training works" | Sec. 6 | V |
| 55 | 2410.08827v3 | "Historically, it has been unclear whether unlearning techniques are removing information from the model weights or just making it harder to access" | Abstract | V |
| 56 | 2410.08827v3 | U "fully removes the information about q from the weights if and only if the mutual information between U(θ) and Y is 0" | Sec. 3.1 | V |
| 57 | 2410.08827v3 | "Our formalization only applies to questions that are practically impossible to guess" | Sec. 3.1 | V |
| 58 | 2410.08827v3 | "Any facts V that the attacker recovers indicate that these facts were hidden, rather than removed" | Sec. 3.2 | V |
| 59 | 2410.08827v3 | independence of T and V means "we do not need to worry about "reteaching" the model the facts, which is a concern if we perform attacks that use access to V, like relearning time" | Sec. 3.2 | V |
| 60 | 2410.08827v3 | "Training on T might reveal information that was hidden by increasing the model's propensity to output the unlearned facts without teaching the model the facts again" | Sec. 3.2 | V |
| 61 | 2410.08827v3 | "training the model on T should not increase accuracy on V for a model that was not trained on either T or V" | Sec. 3.2 | V |
| 62 | 2410.08827v3 | relearning time: "The main problem with this approach is the lack of a reliable baseline for comparison"; "we cannot determine if this number of time steps or samples implies that the information was hidden or removed" | Sec. 2 (White-box unlearning evaluations) | V |
| 63 | 2410.08827v3 | Recovery Rate = acc on V of unlearned model after RTT / acc on V of original model after RTT | Sec. 5.1 | V |
| 64 | 2410.08827v3 | "recovery rates for pretrained information were greater than 88%"; abstract: "can recover 88% of the pre-unlearning accuracy" | Sec. 5.1; Abstract | V |
| 65 | 2410.08827v3 | fine-tuned info (Random Birthdays): "we still get a high recovery rate for RMU, but a recovery rate less than" 35 % (HTML: "35\%") "for GD and RIA" | Sec. 5.2 | V |
| 66 | 2410.08827v3 | negative control (model never fine-tuned on Random Birthdays): "The highest accuracy we are able to get is 31.2%" | App. E | V |
| 67 | 2410.08827v3 | splits of 157 items; T = 628 items | Sec. 4.3 | V |
| 68 | 2410.08827v3 | "we run RTT for 6 epochs, record accuracies after each epoch and report the maximum across the 6 epochs" | App. A, Table 2 | V |
| 69 | 2410.08827v3 | high-granularity hiding stress test: "We were able to recover the forget accuracy back to 83% quickly" | Sec. 6.1 | V |
| 70 | 2410.08827v3 | "Our evaluation does not guarantee that information is removed from the weights; rather, it sets a higher bar than previous evaluation methods for unlearning" | Sec. 7.1 | V |
| 71 | 2410.08827v3 | "if the goal is to unlearn the capability of coding, it's hard to construct T and V with low leakage" | Sec. 7.1 | V |
| 72 | 2405.08989v1 | "a machine learning model has a capability to X just when it would reliably succeed at doing X if it 'tried'" | Abstract | V |
| 73 | 2405.08989v1 | UNIFORM "always produces the next token by sampling uniformly from the token vocabulary"; "very few ability claims are actually true of UNIFORM" | Sec. 3.1 | V |
| 74 | 2405.08989v1 | Reliability: "the more reliably the model φs, the stronger the evidence that it has an ability to φ" | Sec. 3.1, Def. 5 | V |
| 75 | 2405.08989v1 | "a performance failure does not entail the absence of an underlying competence" | Sec. 3.2 | V |
| 76 | 2405.08989v1 | account "should explain how a model's ability to φ can be invariant across different methods for eliciting and measuring that capability" | Sec. 3.2, Def. 6 | V |
| 77 | 2405.08989v1 | Orthodox analysis: "M is able to φ_c iff there exists some set of background conditions B in which M reliably φ_c s across queries" | Sec. 3.3, Def. 7 | V |
| 78 | 2405.08989v1 | CAMA: the conditional "if the output M produces is best explained by its being directed at φ_c ing, then M successfully φ_c s" must hold reliably in some B | Sec. 4.1, Def. 10 | V |
| 79 | 2405.08989v1 | trying test: M "is sensitive to φ_c-relevant perturbations to the input" and insensitive to φ_c-irrelevant ones | Sec. 4.2, Def. 11 | V |
| 80 | 2405.08989v1 | behavioural test "cannot be used to disambiguate between capabilities φ_c and ψ_c which are behaviourally indistinguishable" | Sec. 4.2 | V |
| 81 | 2405.08989v1 | protocol step 4: "Throw away those outputs … which are not explained by being directed at φing" | Sec. 5.1, Def. 12 | V |
| 82 | 2405.08989v1 | "we should be wary of making claims about the model's ability to φ_c in domains in which it only 'tries' to φ_c on some very narrow range of queries" | Sec. 5.1 | V |
| 83 | 2405.08989v1 | "we treat tokenizers, inference procedures, and other scaffolding for p_θ as part of the set of background conditions" | Sec. 2.3 | V |
| 84 | 2405.08989v1 | "a model individuation is principled iff it identifies a fixed-point in the conditions under which the model is evaluated" | Sec. 2.3, Def. 3 | V |
| 85 | 2405.08989v1 | "fine-tuning changes the model being evaluated"; "the difference between tuning and training proper is a matter of degree" | Sec. 5.2.2, at fn. 38 (HTML inlines the footnote; main text vs footnote not separable) | V |
| 86 | 2405.08989v1 | "if a model can be fine-tuned with very little compute (relative to e.g. its original training compute) to φ successfully, this does provide indirect evidence that it already had the ability to φ" | Sec. 5.2.2, at fn. 38 (HTML inlines the footnote; main text vs footnote not separable) | V |
| 87 | 2405.08989v1 | "the fine-tuning compute budget required to elicit a capability might be a natural proxy for the difficulty of eliciting the capability; future work could explore this" | Sec. 5.2.2, at fn. 38 (HTML inlines the footnote; main text vs footnote not separable) | V |
| 88 | 2405.08989v1 | RLHF: the conditional "is unaffected by RLHF (for most capabilities φ), the degree to which the model tries to φ changes" | Sec. 5.2.2 | V |
| 89 | 2405.08989v1 | practical availability "is given by the probability mass of the output strings which count (according to the operationalisation construct) as implementing the action" | Sec. 5.2.2, fn. 42 | V |
| 90 | 2405.08989v1 | "what's relevant is finding – for each model – the set of background conditions on which the model is most successful at φing" | Sec. 5.2.3 | V |
| 91 | sep-abilities (Maier & Kikkert 2025) | "First published Tue Jan 26, 2010; substantive revision Tue Apr 22, 2025"; copyright 2025 John Maier, Sophie Kikkert | header; footer | V |
| 92 | sep-abilities | specific vs general ability: player at the service line "has the specific ability to serve"; away from a court "has the general ability to serve" | Sec. 2.1 | V |
| 93 | sep-abilities | "Local abilities, in contrast, can be exercised in only a narrow range of circumstances"; "general abilities may be more or less global" | Sec. 2.2 | V |
| 94 | sep-abilities | "there is a sense in which agents are able to do whatever they in fact do. For example, an agent who rolls a six with a fair die was able (in the simple sense) to do so"; "Mele calls ability in this more controlled sense an intentional ability" | Sec. 2.2 | V |
| 95 | sep-abilities | "(CA) S has the ability to A iff S would A if S tried to A" | Sec. 3.1 | V |
| 96 | sep-abilities | "psychological shortcomings, just as much as external impediments, may undermine abilities" | Sec. 3.2 | V |
| 97 | sep-abilities | ACA: "there is some practically available action such that if S tries to do it, she does A" (Mandelkern et al. 2017) | Sec. 3.3 | V |
| 98 | sep-abilities | Fara: "S has the ability to A in circumstances C iff she has the disposition to A when, in circumstances C, she tries to A"; styrofoam glass "a case of masking" | Sec. 3.4 | V |
| 99 | sep-abilities | Jaster success view: "S A's in a sufficiently high proportion of the relevant possible situations in which she intends to A" | Sec. 3.4 | V |
| 100 | sep-abilities | "(MA) S has the ability to A iff S does A at some world (or set of worlds) satisfying condition C"; mere possibility: "it seems implausible that this sort of possibility is a sufficient condition" | Sec. 4.1 | V |
| 101 | sep-abilities | Kenny: "A hopeless darts player may, once in a lifetime, hit the bull, but be unable to repeat the performance because he does not have the ability to hit the bull" | Sec. 4.3 | V |
| 102 | sep-abilities | Kenny: "I have the ability to pick out on request a card which is either black or red" (but not red, nor black, on request) | Sec. 4.3 | V |
| 103 | sep-abilities | Brown-style repair: "an agent must succeed reliably enough across certain nearby worlds for their" A-ing (italic A in the HTML, extracted as "Aing") "to count as sufficiently controlled" | Sec. 4.3 | V |
| 104 | firestone2020-performance (PMC7604508; PNAS 117(43):26562-26571) | failure may arise "not because the system lacks the relevant knowledge or internal capacities ("competence"), but instead because of superficial constraints on demonstrating that knowledge ("performance")" | Abstract | V |
| 105 | firestone2020-performance | "Competence is a system's underlying knowledge: the internal rules and states that ultimately explain a given capacity, often in idealized terms" | "Internal Knowledge vs. External Expression" | V |
| 106 | firestone2020-performance | "intelligent creatures often know more than their behavior may indicate, because of "performance constraints"" | "Internal Knowledge vs. External Expression" | V |
| 107 | firestone2020-performance | infants "stare measurably longer at unsupported floating objects"; "infants knew about gravity all along; they just failed to display that knowledge in their natural behavior" | "When Superficial Differences Hide Deep Similarities" | V |
| 108 | firestone2020-performance | "Since accommodating performance constraints still failed to produce humanlike behavior, we might feel safer concluding that chimpanzees lack the competence for language" | "When Superficial Differences Are Deep Ones Too" | V |
| 109 | firestone2020-performance | "Allowing other minds to demonstrate their knowledge requires accommodating their performance constraints, so that their success or failure won't depend on those constraints" | "Fair Comparisons" | V |
| 110 | firestone2020-performance | same/different: "the best performing CNN model for this problem could not get significantly above chance from 1 million training examples" (quoting Kim et al.) | "A Case Study: Same vs. Different." | V |
| 111 | firestone2020-performance | "it is conceivable that some other test could reveal efficient same/different abstraction"; "the space of superficial alternative explanations has narrowed considerably" | "A Case Study: Same vs. Different." | V |
| 112 | firestone2020-performance | shape bias: "it wasn't that CNNs couldn't give shape-based classifications of images; they just didn't employ that humanlike strategy until their environment invited it" | "3. Species-Specific Task Alignment" | V |
| 113 | firestone2020-performance | "performance without competence" vs "different behaviors can arise from similar underlying processes ("competence without performance")" | "What Species-Fair Comparisons Show" | V |
| 114 | 2309.11975v2 | "we introduce measurement layouts which model how task-instance features interact with system capabilities to explain performance" | Abstract | V |
| 115 | 2309.11975v2 | cognitive profile ⟨C,B,R⟩: "Capability levels represent what the system can do, bias values represent some other preferences or limitations that may impact performance in a less monotonic way, and robustness levels account for reliability issues, unexplained or random effects (noise)" | Sec. 2 | V |
| 116 | 2309.11975v2 | σ(capability − demand) link "allows us to interpret a capability with value x as consistently succeeding on the demand with meta-feature x in" 50 % (HTML: "50\%") "of instances" | Sec. 6 | V |
| 117 | 2309.11975v2 | "our capabilities were non-compensatory, so we often opted for taking the product of probabilities" | Sec. 6 | V |
| 118 | 2309.11975v2 | "half-normal priors on these latent capabilities, reflecting the fact that there is a meaningful 0 value" | Sec. 3 | V |
| 119 | 2309.11975v2 | a "random action agent that moves stochastically in the environment" fitted among subjects | Sec. 5 | V |
| 120 | 2309.11975v2 | "we infer the cognitive profile of a single subject from their performance data alone" | App. A | V |
| 121 | 2309.11975v2 | "a correct description of the agent in around 75% of cases" (synthetic agents) | Sec. 4 | V |
| 122 | 2309.11975v2 | "unable to spot "fraudsters" making use of strategies (such as "go to the last seen location of the reward") to mimic OP capability" | Sec. 4 | V |
| 123 | 2309.11975v2 | held-out prediction: "consistently a better predictor of success (lower Brier score" than aggregates; abstract "significantly more predictive" | Sec. 4; Abstract | V |
| 124 | 2309.11975v2 | "All agents except children have low object permanence capability" | Sec. 5 | V |
| 125 | 2309.11975v2 | "If many parameters must be inferred simultaneously then many evaluation instances are needed" | Sec. 8 | V |
| 126 | 2305.15324v2 | "Developers must be able to identify dangerous capabilities (through "dangerous capability evaluations") and the propensity of models to apply their capabilities for harm (through "alignment evaluations")" | Abstract | V |
| 127 | 2305.15324v2 | "two categories: (a) whether a model has certain dangerous capabilities, and (b) whether it has the propensity to harmfully apply its capabilities (alignment)" | Sec. 1 | V |
| 128 | 2305.15324v2 | "a model should be treated as highly dangerous if it has a capability profile that would be sufficient for extreme harm, assuming misuse and/or misalignment" | Sec. 2 | V |
| 129 | 2305.15324v2 | "Researchers will need to bring latent capabilities to the surface (for example, by prompt engineering or fine-tuning)" | Sec. 4, Table 2 | V |
| 130 | 2305.15324v2 | "Capability overhang: Models sometimes have capabilities that the AI research community does not realise" (CoT example) | Sec. 5.1, item 3a | V |
| 131 | 2305.15324v2 | "the results from the end of a long development process will likely fail to convey relevant information about the base model" | Sec. 4, Table 2 | V |
| 132 | 2305.15324v2 | "Evaluations should study models both with and without these augmentations" | Sec. 4, Table 2 | V |
| 133 | 2305.15324v2 | change magnitude "in terms of the amount of additional training that it has gone through (as a percentage of the original training length), or the model's improvement on key performance benchmarks" | Sec. 3.2, fn. 4 | V |
| 134 | 2305.15324v2 | evaluations "should eventually also involve looking mechanistically at how the model produced that behaviour"; "one reason not to rely solely on behavioural evaluations" | Table 2; Sec. 5.1 | V |
| 135 | 2305.15324v2 | "evaluations would ideally provide a quantitative score, although this will not always be practical" | Table 2 | V |
| 136 | 2305.15324v2 | "Partly, agency is a question of the model's capabilities" | Sec. 4 | V |
| 137 | 1408.6908v3 | "This paper is largely superseded by the following paper: "Evaluation in artificial intelligence: from task-oriented to ability-oriented measurement"" | front matter | V |
| 138 | 1408.6908v3 | Φ(π,M,p) = Σ_μ p(μ)·E[R(π,μ)] (average-case performance); rank-based aggregation "is more robust to systems getting good scores on many easy problems but doing poorly on the difficult problems" | Sec. 2.1, Eq. 1, fn. 1 | V |
| 139 | 1408.6908v3 | "we can define a cognitive ability as a property of individuals that allows them to perform well in a range of information-processing tasks"; "the ability is necessary but it does not have to be sufficient" | Sec. 3.1 | V |
| 140 | 1408.6908v3 | "While tasks can be seen as measuring instruments, abilities are constructs" | Sec. 3.1 | V |
| 141 | 1408.6908v3 | evaluate systems not "for what they do but for what they are able to (learn to) do" | Sec. 1 | V |
| 142 | 1408.6908v3 | "Item difficulty is determined by the percentage of subjects that are able to solve the item"; "this difficulty assessment is relative to the population and not derived from the nature of the item itself" | Sec. 3.2 | V |
| 143 | 1408.6908v3 | C-test: "the difficulty of these exercises is intrinsic, and not based on how difficult humans find them" | Sec. 3.3 | V |
| 144 | 1408.6908v3 | abilities "are properties that emanate from (general) classes of tasks, perfectly defined in computational terms"; "measures are absolute and not relativised wrt. a population" | Sec. 3.4 | V |
| 145 | 1408.6908v3 | "An intrinsic difficulty function (even if approximate) is always very useful" | Sec. 4 | V |
| 146 | hernandezorallo2021generality (PMC8613222; Sci. Rep. 11:22822) | "we simply define capability as the area under the curve"; "Capability is just the area of this curve"; it "has the same units as difficulty" | section "Agent characteristic curves and capability", Eq. 1 | V |
| 147 | hernandezorallo2021generality | "we simply define generality as the reciprocal of spread" | section "Spread and (normalised) generality", Eq. 5 | V |
| 148 | hernandezorallo2021generality | "Difficulties can be derived intrinsically from the properties of the instance (e.g., size, number of components, noise, distortions, etc.) or the resources that are expected to solve it" | section "Agent characteristic curves and capability" (preceding paragraph) | V |
| 149 | hernandezorallo2021generality | "an agent that is randomly correct in a given percentage of instances that is independent of difficulty would typically have" constant-ACC normalised generality | section "Spread and (normalised) generality" | V |
| 150 | hernandezorallo2021generality | "an agent can only be called fully general if it covers all tasks up to an equivalent level of difficulty, determined by the resources that are needed for them" | Introduction | V |
| 151 | hernandezorallo2021generality | "This ability in the IRT models is relative to the population used for the estimation" | Introduction (background) | V |
| 152 | hernandezorallo2021generality | "generality and capability can decouple at the individual level"; "The choice of the difficulty function now plays a prominent role" | Abstract | V |
| 153 | Measure of All Minds (book, CUP 2017) | no claim taken from the book | — | UNVERIFIED (no text access) |
| 154 | 2403.13793v2 | "Ideally, we could estimate an upper bound on the level of harm a model could cause, even under pessimistic assumptions" | Sec. 2 | V |
| 155 | 2403.13793v2 | SFT on "researcher-generated trajectories for (benign) tasks adjacent to those tested in Section 6" initially helped, then was outperformed by a newer checkpoint | Sec. 2 | V |
| 156 | 2403.13793v2 | "By measuring how much information the agent needs, we can distinguish between agents very close to having a capability from those very far away" | Sec. 6.2 | V |
| 157 | 2403.13793v2 | ideal: "the absolute minimum amount of information sufficient for success, but this objective trades off against the efficiency and reproducibility of the evaluation process" | Sec. 6.2 | V |
| 158 | 2403.13793v2 | unguided cost ⌈−log₂ ρ⌉ bits: "approximately the number of bits required to, on average, provide a random seed that would allow the agent to succeed in a single attempt" | Sec. 6.2, step 1 | V |
| 159 | 2403.13793v2 | best-of-N with N = 16; "We charge the agent log₂[i(i+1)] bits of expert help per step, where i is the index of the selected action"; "we are simulating the agent getting "lucky"" | Sec. 6.2, step 2, fn. 12 | V |
| 160 | 2403.13793v2 | golden solution: "the negative log probability the agent assigns to the solution is equivalent to the number of nats it would take to communicate the golden solution to the agent"; "we can always give the agent a score (in bits), even if it is far from success" | Sec. 6.2, step 3 | V |
| 161 | 2403.13793v2 | "Pro 1.0 and Ultra 1.0 required only 10 and 9 bits to complete the Bitcoin Wallet task and 126 and 55 bits to complete the Email Setup task" | Sec. 6.3, Fig. 9 caption | V |
| 162 | 2403.13793v2 | "the agents appear to know good actions but rate poorer alternatives more highly" | Sec. 6.4.2 | V |
| 163 | 2403.13793v2 | milestone necessity: "it is still possible that a sufficiently capable agent could find a shortcut"; bound uses "upper bound of a one-sided 97.5% CI" | Sec. 6.2 fn. 11; App. E.4 fn. 18 | V |
| 164 | 2403.13793v2 | "We hypothesise that it is possible to find a (non-linear) correlation between expert bits on a task and easy-to-obtain measures of the model's general performance" | Sec. 6.5 | V |
| 165 | 2403.13793v2 | prototype y-axis "normalised by the number of bits required to compress the golden solutions using gzip" | App. E.5, Fig. 20 | V |
| 166 | 2403.13793v2 | future work: "Better capability elicitation methodology – in particular, working out how best to fine-tune models to elicit their capabilities" | Sec. 9 | V |
| 167 | anthropic2023rsp (Anthropic RSP v1.0, Sept 2023, PDF) | "We define an ASL-3 model as one that can either immediately, or with additional post-training techniques corresponding to less than 1% of the total training cost, do at least one of the following two things" (so the actual-capability wording in claim 27 is van der Weij et al.'s paraphrase) | p. 6 | V |
| 168 | anthropic2023rsp | "By post-training techniques we mean the best capabilities elicitation techniques we are aware of at the time, including but not limited to fine-tuning, scaffolding, tool use, and prompt engineering" | p. 6 | V |
| 169 | anthropic2023rsp | RLHF/constitutional safeguards "can almost certainly be fine-tuned away within the specified 1% of training cost"; "ASL-3 is intended to characterize the model's underlying knowledge and abilities" | p. 7 | V |
| 170 | anthropic2023rsp | "it is not currently possible to truly upper-bound the capabilities of generative models" | p. 12 | V |
| 171 | anthropic2023rsp | "We count a task as "passed" if the model succeeds at least once out of 10 tries, since we expect that a model passing a task 10% of the time can likely be easily improved to achieve a much higher success rate" | p. 15 | V |
| 172 | anthropic2023rsp | elicitation: model "should be trained to be competent at general computer use, including training on tasks in the same vein as but not identical to these specific tasks" | p. 16 | V |
| 173 | anthropic2023rsp | "We have aimed to set the size of our safety buffer to 6x (larger than our 4x evaluation interval)" | p. 11 | V |
| 174 | donoway2025elicitation-abs (NeurIPS 2025 poster page, OpenReview Dkgx2pS4Ww; PDF not accessible) | "we recast elicitation as an information-constrained fine-tuning problem and empirically characterize upper bounds on the minimal number of parameters needed to achieve specific task performances" | Abstract | V |
| 175 | donoway2025elicitation-abs | "training as few as 10–100 randomly chosen parameters" can recover "up to 50" % "of the performance gap between pretrained-only and full fine-tuned models, and 1,000s to 10,000s of parameters can recover 95" % "of this performance gap" (page shows "50\%", "95\%") | Abstract | V |
| 176 | donoway2025elicitation-abs | "a logistic curve fits the relationship between the number of trained parameters and model performance gap recovery"; "offering a potential route to distinguish elicitation from teaching" | Abstract | V |
| 177 | 2605.08368v1 | accessible support = "the set of behaviors that a model can practically produce under finite budgets"; reweighting within it = elicitation, changing it = creation | Abstract | V |
| 178 | 2605.08368v1 | "strict mathematical support is too weak for capability analysis"; regions "are relative to a base model, a decoding policy, an evaluation signal, and a compute budget"; "This is not meant to be a new formal object with a unique estimator" | Sec. 4.1 | V |
| 179 | 2605.08368v1 | tail reweighting: "The key evidence for this regime is whether the base model can already produce the target behavior under a larger sampling budget"; fair comparison is "SFT on reward-selected, verifier-selected, or search-generated samples" | Sec. 4.3 | V |
| 180 | 2402.16835v1 | "higher-than-baseline amounts of knowledge can reliably be extracted" from the WHP unlearned model | Abstract | V |
| 181 | 2402.16835v1 | few-shot fine-tuning: "a small amount of fine-tuning data causes a disproportionately large amount of knowledge to resurface"; "800 sentences and 8,000 sentences representing about 1% and 10% of the complete Harry Potter book corpus" | Sec. 3, method 4 ("Relearning through Fine-tuning") | V |
| 182 | 2402.16835v1 | "Harry Potter knowledge can be probed for internally even if the model answers incorrectly" | App. A.3 | V |
| 183 | 2412.01784v3 | "non-sandbagging models show predictable performance degradation with increasing noise", sandbagging models "exhibit anomalous performance improvements"; noise injection elicits "the full performance of Mistral Large 120B" where it underperforms uninstructed | Abstract | V |
| 184 | 2403.10462v2 | Inability arguments: "AI systems are incapable of causing unacceptable outcomes in any realistic setting." | Sec. 2.2 | V |
| 185 | 2403.10462v2 | "To rule out hidden capabilities, developers can check whether AI systems can be quickly fine-tuned to perform proxy tasks"; "the gold standard fine-tuning strategy for making inability arguments is supervised fine-tuning" | Sec. 5.2.1; Sec. 2.2 fn. 1 | V |
| 186 | 2403.10462v2 | RL "less reliable for estimating capabilities because they don't directly incentivize intelligent exploration" | Sec. 5.2.1, fn. 8 | V |
| 187 | 2403.10462v2 | SFT may fail because systems "might be able to accomplish the proxy task via strategies that differ from those represented in supervised fine-tuning data"; "The assumptions behind using supervised fine-tuning to elicit capabilities are currently not well understood" | Sec. 5.2.1 (subclaim 3) | V |
| 188 | metr2024elicitation (metr.org blog 2024-03-15) | aims "to measure what capabilities might be reachable with moderate amounts of elicitation effort without requiring the evaluator to actually perform all of that elicitation upfront"; recommendations are "somewhat informed and reasoned guesses only" | 1. Overview | V |
| 189 | metr2024elicitation | "Spurious" bottlenecks: "failures that are easily fixable and not "real" capability limitations"; "Real" bottlenecks: "failures where there's no "obvious fix"" | 2.2 Handle remaining failures | V |
| 190 | metr2024elicitation | spurious knowledge-based failures = "mistakes that could be fixed by always including some text in the prompt explaining these facts"; spurious disposition failures "Detectable + correctable by humans who … don't have particular domain expertise or any special knowledge of the task" | 4.1 Spurious failures | V |
| 191 | sep-dispositions (Choi & Fara; 2018 revision) | SCA: "An object is disposed to \(M\) when \(C\) iff it would \(M\) if it were the case that \(C\)"; mimicker: x "mimics the manifestation of disposition \(D\) although it does not possess \(D\)" (math delimiters as in the HTML) | Sec. 1.2 | V |
| 192 | sep-dispositions | PROP (Manley & Wasserman): "is disposed to \(M\) in \(C\) iff some suitable proportion of \(C\)-cases are such that \(x\) would \(M\) in them"; "The suitable proportion of \(C\)-cases is fixed partly by the stimulus condition \(C\) and partly by the context of ascription" | Sec. 1.4 | V |
| 193 | sep-dispositions | "it is extremely difficult to make sense of talk of proportions of \(C\)-cases" | Sec. 1.4 | V |
| 194 | 1912.05511v3 | constructs "cannot be measured directly and must instead be inferred from measurements of observable properties"; fairness debates "appear to be about different operationalizations" but "are, in fact, debates about different theoretical understandings" | Abstract | V |
| 195 | 2502.00561v2 | "our position is that evaluating GenAI systems is a social science measurement challenge"; "a four-level framework … for measuring concepts related to the capabilities, behaviors, and impacts of GenAI systems" | Abstract | V |
| 196 | 2502.00561v2 | levels (background concept, systematized concept, instruments, measurements) "linked by four processes: systematization, operationalization, application, and interrogation"; without separate systematization "it is hard to know precisely what is being measured" | Sec. 3; Sec. 3.1 | V |
| 197 | 2503.05336v3 | "Evaluation metrics must be applicable to real-world performance, metrics must be iteratively refined, and evaluation institutions and norms must be established"; "Commonly used static benchmarks face validity challenges" | Abstract | V |
| 198 | 2401.03910v1 | "ongoing disagreements about the extent to which we can meaningfully ascribe any kind of linguistic or cognitive competence to language models" | Abstract | V |
| 199 | 2401.03910v1 | Redescription Fallacy: arguing a system "cannot model a particular cognitive capacity, simply because its operations can be explained in less abstract and more deflationary terms"; debates "cannot be settled a priori by considering general characteristics of untrained models" | Sec. 3 | V |
| 200 | 2301.06627v3 | "formal linguistic competence -- knowledge of linguistic rules and patterns -- and functional linguistic competence -- understanding and using language in the world"; functional performance "often requires specialized fine-tuning and/or coupling with external modules" | Abstract | V |
| 201 | 2311.12786v2 | "fine-tuning rarely alters the underlying model capabilities"; a "'wrapper', is typically learned on top"; revival: "the model begins reusing these capabilities after only a few gradient steps" (HTML full text; the arXiv abstract page reads "these capability") | Abstract | V |
| 202 | 2311.12786v2 | Def. 1: M "possesses a capability" C if for all x in the sub-domain some layer's linear readout equals f_C(x_d); readout used because "the notion of a capability need not correspond to only input-output behavior" | Sec. 3, Def. 1 | V |
| 203 | 2311.12786v2 | TinyStories: models "relearn to generate stories with" twist (LaTeX \mathtt in the HTML) "more sample-efficiently than the control model pre-trained on data w/o twists"; Tracr/PCFG: reFT "in substantially fewer iterations than the baseline" (Scr.+FT) | Sec. 5.2, Table 1; App. E.3 | V |
| 204 | 2406.19370v4 | hidden capabilities: "latent interventions show the model possesses the capability to manipulate a concept, but these capabilities cannot yet be elicited via naive input prompting"; they "emerge suddenly and consistently during training" | Abstract | V |
| 205 | 2305.11206v1 | Superficial Alignment Hypothesis: "A model's knowledge and capabilities are learnt almost entirely during pretraining, while alignment teaches it which subdistribution of formats should be used when interacting with users" | Sec. 2 | V |
| 206 | 2305.11206v1 | "fine-tuned with the standard supervised loss on only 1,000 carefully curated prompts and responses"; "almost all knowledge in large language models is learned during pretraining" | Abstract | V |
| 207 | 2409.18025v6 | "finetuning on 10 unrelated examples or removing specific directions in the activation space can recover most hazardous capabilities for models edited with RMU" | Abstract | V |
| 208 | 2312.09390v1 | "PGR measures the fraction of the performance gap (the difference in performance between the weak and strong ceiling models) that we can recover with weak supervision" | Sec. 3 | V |
| 209 | 2312.09390v1 | "we do not need the weak supervisor to teach the strong model new capabilities; instead, we simply need the weak supervisor to elicit what the strong model already knows" | Sec. 1 | V |
