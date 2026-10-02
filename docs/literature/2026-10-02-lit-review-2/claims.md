# Claim ledgers (lit-review-2, 2026-10-02)

Sources: `~/lr_sources/<id>.txt` (`.abs.txt` = abstract only), fetched with `lit_review_2/fetch.py`; reprint any quote with `python3 lit_review_2/quote.py <id> '<regex>'`. Sections (b) of the three screening agents' files, merged unchanged. V = found verbatim at the stated location.

## Q1


| claim | id+version | location | verbatim snippet (≤ 25 words) | V / UNVERIFIED |
|---|---|---|---|---|
| Base surpasses RL at large k | 2504.13837v5 | Sec. 1 | "base models consistently surpass RLVR models across all benchmarks and LLM families as $k$ increases" | V |
| pass@256 falls as RL proceeds | 2504.13837v5 | Fig. 1 caption, Table 4 | "the average performance (i.e., pass@1) improves, but the coverage of solvable problems (i.e., pass@256) decreases" | V |
| Base-✗/RL-✓ fraction is tiny | 2504.13837v5 | Table 2 | AIME24 (k=1024) ✗✓ "0.0%"; MATH500 (k=128) "1.0%" | V |
| RL outputs lie in the base's high-likelihood region | 2504.13837v5 | Sec. 4.1, Fig. 6 | "the distribution of PPL_Base(Y_RL\|x) closely matches the lower portion of the PPL_Base(Y_Base\|x) distribution" | V |
| Fig. 6 rests on two problems | 2504.13837v5 | Fig. 6 caption | "We randomly sample two problems from AIME24 … to generate 16 responses for each problem" | V |
| Distillation expands the boundary | 2504.13837v5 | Sec. 4.2 header | "4.2 Distillation Expands the Reasoning Boundary" | V |
| ProRL: RL solves where the base fails entirely | 2505.24864v1 | abstract | "including scenarios where base models fail entirely regardless of the number of attempts" | V |
| ProRL: >2k steps, KL + ref reset | 2505.24864v1 | Sec. 1, 2.3.1 | "continued performance improvements after an unprecedented 2k training steps" | V |
| Gains largest where base is weakest | 2505.24864v1 | Sec. 4, Fig. 3 | "a significant negative correlation between the base model's reasoning boundary and the extent of reasoning improvement after RL training" | V |
| ProRL base had formatting failures | 2505.24864v1 | Sec. 4 / App. | "despite the base model struggles with formatting"; "formatting is relatively easy to learn" | V |
| EDL definition | 2601.04728v1 | abstract | "measures the gap between the bits required to encode training labels sequentially using an evolving model … and the residual encoding cost" | V |
| Format learning is a transient distinct from capability | 2601.04728v1 | abstract, Prop. 5.5 | "format learning creates early transients distinct from capability acquisition" | V |
| Elicit vs teach EDL signatures | 2601.04728v1 | Sec. 7.1.1 | "Elicitation shows monotonically decreasing EDL/token with dataset size; teaching shows an initial increasing phase" | V (as stated; empirics in companion paper UNVERIFIED) |
| Pre-teaching converts teaching to elicitation | 2601.04728v1 | Sec. 7.1.1 | "reducing information thresholds by ~10–100x" | V (stated) / UNVERIFIED (empirics) |
| EDL is SFT-only | 2601.04728v1 | Sec. 7.2 | "Extensions to reinforcement learning, preference optimization, and other post-training methods require additional development." | V |
| Coverage necessary and sufficient for BoN | 2510.15020v2 | Sec. 1 | "We prove that a good coverage profile is necessary and sufficient for Best-of-N to succeed" | V |
| CE can anti-correlate with BoN | 2510.15020v2 | Sec. 1, Fig. 1 | "cross-entropy can be anti-correlated with BoN performance" | V |
| Coverage avoids sequence-length dependence | 2510.15020v2 | abstract | "coverage generalizes faster than cross-entropy, avoiding spurious dependence on problem-dependent parameters such as the sequence length" | V |
| Coverage for GRPO is only conjectured | 2510.15020v2 | Sec. 2 | "some form of coverage is thought to be necessary for the success of post-training methods like GRPO" | V |
| Coverage lower-bounds RL runtime | 2503.07453v2 | abstract item 1 | "coverage, while not necessary for data efficiency, lower bounds the runtime of any algorithm in our framework" | V |
| Multi-turn exploration needs only token-level coverage | 2503.07453v2 | abstract item 4 | "replacing sequence-level coverage with token-level coverage) through multi-turn exploration" | V |
| CoT-Pass@K shows RLVR extends the boundary | 2506.14245v2 | abstract | "RLVR can extend the reasoning boundary for both mathematical and coding tasks" | V |
| Self-improvement cannot create information | 2412.01951v2 | abstract | "It is impossible for this self-improvement to create information that is not already in the model" | V |
| Priming behaviours enables RL gains | 2503.01307v2 | abstract | "priming Llama with examples containing these reasoning behaviors enables substantial improvements during RL" | V |
| Large-k pass@k is misleading on discrete answers | 2510.08325v2 | abstract | "Pass@k at large k reflects the increasingly higher chance of success in the limit of the number of trials rather than genuine reasoning" | V |
| Spurious rewards nearly match true rewards on Qwen-Math | 2506.10947v2 | abstract | "improves MATH-500 performance for Qwen2.5-Math-7B by 21.4 percentage points using randomly assigned rewards, nearly matching the 29.1-point gain" | V |
| Spurious-reward effect is model-dependent | 2506.10947v2 | abstract | "the presence of such amplifiable behaviors is highly model-dependent" | V |
| Clean generator: only accurate reward surpasses base | 2507.10532v3 | abstract | "only accurate reward signals yield steady improvements that surpass the base model's performance boundary in mathematical reasoning" | V |
| BroRL: rollout width revives saturated ProRL | 2510.01180v1 | abstract | "BroRL revives models saturated after 3K ProRL training steps and demonstrates robust, continuous improvement" | V |
| Two-stage view: early shrinkage, later expansion | 2510.04028v1 | abstract | "over-exploitation during the exploitation stage may lead to capability boundary shrinkage, whereas prolonged training into the exploration stage can promote an expansion" | V |
| Curriculum RL raises pass@256 | 2606.22317v1 | abstract | "average pass@256 improves by 9.8 percentage points over the base models and by 10.3 percentage points over Vanilla RL" | V |


## Q2


| claim | id+version | location | verbatim snippet (≤ 25 words) | V / UNVERIFIED |
|---|---|---|---|---|
| Go-Explore diagnoses detachment and derailment | 2004.12919v6 | Abstract | "forgetting how to reach previously visited states (“detachment”) and from failing to first return to a state before exploring from it (“derailment”)" | V |
| Explore step: random actions or policy sampling | 2004.12919v6 | Fig. 1 caption | "(c) Explore from that state by taking random actions or sampling from a policy." | V |
| Prior methods mix random actions throughout an episode | 2004.12919v6 | Introduction | "mix in exploration throughout an episode, usually by adding random actions a fraction of the time" | V |
| Montezuma solved; scores after robustification | 2004.12919v6 | main text, domain-knowledge results | "completely solving Montezuma’s Revenge by reaching the end of level 3. After robustification, the resulting policies achieve a mean score of 102,571 on Pitfall" | V |
| Robustification = backward algorithm (LfD) | 2004.12919v6 | main text, Atari section | "the robustification phase consists of a modified version of the “backward algorithm”" | V |
| AZ novelty = information absent from human games | 2310.16410v1 | §4 overview | "novelty (whether it contains some information that is not present in human games)" | V |
| Grandmaster Phase-1→3 scores 0→42, 33→58, 25→42, 38→44 | 2310.16410v1 | §6.1 Table 4 | "1 | 0 | 42 | +42 | 36 | 2 | 33 | 58 | +25 | 36 | 3 | 25 | 42 | +16" | V |
| Phase 3 uses unseen puzzles | 2310.16410v1 | §6 protocol | "Grandmasters are tasked with providing solutions for a test set of unseen puzzles sampled from the same concepts" | V |
| Many human concepts are predictable from AZ activations | 2111.09259v3 | Introduction | "many human concepts can be predicted from AlphaZero’s activations after training" | V |
| AZ independently discovered common human openings | 1712.01815v1 | main text (human openings analysis) | "Each of these openings is independently discovered and played frequently by AlphaZero during self-play training." | V |
| AZ root Dirichlet noise, α 0.3/0.15/0.03 | 1712.01815v1 | Methods | "Dirichlet noise Dir(α) was added to the prior probabilities in the root node" | V |
| AZ starts from random play with rules only | 1712.01815v1 | Abstract | "Starting from random play, and given no domain knowledge except the game rules" | V |
| No Dirichlet ablation in AZ paper | 1712.01815v1 | full text | (grep for "ablat", "without noise": no hits) | V (absence) |
| MuZero reuses AZ Dirichlet noise for board games | 1911.08265v2 | Appendix (hyperparameters) | "we use the same UCB constants, dirichlet exploration noise and the same 800 simulations per search as in AlphaZero" | V |
| RepExp question: discovery vs sharpening | 2510.11686v2 | Abstract | "unclear if current RL techniques promote the discovery of novel behaviors, or simply sharpen those already present in the base model" | V |
| RepExp >50% verifier efficiency, 14B | 2510.11686v2 | §1, RF1 | "we obtain over 50% improvement in verifier efficiency on almost all tasks" | V |
| RepExp hurts weak models | 2510.11686v2 | §4.1 RF2 | "weaker models (e.g., Qwen-2.5-0.5B) experience no benefit or even degradation" | V |
| GRPO degrades large-k; RepExp preserves/improves | 2510.11686v2 | RF6, Fig. 2 | "policies fit using RepExp preserve or improve pass@k for large values of k with limited reductions for small k" | V |
| RepExp pass@80 = GRPO pass@256 (AIME24) | 2510.11686v2 | Abstract | "pass@80 matches the pass@256 of GRPO on the same model" | V |
| RepExp responses less likely under base | 2510.11686v2 | RF7, Fig. 9 | "responses from RepExp tend to be less likely under the base model" | V |
| Zero-RL prover stalls without new proofs | 1905.10501v3 | §4.1 | "With no new training data, the learning process would stall." | V |
| Exploration = interleaving learnt and tf-idf premise lists | 1905.10501v3 | text at Fig. 3b | "The final set of premises is obtained by interleaving the two premise lists." | V |
| Zero reference 7.0% vs Zero explore 56.3% (final) | 1905.10501v3 | Fig. 4 | "Zero reference | 7.0% | 7.3% | Zero explore | 56.3% | 64.2%" | V |
| One-off seeding < exploring throughout | 1905.10501v3 | §6, Fig. 5a | "it does not reach the same level of performance as the Zero Explore, which explores throughout" | V |
| Heuristic-only premises 43% vs mixed 64% | 1905.10501v3 | §6 | "it manages to prove 43% of the statements cumulatively, compared with 64%" | V |
| Hard problems give zero signal on-policy | 2601.18779v1 | Abstract | "On hard problems, on-policy RL rarely explores even a single correct rollout, yielding zero reward and no learning signal" | V |
| Entropy bonus / clipping / pass@k fail on hard problems | 2601.18779v1 | Abstract | "such as entropy bonuses, more permissive clipping of the importance ratio, or direct optimization of pass@k objectives, do not resolve this issue" | V |
| Easy+hard mixing hurts (ray interference) | 2601.18779v1 | Abstract | "mixing easy and hard problems during RL training is counterproductive due to ray interference" | V |
| POPE: oracle prefixes, transfer to unguided | 2601.18779v1 | Abstract | "POPE augments hard problems with prefixes of oracle solutions"; "the resulting behaviors transfer back to the original, unguided problems" | V |
| pass@k optimisation needs non-zero pass@1 | 2601.18779v1 | §(pass@k analysis) | "pass@k optimization can improve solvability only if the model achieves non-zero pass@1 on each problem" | V |
| SvS +18.3 / +22.8 Pass@32 AIME24/25 | 2508.14029v4 | Abstract | "absolute gains of 18.3% and 22.8% in Pass@32 performance on the competition-level AIME 24 and AIME 25" | V |
| SvS beats initial model at large k on MATH-500 | 2508.14029v4 | §5.2, Fig. 6 | "SvS consistently outperforms both RLVR and the initial model as k increases" | V |
| SvS augments only problems at 12.5–50% acc | 2508.14029v4 | Appendix (hyperparameters) | "The underperforming problem range [acc_l, acc_h] is set to 12.5%–50.0%" | V |
| AlphaEvolve beats SOTA on ~20% of >50 problems | 2506.13131v1 | §1 / §3 | "On ∼20% of the problems, AlphaEvolve surpasses the SOTA and discovers new, provably better constructions." | V |
| AlphaEvolve 4×4 complex matmul in 48 mults | 2506.13131v1 | Abstract/§1 | "multiply two 4×4 complex-valued matrices using 48 scalar multiplications" | V |
| AlphaEvolve improves with stronger LLM | 2506.13131v1 | §2 / §4 | "AlphaEvolve performs increasingly better as the underlying LLM improves" | V |
| Pseudo-counts: 15 of 24 rooms of Montezuma | 1606.01868v2 | §5 / Fig. 3 | "within 50 million frames our agent learns a policy which consistently navigates through 15 rooms" | V |
| RND: first > human avg on Montezuma w/o demos | 1810.12894v1 | Abstract | "first method that achieves better than average human performance on this game without using demonstrations" | V |
| Agent57 > human benchmark on all 57 games | 2003.13350v1 | Abstract | "the first deep RL agent that outperforms the standard human benchmark on all 57 Atari games" | V |
| ICM curiosity ignores uncontrollable aspects | 1705.05363v1 | Abstract | "ignores the aspects of the environment that cannot affect the agent" | V |
| CDE ≈ +3 points AIME over GRPO/PPO | 2509.09675v1 | Abstract | "approximate +3 point improvement over standard RLVR using GRPO/PPO on AIME benchmarks" | V |
| IGE uses FM notions of interestingness | 2405.15143v4 | Abstract | "internalized human notions of interestingness captured by giant pretrained foundation models" | V |
| TacticZero: replay buffer of past successful proofs | 2102.09756v2 | training setup section | "we maintain a replay buffer of earlier successful proofs of each theorem" | V |
| TacticZero outperforms Hol4 hammers on unseen problems | 2102.09756v2 | Abstract | "outperforms existing automated theorem provers (i.e. hammers) available in Hol4 when evaluated on unseen problems" | V |
| ε-random actions used for LLM policy RL (any source) | — | WebSearch, 2 queries | no source found | UNVERIFIED (none found) |


## Q3


| claim | id+version | location | verbatim snippet (≤ 25 words) | V / UNVERIFIED |
|---|---|---|---|---|
| Rewrite organism: pretraining excludes the shortcuts later analysed | 2607.07646v1 | Sec 3.1 | "exposes the model to local grammar-consistent rewrite dynamics, but not to the later non-primitive shortcuts" | V |
| Base model 0% on hardest buckets even at pass@1024; RL solves at pass@16 | 2607.07646v1 | Sec 6 / Discussion | "even at pass@1024, the pretrained policy gets $0\%$ on Buckets 4–5, while RL later solves them at pass@16" | V |
| RFT plateaus while GRPO keeps improving | 2607.07646v1 | Sec 6 | "RFT then plateaus, while RL improves more slowly at first but continues climbing after RFT saturates" | V |
| Macro contractions overtake primitives ~iteration 12,500 | 2607.07646v1 | Sec 4, Fig 2 | "Macro contractions accelerate and overtake primitive contractions around iteration 12,500" | V |
| Difference RL vs RFT is selectivity | 2607.07646v1 | Abstract | "the key difference is not exploration volume but selectivity" | V |
| Pretraining chaining gates emergence | 2607.07646v1 | Sec 7 | "Low- $\rho$ pretraining never reliably enters the macro or parallel regime" | V |
| Model size 12 layers, hidden 512; 4xH100 | 2607.07646v1 | App A | "The pretrained model uses 12 layers, hidden dimension 512, 8 attention heads" | V |
| NTP alone fails on parity even at ~1e7 samples | 2510.11495v2 | Sec 3.1 | "does not result in a generalizing model (and the median greedy response remains short) even with access to $\sim$ $10^{7}$ training samples" | V |
| Pretrained model generates long responses at rate p_cot | 2510.11495v2 | Sec 3.2 | "the models are calibrated with respect to length, i.e., they generate long responses with probability $p_{\mathrm{cot}}$" | V |
| STaR amplification recursion converges exponentially | 2510.11495v2 | Sec 3.2 | "$p_{n}=\frac{2p_{n-1}}{1+p_{n-1}}$ converges to $1$ exponentially fast" | V |
| Efficient if long-demo share not exponentially small | 2510.11495v2 | Abstract | "as long as the proportion of long demonstrations in the data mix is not exponentially small in the input dimension $d$" | V |
| Starting RL too early fails | 2510.11495v2 | Sec 3.1 | "if post-training starts too early, then it might not lead to a generalizing model despite the length increase" | V |
| Tsilivis tests p_cot = 0 (no long demos) | 2510.11495v2 | — | lowest found: "For $p_{\mathrm{cot}}$ =0.01, we observe length increase for almost all checkpoints" | UNVERIFIED (no p_cot=0 run found) |
| Wang: 1-layer single-head transformer, d=120 | 2509.22613v2 | Sec 2.1 | "we use a one-layer, single-head Transformer as the backbone model. The embedding size is set to $d=120$" | V |
| SFT cannot exploit transitivity absent from data | 2509.22613v2 | Sec 3.2 | "SFT will fail to exploit transitivity information (which never appears in $\mathcal{D}^{\text{SFT}}$ )" | V |
| PG discovers paths absent from SFT data | 2509.22613v2 | Sec 4.1 | "it can explore and discover new correct paths that were absent from the initial training set" | V |
| Continual SFT degrades, PG improves test accuracy | 2509.22613v2 | Sec 4.2 | "the test accuracy of Continual SFT constantly decreases, while all the PG methods can achieve an improvement" | V |
| PG without KL collapses to one path | 2509.22613v2 | Sec 4.2 | "In the end, the model eventually produces only one path per pair." | V |
| Outcome-only Q-learning biased; process reward fixes | 2509.22613v2 | Takeaway 5 | "relying solely on the outcome reward signal can cause Q-value bias, whereas introducing process rewards mitigates" | V |
| Outcome RL CoT emergence needs simple examples | 2601.15158v4 | Abstract | "when this mass vanishes, policy gradient learning becomes infeasible" | V |
| Coverage generalises faster than CE | 2510.15020v2 | Abstract | "coverage generalizes faster than cross-entropy" | V |
| RL synthesises only with atomic skills mastered | 2512.01970v3 | Abstract | "it synthesizes new composite strategies only when the base model has first mastered the independent atomic skills via SFT" | V |
| Atomic-skills study uses pretrained Qwen, not from scratch | 2512.01970v3 | Sec 4.2 | "We adopt a standard SFT-then-RL pipeline using Qwen-2.5-1.5B" | V |
| RLVR grokking-type plateaus with difficulty gaps | 2602.14872v3 | Abstract | "training undergoes grokking-type phase transitions with prolonged plateaus before progress recurs" | V |
| ChessFormer 1000/1300 transcend; 1500 does not | 2406.11741v4 | Sec 4.1 | "ChessFormer 1500 is unable to transcend at test time" | V |
| Transcendence via low-temperature sampling | 2406.11741v4 | Abstract | "transcendence can be enabled by low-temperature sampling, and rigorously assess this claim experimentally" | V |
| SFT needed before RL for format | 2501.17161v2 | Abstract | "SFT remains essential for effective RL training; SFT stabilizes the model's output format" | V |
| CFGs require DP to parse | 2305.13673v4 | Abstract | "locally ambiguous and require dynamic programming to parse" | V |
| Othello-GPT world model | 2210.13382v5 | Abstract | "an emergent nonlinear internal representation of the board state" | V |
| Grokking phases | 2301.05217v3 | Abstract | "memorization, circuit formation, and cleanup" | V |
| Teacher forcing can fail to learn the predictor | 2403.06963v3 | Abstract | "teacher-forcing can simply fail to learn an accurate next-token predictor in the first place" | V |
| NTP learns incomplete reachability | 2405.09220v3 | Abstract | "learn the adjacency and an incomplete reachability matrices" | V |
| Grokking past overfitting | 2201.02177v1 | Abstract | "can happen well past the point of overfitting" | V |
| Physics of LMs has an RL part (2025) | — | — | Part 4.1 (2512.17351, seen in search only) mentions RL confounds; not fetched | UNVERIFIED |


## Executor's independent re-check (33 claims, all V)

Re-run with `quote.py` by the executor after the agents finished; each regex below matched at the stated place.

| id | regex | ledger claim |
|---|---|---|
| 2504.13837 | `consistently surpass RLVR models` | base surpasses RL at large k |
| 2504.13837 | `randomly sample two problems` | Fig. 6 rests on two problems |
| 2504.13837 | `closely matches the lower portion` | RL outputs in base high-likelihood region |
| 2504.13837 | Table 2 row (✗ ✓ 0.0 % / 1.0 %) | base-✗ RL-✓ share |
| 2505.24864 | `fail entirely regardless` | ProRL abstract claim |
| 2505.24864 | `significant negative correlation` | gains largest where base weakest |
| 2505.24864 | `struggles with formatting` | format confound |
| 2601.04728 | `Elicitation shows monotonically` | EDL signatures (companion paper) |
| 2601.04728 | `require additional development` | EDL is SFT-only |
| 2510.15020 | `necessary and sufficient for Best-of-N` | coverage principle |
| 2503.07453 | `lower bounds the runtime` | coverage lower-bounds RL runtime |
| 2412.01951 (abs) | `impossible for this self-improvement` | sharpening creates no information |
| 2510.01180 (abs) | `revives models saturated` | BroRL |
| 2607.07646 | `even at pass@1024` | base 0 % on buckets 4–5; RL at pass@16 |
| 2607.07646 | `RFT then plateaus` | RFT plateaus |
| 2607.07646 | `not exploration volume but selectivity` | selectivity |
| 2607.07646 | `not to the later non-primitive shortcuts` | pretraining excludes shortcuts |
| 2607.07646 | `12 layers, hidden dimension 512` | model size |
| 2510.11495 | `generate long responses with probability` | calibrated to p_cot |
| 2510.11495 | `not exponentially small` | efficiency condition |
| 2509.22613 | `only one path per pair` | diversity collapse |
| 2509.22613 | `absent from the initial training set` | PG finds new paths |
| 1905.10501 | `Zero reference` (Fig. 4: 7.0 % / 56.3 %) | exploration vs none |
| 1905.10501 | `learning process would stall` | stall without exploration |
| 1905.10501 | `explores throughout` | seeding alone insufficient |
| 2601.18779 | `do not resolve this issue` | entropy / clip / pass@k fail on hard problems |
| 2601.18779 | `non-zero pass@1 on each problem` | pass@k needs nonzero pass@1 |
| 2510.11686 | `less likely under the base model` | RepExp responses lower base likelihood |
| 2510.11686 | `no benefit or even degradation` | weak models do not benefit |
| 2508.14029 | `12.5%` | SvS augments only 12.5–50 % problems |
| 2310.16410 | `not present in human games` | novelty definition |
| 1712.01815 | `Dirichlet noise` | AZ root noise |
| 2406.11741 | `unable to transcend` | ChessFormer 1500 |

Abstract-level claims for the four papers the executor added (2601.21590v1, 2605.04542v1, 2512.21625v1, 2508.21188v2) are quoted in REVIEW.md from the fetched abstracts.
