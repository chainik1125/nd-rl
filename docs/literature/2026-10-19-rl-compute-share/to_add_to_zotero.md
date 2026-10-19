# Sources to add to Zotero (run rl-compute-share, 2026-10-19)

One identifier per line, for Dan, in the order of the screened table in `screened.md`. These are the screened sources not
already in `references.bib`: 47 arXiv papers, then 43 web sources to add by URL. Excluded because they are already
there: Yue et al. 2504.13837 (`yueDoesReinforcementLearning2025`), ProRL 2505.24864 (`liuProRLProlongedReinforcement2025`),
DeepSeek-R1 2501.12948 and its Nature version (`guoDeepSeekR1IncentivizesReasoning2025`), DeepSeek-V4 2606.19348
(`deepseek-aiDeepSeekV4HighlyEfficient2026`), Demystifying RL post-training 2608.24949
(`clayDemystifyingReinforcementLearning2026`) and the DeepSeek-V4.1-Flash technical report on Hugging Face
(`deepseekai2026deepseekv41flash`). Two supporting files are in the bib under their own keys, but the papers they support
are not, so the papers are listed: the verl-recipe DAPO script (`tongDAPORecipe2025`; DAPO is 2503.14476) and the
open-instruct OLMo 3 RL-Zero scripts (`allenaiOLMo3RLZero2025`; Olmo 3 is 2512.13961). Supporting files (configs, scripts,
commit logs, model and dataset cards, hardware datasheets, the OpenRouter listing) are not listed. "(also listed by ...)"
marks the 19 items already in an earlier run's list (8 lit-review, 5 lit-review-2, 6 capabilities); skip them if already imported.
Citekeys are proposals. Unmarked web keys below come from the screening agents' summaries (agents A, C and D); keys
marked † are proposed here in the same style (agent B proposed none). Keys used in `notes/` are proposals too; pin them
in Zotero or rename them in the notes after import.

arxiv:2503.18892
arxiv:2503.20783 (also listed by lit-review)
arxiv:2503.14476 (also listed by lit-review)
arxiv:2506.10910
arxiv:2409.12122
arxiv:2412.15115
arxiv:2407.21783
arxiv:2502.13923
arxiv:2501.15383
arxiv:2412.19437
arxiv:2512.02556
arxiv:2503.24290
arxiv:2505.22312
arxiv:2506.13585
arxiv:2501.08313
arxiv:2505.09388
arxiv:2501.12599
arxiv:2510.13786 (also listed by capabilities)
arxiv:2512.13961
arxiv:2505.16400
arxiv:2504.21318
arxiv:2412.08905
arxiv:2505.00949
arxiv:2505.07291
arxiv:2508.06471
arxiv:2512.16649
arxiv:2610.15542
arxiv:2504.12501
arxiv:2510.01180 (also listed by lit-review-2)
arxiv:2509.21128 (also listed by lit-review)
arxiv:2507.14843 (also listed by lit-review)
arxiv:2506.01347 (also listed by lit-review)
arxiv:2509.25123 (also listed by lit-review)
arxiv:2512.07783 (also listed by lit-review)
arxiv:2510.04028 (also listed by lit-review-2)
arxiv:2504.07912 (also listed by lit-review)
arxiv:2506.10947 (also listed by lit-review-2)
arxiv:2504.20571 (also listed by capabilities)
arxiv:2510.11495 (also listed by lit-review-2)
arxiv:2607.07646 (also listed by lit-review-2)
arxiv:2607.12395
arxiv:2606.15455
arxiv:2507.06261
arxiv:2509.04259 (also listed by capabilities)
arxiv:2505.11711 (also listed by capabilities)
arxiv:2511.08567 (also listed by capabilities)
arxiv:2509.18521

## Non-arXiv sources (add by URL)

https://github.com/ganler/code-r1 — Code-R1: Reproducing R1 for Code with Reliable Rewards, Liu & Zhang (`liuCodeR1ReproducingR12025`)
https://www.together.ai/blog/deepcoder — DeepCoder: A Fully Open-Source 14B Coder at O3-mini Level, Luo et al. (Agentica, Together AI) (`luoDeepCoderFullyOpenSource2025`)
https://github.com/hiyouga/EasyR1 — EasyR1: An Efficient, Scalable, Multi-Modality RL Training Framework, Zheng et al. (`zhengEasyR1EfficientScalable2025`)
https://mistral.ai/news/mistral-medium-3 — Medium is the new large (Mistral Medium 3), Mistral AI (`mistralaiMediumNewLarge2025`†)
https://hkunlp.github.io/blog/2025/Polaris/ — POLARIS: A Post-training Recipe for Scaling RL on Advanced Reasoning Models, An et al. (HKU NLP) (`anPOLARISPostTrainingRecipe2025`†)
https://storage.googleapis.com/intellect-3-paper/INTELLECT_3_Technical_Report.pdf — INTELLECT-3: Technical Report, Prime Intellect (`primeintellectINTELLECT3TechnicalReport2025`†)
https://reflection.ai/blog/introducing-beam — Introducing Beam: Reflection's 501B open-weight model, Reflection AI (`reflectionIntroducingBeam2026`)
https://developer.nvidia.com/blog/scaling-llm-reinforcement-learning-with-prolonged-training-using-prorl-v2/ — Scaling LLM Reinforcement Learning with Prolonged Training Using ProRL v2, Hu et al. (NVIDIA) (`huScalingLLMReinforcement2025`†)
https://research.nvidia.com/labs/lpr/prorlv2/ — Stabilizing Long-Horizon Fine Tuning: ProRLv2's Strategies, NVIDIA (`nvidiaStabilizingLongHorizonFine2025`†)
https://openai.com/index/learning-to-reason-with-llms/ — Learning to Reason with LLMs, OpenAI (`openaiLearningReasonLLMs2024`)
https://www.darioamodei.com/post/on-deepseek-and-export-controls — On DeepSeek and Export Controls, Dario Amodei (`amodeiDeepSeekExportControls2025`)
https://x.ai/news/grok-3 — Grok 3 Beta: The Age of Reasoning Agents, xAI (`xaiGrok3Beta2025`†)
https://openai.com/index/introducing-o3-and-o4-mini/ — Introducing OpenAI o3 and o4-mini, OpenAI (`openaiIntroducingO3O4mini2025`)
https://inferencebysequoia.substack.com/p/9-years-to-agi-openais-dan-roberts — 9 Years to AGI? OpenAI's Dan Roberts Reasons About Emulating Einstein, Sequoia (`roberts9YearsAGI2025`†)
https://www.dwarkesh.com/p/sholto-trenton-2 — How Does Claude 4 Think? Sholto Douglas & Trenton Bricken, Dwarkesh Podcast (`patelHowDoesClaude2025`†)
https://x.ai/news/grok-4 — Grok 4, xAI (`xaiGrok42025`)
https://openai.com/index/introducing-gpt-5/ — Introducing GPT-5, OpenAI (`openaiIntroducingGPT52025`†)
https://x.ai/news/grok-4-1 — Grok 4.1, xAI (`xaiGrok412025`†)
https://cursor.com/blog/composer-1-5 — Introducing Composer 1.5, Cursor (`cursorIntroducingComposer152026`)
https://www.dwarkesh.com/p/dario-amodei-2 — Dario Amodei: "We are near the end of the exponential", Dwarkesh Podcast (`patelDarioAmodeiWe2026`†)
https://openai.com/index/pacing-model-development-cyber-capabilities/ — Pacing model development in an era of cyber-critical capabilities, OpenAI (`openaiPacingModelDevelopment2026`†)
https://epoch.ai/gradient-updates/what-went-into-training-deepseek-r1 — What went into training DeepSeek-R1?, Erdil (Epoch AI) (`epochWhatWentTraining2025`†)
https://epoch.ai/gradient-updates/how-far-can-reasoning-models-scale — How far can reasoning models scale?, You (Epoch AI) (`epochHowFarCanReasoning2025`)
https://epoch.ai/gradient-updates/three-issues-undermining-compute-based-ai-policies — Three challenges facing compute-based AI policies, Epoch AI (`epochThreeChallengesFacing2025`†)
https://epoch.ai/gradient-updates/why-gpt5-used-less-training-compute-than-gpt45-but-gpt6-probably-wont — Why GPT-5 used less training compute than GPT-4.5 (but GPT-6 probably won't), Ho (Epoch AI) (`epochWhyGPT5Used2025`)
https://epoch.ai/data-insights/openai-compute-spend — Most of OpenAI's 2024 compute went to experiments, Epoch AI (`epochMostOpenAIsCompute2025`)
https://epochai.substack.com/p/notes-on-gpt-5-training-compute — Notes on GPT-5 training compute, Epoch AI (`epochNotesGPT5Training2025`)
https://epoch.ai/data-insights/grok-4-training-resources — What did it take to train Grok 4?, Epoch AI (`epochWhatDidIt2025`†)
https://epoch.ai/data/large_scale_ai_models.csv — Large-Scale AI Models database, Epoch AI (`epochLargeScaleAIModels2026`†)
https://www.interconnects.ai/p/olmoe-and-building-better-llms — OLMoE and the hidden simplicity in training better foundation models, Lambert (Interconnects) (`lambertOLMoEHiddenSimplicity2024`†)
https://www.understandingai.org/p/nathan-lambert-on-the-rise-of-thinking — Nathan Lambert on the rise of "thinking" language models, Lee (Understanding AI) (`leeNathanLambertRise2025`†)
https://newsletter.semianalysis.com/p/scaling-reinforcement-learning-environments-reward-hacking-agents-scaling-data — Scaling Reinforcement Learning: Environments, Reward Hacking, Agents, Scaling Data, Patel (SemiAnalysis) (`patelScalingReinforcementLearning2025`†)
https://www.tobyord.com/writing — Evidence that Recent AI Gains are Mostly from Inference-Scaling, Ord (read via the writing index; essay URL not fetched) (`ordEvidenceRecentAI2025`†)
https://www.tobyord.com/writing/how-well-does-rl-scale — How Well Does RL Scale?, Ord (`ordHowWellDoes2025`)
https://texxr.com/892700 — Gemini 3 Gains Challenge Pre-Training Plateau Claim, texxr (`texxrGemini3Gains2025`†)
https://www.dwarkesh.com/p/ilya-sutskever-2 — Ilya Sutskever: We're moving from the age of scaling to the age of research, Dwarkesh Podcast (`patelIlyaSutskeverWere2025`†)
https://www.interconnects.ai/p/frontier-post-training-recipe-review — Frontier post-training recipe review with Finbarr Timbers, Lambert (Interconnects) (`lambertFrontierPostTrainingRecipe2026`†)
https://gigazine.net/gsc_news/en/20260819-openai-pacing-model/ — OpenAI says it halted AI reinforcement learning to pace model development, Gigazine (`gigazineOpenAIHasHighlighted2026`†)
https://newsletter.semianalysis.com/p/long-live-the-short-king-why-4-hi — Long Live the Short King: Why 4-hi HBM Wins, Xie et al. (SemiAnalysis) (`xieLongLiveShort2026`)
https://the-decoder.com/reflections-beam-becomes-the-most-capable-open-weight-model-built-outside-china/ — Reflection's Beam becomes the most capable open-weight model built outside China, The Decoder (`decoderReflectionsBeamBecomes2026`†)
https://startupfortune.com/ai-labs-are-now-spending-more-compute-teaching-models-to-think-than-to-learn/ — AI labs are now spending more compute teaching models to think than to learn, Startup Fortune (`startupfortuneAILabsAre2026`†)
https://officechai.com/ai/while-share-of-inference-has-held-steady-over-the-last-2-years-pre-training-compute-has-moved-to-post-training-semi-analysis/ — While share of inference has held steady, pre-training compute has moved to post-training (SemiAnalysis), OfficeChai (`officechaiWhileShareInference2026`†)
https://thinkingmachines.ai/blog/lora/ — LoRA Without Regret, Schulman (Thinking Machines Lab) (`schulmanLoRAWithoutRegret2025`) (also listed by capabilities)
