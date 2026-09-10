---
written_on: 2026-09-08
written_by: agent:codex
---

# Research log

- 05:14 UTC: Began ten-hour sprint and read sprint guidance. Local `nd-rl` is an uncommitted workspace scaffold; older local experiments are from June.
- 05:20 UTC: simplex1 accessible; four idle H100s (0,2,3,4), other devices have existing work. Reserved 0 and 2. Located newer September 6–7 experiments remotely.
- 05:30 UTC: Read aggregate recent summaries. Prior long conjunction-chain success did not transfer: all selected models solved 1/33 additional textbook classes, 0/30 at reference length ≥8 under greedy decoding. These are historical results.
- 05:35 UTC: Strategy recommends symbolic backward plans plus verifier-only policy learning and matched direct controls. Review confirms legacy verifier's future-reference loophole is closed by the newer strict wrapper; all new rewards will use strict verification. Implementer building scratchpad SFT/RL; reviewer preparing blind splits and guarded data. Chosen primary parents: cap-six 8M `stage1.pt` and fresh cap-six 85M checkpoint, subject to source-data overlap audit. Inherited longer-proof checkpoints remain secondary.

- 05:39–05:41 UTC: Matched 100-step direct/scratch SFT smoke passed on GPUs3/4. Four scratch RL updates produced mixed-reward groups and valid proofs (6–19% sampled success on synthetic fixtures). Independent six-test CUDA/BF16 review passed. Trained-model BF16 cached/teacher log-probability drift was0.12–0.16; production policy sampling and scoring use FP32 instead, observed drift around0.00005.
- 05:42 UTC: Launched pair1: shared expanded cap6 8M parent;3,000 cap6 SFT updates and256 LOO/PPO updates; direct vs reverse-plan scratch, same random record/theorem schedule. RL mixture50% short synthetic,45% longer synthetic prompts,5% permitted textbook draws (14 distinct classes max);64 samples/update,temperature0.8,1,024-token generation cap,FP32; retention0.1. Complete configs and immutable source snapshotv1 saved remotely. Textbook evaluation occurs after each arm's RL completion.
- 05:48 UTC: Queued pair2:85M cap6 parent,3,000 SFT and512 RL updates,dependency-plan scratch vs direct. Snapshotv2 records all settings. Prepared explicitly separate cap8/cap12 synthetic label extensions:1,922 and4,969 added proofs, with1,303 disjoint RL prompts.
- 05:50 UTC: Hugging Face archival preflight found no authenticated Hub token on simplex1 or local launcher. Artifacts remain durable on simplex1 and report/code will be copied locally; no credentials requested during autonomous sprint. Repository-wide checks encounter preexisting missing scaffold packages and dependencies; scoped sprint checks pass (review note has details).
