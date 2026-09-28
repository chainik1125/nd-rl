# exp-00c status

- 2026-09-27T23:36:25Z proposal.md, configs, launch.sh, split and scripts committed (gate A); make_split.py already run (no model); nothing sampled yet
- 2026-09-27T23:36:39Z launched: nohup bash experiments/ws3/exp-00c-lr-ei-cpu/launch.sh (pool filter, 3 GRPO arms, EI, analysis; sequential)
- 2026-09-27T23:40:42Z pool filter done (242 s): pool 198 (L6 63, L7 131, L8 4); EI k = 21
- 2026-09-27T23:44:07Z frozen baseline done (202 s): eval-set L7 avg 0.122, 39/160; L8 0/40
- 2026-09-27T23:57Z lr3e-5 done (train 531 s, eval 255 s): L7 avg 0.430 -> surprisingly good; stop-and-check started (see results.md, verify_lr3e-5/)
- 2026-09-28T00:09Z lr1e-4 done (488 s, 250 s); 00:23Z lr3e-4 done (576 s, 260 s; greedy 2-6 dropped 0.097); 00:32Z EI done (238 s, 269 s); analyze.py ran; selection rule: lr 1e-4
- independent verifier (separate session, own code): no reward bug, loophole or leak for lr3e-5 and EI (verify_lr3e-5/, verify_ei/)
- gate B: fetch.sh --verify all hashes match; vendor dir unchanged vs charles-pre-ws3; no code change since gate A commit 487f43d (only .gitignore); metric coverage complete for all 4 arms; training prompts = pool.jsonl (train half), leak-checked at load and by the verifier
- dumps > 200 KB gzipped in place; 20 raw eval samples per arm read by the executor
DONE 2026-09-28T00:36:16Z
