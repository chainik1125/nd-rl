#!/usr/bin/env bash
# After queue.sh: the CPT control arm (cpt.py), its rollouts, and its crosscoders against the same base.
set -uo pipefail
cd /workspace/job/crosscoder_ladder_20261106
export ND_FORK=/workspace/nd-takehome CL_CKPTS=/workspace/ckpts CL_ART=/workspace/art
until grep -q QUEUE_DONE /workspace/cc.log; do sleep 60; done
python cpt.py --base v4_untied-ei_s0:pretrain --mirror v4_untied-ei_s0 --out /workspace/ckpts/cpt_s0
for r in 005 010 015 020 025 030 035; do python collect.py roll --run cpt_s0 --stem rl_round_$r --n_targets 2000 --k 4; done
python collect.py roll --run cpt_s0 --stem final --n_targets 2000 --k 4
python crosscoder.py --layer 3 --a v4_untied-ei_s0:pretrain --b cpt_s0:final
for r in 005 010 015 020 025 030 035; do python crosscoder.py --layer 3 --a v4_untied-ei_s0:pretrain --b cpt_s0:rl_round_$r; done
echo ALL_DONE
