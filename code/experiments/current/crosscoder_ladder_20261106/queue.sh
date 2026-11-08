#!/usr/bin/env bash
# The whole job on one A40, resumable (every step skips when its output exists):
#   1. sequences: pretraining records; each ladder point's own rollouts (the base's double as a_roll)
#   2. crosscoders at layer 3: base vs every point of its run; the two-pretrains null; planted controls on the finals
# Order: the four finals first (the longest-RL points), then the rounds, so an early stop still leaves the curve's ends.
set -uo pipefail
cd /workspace/job/crosscoder_ladder_20261106
export ND_FORK=/workspace/nd-takehome CL_CKPTS=/workspace/ckpts CL_ART=/workspace/art
L3="--layer 3"
pt() { python collect.py pretrain --n 20000; }
roll() { python collect.py roll --run "$1" --stem "$2" --n_targets 2000 --k 4; }
cc() { python crosscoder.py $L3 "$@"; }

pt
for run in v4_untied-ei_s0 v4_untied-ei_s1 v2_untied-ei_s0 v2_untied-ei_s1; do roll $run pretrain; done
for run in v4_untied-ei_s0 v4_untied-ei_s1 v5_untied-ei_s0 v5_untied-ei_s1; do roll $run final; done
for run in v2_untied-ei_s0 v2_untied-ei_s1; do roll $run final; done
for run in v4_untied-ei_s0 v4_untied-ei_s1; do for r in 005 010 015 020 025 030 035; do roll $run rl_round_$r; done; done
for run in v5_untied-ei_s0 v5_untied-ei_s1; do for r in 005 010 015 020 025 030 035 040; do roll $run rl_round_$r; done; done

base() { case $1 in v4_untied-ei_s0) echo v4_untied-ei_s0:pretrain;; v4_untied-ei_s1) echo v4_untied-ei_s1:pretrain;;
                    v5_untied-ei_s0) echo v2_untied-ei_s0:pretrain;; v5_untied-ei_s1) echo v2_untied-ei_s1:pretrain;; esac; }
for run in v4_untied-ei_s0 v4_untied-ei_s1 v5_untied-ei_s0 v5_untied-ei_s1; do cc --a "$(base $run)" --b "$run:final"; done
# the null: two pretraining runs of the same seed (v4's and v2's), same data and recipe, GPU nondeterminism apart
cc --a v4_untied-ei_s0:pretrain --b v2_untied-ei_s0:pretrain --tag _null
cc --a v4_untied-ei_s1:pretrain --b v2_untied-ei_s1:pretrain --tag _null
for run in v5_untied-ei_s0 v5_untied-ei_s1; do cc --a "$(base $run)" --b "${run/v5/v2}:final" --tag _prior; done
for run in v4_untied-ei_s0 v4_untied-ei_s1; do for r in 005 010 015 020 025 030 035; do cc --a "$(base $run)" --b "$run:rl_round_$r"; done; done
for run in v5_untied-ei_s0 v5_untied-ei_s1; do for r in 005 010 015 020 025 030 035 040; do cc --a "$(base $run)" --b "$run:rl_round_$r"; done; done
for run in v4_untied-ei_s0 v5_untied-ei_s0; do cc --a "$(base $run)" --b "$run:final" --plant --tag _plant; done
echo QUEUE_DONE
