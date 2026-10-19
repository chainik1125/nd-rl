#!/usr/bin/env bash
# GRPO v2 (Leon, 2026-10-09): length curriculum L2 (10-12 lines) -> L3 (13-15), and GRPO with the verifier.
#   bash line.sh GPU SEED ARM
#   ARM: grpo_think (S1g3, plain GRPO)       grpo_think_v (S1g3, logical arm + step penalty)
#        ei_think   (S1g3, plain EI)         ei_think_v   (S1g3, guided EI: logical arm, accepted proofs only)
#        grpo_nothink_v (T0S, guided GRPO)   ei_nothink_v (T0S, guided EI)   [optional: grpo_nothink, ei_nothink]
#        grpo_think_vs, grpo_nothink_vs: the verifier without redraws (2026-10-10): logical arm with --max_rej 0, so
#        the first step the checker rejects ends the attempt and gets the step penalty (training only; reads as above)
#        grpo_nothink_vs3: grpo_nothink_vs plus 3-premise rewrites of the pool rows (runs/pools/curriculum_3p, goal
#        antecedents moved into the premises) as --targets_extra at XFRAC (0.33) of each iteration's groups (2026-10-10)
#        ARM_h, for any grpo_* ARM above (grpo_think_h, grpo_think_vs_h, grpo_nothink_h, ...; 2026-10-10): ARM with the
#        GRPO lr halved, 7.5e-5 instead of 1.5e-4 in both stages (a rollback halves from there); all else as ARM. Every
#        run, log and eval name keeps the full ARM_h, so nothing collides with ARM's. ei_*_h is refused (no GRPO lr).
#        ARM_b, for any grpo_* ARM above, ARM_h included (grpo_think_b, grpo_think_h_b, grpo_think_vs_b, ...; 2026-10-11):
#        stage L3 of ARM again, from ARM's finished L2 checkpoint (its init and KL anchor, as in ARM's own L3), with the
#        think ban flags below; all else as ARM, then the reads. Every run, log and eval name keeps the full ARM_b.
#        Refused for EI, without ARM's L2 checkpoint, or when the last guard line of ARM's L2 log is not at
#        'lr x1 (0 rollbacks)' (an alarm's rollback may have shaped that L2: rerun the whole line instead).
#        ARM_c, ARM_w, ARM_wb, for a grpo_* ARM above, never with _h or _b (2026-10-13): GRPO lr schedules (grpo.py
#        --lr_schedule; a rollback halves on top), their flags in SCHED_C / SCHED_W below and nowhere else. ARM_c:
#        cosine in both stages, warmup over 3 iterations, down to 0.1x at each stage's last iteration. ARM_w: L2 as
#        ARM's (no schedule flags), L3 warmup-stable-decay: the 10-step warmup and lr of today, then its last 20% of
#        iterations (8 of 39) down to 0.3x along 1-sqrt (Hagele et al. 2024). ARM_wb: ARM_w's L3 from ARM's finished
#        L2 checkpoint, which ARM_w's L2 repeats, checked and refused as for ARM_b. Every run, log and eval name keeps
#        the full ARM_c / ARM_w / ARM_wb. Refused for EI (its fine-tune has its own cosine).
#        grpo_think_vs3 (2026-10-13): grpo_think_vs plus the 3-premise rewrites as --targets_extra, as grpo_nothink_vs3.
#        ARM_rfb, ARM_ulb, ARM_3pb, for a grpo_* ARM above, never with another suffix (L3 pilots, 2026-10-13): stage L3
#        of ARM from ARM's finished L2 checkpoint, checked and refused as for ARM_b, with grpo.py's pilot flags, set in
#        PILOT_RFB / PILOT_ULB below and nowhere else. ARM_rfb: refill (dynamic sampling) up to 2 extra rounds and the
#        pass-rate filter --max_pass 0.9. ARM_ulb: the unlikeliness reward, beta 0.25 (He et al. 2025). ARM_3pb: the L3
#        3-premise rows (L3_3p) as --targets_extra at XFRAC, exactly as _vs3 passes them, after _vs3's extra-targets
#        preflight; refused for *_vs3 (it has them). Every run, log and eval name keeps the full ARM_rfb / ARM_ulb /
#        ARM_3pb. Refused for EI. An arm with 3-premise targets (_vs3, _3pb) is refused while a curriculum_3p file is
#        missing.
#        ARM_pb, for a grpo_* ARM above, never with another suffix (L3 pilot, 2026-10-13): as ARM_rfb, with grpo.py's
#        --unseen_prior mean (PILOT_PB): a never-drawn pair's pass rate starts at its budget's mean over the seen pairs
#        instead of 1/2. Every run, log and eval name keeps the full ARM_pb. Refused for EI.
#        ARM_mb, for a grpo_* ARM above but *_vs3, never with another suffix (the merged recipe, 2026-10-14): stage L3 of
#        ARM from ARM's finished L2 checkpoint, checked and refused as ARM_3pb, with _wb's cooldown (SCHED_W), _3pb's
#        3-premise extra targets and _ulb's unlikeliness reward (PILOT_ULB). After the usual reads, the mean of its L3
#        checkpoints at AVG_ITS and the last (pilot4.sh's averaging; skipped when it exists) is read plain on t1320,
#        tb72, charles173 and t2130 under the arm name ARM_mb_avg.
#        ARM_fbb, for a grpo_think_* ARM with the verifier (grpo_think_v, grpo_think_vs, grpo_think_vs3), never with another
#        suffix (L3 pilot, 2026-10-14; grpo_think_vs_fbb is the one asked for): as ARM_rfb, with grpo.py's in-context
#        verifier feedback (PILOT_FBB: --feedback context; a rejected step is retried with its action in the think block
#        as a dead-end line, feedback.py) and --max_rej FB_MAXREJ retries per step; then the usual reads plus the feedback
#        read arm (eval.py --arms feedback, --max_rej FB_MAXREJ) on t1320 and charles173, alongside the logical reads.
#        Refused for EI, for no-think ARMs (no think block) and for ARMs without the verifier, and until $WT's grpo.py
#        and eval.py have the feedback options (the guard checks FEEDBACK_MODES and READ_ARMS).
#        ARM_mb_kKNOB, for an ARM_mb above (the knob pilots, 2026-10-15; grpo_think_vs_mb_kKNOB at seed 0 asked for):
#        ARM_mb's L3, checked and refused as ARM_mb, with one setting changed, set in the knob block below and nowhere
#        else: x66 --targets_extra_frac 0.66 (XFRAC, 0.33); g64 --prompts 64 --group 64 (256 x 16: the same attempts per
#        iteration); t12 --temperature 1.2 (training only; grpo.py's default 1.0); kl0, kl1 --kl_anchor_coef 0, 0.1
#        (0.02); ch4 --clip_eps_high 0.4 (grpo.py's 0.28); p3 --rej_penalty 0.3 (0.1; refused for an ARM without the step
#        penalty); i78 --iters 78 in L3 (39; the WSD cooldown is its last 20%, the reads take its it78 checkpoint, the L2
#        it starts from stays it39); rr nothing (ARM_mb's L3 again: the noise control). The knob overrides the recipe's
#        value, so every flag is in the command once. Then plain reads only, t1320 k16, t2130 k8 and charles173 k64
#        (never textbook72, which knob selection must not see; no guided, hand14 or averaged-checkpoint reads). Every
#        run, log and eval name keeps the full ARM_mb_kKNOB. Another knob, or any suffix after the knob, is refused.
#        KNOB_FULL=1 (2026-10-15, for a knob that won the screening): ARM_mb_kKNOB's usual reads instead, all of ARM_mb's
#        but the averaged checkpoint's (a rollback can leave a hole in AVG_ITS); a finished L3 and reads already there
#        are skipped, so on a finished pilot this adds textbook72, the guided reads and hand14; HEAD goes to
#        ARM_sSEED.reads_commit.txt, leaving the training's base_commit.txt as it was.
#        ARM_kKNOB, for an ei_* ARM above, never with another suffix (the EI knob pilots, 2026-10-16; ei_think_v_kKNOB at
#        seed 0 asked for): stage L3 of ARM from ARM's finished L2 checkpoint (ARM_sSEED_L2_r1.pt; refused without it,
#        never retrained), with one setting changed, set in the knob block below and nowhere else: rr nothing (ARM's L3
#        again: the noise control); 3p the targets are a fixed mix of 5,000 (M3: 3,333 L3 pool rows and 1,667 L3_3p
#        rows, the third ARM_mb's GRPO draws from them; the same mix for every seed), built by grpo_v2/build_mix_ei.py,
#        refused unless its pass record (M3P, written only when the build's preflight passed) names the mix's md5, and
#        preflighted again at launch as the extra GRPO targets are; ret0 --retain 0 (ei.py's 20,000 Stage-1 records: the
#        fine-tune mix is the found proofs alone); t10 --temperature 1.0 (training samples only; 0.8); r2 --rounds 2 --k 16
#        --ft_steps 300 (2 x 16 attempts and 2 x 300 fine-tune steps against 1 x 32 and 1 x 600: the same 160k attempts
#        and 600 steps, so only how on-policy round 2's samples are differs; round 2 samples from round 1's checkpoint
#        and fine-tunes it again on the cumulative found proofs; each fine-tune has its own warmup and cosine; the stage
#        ends at, and the reads take, _r2.pt; about one round's time, inside the 6 h timeout). Then ARM_mb_kKNOB's plain
#        reads (KNOB_FULL=1: ARM's usual reads, as above). Every run, log and eval name keeps the full ARM_kKNOB. Another
#        knob, any suffix before or after the knob, and an EI knob on a grpo_* ARM are refused.
#        grpo_think_vs_mini, grpo_think_vs_mini_kKNOB (the mini runs, 2026-10-16; the reference at seeds 0-3, knobs at 0):
#        the primary recipe's two stages, MINI_ITERS (10) iterations each, chained, on fixed fifths of the data
#        (runs/pools/mini, grpo_v2/build_mini.py: 1,000 L2 rows, 1,000 L3 rows and the 369 3-premise rewrites of those;
#        refused unless its pass record names each subset's md5; preflighted again at launch as the pools and the extra
#        targets are), always from Stage-1 S1g3_s0: SEED sets the RL sampling and data order only. L2-mini is
#        grpo_think_vs's L2 command (KL anchor 0.02, constant lr after the 10-step warmup: 2.5 of the 10 iterations),
#        L3-mini grpo_think_vs_mb_kkl0's L3 command (the merged recipe at KL 0: SCHED_W, whose --decay_frac 0.2 makes the
#        cooldown the last 2 of 10 iterations, the 3-premise subset as extra targets at XFRAC, PILOT_ULB), from L2-mini's
#        last checkpoint; only --iters / --save_every, the targets, --save_rolling 2 (every rolling checkpoint kept) and
#        the timeout (2 h, watchdog 1 h) differ. KNOB, in both stages unless said, set in the knob block below and
#        nowhere else: rr nothing (the noise control); t07, t085, t115, t13 --temperature 0.7 / 0.85 / 1.15 / 1.3
#        (training only); lpNN, lpmNN --len_penalty +NN / -NN hundredths (two digits: lp20 0.2, lpm10 -0.1; per 1,000
#        sampled tokens of a success; grpo.py's length-only groups); g64b --prompts 64 --group 64 (the same attempts per
#        iteration); pool250, pool500 the nested 250- / 500-row subsets of each stage's targets (and the same fraction
#        of the 3-premise rows: the same attempts, more draws per row); size the size-twin mix (2026-10-18: a fixed
#        half of each subset's rows, by a hash of the row name, as their big-statement twins, the 3-premise extras
#        following their parent; grpo_v2/build_size.py, its own pass record), which also reads SIZE_READS; lossact,
#        lossthink --loss_tokens action / think (the policy loss and its divisor on that phase's
#        tokens alone; grpo.py's mini-run knobs, guarded). The controls kl1 (--kl_anchor_coef 0.1, 0 in the reference)
#        and g64 (64 x 64) and the advantage knobs advpos (--adv_sign pos: the positive outcome advantages alone, at
#        --rej_penalty 0) and advneg (--adv_sign neg: the failed attempts' alone; guarded as lossact) are L3-mini forks:
#        from the reference's L2-mini at the same seed (its _it10.pt, also the KL anchor's reference), no L2-mini of their
#        own;
#        a fork waits up to 3 h for that L2-mini to finish (the checkpoint and its line's exit-0 line in pull_queue.sh's
#        job log) and refuses one that did not end at lr x1 (0 rollbacks). After each stage its ALARM lines, rollbacks
#        and, without its last checkpoint (--ban_stop's exit 3, a timeout), a FAILED line. Then plain reads only (never
#        textbook72) of L3-mini's last checkpoint: MINI_READS (L2 and L3 test_1k k8, t1320 k8, t2130 k8, charles173
#        k64); for the reference, rr and the pool arms also MINI_TRAIN (their own L2 / L3 training subsets, k8); for the
#        t arms also MINI_PANEL at their training temperature (dirs ending _T07 ... _T13); with MINI_ROLL (env, none by
#        default: STAGE:IT, e.g. "L2:4 L3:2 L3:6") MINI_PANEL of those rolling checkpoints. Refused: another base arm,
#        another knob, anything after it, KNOB_FULL=1. The training's base_commit.txt is written once: a finished
#        line's rerun records HEAD in reads_commit.txt, one that trains again in rerun_commit.txt.
#        grpo_think_vs_mini20[_kKNOB] (2026-10-18): the same but MINI20_ITERS (20) iterations a stage (--iters /
#        --save_every 20; SCHED_W's cooldown the last 4 of L3-mini's 20; the warmup still 10 AdamW steps), 3 h a stage
#        (watchdog 2 h); its forks from grpo_think_vs_mini20_sSEED_L2_it20.pt; the same subsets, knobs and reads, and
#        grpo_think_vs_mini20_tpanel / _szpanel read its reference.
#        grpo_think_vs_mini_tpanel: no training; MINI_PANEL of the reference's L3-mini last at each TPANEL_TS (0.7, 1.3),
#        named as the t arms' training-T reads (_T07, _T13): their reference. grpo_think_vs_mini_szpanel: no training;
#        SIZE_READS (the L2 / L3 test_1k twins, l2test_sz / l3test_sz, and their matched originals, l2test_om /
#        l3test_om, k8) of the reference's L3-mini last: the size knob's reference.
#        ARM_ckreads (2026-10-16): no training; plain reads of a finished full GRPO L3 run that started from another
#        arm's L2 (the knob pilots, ARM_mb, ARM_fbb, ...): CKR_PANEL (L2 and L3 test_1k k8, t1320 k4, t2130 k4,
#        charles173 k16) of its rolling checkpoints at CKR_ITS (every 10 before its last; a missing one: the nearest
#        earlier rolling one, said and recorded in ARM_ckreads_sSEED.ckpts.txt) under ARM_itN, and CKR_FINAL (the test_1k
#        pair) of its last under ARM (its pilot reads are there). Its it0, the base arm's L2 checkpoint, is shared:
#        ARM_l2_ckreads reads a base arm's L2 last (grpo_think_vs_l2_ckreads, as grpo_think_vs_L2) and s1g3_ckreads the
#        Stage-1 base S1g3_sSEED (as s1g3), each on CKR_PANEL and MINI_READS. HEAD in ARM_ckreads_sSEED.base_commit.txt.
#        Refused for other arms, without the checkpoint, KNOB_FULL=1.
# One stage = 160k sampled attempts: 39 GRPO iterations x 256 targets x 16, or one EI round at k 32 over 5,000.
# GRPO keeps tonight's settings and safeguards (grpo_think/line.sh): lr 1.5e-4, --overlong fail, canary + rollback,
# KL anchor 0.02 to each stage's init, effort budgets 0,64,256 for thinking; plus --monitor. EI as seed2/line.sh.
# A GRPO stage that an earlier run left without its last checkpoint (its dir has args.json or metrics.jsonl: a crash,
# a stop) has its dir and log moved aside to *.crashed_<UTC> before it reruns (2026-10-18), so the rerun starts clean.
# Think ban (2026-10-11, every GRPO stage): --ban_restore_check (a think ban alarm alone does not roll back when the
# checkpoint it would restore is over the limit too) and --ban_stop 0.005 (the stage ends, exit 3, and the line stops).
# Reads after the last stage on the same GPU (one queue per GPU): t1320 (986 valid; 16-20 is a min_lines filter of
# the rows) k16, textbook72 k64, charles173 (Charles's 2026-10-02 release: candidate_v0 + v1 + batch3) k64, each with
# the plain and the guided (logical) arm; hand_proved_14 k64 and a 1,000-item sample of test_21_30 k8, plain only.
# Thinking models read at B64, no-think at B0. Rows under runs/eval/v2/<set>_<arm>_s<seed>_<readarm>_k<k>/.
# env: POOL_L2, POOL_L3 (default runs/pools/curriculum/L{2,3}/pool.jsonl), REJ_PENALTY (0.1),
#      SMOKE=1: a tiny CPU dry run of the whole line (2 iterations, k 2, 2 eval problems) under runs/grpo_v2/smoke/.
source /scratch/cowifornia/nd-rl/think/runs/followup_20261007/lib.sh
WT=/scratch/cowifornia/nd-rl/think/nd-rl-v2; BR=leon/grpo-v2  # the code every line imports
[ "${SMOKE:-0}" = 1 ] && [ -n "${SMOKE_WT:-}" ] && { WT=$SMOKE_WT; BR=leon/grpo-v2-dev; }  # SMOKE=1 only: a dev worktree
cd $WT || exit 1
[ "$(git -C $WT rev-parse --abbrev-ref HEAD)" = $BR ] || { say "$WT is not on $BR: refusing"; exit 1; }
PY=/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python
"$PY" -c "import code.thinking, sys; sys.exit(0 if code.thinking.__file__.startswith('$WT/') else 1)" \
  || { say "code.thinking does not resolve to $WT: refusing"; exit 1; }
g=$1; s=$2; arm=$3
[ -n "$g" ] && [ -n "$s" ] && [ -n "$arm" ] || { echo "usage: line.sh GPU SEED ARM"; exit 2; }
job=$arm; ckr=""  # ARM_ckreads (l3: a full L3's checkpoints), ARM_l2_ckreads (l2), s1g3_ckreads (base): $arm, the arm read
                  # (job names in lower case: pull_queue.sh takes [a-z0-9_]+:[0-9]+ lines only)
case $arm in s1g3_ckreads) ckr=base; arm=grpo_think_vs ;; *_l2_ckreads) ckr=l2; arm=${arm%_l2_ckreads} ;;
  *_ckreads) ckr=l3; arm=${arm%_ckreads} ;; *_tpanel) ckr=tpanel; arm=${arm%_tpanel} ;;
  *_szpanel) ckr=szpanel; arm=${arm%_szpanel} ;; esac  # (tpanel: TPANEL_TS; szpanel: SIZE_READS)
knob=""; pa=$arm  # ARM_mb_kKNOB, ARM_kKNOB (EI): the suffixes below are parsed on ARM_mb / ARM ($pa); $arm names every output
mini=""; mk=""; mfork=""; MPN=""; MV=""  # ARM_mini[20][_kKNOB]: parsed first, then as grpo_think_vs_mb ($pa: L3-mini's
  # recipe); its knob $mk, a fork of the reference's L2-mini, the pool arms' subset size, the variant (MV=20: mini20)
case $arm in *_mini|*_mini_*|*_mini20|*_mini20_*) mini=1; mb=${arm%%_mini*}; mk=${arm#"${mb}"_mini}
  case $mk in 20|20_*) MV=20; mk=${mk#20} ;; esac  # grpo_think_vs_mini20[_kKNOB] (2026-10-18): 20 iterations a stage
  [ "$mb" = grpo_think_vs ] || { say "$arm: _mini runs the primary recipe, grpo_think_vs's: refusing"; exit 1; }
  case $mk in ""|_krr|_kt07|_kt085|_kt115|_kt13|_klp[0-9][0-9]|_klpm[0-9][0-9]|_kkl1|_kg64|_kg64b|_kpool250|_kpool500|_ksize|_klossact|_klossthink|_kadvpos|_kadvneg)
    mk=${mk#_k} ;; *)
    say "$arm: _mini takes one knob _kKNOB of rr t07 t085 t115 t13 lpNN lpmNN kl1 g64 g64b pool250 pool500 size lossact lossthink advpos advneg, with nothing after it: refusing"; exit 1 ;;
  esac
  case $mk in lp00|lpm00) say "$arm: _k$mk is no length penalty (grpo_think_vs_mini): refusing"; exit 1 ;;
    size) MPN=_size ;;  # the size-twin mix, grpo_v2/build_size.py's (its own pass record, MP below)
    kl1|g64|advpos|advneg) mfork=1 ;; pool250) MPN=250 ;; pool500) MPN=500 ;; esac
  [ "${KNOB_FULL:-0}" = 1 ] && { say "$arm: KNOB_FULL=1 is the knob pilots' (a mini run has one set of reads): refusing"; exit 1; }
  pa=${mb}_mb ;;
  *_mb_k*) knob=${arm##*_mb_k}; pa=${arm%_k$knob}
  case $knob in x66|g64|t12|kl0|kl1|ch4|p3|i78|rr) ;; *)
    say "$arm: _mb_k takes one knob of x66 g64 t12 kl0 kl1 ch4 p3 i78 rr, with nothing after it: refusing"; exit 1 ;;
  esac ;;
  ei_*_k*) knob=${arm#*_k}; pa=${arm%%_k*}  # at the first _k (no EI arm name has one)
  case $knob in rr|3p|ret0|t10|r2) ;; *)
    say "$arm: _k on an EI arm takes one EI knob of rr 3p ret0 t10 r2, with nothing after it: refusing"; exit 1 ;;
  esac
  case $pa in ei_think|ei_think_v|ei_nothink|ei_nothink_v) ;; *)  # ARM_kKNOB: ARM is an EI arm as listed above
    say "$arm: _k$knob (an EI knob pilot) does not combine with other suffixes: refusing"; exit 1 ;;
  esac ;;
  grpo_*_k*) case ${arm##*_k} in rr|3p|ret0|t10|r2)  # an EI knob on a GRPO arm (other names: unknown arm, below)
    say "$arm: _k${arm##*_k} is an EI knob (ARM_kKNOB, for ei_* arms); GRPO's knob pilots are ARM_mb_kKNOB: refusing"; exit 1 ;;
  esac ;;
esac
pil=""; case $pa in *_rfb) pil=rfb ;; *_ulb) pil=ulb ;; *_3pb) pil=3pb ;; *_pb) pil=pb ;; *_mb) pil=mb ;; *_fbb) pil=fbb ;; esac  # first
case $pa in *_rfb_*|*_ulb_*|*_3pb_*)  # a pilot suffix with another after it
  say "$arm: _rfb, _ulb and _3pb (L3 pilots) do not combine with other suffixes: refusing"; exit 1 ;;
esac
case $pa in *_pb_*) say "$arm: _pb (an L3 pilot) does not combine with other suffixes: refusing"; exit 1 ;; esac
case $pa in *_mb_*) say "$arm: _mb (an L3 pilot) does not combine with other suffixes: refusing"; exit 1 ;; esac
case $pa in *_fbb_*) say "$arm: _fbb (an L3 pilot) does not combine with other suffixes: refusing"; exit 1 ;; esac
core=${pa%_$pil}  # ARM of ARM_$pil
[ -n "$pil" ] && case $core in *_[hbcw]|*_wb|*_pb|*_mb)
  case $pil in pb|mb|fbb) say "$arm: _$pil (an L3 pilot) does not combine with other suffixes: refusing"; exit 1 ;; esac
  say "$arm: _rfb, _ulb and _3pb (L3 pilots) do not combine with other suffixes: refusing"; exit 1 ;;
esac
case ${pa}_ in *_[hb]_c_|*_[hb]_w_|*_[hb]_wb_|*_[cw]_[hb]_*|*_wb_[hb]_*)  # in either order
  say "$arm: _c, _w and _wb (lr schedules) do not combine with _h or _b: refusing"; exit 1 ;;
esac
sch=""; case $pa in *_c) sch=c ;; *_w) sch=w ;; *_wb) sch=wb ;; esac  # ARM_c / ARM_w / ARM_wb: parsed first
x=$pa; [ -n "$sch" ] && x=${pa%_$sch}
src=${x%_b}  # ARM_b: L3 only, from $src's L2; parsed before _h
base=${src%_h}  # ARM_h: $base decides the behaviour below; the full $arm names every output
[ -n "$pil" ] && { src=$core; base=$core; }  # ARM_$pil: L3 only, from $core's L2, as ARM_b
LR=1.5e-4; LRN=""; [ "$base" = "$src" ] || { LR=7.5e-5; LRN=", lr $LR"; }  # GRPO lr, both stages
# GRPO lr schedules (grpo.py --lr_schedule): set here only. ARM_c: both stages, a 3-iteration warmup instead of
# --warmup's 10 AdamW steps; ARM_w / ARM_wb: stage L3 only, with the 10-step warmup.
SCHED_C="--lr_schedule cosine --warmup_iters 3 --min_lr_frac 0.1"
SCHED_W="--lr_schedule wsd --decay_frac 0.2 --min_lr_frac 0.3 --decay_shape sqrt"
S2=""; S3=""  # the schedule flags of stage L2 / L3
case $sch in c) S2=$SCHED_C; S3=$SCHED_C; src=$arm ;; w) S3=$SCHED_W; src=$arm ;; wb) S3=$SCHED_W ;; esac
# (ARM_wb keeps src=$base: L3 only, from $base's L2, as ARM_b)
[ "$pil" = mb ] && S3=$SCHED_W  # ARM_mb: _wb's L3 cooldown
[ -n "$mini" ] && src=$arm  # ARM_mini: both stages (L2-mini without schedule flags, as grpo_think_vs's L2) ...
[ -n "$mfork" ] && src=${mb}_mini$MV  # ... but a fork: L3-mini only, from the reference's L2-mini (mini20: its own)
# L3 pilots (grpo.py's pilot options): set here only. ARM_rfb: refill up to 2 extra rounds, pass-rate filter 0.9;
# ARM_ulb: the unlikeliness reward at the paper's beta. ARM_3pb has no flags of its own (its extra targets below).
PILOT_RFB="--refill --refill_max 2 --max_pass 0.9"
PILOT_ULB="--unlikely_beta 0.25"
PILOT_PB="--unseen_prior mean"  # ARM_pb: the unseen-pair prior at the budget's mean pass rate
PILOT_FBB="--feedback context"  # ARM_fbb: in-context verifier feedback, with ...
FB_MAXREJ=2  # ... at most this many retries per step, in L3 and in the feedback reads
P3=""; case $pil in rfb) P3=$PILOT_RFB ;; ulb|mb) P3=$PILOT_ULB ;; pb) P3=$PILOT_PB ;; fbb) P3=$PILOT_FBB ;; esac  # stage L3's pilot flags
AVG_ITS="25 30 35"  # ARM_mb: the L3 rolling checkpoints averaged with the last one (pilot4.sh's)
MINI_ITERS=10  # ARM_mini: each stage's iterations (the recipe's 39)
MINI20_ITERS=20  # ARM_mini20 (2026-10-18): each stage's; SCHED_W's --decay_frac 0.2 then cools the last 4 of 20
MINI_READS="l2test:8 l3test:8 t1320:8 t2130:8 charles173:64"  # ARM_mini: the plain reads (SET:K, rsets) of L3-mini's last;
MINI_TRAIN="l2train:8 l3train:8"  # the reference's, rr's and the pool arms' training subsets (memorisation vs held out);
MINI_PANEL="l3test:8 t1320:8 t2130:4 charles173:16"  # the T arms' reads at their T, and of each MINI_ROLL checkpoint:
MINI_ROLL=${MINI_ROLL:-}  # env: STAGE:IT rolling checkpoints of the mini stages (--save_rolling 2) to read; none by default
TPANEL_TS="0.7 1.3"  # grpo_think_vs_mini_tpanel: the temperatures it reads the reference at (the T arms' extremes)
SIZE_READS="l2test_sz:8 l3test_sz:8 l2test_om:8 l3test_om:8"  # _ksize, grpo_think_vs_mini_szpanel: the L2 / L3 test_1k
                                                             # twins and their matched originals (paired size effects)
CKR_PANEL="l2test:8 l3test:8 t1320:4 t2130:4 charles173:16"  # ARM_ckreads: the dynamics panel of each checkpoint, and
CKR_FINAL="l2test:8 l3test:8"  # of a full L3's last (its pilot reads hold the out-of-distribution ones)
SMOKE=${SMOKE:-0}
P=${REJ_PENALTY:-0.1}
L2=${POOL_L2:-$R/pools/curriculum/L2/pool.jsonl}; L3=${POOL_L3:-$R/pools/curriculum/L3/pool.jsonl}
C=$R/grpo_v2; EV=$R/eval/v2; WD=$L/watchdog_grpo_v2.log
TB=/scratch/cowifornia/nd-rl/think/nd-takehome/data/eval_only/textbook72; CH=$R/pools/charles_1002
case $base in
  grpo_think|grpo_think_v|grpo_think_vs|grpo_think_vs3|ei_think|ei_think_v) kind=think; init0=$R/stage1/S1g3_s$s/model.pt; pkl=S1g3.pkl ;;
  grpo_nothink|grpo_nothink_v|grpo_nothink_vs|grpo_nothink_vs3|ei_nothink|ei_nothink_v) kind=nothink; init0=$R/stage1/T0S_s$s/model.pt; pkl=T0S.pkl ;;
  *) echo "unknown arm $arm"; exit 2 ;;
esac
[ -n "$mini" ] && init0=$R/stage1/S1g3_s0/model.pt  # ARM_mini: one Stage-1 init for every seed (SEED: the RL's only)
case $base in grpo_*) method=grpo ;; *) method=ei ;; esac
[ $method = ei ] && [ -n "$pil" ] && { say "$arm: _$pil is a GRPO L3 pilot, and $base is EI: refusing"; exit 1; }
[ $method = ei ] && [ -n "$sch" ] && { say "$arm: _$sch schedules the GRPO lr, and EI's fine-tune has its own (run $base): refusing"; exit 1; }
[ $method = ei ] && [ "$src" != "$arm" ] && [ -z "$knob" ] && { say "$arm: _b reruns a GRPO L3 stage, and $src is EI: refusing"; exit 1; }
[ $method = ei ] && [ "$base" != "$src" ] && { say "$arm: _h halves the GRPO lr, which EI does not use (run $base): refusing"; exit 1; }
if [ "$ckr" = tpanel ]; then  # the mini reference's last checkpoint (mini or mini20), read at the T arms' temperatures
  { [ "$arm" = grpo_think_vs_mini ] || [ "$arm" = grpo_think_vs_mini20 ]; } \
    || { say "$job: _tpanel reads the mini reference's last checkpoint (grpo_think_vs_mini_tpanel): refusing"; exit 1; }
elif [ "$ckr" = szpanel ]; then  # the mini reference's last checkpoint (mini or mini20), read on the size-twin test sets
  { [ "$arm" = grpo_think_vs_mini ] || [ "$arm" = grpo_think_vs_mini20 ]; } \
    || { say "$job: _szpanel reads the mini reference's last checkpoint (grpo_think_vs_mini_szpanel): refusing"; exit 1; }
elif [ "$ckr" = l3 ]; then  # a full GRPO L3 that started from another arm's L2 (ARM_b, ARM_wb, the L3 pilots, ARM_mb_kKNOB)
  { [ "$src" != "$arm" ] && [ $method = grpo ] && [ -z "$mini" ]; } \
    || { say "$job: _ckreads reads a full GRPO L3 that started from another arm's L2 (a knob pilot, ARM_mb, ARM_fbb, ...): refusing"; exit 1; }
elif [ -n "$ckr" ]; then  # a base GRPO arm's own L2 (ARM_l2_ckreads), or the Stage-1 base (s1g3_ckreads)
  { [ "$src" = "$arm" ] && [ "$base" = "$arm" ] && [ $method = grpo ] && [ -z "$mini$sch" ]; } \
    || { say "$job: _l2_ckreads reads a base GRPO arm's own L2 (grpo_think_vs_l2_ckreads): refusing"; exit 1; }
fi
[ -n "$ckr" ] && [ "${KNOB_FULL:-0}" = 1 ] && { say "$job: KNOB_FULL=1 is the knob pilots' (_ckreads has its own reads): refusing"; exit 1; }
if [ "$pil" = fbb ]; then  # ARM_fbb: the lines go into the think block, and the verifier writes them
  [ $kind = think ] || { say "$arm: _fbb writes the verifier's feedback into the think block, and $base has none: refusing"; exit 1; }
  case $base in *_v|*_vs|*_vs3) ;; *) say "$arm: _fbb needs the verifier (the logical arm), and $base samples plain: refusing"; exit 1 ;; esac
  "$PY" -c "import sys, code.thinking.grpo as g, code.thinking.eval as e; sys.exit(0 if hasattr(g, 'FEEDBACK_MODES') and 'feedback' in getattr(e, 'READ_ARMS', ()) else 1)" > /dev/null 2>&1 \
    || { say "$arm: $WT's grpo.py / eval.py have no in-context feedback (--feedback, --arms feedback): refusing"; exit 1; }
elif [ -n "$P3" ]; then  # ARM_rfb / ARM_ulb / ARM_pb need their grpo.py options in $WT (leon/grpo-v2-pilots, -prior)
  PA=pilot_options; PN="--refill, --unlikely_beta"; [ "$pil" = pb ] && { PA=UNSEEN_PRIORS; PN="--unseen_prior"; }
  "$PY" -c "import sys, code.thinking.grpo as g; sys.exit(0 if hasattr(g, '$PA') else 1)" > /dev/null 2>&1 \
    || { say "$arm: $WT's grpo.py has no pilot options ($PN): refusing"; exit 1; }
fi
case $mk in lossact|lossthink|advpos|advneg)  # ARM_mini's grpo.py knobs (--loss_tokens, --adv_sign) need them in $WT
  "$PY" -c "import sys, code.thinking.grpo as g; sys.exit(0 if hasattr(g, 'LOSS_TOKENS') and hasattr(g, 'ADV_SIGNS') else 1)" > /dev/null 2>&1 \
    || { say "$arm: $WT's grpo.py has no mini-run knobs (--loss_tokens, --adv_sign): refusing"; exit 1; } ;;
esac
case $base in *_v|*_vs|*_vs3) samp=logical ;; *) samp=plain ;; esac
[ "$knob" = p3 ] && [ $samp = plain ] && {  # ARM_mb_kp3: the step penalty needs the verifier
  say "$arm: _kp3 raises the step penalty, and $base samples plain (no penalty): refusing"; exit 1; }
case $base in *_vs|*_vs3) MAXREJ_TRAIN=0 ;; *) MAXREJ_TRAIN=10 ;; esac  # GRPO training only
[ "$pil" = fbb ] && MAXREJ_TRAIN=$FB_MAXREJ  # ARM_fbb: retries per step (with --feedback)
X2=""; X3=""; XFRAC=${XFRAC:-0.33}  # extra GRPO targets per stage (--targets_extra)
case $base in *_vs3) X2=$R/pools/curriculum_3p/L2_3p.jsonl; X3=$R/pools/curriculum_3p/L3_3p.jsonl ;; esac
if [ "$pil" = 3pb ] || [ "$pil" = mb ]; then  # ARM_3pb, ARM_mb: _vs3's files (both for its extra-targets preflight)
  case $base in *_vs3) say "$arm: $base already trains on the 3-premise rows: refusing"; exit 1 ;; esac
  X2=$R/pools/curriculum_3p/L2_3p.jsonl; X3=$R/pools/curriculum_3p/L3_3p.jsonl
fi
MINI=$R/pools/mini; MP=$MINI/preflight_pass.txt; MB=build_mini.py  # ARM_mini's targets, their pass record, its builder
[ "$mk" = size ] && { MP=$MINI/preflight_pass_size.txt; MB=build_size.py; }  # _ksize: the size-twin mix's
if [ -n "$mini" ]; then  # the L2 and L3 subsets; L3-mini's extra targets the 3-premise subset (L2-mini has none)
  L2=$MINI/L2_mini$MPN.jsonl; L3=$MINI/L3_mini$MPN.jsonl; X2=""; X3=$MINI/L3_3p_mini$MPN.jsonl  # (pool arms: MPN rows)
  for xf in $L2 $L3 $X3; do  # each as its pass record ('PREFLIGHT PASSED <md5> <file>') names it
    [ -f $xf ] || { say "$arm: no mini subset $xf (grpo_v2/$MB builds it): refusing"; exit 1; }
    grep -qx "PREFLIGHT PASSED $(md5sum < $xf | cut -c1-32) $(basename $xf)" $MP 2> /dev/null \
      || { say "$arm: no preflight pass record of $xf as it is ($MP; grpo_v2/$MB): refusing"; exit 1; }
  done
fi
SZ=$R/pools/size_twins  # the size-twin test sets (leon/size-twins) and their matched originals: SIZE_READS's
SZ_SETS="$SZ/L2_test_1k_sz.jsonl $SZ/L3_test_1k_sz.jsonl $SZ/L2_test_1k_orig_matched.jsonl $SZ/L3_test_1k_orig_matched.jsonl"
if [ "$mk" = size ] || [ "$ckr" = szpanel ]; then
  for xf in $SZ_SETS; do [ -f $xf ] || { say "$job: no size-twin test set $xf (runs/pools/size_twins): refusing"; exit 1; }; done
fi
for xf in $X2 $X3; do [ -f $xf ] || { say "$arm: no 3-premise pool file $xf (runs/pools/curriculum_3p): refusing"; exit 1; }; done
L2T=$L2; L3T=$L3  # the training targets as read (MINI_TRAIN), before SMOKE=1 takes their heads
M3=$R/pools/curriculum_3p/L3_mix_ei.jsonl; M3P=$R/pools/curriculum_3p/L3_mix_ei.preflight_pass.txt  # EI ARM_k3p's
if [ "$knob" = 3p ]; then  # targets and their pass record ('PREFLIGHT PASSED <md5>'), both from grpo_v2/build_mix_ei.py
  [ -f $M3 ] || { say "$arm: no 3-premise mix $M3 (grpo_v2/build_mix_ei.py builds it): refusing"; exit 1; }
  grep -qx "PREFLIGHT PASSED $(md5sum < $M3 | cut -c1-32)" $M3P 2> /dev/null \
    || { say "$arm: no preflight pass record of $M3 as it is ($M3P; grpo_v2/build_mix_ei.py): refusing"; exit 1; }
fi
if [ $kind = think ]; then EFF="--effort --budgets 0,64,256"; RB=64; else EFF="--budgets 0"; RB=0; fi
ITERS=39; PROMPTS=256; GROUP=16; K=32; BATCH=1024; FT=""; CAN="--canary_n 1000"; KLN=""; LIM=""; KS=(16 64 64 64 8)
[ -n "$mini" ] && ITERS=$MINI_ITERS  # ARM_mini: both stages
[ "$MV" = 20 ] && ITERS=$MINI20_ITERS  # ARM_mini20: both stages
DEV=$g; MAXSTEPS_H14=160
RET=""  # EI --retain (Stage-1 records in each fine-tune; empty: ei.py's 20,000)
FS=""  # EI --ft_steps (each round's fine-tune; empty: ei.py's 600)
MAXSTEPS=96  # training step cap, GRPO and EI alike (eval's value; v1 used the default 48, which 13-15-line
             # problems exceed in 3-6% of B64 attempts: capped attempts would raise false collapse alarms)
if [ "$SMOKE" = 1 ]; then
  C=$R/grpo_v2/smoke; EV=$R/eval/v2_smoke; WD=$C/watchdog.log; L=$C/logs; mkdir -p $C $L
  head -n 8 $L2 > $C/L2_head.jsonl; head -n 8 $L3 > $C/L3_head.jsonl 2>/dev/null || cp $C/L2_head.jsonl $C/L3_head.jsonl
  [ -s $C/L3_head.jsonl ] || cp $C/L2_head.jsonl $C/L3_head.jsonl
  L2=$C/L2_head.jsonl; L3=$C/L3_head.jsonl
  if [ -n "$X2" ]; then head -n 8 $X2 > $C/X2_head.jsonl; head -n 8 $X3 > $C/X3_head.jsonl; X2=$C/X2_head.jsonl; X3=$C/X3_head.jsonl
  elif [ -n "$X3" ]; then head -n 8 $X3 > $C/X3_head.jsonl; X3=$C/X3_head.jsonl; fi  # ARM_mini: L3-mini's alone
  ITERS=2; PROMPTS=4; GROUP=4; K=2; BATCH=16; FT="--ft_recs 8"; FS=3; RET=64
  CAN="--canary_n 40"; KLN="--kl_anchor_n 16"; LIM="--limit 2"; KS=(2 2 2 2 2); DEV=""
  AVG_ITS=""  # a 2-iteration stage keeps no rolling checkpoint: ARM_mb's mean is of the last one alone
  waitfree() { :; }  # CPU only
fi
# ARM_mb_kKNOB, ARM_kKNOB (EI): the knob, set here only (stage L3; a knob arm has no L2 of its own). It overrides the
# recipe's value (XFRAC, P, PROMPTS x GROUP, KLC, IT3; EI: T3, RT3, ET, RD3 x K3, FS3), or KF adds a flag the recipe leaves at
# grpo.py's default: one copy of each flag. ARM_mini_kKNOB's knob ($mk, below) is in both mini stages.
KLC=0.02; KL3=""  # --kl_anchor_coef, both stages; KL3: stage L3's when set (ARM_mini: kl0's)
KF=""; KD=""; IT3=$ITERS; TO3=21600; WH3=3.5  # stage L3: knob flags, their log text, iterations, timeout s, watchdog h
TO2=21600; WH2=3.5  # stage L2: timeout s, watchdog h
T3=$L3; RT3=$RET; ET=0.8; RD3=1; K3=$K; FS3=$FS  # stage L3: targets; EI: --retain, sampling temperature, rounds x k
                                                 # per round, fine-tune steps per round
case $knob in
  x66) XFRAC=0.66; KD="--targets_extra_frac $XFRAC" ;;
  g64) if [ "$SMOKE" = 1 ]; then PROMPTS=2; GROUP=8; else PROMPTS=64; GROUP=64; fi  # the same attempts per iteration
       KD="--prompts $PROMPTS --group $GROUP" ;;
  t12) KF="--temperature 1.2"; KD=$KF ;;
  kl0) KLC=0; KD="--kl_anchor_coef $KLC" ;;
  kl1) KLC=0.1; KD="--kl_anchor_coef $KLC" ;;
  ch4) KF="--clip_eps_high 0.4"; KD=$KF ;;
  p3) P=0.3; KD="--rej_penalty $P" ;;
  i78) IT3=$((2 * ITERS)); TO3=43200; WH3=7; KD="--iters $IT3" ;;  # twice the iterations (SMOKE=1: 4), timeout, watchdog h
  rr) KD="nothing changed" ;;  # the recipe's L3 again (GRPO's or EI's)
  # EI ARM_kKNOB's from here on
  3p) T3=$M3; if [ "$SMOKE" = 1 ]; then head -n 8 $M3 > $C/M3_head.jsonl; T3=$C/M3_head.jsonl; fi  # the mix (SMOKE=1:
      KD="--targets $(basename $T3)" ;;                                                               # its first 8 rows)
  ret0) RT3=0; KD="--retain $RT3" ;;  # the found proofs alone
  t10) ET=1.0; KD="--temperature $ET" ;;  # training samples only (the reads stay at 0.8)
  r2) RD3=2; K3=$((K / 2)); if [ "$SMOKE" = 1 ]; then FS3=1; else FS3=300; fi  # the same attempts and fine-tune steps
      KD="--rounds $RD3 --k $K3 --ft_steps $FS3" ;;  # (SMOKE=1: k 1 and 1 step a round, for k 2 and 3); ends at _r2
esac
[ -n "$knob" ] && KD="knob $knob: $KD"
ROLL=""  # --save_rolling (grpo.py's 5 when empty)
if [ -n "$mini" ]; then  # ARM_mini: L3-mini at kl0's KL anchor (L2-mini at KLC's, as grpo_think_vs's L2); 2 h a stage;
  KL3=0; TO2=7200; WH2=1; TO3=7200; WH3=1; ROLL=2  # a rolling checkpoint every 2 iterations (MINI_ROLL's)
  [ "$MV" = 20 ] && { TO2=10800; WH2=2; TO3=10800; WH3=2; }  # ARM_mini20: 3 h a stage, watchdog 2 h
  case $mk in  # ARM_mini_kKNOB, both stages unless said
    rr) KD="nothing changed" ;;  # the reference's two stages again: the noise control
    t07) KF="--temperature 0.7"; KD=$KF ;;  # training only (grpo.py's default 1.0; the reads stay at 0.8)
    t085) KF="--temperature 0.85"; KD=$KF ;;
    t115) KF="--temperature 1.15"; KD=$KF ;;
    t13) KF="--temperature 1.3"; KD=$KF ;;
    lp[0-9][0-9]|lpm[0-9][0-9])  # per 1,000 sampled tokens of a success, NN hundredths (lpm: negative, a length bonus)
      KF="--len_penalty $([ "${mk#lpm}" = "$mk" ] || echo -)$(awk -v n=$((10#${mk: -2})) 'BEGIN { print n / 100 }')"; KD=$KF ;;
    kl1) KL3=0.1; KD="--kl_anchor_coef $KL3" ;;  # a fork, L3-mini only (the full-scale kl1 pilot's change)
    g64|g64b) if [ "$SMOKE" = 1 ]; then PROMPTS=2; GROUP=8; else PROMPTS=64; GROUP=64; fi  # the same attempts per
              KD="--prompts $PROMPTS --group $GROUP" ;;  # iteration; g64: a fork, L3-mini only; g64b: both stages
    pool250|pool500) KD="the $MPN-row subsets" ;;  # L2T / L3T (above), the same iterations and attempts
    size) KD="the size-twin mix (a fixed half of the rows as their big-statement twins)" ;;  # L2T / L3T, as the pools
    lossact) KF="--loss_tokens action"; KD=$KF ;;  # the policy loss on the action tokens alone (its divisor too)
    lossthink) KF="--loss_tokens think"; KD=$KF ;;  # ... on the think-block tokens alone (budget 0 trains nothing)
    advpos) P=0; KF="--adv_sign pos"; KD="$KF --rej_penalty 0" ;;  # a fork: the positive outcome advantages alone,
            # without the step penalty (its -P would be the one negative signal left, at twice its relative weight)
    advneg) KF="--adv_sign neg"; KD=$KF ;;  # a fork: the failed attempts' outcome advantages alone (by outcome)
  esac
  [ -n "$mk" ] && KD="knob $mk: $KD"
fi
MT=""; case $mk in t07|t085|t115|t13) MT=${KF#--temperature } ;; esac  # the T arms' training temperature (their reads)
CKR_ITS=$(seq 10 10 $((IT3 - 1)))  # ARM_ckreads: the L3 checkpoints read before its last, every 10 (i78: 10 ... 70)
if [ "$src" != "$arm" ]; then  # ARM_b, ARM_wb, ARM_$pil, ARM_kKNOB: $src's finished L2, from a run no rollback touched
  L2CK=$C/${src}_s${s}_L2/ckpts/${src}_s${s}_L2_it$ITERS.pt
  [ $method = ei ] && L2CK=$C/${src}_s${s}_L2/ckpts/${src}_s${s}_L2_r1.pt  # EI ARM_kKNOB: its one round's
  if [ -n "$mfork" ]; then  # an ARM_mini fork: wait until the reference's L2-mini at this seed has finished, its
    rl=$L/v2_${src}_s${s}.log  # checkpoint there and its line's exit line in pull_queue.sh's job log (save_ckpt is not
    fin() { [ -f $L2CK ] && grep -q "GRPO ${src}_s${s}_L2 exit 0:" $rl 2> /dev/null; }  # atomic), for up to 3 h
    fin || { say "$arm: waiting for $src's L2-mini at seed $s to finish ($L2CK, 'GRPO ${src}_s${s}_L2 exit 0' in $rl), up to 3 h"
      for i in $(seq 1 180); do sleep 60; fin && break; done
      fin || { say "$arm: $src's L2-mini at seed $s did not finish in 3 h: FAILED"; exit 1; }; }
  fi
  [ -f $L2CK ] || { say "$arm: no L2 checkpoint of $src ($L2CK): refusing"; exit 1; }
  gl=$(grep -E '^guard it' $L/v2_${src}_s${s}_L2.log 2>/dev/null | tail -1)
  [ $method = ei ] || case $gl in *"lr x1 (0 rollbacks)"*) ;; *)  # (EI has no guard and no rollbacks)
    [ -n "$mfork" ] && { say "$arm: $src's L2-mini at seed $s ended '${gl:-no guard line}', not lr x1 (0 rollbacks): a fork needs a reference L2-mini without rollbacks (run the control at another seed): refusing"; exit 1; }
    say "$arm: $src's L2 log ends '${gl:-no guard line}', not lr x1 (0 rollbacks): rerun the whole line: refusing"; exit 1 ;;
  esac
fi
if [ -n "$ckr" ]; then  # ARM_ckreads: refused, with nothing written, without the checkpoint it reads last
  CK3=$C/${arm}_s${s}_L3/ckpts/${arm}_s${s}_L3_it$IT3.pt; CK2=$C/${arm}_s${s}_L2/ckpts/${arm}_s${s}_L2_it$ITERS.pt
  case $ckr in l3|tpanel|szpanel) [ -f $CK3 ] || { say "$job: no finished L3 of $arm ($CK3) to read: refusing"; exit 1; } ;; esac
  [ $ckr = l2 ] && [ ! -f $CK2 ] && { say "$job: no checkpoint $CK2 to read: refusing"; exit 1; }
fi
mkdir -p $C $EV
if [ "$SMOKE" != 1 ] && [ -z "$ckr" ]; then  # (ARM_ckreads trains on nothing)
  "$PY" -m code.experiments.current.thinking_traces_20261005.grpo_v2.preflight $L2 $L3 > $C/preflight_${arm}_s$s.log 2>&1 \
    || { say "preflight failed (see $C/preflight_${arm}_s$s.log)"; exit 1; }
  if [ -n "$X2" ] || [ "$knob" = 3p ] || [ -n "$mini" ]; then  # the extra rows (EI ARM_k3p: a third of its mix) are
    xl=$C/preflight_${arm}_s${s}_extra.log  # shorter than the bands by construction: only the band lines may fail
    xp="$X2 $X3"; [ "$knob" = 3p ] && xp="$L2 $M3"  # ARM_k3p: the mix against L2 and every evaluation set
    [ -n "$mini" ] && xp="$L2 $X3"  # ARM_mini: the 3-premise subset against the L2 subset and every evaluation set
    "$PY" -m code.experiments.current.thinking_traces_20261005.grpo_v2.preflight $xp > $xl 2>&1
    if grep -vE '^  BAD: [0-9]+ rows outside' $xl | grep -qE '^(BAD|MISSING|\(skip)' || [ $(grep -c '^ok ' $xl) -lt 22 ]; then
      say "extra-targets preflight failed (see $xl)"; exit 1; fi
  fi
fi
[ -f $init0 ] || { say "no Stage-1 init $init0"; exit 1; }
bc=base_commit; [ -n "$knob" ] && [ "${KNOB_FULL:-0}" = 1 ] && bc=reads_commit  # KNOB_FULL keeps the training's record
if [ -n "$mini" ]; then  # ARM_mini: the training's base_commit.txt is written once. A finished line's rerun (MINI_TRAIN,
  # MINI_ROLL reads) records HEAD in reads_commit.txt; a rerun that trains again (a crash, a stop) in rerun_commit.txt
  if [ -f $C/${arm}_s${s}_L3/ckpts/${arm}_s${s}_L3_it$IT3.pt ]; then bc=reads_commit
  elif [ -f $C/${arm}_s$s.base_commit.txt ]; then bc=rerun_commit; fi
fi
cr=$C/${arm}_s$s.$bc.txt; [ -n "$ckr" ] && cr=$C/${job}_s$s.base_commit.txt  # ARM_ckreads: its own, not the arm's
git -C $WT rev-parse HEAD > $cr
watch() {  # NAME PID LOG EXPECTED_H
  [ "$SMOKE" = 1 ] && return 0
  python3 $R/watchdog.py $1:$2:$g:$3:$4 >> $WD 2>&1 & }
grpo_stage() {  # NAME INIT TARGETS [EXTRA_TARGETS] -> ckpts/NAME_it$ITERS.pt (stage L3: _it$IT3)
  local n=$1 O=$C/$1 it=$ITERS to=$TO2 wh=$WH2 kl=$KLC; case $1 in *_L3) it=$IT3; to=$TO3; wh=$WH3; kl=${KL3:-$KLC} ;; esac
  local last=$C/$1/ckpts/${1}_it$it.pt
  [ -f $last ] && return 0
  if [ -f $O/args.json ] || [ -f $O/metrics.jsonl ]; then  # an earlier run of this stage that did not finish (a crash,
    local t=$(date -u +%Y%m%dT%H%M%SZ)  # a stop): its dir (rolling checkpoints too) and log aside, so the rerun starts
    mv $O $O.crashed_$t; [ -f $L/v2_$n.log ] && mv $L/v2_$n.log $L/v2_$n.crashed_$t.log  # clean (grpo.py appends
    say "GRPO $n: an earlier run left no $(basename $last): its dir moved aside to $O.crashed_$t"; fi  # metrics.jsonl)
  waitfree $g
  local GUIDE=""; [ $samp = logical ] && GUIDE="--arm logical --max_rej $MAXREJ_TRAIN --rej_penalty $P"
  local XT=""; [ -n "$4" ] && XT="--targets_extra $4 --targets_extra_frac $XFRAC"
  local SC=$S3; case $n in *_L2) SC=$S2 ;; esac  # the stage's lr schedule flags (none: constant)
  local PF=$P3; case $n in *_L2) PF="" ;; esac  # the stage's pilot flags (L3 only)
  local KN=$KF KT=$KD; case $n in *_L2) [ -n "$mini" ] || { KN=""; KT=""; } ;; esac  # the knob's flags and their text
                                                                                    # (L3 only; ARM_mini_kKNOB: both)
  ( CUDA_VISIBLE_DEVICES=$DEV timeout $to "$PY" -m code.thinking.grpo --init $2 --name $n --out $O --targets $3 \
      --iters $it --prompts $PROMPTS --group $GROUP --batch $BATCH --lr $LR $SC $PF $KN --overlong fail --save_every $it \
      --max_steps $MAXSTEPS \
      --seed $s --keep_rolling 100 ${ROLL:+--save_rolling $ROLL} $EFF --canary $R/pools/canary_long.jsonl $CAN --canary_every 1 \
      --rollback_on_alarm --max_rollbacks 3 --ban_restore_check --ban_stop 0.005 \
      --kl_anchor_coef $kl --kl_anchor_ref $2 $KLN --monitor $GUIDE $XT \
      > $L/v2_$n.log 2>&1 ) & local p=$!
  sleep 5; say "launched GRPO $n ($samp${GUIDE:+, P $P}${XT:+, extra $(basename $4) x $XFRAC}$LRN${SC:+, $SC}${PF:+, $PF}${KT:+, $KT}) on GPU $g (pid $p)"; watch $n $p $L/v2_$n.log $wh
  wait $p; local rc=$?; say "GRPO $n exit $rc: $(grep -E '^it [0-9]+:' $L/v2_$n.log | tail -1 | cut -c1-150)"
  [ -n "$mini" ] && stage_alarms $n $L/v2_$n.log
  [ -n "$mini" ] && [ ! -f $last ] && say "GRPO $n FAILED: exit $rc$([ $rc = 3 ] && echo ' (--ban_stop: the think ban mass passed 0.005)'), no $(basename $last): the run counts as failed"
  [ -f $last ]; }
stage_alarms() {  # NAME LOG: ARM_mini's record of a stage's alarms (grpo.py's ALARM lines: each alarm, the rollback
  local x                # it led to or why none) and the rollbacks it ended with (its last guard line)
  while IFS= read -r x; do say "GRPO $1 $x"; done < <(grep -E '^ALARM it' $2)
  x=$(grep -E '^guard it' $2 | tail -1 | grep -oE 'lr x[0-9.e-]+ \([0-9]+ rollbacks\)')
  say "GRPO $1 rollbacks: ${x:-no guard line}"; }
ei_stage() {  # NAME INIT TARGETS -> ckpts/NAME_r1.pt (stage L3: _r$RD3)
  local n=$1 O=$C/$1 rd=1 k=$K t=0.8 rt=$RET fs=$FS KT=""  # stage L3: ARM_kKNOB's values and their text
  case $1 in *_L3) rd=$RD3; k=$K3; t=$ET; rt=$RT3; fs=$FS3; KT=$KD ;; esac
  local last=$C/$1/ckpts/${1}_r$rd.pt
  [ -f $last ] && return 0; waitfree $g
  ( CUDA_VISIBLE_DEVICES=$DEV timeout 21600 "$PY" -m code.thinking.ei --init $2 --name $n --out $O --targets $3 \
      --train $R/data/$pkl --rounds $rd --k $k --temperature $t --arm $samp --max_rej 10 --batch $BATCH --seed $s \
      --max_steps $MAXSTEPS \
      --select shortest $EFF ${fs:+--ft_steps $fs} $FT ${rt:+--retain $rt} > $L/v2_$n.log 2>&1 ) & local p=$!
  sleep 5; say "launched EI $n ($samp${KT:+, $KT}) on GPU $g (pid $p)"; watch $n $p $L/v2_$n.log 4
  wait $p; say "EI $n exit $?: $(tail -1 $L/v2_$n.log | cut -c1-150)"
  [ -f $last ]; }
ck() { local it=$ITERS rd=1; [ "$1" = L3 ] && { it=$IT3; rd=$RD3; }  # STAGE -> the stage's last checkpoint
  if [ $method = grpo ]; then echo $C/${arm}_s${s}_$1/ckpts/${arm}_s${s}_$1_it$it.pt; else echo $C/${arm}_s${s}_$1/ckpts/${arm}_s${s}_$1_r$rd.pt; fi; }
read1() {  # SETNAME SETS K READARM [MAXSTEPS [ARMNAME CKPT [MAXREJ [T]]]]  (default: $arm, its L3 checkpoint, 10, 0.8;
  local o=$EV/$1_${6:-$arm}_s${s}_$4_k$3${9:+_T${9//./}} ms=${5:-96} cp=${7:-$(ck L3)} mr=${8:-10} tp=${9:-0.8}  # T: _T13)
  [ -f $o/rows_$4_B$RB.jsonl ] && return 0; waitfree $g; local t=$(date +%s); rm -rf $o
  CUDA_VISIBLE_DEVICES=$DEV timeout 9000 "$PY" -m code.thinking.eval --ckpt $cp --sets $2 --out $o --k $3 \
    --budgets $RB --arms $4 --batch $BATCH --seed 0 --temperature $tp --max_action 512 --max_steps $ms --max_rej $mr \
    $LIM > $L/v2_eval_$(basename $o).log 2>&1
  local rc=$?; say "read $(basename $o) exit $rc $(( $(date +%s) - t ))s: $(grep -E '^\[' $L/v2_eval_$(basename $o).log | tr '\n' ' ' | cut -c1-200)"; }
AV=$C/${arm}_avg_s${s}_L3/ckpts/${arm}_avg_s${s}_L3_it$ITERS.pt  # ARM_mb's averaged checkpoint
avg_ckpt() {  # ARM_mb: the mean of stage L3's checkpoints at AVG_ITS and its last -> $AV; skipped when it exists
  [ -f $AV ] && return 0
  local K=$C/${arm}_s${s}_L3/ckpts fs=() f i
  for i in $AVG_ITS; do fs+=($K/${arm}_s${s}_L3_it${i}_rolling.pt); done; fs+=($(ck L3))
  for f in "${fs[@]}"; do [ -f $f ] || { say "$arm: no $f to average (an alarm iteration?): no averaged reads"; exit 1; }; done
  mkdir -p $(dirname $AV)
  # every floating-point tensor of "state" averaged (in float64, back to its dtype), the rest of the last checkpoint
  # kept, train_state dropped (pilot4.sh); written to a temporary file first, so a rerun never reads half a file
  local al=$L/v2_avg_${arm}_s$s.log
  CUDA_VISIBLE_DEVICES="" "$PY" - $AV "${fs[@]}" > $al 2>&1 <<'EOF' || { say "$arm: averaging failed (see $al)"; exit 1; }
import os, sys, torch
out, paths = sys.argv[1], sys.argv[2:]
cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
keys = list(cks[-1]["state"])
assert all(list(c["state"]) == keys for c in cks)
avg = {k: (sum(c["state"][k].double() for c in cks) / len(cks)).to(cks[-1]["state"][k].dtype)
       if cks[-1]["state"][k].is_floating_point() else cks[-1]["state"][k] for k in keys}
d = {k: v for k, v in cks[-1].items() if k != "train_state"}
d["state"] = avg
d["extra"] = dict(d.get("extra") or {}, averaged_from=paths)
torch.save(d, out + ".tmp")
os.replace(out + ".tmp", out)
print("averaged", len(cks), "checkpoints ->", out)
EOF
  say "averaged ${arm}_s${s}_L3 it ${AVG_ITS:+$AVG_ITS }$ITERS -> $AV"; }
CH173=$CH/candidate_v0.jsonl,$CH/candidate_v1.jsonl,$CH/batch3.jsonl
rsets() {  # SET -> its files: ARM_mini's and ARM_ckreads's reads (MINI_READS, CKR_READS)
  case $1 in
    l2test) echo $R/pools/curriculum/L2/test_1k.jsonl ;;  # the pools' held-out splits (in distribution: preflight.py
    l3test) echo $R/pools/curriculum/L3/test_1k.jsonl ;;  # checks the pools against them)
    t1320) echo $R/eval/matched/sets/t1320_n1000.jsonl ;; t2130) echo $R/eval/matched/sets/t2130_n1000.jsonl ;;
    charles173) echo $CH173 ;;
    l2train) echo $L2T ;; l3train) echo $L3T ;;  # a mini run's own training subsets
    l2test_sz) echo $SZ/L2_test_1k_sz.jsonl ;; l3test_sz) echo $SZ/L3_test_1k_sz.jsonl ;;  # the test_1k twins and
    l2test_om) echo $SZ/L2_test_1k_orig_matched.jsonl ;; l3test_om) echo $SZ/L3_test_1k_orig_matched.jsonl ;;  # the
                                                                                     # originals of those, in order
  esac; }
RDONE=" "  # the eval dirs reads_of has read (or tried) in this line: a read in two of its lists runs once
reads_of() {  # NAME CKPT READS [T]: plain reads (SET:K each; SMOKE=1: k 2) of CKPT, named after NAME, at T (0.8)
  local r k o; for r in $3; do k=${r#*:}; [ "$SMOKE" = 1 ] && k=2
    o=${r%:*}_$1_s${s}_plain_k$k${4:+_T${4//./}}  # (read1's name)
    case $RDONE in *" $o "*) continue ;; esac; RDONE="$RDONE$o "
    read1 ${r%:*} $(rsets ${r%:*}) $k plain 96 $1 $2 "" $4; done; }
if [ "$ckr" = l3 ]; then  # ARM_ckreads: no training; CKR_PANEL of $arm's L3 at CKR_ITS, CKR_FINAL of its last
  rec=$C/${job}_s$s.ckpts.txt; : > $rec  # the iteration asked for and the checkpoint read
  for i in $CKR_ITS; do  # a missing rolling checkpoint (an alarm iteration saves none): the nearest earlier one
    j=$i; while [ $j -gt 0 ] && [ ! -f $C/${arm}_s${s}_L3/ckpts/${arm}_s${s}_L3_it${j}_rolling.pt ]; do j=$((j - 1)); done
    [ $j -gt 0 ] || { say "$job: no rolling checkpoint at or before it$i: its reads skipped"; continue; }
    [ $j = $i ] || say "$job: no it$i rolling checkpoint (an alarm iteration): it$j's instead"
    cp=$C/${arm}_s${s}_L3/ckpts/${arm}_s${s}_L3_it${j}_rolling.pt; echo "it$i $cp" >> $rec
    reads_of ${arm}_it$j $cp "$CKR_PANEL"
  done
  echo "it$IT3 $(ck L3)" >> $rec; reads_of $arm $(ck L3) "$CKR_FINAL"
  say "grpo_v2 line done ($job seed $s)"; exit 0
elif [ "$ckr" = szpanel ]; then  # grpo_think_vs_mini_szpanel: SIZE_READS of the reference's L3-mini last
  reads_of $arm $(ck L3) "$SIZE_READS"
  say "grpo_v2 line done ($job seed $s)"; exit 0
elif [ "$ckr" = tpanel ]; then  # grpo_think_vs_mini_tpanel: MINI_PANEL of the reference's L3-mini last at each TPANEL_TS
  for t in $TPANEL_TS; do reads_of $arm $(ck L3) "$MINI_PANEL" $t; done  # (the T arms' names: _T07, _T13)
  say "grpo_v2 line done ($job seed $s)"; exit 0
elif [ -n "$ckr" ]; then  # ARM_l2_ckreads: $arm's L2 last, as ARM_L2; s1g3_ckreads: Stage-1 S1g3_sSEED, as s1g3
  if [ $ckr = l2 ]; then cp=$(ck L2); nm=${arm}_L2; else cp=$init0; nm=s1g3; fi  # (both checked above)
  reads_of $nm $cp "$CKR_PANEL"; reads_of $nm $cp "$MINI_READS"  # (a read in both: once)
  say "grpo_v2 line done ($job seed $s)"; exit 0
fi
stage=${method}_stage
if [ "$src" = "$arm" ]; then  # ARM_b, ARM_wb, ARM_$pil, ARM_kKNOB have no L2 of their own: L2CK is $src's (checked above)
  $stage ${arm}_s${s}_L2 $init0 $L2 $X2 || { say "${arm}_s${s}_L2 failed"; exit 1; }
  L2CK=$(ck L2)
fi
$stage ${arm}_s${s}_L3 $L2CK $T3 $X3 || { say "${arm}_s${s}_L3 failed"; exit 1; }
if [ -n "$mini" ]; then  # ARM_mini: MINI_READS of L3-mini's last checkpoint under $arm (never tb72); MINI_TRAIN and
  reads_of $arm $(ck L3) "$MINI_READS"  # MINI_PANEL at the training T (as above); MINI_PANEL of each MINI_ROLL one
  case $mk in ""|rr|pool250|pool500) reads_of $arm $(ck L3) "$MINI_TRAIN" ;; esac
  [ "$mk" = size ] && reads_of $arm $(ck L3) "$SIZE_READS"  # paired size reads in distribution
  [ -n "$MT" ] && reads_of $arm $(ck L3) "$MINI_PANEL" $MT
  for c in $MINI_ROLL; do  # under ARM_STAGEitIT
    st=${c%%:*}; i=${c#*:}; cp=$C/${arm}_s${s}_$st/ckpts/${arm}_s${s}_${st}_it${i}_rolling.pt
    [ -f $cp ] || { say "$arm: no $cp (an alarm iteration?): its reads skipped"; continue; }
    reads_of ${arm}_${st}it$i $cp "$MINI_PANEL"
  done
  say "grpo_v2 line done ($arm seed $s)"; exit 0
fi
if [ -n "$knob" ] && [ "${KNOB_FULL:-0}" != 1 ]; then  # ARM_mb_kKNOB, ARM_kKNOB: plain reads on 13-20, 21-30 and charles173 only (knob selection never sees tb72)
  read1 t1320 $R/eval/matched/sets/t1320_n1000.jsonl ${KS[0]} plain
  read1 t2130 $R/eval/matched/sets/t2130_n1000.jsonl ${KS[4]} plain
  read1 charles173 $CH173 ${KS[2]} plain
  say "grpo_v2 line done ($arm seed $s)"; exit 0
fi
for ra in plain logical; do
  read1 t1320 $R/eval/matched/sets/t1320_n1000.jsonl ${KS[0]} $ra
  read1 tb72 $TB/textbook_dev.jsonl,$TB/textbook_train.jsonl ${KS[1]} $ra
  read1 charles173 $CH173 ${KS[2]} $ra
done
if [ "$pil" = fbb ]; then  # ARM_fbb: the feedback read arm too (FB_MAXREJ retries per step), next to the logical reads
  read1 t1320 $R/eval/matched/sets/t1320_n1000.jsonl ${KS[0]} feedback 96 "" "" $FB_MAXREJ
  read1 charles173 $CH173 ${KS[2]} feedback 96 "" "" $FB_MAXREJ
fi
read1 hand14 $CH/hand_proved_14.jsonl ${KS[3]} plain $MAXSTEPS_H14
read1 t2130 $R/eval/matched/sets/t2130_n1000.jsonl ${KS[4]} plain
if [ "$pil" = mb ] && [ -z "$knob" ]; then  # ARM_mb: the averaged checkpoint, read plain under ARM_mb_avg (pilot4.sh's reads)
  avg_ckpt
  read1 t1320 $R/eval/matched/sets/t1320_n1000.jsonl ${KS[0]} plain 96 ${arm}_avg $AV
  read1 tb72 $TB/textbook_dev.jsonl,$TB/textbook_train.jsonl ${KS[1]} plain 96 ${arm}_avg $AV
  read1 charles173 $CH173 ${KS[2]} plain 96 ${arm}_avg $AV
  read1 t2130 $R/eval/matched/sets/t2130_n1000.jsonl ${KS[4]} plain 96 ${arm}_avg $AV
fi
say "grpo_v2 line done ($arm seed $s)"
