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
#        sampled tokens of a success; grpo.py's length-only groups); kl1 --kl_anchor_coef 0.1 in L3-mini (0; L2-mini
#        keeps 0.02: the full-scale kl1 pilot); g64 --prompts 64 --group 64 (the same attempts per iteration). After each
#        stage its ALARM lines and rollbacks; then plain reads only (never textbook72): MINI_READS of L3-mini's last
#        checkpoint, L2 and L3 test_1k k8, t1320 k8, t2130 k8, charles173 k64; with MINI_ROLL (env, none by default:
#        STAGE:IT, e.g. "L2:4 L3:2 L3:6") also the MINI_PANEL reads of those rolling checkpoints. Refused: another base
#        arm, another knob, anything after it, KNOB_FULL=1.
#        ARM_mb_kKNOB_ckreads (2026-10-16; grpo_think_vs_mb_kkl0_ckreads asked for): no training; plain reads CKR_READS
#        (L2 and L3 test_1k k8, t1320 k16, t2130 k8, charles173 k64) of the finished knob pilot's L3 at CKR_ITS: it0 (its
#        init, the base arm's L2 checkpoint), the it10 / it20 / it30 rolling checkpoints (one missing: its reads skipped)
#        and the last, under ARM_mb_kKNOB_itN (the last: ARM_mb_kKNOB, so its pilot reads are skipped); HEAD in
#        ARM_mb_kKNOB_sSEED.ckreads_commit.txt. Refused for other arms, i78, without the finished L3, KNOB_FULL=1.
# One stage = 160k sampled attempts: 39 GRPO iterations x 256 targets x 16, or one EI round at k 32 over 5,000.
# GRPO keeps tonight's settings and safeguards (grpo_think/line.sh): lr 1.5e-4, --overlong fail, canary + rollback,
# KL anchor 0.02 to each stage's init, effort budgets 0,64,256 for thinking; plus --monitor. EI as seed2/line.sh.
# Think ban (2026-10-11, every GRPO stage): --ban_restore_check (a think ban alarm alone does not roll back when the
# checkpoint it would restore is over the limit too) and --ban_stop 0.005 (the stage ends, exit 3, and the line stops).
# Reads after the last stage on the same GPU (one queue per GPU): t1320 (986 valid; 16-20 is a min_lines filter of
# the rows) k16, textbook72 k64, charles173 (Charles's 2026-10-02 release: candidate_v0 + v1 + batch3) k64, each with
# the plain and the guided (logical) arm; hand_proved_14 k64 and a 1,000-item sample of test_21_30 k8, plain only.
# Thinking models read at B64, no-think at B0. Rows under runs/eval/v2/<set>_<arm>_s<seed>_<readarm>_k<k>/.
# env: POOL_L2, POOL_L3 (default runs/pools/curriculum/L{2,3}/pool.jsonl), REJ_PENALTY (0.1),
#      SMOKE=1: a tiny CPU dry run of the whole line (2 iterations, k 2, 2 eval problems) under runs/grpo_v2/smoke/.
source /scratch/cowifornia/nd-rl/think/runs/followup_20261007/lib.sh
WT=/scratch/cowifornia/nd-rl/think/nd-rl-v2
cd $WT || exit 1
[ "$(git -C $WT rev-parse --abbrev-ref HEAD)" = leon/grpo-v2 ] || { say "$WT is not on leon/grpo-v2: refusing"; exit 1; }
PY=/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python
"$PY" -c "import code.thinking, sys; sys.exit(0 if code.thinking.__file__.startswith('$WT/') else 1)" \
  || { say "code.thinking does not resolve to $WT: refusing"; exit 1; }
g=$1; s=$2; arm=$3
[ -n "$g" ] && [ -n "$s" ] && [ -n "$arm" ] || { echo "usage: line.sh GPU SEED ARM"; exit 2; }
job=$arm; ckr=""; case $arm in *_ckreads) ckr=1; arm=${arm%_ckreads} ;; esac  # ARM_ckreads: $arm is the arm it reads
knob=""; pa=$arm  # ARM_mb_kKNOB, ARM_kKNOB (EI): the suffixes below are parsed on ARM_mb / ARM ($pa); $arm names every output
mini=""; mk=""  # ARM_mini[_kKNOB]: parsed first, then as grpo_think_vs_mb ($pa: L3-mini's recipe), its knob in $mk
case $arm in *_mini|*_mini_*) mini=1; mb=${arm%%_mini*}; mk=${arm#"${mb}"_mini}
  [ "$mb" = grpo_think_vs ] || { say "$arm: _mini runs the primary recipe, grpo_think_vs's: refusing"; exit 1; }
  case $mk in ""|_krr|_kt07|_kt085|_kt115|_kt13|_klp[0-9][0-9]|_klpm[0-9][0-9]|_kkl1|_kg64) mk=${mk#_k} ;; *)
    say "$arm: _mini takes one knob _kKNOB of rr t07 t085 t115 t13 lpNN lpmNN kl1 g64, with nothing after it: refusing"; exit 1 ;;
  esac
  case $mk in lp00|lpm00) say "$arm: _k$mk is no length penalty (grpo_think_vs_mini): refusing"; exit 1 ;; esac
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
[ -n "$mini" ] && src=$arm  # ARM_mini: both stages (L2-mini without schedule flags, as grpo_think_vs's L2)
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
MINI_READS="l2test:8 l3test:8 t1320:8 t2130:8 charles173:64"  # ARM_mini: the plain reads (SET:K, rsets) of L3-mini's last
MINI_PANEL="l3test:8 t1320:8"  # ... and of each MINI_ROLL checkpoint, a smaller fixed panel (a hook for time-resolved reads)
MINI_ROLL=${MINI_ROLL:-}  # env: STAGE:IT rolling checkpoints of the mini stages (--save_rolling 2) to read; none by default
CKR_READS="l2test:8 l3test:8 t1320:16 t2130:8 charles173:64"  # ARM_ckreads: the knob pilots' t1320 / t2130 / ch173 k
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
if [ -n "$ckr" ]; then  # ARM_ckreads: a GRPO knob pilot's 39-iteration L3
  { [ -n "$knob" ] && [ $method = grpo ] && [ -z "$mini" ]; } \
    || { say "$job: _ckreads reads a GRPO knob pilot's L3 checkpoints (ARM_mb_kKNOB_ckreads): refusing"; exit 1; }
  [ "$knob" = i78 ] && { say "$job: _ckreads reads 39-iteration L3s (CKR_ITS), and _ki78's has 78: refusing"; exit 1; }
  [ "${KNOB_FULL:-0}" = 1 ] && { say "$job: KNOB_FULL=1 is the knob pilots' (_ckreads has its own reads): refusing"; exit 1; }
fi
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
MINI=$R/pools/mini; MP=$MINI/preflight_pass.txt  # ARM_mini's targets and their pass record, grpo_v2/build_mini.py's
if [ -n "$mini" ]; then  # the L2 and L3 subsets; L3-mini's extra targets the 3-premise subset (L2-mini has none)
  L2=$MINI/L2_mini.jsonl; L3=$MINI/L3_mini.jsonl; X2=""; X3=$MINI/L3_3p_mini.jsonl
  for xf in $L2 $L3 $X3; do  # each as its pass record ('PREFLIGHT PASSED <md5> <file>') names it
    [ -f $xf ] || { say "$arm: no mini subset $xf (grpo_v2/build_mini.py builds it): refusing"; exit 1; }
    grep -qx "PREFLIGHT PASSED $(md5sum < $xf | cut -c1-32) $(basename $xf)" $MP 2> /dev/null \
      || { say "$arm: no preflight pass record of $xf as it is ($MP; grpo_v2/build_mini.py): refusing"; exit 1; }
  done
fi
for xf in $X2 $X3; do [ -f $xf ] || { say "$arm: no 3-premise pool file $xf (runs/pools/curriculum_3p): refusing"; exit 1; }; done
M3=$R/pools/curriculum_3p/L3_mix_ei.jsonl; M3P=$R/pools/curriculum_3p/L3_mix_ei.preflight_pass.txt  # EI ARM_k3p's
if [ "$knob" = 3p ]; then  # targets and their pass record ('PREFLIGHT PASSED <md5>'), both from grpo_v2/build_mix_ei.py
  [ -f $M3 ] || { say "$arm: no 3-premise mix $M3 (grpo_v2/build_mix_ei.py builds it): refusing"; exit 1; }
  grep -qx "PREFLIGHT PASSED $(md5sum < $M3 | cut -c1-32)" $M3P 2> /dev/null \
    || { say "$arm: no preflight pass record of $M3 as it is ($M3P; grpo_v2/build_mix_ei.py): refusing"; exit 1; }
fi
if [ $kind = think ]; then EFF="--effort --budgets 0,64,256"; RB=64; else EFF="--budgets 0"; RB=0; fi
ITERS=39; PROMPTS=256; GROUP=16; K=32; BATCH=1024; FT=""; CAN="--canary_n 1000"; KLN=""; LIM=""; KS=(16 64 64 64 8)
[ -n "$mini" ] && ITERS=$MINI_ITERS  # ARM_mini: both stages
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
  case $mk in  # ARM_mini_kKNOB, both stages unless said
    rr) KD="nothing changed" ;;  # the reference's two stages again: the noise control
    t07) KF="--temperature 0.7"; KD=$KF ;;  # training only (grpo.py's default 1.0; the reads stay at 0.8)
    t085) KF="--temperature 0.85"; KD=$KF ;;
    t115) KF="--temperature 1.15"; KD=$KF ;;
    t13) KF="--temperature 1.3"; KD=$KF ;;
    lp[0-9][0-9]|lpm[0-9][0-9])  # per 1,000 sampled tokens of a success, NN hundredths (lpm: negative, a length bonus)
      KF="--len_penalty $([ "${mk#lpm}" = "$mk" ] || echo -)$(awk -v n=$((10#${mk: -2})) 'BEGIN { print n / 100 }')"; KD=$KF ;;
    kl1) KL3=0.1; KD="--kl_anchor_coef $KL3 in L3" ;;  # L3-mini only, as the full-scale kl1 pilot (L2 keeps KLC)
    g64) if [ "$SMOKE" = 1 ]; then PROMPTS=2; GROUP=8; else PROMPTS=64; GROUP=64; fi  # the same attempts per iteration
         KD="--prompts $PROMPTS --group $GROUP" ;;
  esac
  [ -n "$mk" ] && KD="knob $mk: $KD"
fi
CKR_ITS="0 10 20 30 $IT3"  # ARM_ckreads: the L3 iterations read (0: its init; the last: the finished L3)
if [ "$src" != "$arm" ]; then  # ARM_b, ARM_wb, ARM_$pil, ARM_kKNOB: $src's finished L2, from a run no rollback touched
  L2CK=$C/${src}_s${s}_L2/ckpts/${src}_s${s}_L2_it$ITERS.pt
  [ $method = ei ] && L2CK=$C/${src}_s${s}_L2/ckpts/${src}_s${s}_L2_r1.pt  # EI ARM_kKNOB: its one round's
  [ -f $L2CK ] || { say "$arm: no L2 checkpoint of $src ($L2CK): refusing"; exit 1; }
  gl=$(grep -E '^guard it' $L/v2_${src}_s${s}_L2.log 2>/dev/null | tail -1)
  [ $method = ei ] || case $gl in *"lr x1 (0 rollbacks)"*) ;; *)  # (EI has no guard and no rollbacks)
    say "$arm: $src's L2 log ends '${gl:-no guard line}', not lr x1 (0 rollbacks): rerun the whole line: refusing"; exit 1 ;;
  esac
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
[ -n "$ckr" ] && bc=ckreads_commit  # and ARM_ckreads, the arm it reads
git -C $WT rev-parse HEAD > $C/${arm}_s$s.$bc.txt
watch() {  # NAME PID LOG EXPECTED_H
  [ "$SMOKE" = 1 ] && return 0
  python3 $R/watchdog.py $1:$2:$g:$3:$4 >> $WD 2>&1 & }
grpo_stage() {  # NAME INIT TARGETS [EXTRA_TARGETS] -> ckpts/NAME_it$ITERS.pt (stage L3: _it$IT3)
  local n=$1 O=$C/$1 it=$ITERS to=$TO2 wh=$WH2 kl=$KLC; case $1 in *_L3) it=$IT3; to=$TO3; wh=$WH3; kl=${KL3:-$KLC} ;; esac
  local last=$C/$1/ckpts/${1}_it$it.pt
  [ -f $last ] && return 0; waitfree $g
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
  wait $p; say "GRPO $n exit $?: $(grep -E '^it [0-9]+:' $L/v2_$n.log | tail -1 | cut -c1-150)"
  [ -n "$mini" ] && stage_alarms $n $L/v2_$n.log
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
read1() {  # SETNAME SETS K READARM [MAXSTEPS [ARMNAME CKPT [MAXREJ]]]  (default: $arm, its L3 checkpoint, 10)
  local o=$EV/$1_${6:-$arm}_s${s}_$4_k$3 ms=${5:-96} cp=${7:-$(ck L3)} mr=${8:-10}
  [ -f $o/rows_$4_B$RB.jsonl ] && return 0; waitfree $g; local t=$(date +%s); rm -rf $o
  CUDA_VISIBLE_DEVICES=$DEV timeout 9000 "$PY" -m code.thinking.eval --ckpt $cp --sets $2 --out $o --k $3 \
    --budgets $RB --arms $4 --batch $BATCH --seed 0 --temperature 0.8 --max_action 512 --max_steps $ms --max_rej $mr \
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
  esac; }
reads_of() {  # NAME CKPT READS: plain reads (SET:K each; SMOKE=1: k 2) of CKPT, named after NAME
  local r k; for r in $3; do k=${r#*:}; [ "$SMOKE" = 1 ] && k=2; read1 ${r%:*} $(rsets ${r%:*}) $k plain 96 $1 $2; done; }
if [ -n "$ckr" ]; then  # ARM_ckreads: no training; CKR_READS of $arm's L3 at each of CKR_ITS
  [ -f $(ck L3) ] || { say "$job: no finished L3 of $arm ($(ck L3)) to read: refusing"; exit 1; }
  for i in $CKR_ITS; do  # 0: the L3's init ($src's L2); the last: $arm's own names (its pilot reads are there)
    case $i in 0) cp=$L2CK; nm=${arm}_it0 ;; $IT3) cp=$(ck L3); nm=$arm ;;
      *) cp=$C/${arm}_s${s}_L3/ckpts/${arm}_s${s}_L3_it${i}_rolling.pt; nm=${arm}_it$i ;; esac
    [ -f $cp ] || { say "$job: no $cp (an alarm iteration?): its reads skipped"; continue; }
    reads_of $nm $cp "$CKR_READS"
  done
  say "grpo_v2 line done ($job seed $s)"; exit 0
fi
stage=${method}_stage
if [ "$src" = "$arm" ]; then  # ARM_b, ARM_wb, ARM_$pil, ARM_kKNOB have no L2 of their own: L2CK is $src's (checked above)
  $stage ${arm}_s${s}_L2 $init0 $L2 $X2 || { say "${arm}_s${s}_L2 failed"; exit 1; }
  L2CK=$(ck L2)
fi
$stage ${arm}_s${s}_L3 $L2CK $T3 $X3 || { say "${arm}_s${s}_L3 failed"; exit 1; }
if [ -n "$mini" ]; then  # ARM_mini: MINI_READS of L3-mini's last checkpoint under $arm (never tb72); MINI_PANEL of
  reads_of $arm $(ck L3) "$MINI_READS"  # each MINI_ROLL one under ARM_STAGEitIT
  for c in $MINI_ROLL; do
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
