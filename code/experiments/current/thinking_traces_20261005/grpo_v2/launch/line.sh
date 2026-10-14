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
pil=""; case $arm in *_rfb) pil=rfb ;; *_ulb) pil=ulb ;; *_3pb) pil=3pb ;; *_pb) pil=pb ;; *_mb) pil=mb ;; esac  # first
case $arm in *_rfb_*|*_ulb_*|*_3pb_*)  # a pilot suffix with another after it
  say "$arm: _rfb, _ulb and _3pb (L3 pilots) do not combine with other suffixes: refusing"; exit 1 ;;
esac
case $arm in *_pb_*) say "$arm: _pb (an L3 pilot) does not combine with other suffixes: refusing"; exit 1 ;; esac
case $arm in *_mb_*) say "$arm: _mb (an L3 pilot) does not combine with other suffixes: refusing"; exit 1 ;; esac
core=${arm%_$pil}  # ARM of ARM_$pil
[ -n "$pil" ] && case $core in *_[hbcw]|*_wb|*_pb|*_mb)
  case $pil in pb|mb) say "$arm: _$pil (an L3 pilot) does not combine with other suffixes: refusing"; exit 1 ;; esac
  say "$arm: _rfb, _ulb and _3pb (L3 pilots) do not combine with other suffixes: refusing"; exit 1 ;;
esac
case ${arm}_ in *_[hb]_c_|*_[hb]_w_|*_[hb]_wb_|*_[cw]_[hb]_*|*_wb_[hb]_*)  # in either order
  say "$arm: _c, _w and _wb (lr schedules) do not combine with _h or _b: refusing"; exit 1 ;;
esac
sch=""; case $arm in *_c) sch=c ;; *_w) sch=w ;; *_wb) sch=wb ;; esac  # ARM_c / ARM_w / ARM_wb: parsed first
x=$arm; [ -n "$sch" ] && x=${arm%_$sch}
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
# L3 pilots (grpo.py's pilot options): set here only. ARM_rfb: refill up to 2 extra rounds, pass-rate filter 0.9;
# ARM_ulb: the unlikeliness reward at the paper's beta. ARM_3pb has no flags of its own (its extra targets below).
PILOT_RFB="--refill --refill_max 2 --max_pass 0.9"
PILOT_ULB="--unlikely_beta 0.25"
PILOT_PB="--unseen_prior mean"  # ARM_pb: the unseen-pair prior at the budget's mean pass rate
P3=""; case $pil in rfb) P3=$PILOT_RFB ;; ulb|mb) P3=$PILOT_ULB ;; pb) P3=$PILOT_PB ;; esac  # stage L3's pilot flags
AVG_ITS="25 30 35"  # ARM_mb: the L3 rolling checkpoints averaged with the last one (pilot4.sh's)
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
case $base in grpo_*) method=grpo ;; *) method=ei ;; esac
[ $method = ei ] && [ -n "$pil" ] && { say "$arm: _$pil is a GRPO L3 pilot, and $base is EI: refusing"; exit 1; }
[ $method = ei ] && [ -n "$sch" ] && { say "$arm: _$sch schedules the GRPO lr, and EI's fine-tune has its own (run $base): refusing"; exit 1; }
[ $method = ei ] && [ "$src" != "$arm" ] && { say "$arm: _b reruns a GRPO L3 stage, and $src is EI: refusing"; exit 1; }
[ $method = ei ] && [ "$base" != "$src" ] && { say "$arm: _h halves the GRPO lr, which EI does not use (run $base): refusing"; exit 1; }
if [ -n "$P3" ]; then  # ARM_rfb / ARM_ulb / ARM_pb need their grpo.py options in $WT (leon/grpo-v2-pilots, -prior)
  PA=pilot_options; PN="--refill, --unlikely_beta"; [ "$pil" = pb ] && { PA=UNSEEN_PRIORS; PN="--unseen_prior"; }
  "$PY" -c "import sys, code.thinking.grpo as g; sys.exit(0 if hasattr(g, '$PA') else 1)" > /dev/null 2>&1 \
    || { say "$arm: $WT's grpo.py has no pilot options ($PN): refusing"; exit 1; }
fi
case $base in *_v|*_vs|*_vs3) samp=logical ;; *) samp=plain ;; esac
case $base in *_vs|*_vs3) MAXREJ_TRAIN=0 ;; *) MAXREJ_TRAIN=10 ;; esac  # GRPO training only
X2=""; X3=""; XFRAC=${XFRAC:-0.33}  # extra GRPO targets per stage (--targets_extra)
case $base in *_vs3) X2=$R/pools/curriculum_3p/L2_3p.jsonl; X3=$R/pools/curriculum_3p/L3_3p.jsonl ;; esac
if [ "$pil" = 3pb ] || [ "$pil" = mb ]; then  # ARM_3pb, ARM_mb: _vs3's files (both for its extra-targets preflight)
  case $base in *_vs3) say "$arm: $base already trains on the 3-premise rows: refusing"; exit 1 ;; esac
  X2=$R/pools/curriculum_3p/L2_3p.jsonl; X3=$R/pools/curriculum_3p/L3_3p.jsonl
fi
for xf in $X2 $X3; do [ -f $xf ] || { say "$arm: no 3-premise pool file $xf (runs/pools/curriculum_3p): refusing"; exit 1; }; done
if [ $kind = think ]; then EFF="--effort --budgets 0,64,256"; RB=64; else EFF="--budgets 0"; RB=0; fi
ITERS=39; PROMPTS=256; GROUP=16; K=32; BATCH=1024; FT=""; CAN="--canary_n 1000"; KLN=""; LIM=""; KS=(16 64 64 64 8)
DEV=$g; MAXSTEPS_H14=160
MAXSTEPS=96  # training step cap, GRPO and EI alike (eval's value; v1 used the default 48, which 13-15-line
             # problems exceed in 3-6% of B64 attempts: capped attempts would raise false collapse alarms)
if [ "$SMOKE" = 1 ]; then
  C=$R/grpo_v2/smoke; EV=$R/eval/v2_smoke; WD=$C/watchdog.log; L=$C/logs; mkdir -p $C $L
  head -n 8 $L2 > $C/L2_head.jsonl; head -n 8 $L3 > $C/L3_head.jsonl 2>/dev/null || cp $C/L2_head.jsonl $C/L3_head.jsonl
  [ -s $C/L3_head.jsonl ] || cp $C/L2_head.jsonl $C/L3_head.jsonl
  L2=$C/L2_head.jsonl; L3=$C/L3_head.jsonl
  if [ -n "$X2" ]; then head -n 8 $X2 > $C/X2_head.jsonl; head -n 8 $X3 > $C/X3_head.jsonl; X2=$C/X2_head.jsonl; X3=$C/X3_head.jsonl; fi
  ITERS=2; PROMPTS=4; GROUP=4; K=2; BATCH=16; FT="--ft_steps 3 --ft_recs 8 --retain 64"
  CAN="--canary_n 40"; KLN="--kl_anchor_n 16"; LIM="--limit 2"; KS=(2 2 2 2 2); DEV=""
  AVG_ITS=""  # a 2-iteration stage keeps no rolling checkpoint: ARM_mb's mean is of the last one alone
  waitfree() { :; }  # CPU only
fi
if [ "$src" != "$arm" ]; then  # ARM_b, ARM_wb, ARM_$pil: $src's finished L2, from a run no rollback touched
  L2CK=$C/${src}_s${s}_L2/ckpts/${src}_s${s}_L2_it$ITERS.pt
  [ -f $L2CK ] || { say "$arm: no L2 checkpoint of $src ($L2CK): refusing"; exit 1; }
  gl=$(grep -E '^guard it' $L/v2_${src}_s${s}_L2.log 2>/dev/null | tail -1)
  case $gl in *"lr x1 (0 rollbacks)"*) ;; *)
    say "$arm: $src's L2 log ends '${gl:-no guard line}', not lr x1 (0 rollbacks): rerun the whole line: refusing"; exit 1 ;;
  esac
fi
mkdir -p $C $EV
if [ "$SMOKE" != 1 ]; then
  "$PY" -m code.experiments.current.thinking_traces_20261005.grpo_v2.preflight $L2 $L3 > $C/preflight_${arm}_s$s.log 2>&1 \
    || { say "preflight failed (see $C/preflight_${arm}_s$s.log)"; exit 1; }
  if [ -n "$X2" ]; then  # the extra rows are shorter than the bands by construction: only the band lines may fail
    xl=$C/preflight_${arm}_s${s}_extra.log
    "$PY" -m code.experiments.current.thinking_traces_20261005.grpo_v2.preflight $X2 $X3 > $xl 2>&1
    if grep -vE '^  BAD: [0-9]+ rows outside' $xl | grep -qE '^(BAD|MISSING|\(skip)' || [ $(grep -c '^ok ' $xl) -lt 22 ]; then
      say "extra-targets preflight failed (see $xl)"; exit 1; fi
  fi
fi
[ -f $init0 ] || { say "no Stage-1 init $init0"; exit 1; }
git -C $WT rev-parse HEAD > $C/${arm}_s$s.base_commit.txt
watch() {  # NAME PID LOG EXPECTED_H
  [ "$SMOKE" = 1 ] && return 0
  python3 $R/watchdog.py $1:$2:$g:$3:$4 >> $WD 2>&1 & }
grpo_stage() {  # NAME INIT TARGETS [EXTRA_TARGETS] -> ckpts/NAME_it$ITERS.pt
  local n=$1 O=$C/$1 last=$C/$1/ckpts/${1}_it$ITERS.pt
  [ -f $last ] && return 0; waitfree $g
  local GUIDE=""; [ $samp = logical ] && GUIDE="--arm logical --max_rej $MAXREJ_TRAIN --rej_penalty $P"
  local XT=""; [ -n "$4" ] && XT="--targets_extra $4 --targets_extra_frac $XFRAC"
  local SC=$S3; case $n in *_L2) SC=$S2 ;; esac  # the stage's lr schedule flags (none: constant)
  local PF=$P3; case $n in *_L2) PF="" ;; esac  # the stage's pilot flags (L3 only)
  ( CUDA_VISIBLE_DEVICES=$DEV timeout 21600 "$PY" -m code.thinking.grpo --init $2 --name $n --out $O --targets $3 \
      --iters $ITERS --prompts $PROMPTS --group $GROUP --batch $BATCH --lr $LR $SC $PF --overlong fail --save_every $ITERS \
      --max_steps $MAXSTEPS \
      --seed $s --keep_rolling 100 $EFF --canary $R/pools/canary_long.jsonl $CAN --canary_every 1 \
      --rollback_on_alarm --max_rollbacks 3 --ban_restore_check --ban_stop 0.005 \
      --kl_anchor_coef 0.02 --kl_anchor_ref $2 $KLN --monitor $GUIDE $XT \
      > $L/v2_$n.log 2>&1 ) & local p=$!
  sleep 5; say "launched GRPO $n ($samp${GUIDE:+, P $P}${XT:+, extra $(basename $4) x $XFRAC}$LRN${SC:+, $SC}${PF:+, $PF}) on GPU $g (pid $p)"; watch $n $p $L/v2_$n.log 3.5
  wait $p; say "GRPO $n exit $?: $(grep -E '^it [0-9]+:' $L/v2_$n.log | tail -1 | cut -c1-150)"
  [ -f $last ]; }
ei_stage() {  # NAME INIT TARGETS -> ckpts/NAME_r1.pt
  local n=$1 O=$C/$1 last=$C/$1/ckpts/${1}_r1.pt
  [ -f $last ] && return 0; waitfree $g
  ( CUDA_VISIBLE_DEVICES=$DEV timeout 21600 "$PY" -m code.thinking.ei --init $2 --name $n --out $O --targets $3 \
      --train $R/data/$pkl --rounds 1 --k $K --temperature 0.8 --arm $samp --max_rej 10 --batch $BATCH --seed $s \
      --max_steps $MAXSTEPS \
      --select shortest $EFF $FT > $L/v2_$n.log 2>&1 ) & local p=$!
  sleep 5; say "launched EI $n ($samp) on GPU $g (pid $p)"; watch $n $p $L/v2_$n.log 4
  wait $p; say "EI $n exit $?: $(tail -1 $L/v2_$n.log | cut -c1-150)"
  [ -f $last ]; }
ck() { if [ $method = grpo ]; then echo $C/${arm}_s${s}_$1/ckpts/${arm}_s${s}_$1_it$ITERS.pt; else echo $C/${arm}_s${s}_$1/ckpts/${arm}_s${s}_$1_r1.pt; fi; }
read1() {  # SETNAME SETS K READARM [MAXSTEPS [ARMNAME CKPT]]  (default: $arm and its L3 checkpoint)
  local o=$EV/$1_${6:-$arm}_s${s}_$4_k$3 ms=${5:-96} cp=${7:-$(ck L3)}
  [ -f $o/rows_$4_B$RB.jsonl ] && return 0; waitfree $g; local t=$(date +%s); rm -rf $o
  CUDA_VISIBLE_DEVICES=$DEV timeout 9000 "$PY" -m code.thinking.eval --ckpt $cp --sets $2 --out $o --k $3 \
    --budgets $RB --arms $4 --batch $BATCH --seed 0 --temperature 0.8 --max_action 512 --max_steps $ms --max_rej 10 \
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
stage=${method}_stage
if [ "$src" = "$arm" ]; then  # ARM_b, ARM_wb, ARM_$pil have no L2 of their own: L2CK is $src's (checked above)
  $stage ${arm}_s${s}_L2 $init0 $L2 $X2 || { say "${arm}_s${s}_L2 failed"; exit 1; }
  L2CK=$(ck L2)
fi
$stage ${arm}_s${s}_L3 $L2CK $L3 $X3 || { say "${arm}_s${s}_L3 failed"; exit 1; }
CH173=$CH/candidate_v0.jsonl,$CH/candidate_v1.jsonl,$CH/batch3.jsonl
for ra in plain logical; do
  read1 t1320 $R/eval/matched/sets/t1320_n1000.jsonl ${KS[0]} $ra
  read1 tb72 $TB/textbook_dev.jsonl,$TB/textbook_train.jsonl ${KS[1]} $ra
  read1 charles173 $CH173 ${KS[2]} $ra
done
read1 hand14 $CH/hand_proved_14.jsonl ${KS[3]} plain $MAXSTEPS_H14
read1 t2130 $R/eval/matched/sets/t2130_n1000.jsonl ${KS[4]} plain
if [ "$pil" = mb ]; then  # ARM_mb: the averaged checkpoint, read plain under ARM_mb_avg (pilot4.sh's reads)
  avg_ckpt
  read1 t1320 $R/eval/matched/sets/t1320_n1000.jsonl ${KS[0]} plain 96 ${arm}_avg $AV
  read1 tb72 $TB/textbook_dev.jsonl,$TB/textbook_train.jsonl ${KS[1]} plain 96 ${arm}_avg $AV
  read1 charles173 $CH173 ${KS[2]} plain 96 ${arm}_avg $AV
  read1 t2130 $R/eval/matched/sets/t2130_n1000.jsonl ${KS[4]} plain 96 ${arm}_avg $AV
fi
say "grpo_v2 line done ($arm seed $s)"
