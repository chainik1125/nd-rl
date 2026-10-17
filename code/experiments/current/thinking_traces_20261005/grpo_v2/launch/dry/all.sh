#!/bin/bash
# all.sh: dry-run every GRPO v2 arm (the L3 pilots, ARM_mb, ARM_fbb, ARM_mb_kKNOB, EI ARM_kKNOB, the mini runs and ARM_ckreads too), seeds 0-2, normal and SMOKE=1, plus the refusals and queue.sh; results under
# $H/res/. No python, no GPU (make.sh's stubs). Then check.py compares the records with the transcript evidence.
#   bash make.sh [DIR] && bash all.sh && python3 check.py
H=${DRY_HOME:-/scratch/cowifornia/nd-rl/think/dry}  # outputs and the built harness
D=$(cd "$(dirname "$0")" && pwd)  # this script's directory (run.sh)
BASE=(grpo_think grpo_think_v grpo_think_vs grpo_nothink grpo_nothink_v grpo_nothink_vs grpo_nothink_vs3
      ei_think ei_think_v ei_nothink ei_nothink_v grpo_think_vs3)
HARMS=(grpo_think_h grpo_think_v_h grpo_think_vs_h grpo_nothink_h grpo_nothink_v_h grpo_nothink_vs_h grpo_nothink_vs3_h
       grpo_think_vs3_h)
GRPO=(grpo_think grpo_think_v grpo_think_vs grpo_nothink grpo_nothink_v grpo_nothink_vs grpo_nothink_vs3 grpo_think_vs3
      "${HARMS[@]}")
EIH=(ei_think_h ei_think_v_h ei_nothink_h ei_nothink_v_h)
EIB=(ei_think_b ei_think_v_b ei_nothink_b ei_nothink_v_b ei_think_h_b)
# the lr schedules (2026-10-13): ARM_c, ARM_w for every GRPO base arm; ARM_wb needs the base arm's L2 (a setup)
GBASE=(grpo_think grpo_think_v grpo_think_vs grpo_nothink grpo_nothink_v grpo_nothink_vs grpo_nothink_vs3 grpo_think_vs3)
CARMS=("${GBASE[@]/%/_c}"); WARMS=("${GBASE[@]/%/_w}"); WBARMS=("${GBASE[@]/%/_wb}")
EIS=(ei_think_c ei_think_v_c ei_nothink_c ei_nothink_v_c ei_think_w ei_think_v_w ei_nothink_w ei_nothink_v_w
     ei_think_wb ei_think_v_wb ei_nothink_wb ei_nothink_v_wb)
COMBO=(grpo_think_c_h grpo_think_c_b grpo_think_h_c grpo_think_b_c grpo_think_vs_w_h grpo_think_vs_w_b grpo_think_vs_h_w
       grpo_think_vs_wb_b grpo_think_vs_h_wb grpo_think_vs_b_wb grpo_think_c_h_b grpo_think_vs_h_b_w ei_think_c_h)
# the L3 pilots (2026-10-13): ARM_rfb, ARM_ulb for every GRPO base arm, ARM_3pb for those without the 3-premise rows; each
# needs the base arm's L2 (a setup)
PARMS=("${GBASE[@]/%/_rfb}" "${GBASE[@]/%/_ulb}" grpo_think_3pb grpo_think_v_3pb grpo_think_vs_3pb grpo_nothink_3pb
       grpo_nothink_v_3pb grpo_nothink_vs_3pb "${GBASE[@]/%/_pb}")
# ARM_mb (2026-10-14), every GRPO base arm but the _vs3 ones; its setup also holds the L3 rolling checkpoints it averages
MBARMS=(grpo_think_mb grpo_think_v_mb grpo_think_vs_mb grpo_nothink_mb grpo_nothink_v_mb grpo_nothink_vs_mb)
PREF=(ei_think_rfb ei_think_v_ulb ei_nothink_3pb ei_nothink_v_rfb ei_think_h_ulb
      grpo_think_h_rfb grpo_think_rfb_h grpo_think_b_ulb grpo_think_ulb_b grpo_think_c_rfb grpo_think_rfb_c
      grpo_think_vs_w_3pb grpo_think_vs_3pb_wb grpo_think_wb_ulb grpo_think_rfb_ulb grpo_think_vs_3pb_rfb grpo_think_ulb_3pb
      grpo_think_h_b_rfb grpo_think_vs3_3pb grpo_nothink_vs3_3pb
      ei_think_pb ei_nothink_v_pb grpo_think_h_pb grpo_think_pb_h grpo_think_b_pb grpo_think_pb_rfb grpo_think_rfb_pb
      grpo_think_vs_pb_3pb grpo_think_c_pb grpo_think_wb_pb  # ARM_pb (2026-10-13): EI, other suffixes
      ei_think_mb ei_nothink_v_mb grpo_think_h_mb grpo_think_mb_h grpo_think_wb_mb grpo_think_mb_ulb grpo_think_ulb_mb
      grpo_think_vs_mb_pb grpo_think_pb_mb grpo_think_vs3_mb grpo_nothink_vs3_mb)  # ARM_mb (2026-10-14)
# ARM_fbb (2026-10-14): the thinking verifier arms, each from its base arm's L2 (a setup); refused for EI, no-think and
# plain-sampling arms, and with another suffix in either order
FBARMS=(grpo_think_v_fbb grpo_think_vs_fbb grpo_think_vs3_fbb)
FBREF=(ei_think_v_fbb ei_think_fbb grpo_nothink_vs_fbb grpo_nothink_v_fbb grpo_nothink_fbb grpo_think_fbb grpo_think_h_fbb
       grpo_think_vs_h_fbb grpo_think_vs_fbb_h grpo_think_vs_b_fbb grpo_think_vs_fbb_b grpo_think_vs_c_fbb grpo_think_vs_fbb_c
       grpo_think_vs_wb_fbb grpo_think_vs_fbb_w grpo_think_vs_rfb_fbb grpo_think_vs_fbb_rfb grpo_think_vs_pb_fbb
       grpo_think_vs_fbb_pb grpo_think_vs_3pb_fbb grpo_think_vs_fbb_3pb grpo_think_vs_ulb_fbb grpo_think_vs_fbb_ulb
       grpo_think_vs_mb_fbb grpo_think_vs_fbb_mb grpo_think_vs_fbb_fbb)
# ARM_mb_kKNOB (2026-10-15): every knob of grpo_think_vs_mb (the ones asked for), and some on other ARM_mb arms, each from
# its base arm's L2 (a setup); refused: another knob or a suffix after it, another suffix before _mb, EI, _vs3, _kp3 for a
# plain-sampling arm, a knob without _mb, as ARM_mb's L2 and 3-premise checks
KNOBS=(x66 g64 t12 kl0 kl1 ch4 p3 i78 rr)
KARMS=("${KNOBS[@]/#/grpo_think_vs_mb_k}" grpo_think_mb_kg64 grpo_think_v_mb_kp3 grpo_nothink_vs_mb_kt12 grpo_nothink_mb_ki78
       grpo_nothink_v_mb_kx66 grpo_think_mb_krr grpo_nothink_vs_mb_kkl1)
KREF=(grpo_think_vs_mb_kfoo grpo_think_vs_mb_k grpo_think_vs_mb_kx66_h grpo_think_vs_mb_kx66_krr grpo_think_vs_mb_kmb
      grpo_think_vs_mb_krr_mb grpo_think_vs_mb_kt12_b grpo_think_vs_h_mb_kx66 grpo_think_vs_wb_mb_ki78 grpo_think_vs_mb_mb_krr
      grpo_think_vs_ulb_mb_krr grpo_think_vs_pb_mb_kt12 grpo_think_vs_fbb_mb_kkl0 ei_think_mb_krr ei_nothink_v_mb_kg64
      grpo_think_vs3_mb_kx66 grpo_think_mb_kp3 grpo_nothink_mb_kp3 grpo_think_vs_kx66 grpo_think_vs_fbb_kx66
      grpo_think_vs_ulb_kp3)
# EI ARM_kKNOB (2026-10-16): every knob of ei_think_v (the ones asked for), and some on the other EI arms, each from its
# base arm's L2 checkpoint (a setup; ARM_k3p's also holds the mix and its pass record); refused: an unknown knob or a
# suffix after it, a suffix before it, a GRPO knob on an EI arm, an EI knob on a GRPO arm, an EI arm with _mb_k
EKNOBS=(rr 3p ret0 t10 r2)
EKARMS=("${EKNOBS[@]/#/ei_think_v_k}" ei_think_kr2 ei_think_k3p ei_nothink_v_kret0 ei_nothink_kt10 ei_nothink_v_krr)
EKREF=(ei_think_v_kfoo ei_think_v_k ei_think_v_kx66 ei_think_v_kp3 ei_think_v_k3p_h ei_think_v_kret0_krr ei_think_v_kt10_b
       ei_think_v_kr2_k3p ei_think_v_h_krr ei_think_v_b_k3p ei_think_v_c_kr2 ei_think_v_wb_kt10 ei_think_v_rfb_krr
       ei_think_v_fbb_kret0 ei_think_v_mb_k3p ei_nothink_v_mb_kret0 grpo_think_v_k3p grpo_think_vs_kret0 grpo_nothink_v_kt10
       grpo_think_kr2 grpo_think_vs_krr grpo_nothink_vs3_k3p grpo_think_vs_mb_kret0)
# the mini runs (2026-10-16): grpo_think_vs_mini and each of its knobs, with the mini subsets and their pass record (a
# setup; the forks kl1 and g64 also the reference's finished L2-mini); refused: another base arm, a suffix before
# _mini, another knob or anything after it, size, KNOB_FULL=1 (MREF), a missing subset, no pass record, a stale one, a
# fork whose reference L2-mini rolled back or never finishes. ARM_ckreads: a full L3 run's checkpoints, a base arm's L2
# (ARM_l2_ckreads), the Stage-1 base (s1g3_ckreads), each with a setup; refused (CKREF) for other arms, KNOB_FULL=1,
# without the checkpoints
MKNOBS=(rr t07 t085 t115 t13 lpm10 lp20 lp50 lp05 g64b pool250 pool500 lossact lossthink advpos advneg)
MARMS=(grpo_think_vs_mini "${MKNOBS[@]/#/grpo_think_vs_mini_k}")
MFORKS=(grpo_think_vs_mini_kkl1 grpo_think_vs_mini_kg64)
MREF=(grpo_think_mini grpo_nothink_vs_mini grpo_think_vs_h_mini grpo_think_vs_mb_mini grpo_think_vs_mb_kkl0_mini ei_think_v_mini
      grpo_think_vs_mini_kfoo grpo_think_vs_mini_k grpo_think_vs_mini_kt07_krr grpo_think_vs_mini_kt07_h grpo_think_vs_mini_h
      grpo_think_vs_mini_kx66 grpo_think_vs_mini_kkl0 grpo_think_vs_mini_mini grpo_think_vs_mini_kg64_mini grpo_think_vs_mini_ku1
      grpo_think_vs_mini_kg8 grpo_think_vs_mini_klp5 grpo_think_vs_mini_klp100 grpo_think_vs_mini_klpx1 grpo_think_vs_mini_klpm
      grpo_think_vs_mini_klp00 grpo_think_vs_mini_klpm00 grpo_think_vs_mini_ksize grpo_think_vs_mini_kpool100
      grpo_think_vs_mini_kpool250_krr)
CKREF=(grpo_think_vs_ckreads grpo_think_vs_mb_l2_ckreads grpo_think_vs_mb_kkl0_l2_ckreads ei_think_v_krr_ckreads
       ei_think_v_l2_ckreads grpo_think_vs_mini_ckreads grpo_think_vs_mini_kt07_ckreads grpo_think_vs_mini_l2_ckreads
       grpo_think_vs_mb_kfoo_ckreads grpo_think_vs_c_l2_ckreads)
MODES=(normal smoke); SEEDS=(0 1 2)
OKLOG='guard it 39: kl anchor 0.0004 x 0.02 (gn policy 0.014); lr x1 (0 rollbacks); capped B0 0.0% B64 0.0% B256 0.0%'
RBLOG='guard it 39: kl anchor 0.0006 x 0.02 (gn policy 0.013); lr x0.5 (1 rollbacks); capped B0 0.0% B64 0.0% B256 0.0%'

setup() {  # NAME MODE SEED SRC CKPT(yes|no) GUARDLINE|none -> a SETUP file: SRC's stub L2 checkpoint and L2 log
  local f=$H/setups/$1.sh c=runs/grpo_v2 l=runs/logs it=39
  [ $2 = smoke ] && { c=runs/grpo_v2/smoke; l=$c/logs; it=2; }
  mkdir -p $H/setups
  { [ $5 = yes ] && echo "mkdir -p \"\$SB/$c/${4}_s${3}_L2/ckpts\" && : > \"\$SB/$c/${4}_s${3}_L2/ckpts/${4}_s${3}_L2_it$it.pt\""
    [ "$6" != none ] && printf 'mkdir -p "$SB/%s" && printf "%%s\\n" "it 39: dry" %q >> "$SB/%s/v2_%s_s%s_L2.log"\n' "$l" "$6" "$l" "$4" "$3"
    echo true; } > $f
  echo $f; }
eisetup() {  # NAME MODE SEED SRC CKPT(yes|no) [mix|nopass|stale] -> a SETUP file: SRC's stub EI L2 checkpoints (_r0.pt,
  local f=$H/setups/$1.sh c=runs/grpo_v2 m='$SB/runs/pools/curriculum_3p/L3_mix_ei'  # the effort init, always; _r1.pt,
  [ $2 = smoke ] && c=runs/grpo_v2/smoke                                              # the round's, with yes)
  mkdir -p $H/setups
  { echo "mkdir -p \"\$SB/$c/${4}_s${3}_L2/ckpts\" && : > \"\$SB/$c/${4}_s${3}_L2/ckpts/${4}_s${3}_L2_r0.pt\""
    [ $5 = yes ] && echo ": > \"\$SB/$c/${4}_s${3}_L2/ckpts/${4}_s${3}_L2_r1.pt\""
    case ${6:-} in  # ARM_k3p's mix (10 rows) and its pass record: the mix's md5 (mix), none (nopass), another md5 (stale)
      mix|nopass|stale) echo "for i in 1 2 3 4 5 6 7 8 9 10; do echo \"{\\\"dry\\\": \\\"L3_mix_ei row \$i\\\"}\"; done > \"$m.jsonl\"" ;;
    esac
    case ${6:-} in
      mix) echo "echo \"PREFLIGHT PASSED \$(md5sum < \"$m.jsonl\" | cut -c1-32)\" > \"$m.preflight_pass.txt\"" ;;
      stale) echo "echo \"PREFLIGHT PASSED \$(echo stale | md5sum | cut -c1-32)\" > \"$m.preflight_pass.txt\"" ;;
    esac
    echo true; } > $f
  echo $f; }
donesetup() {  # SETUP MODE SEED ARM CKPT [SET_K...]: append ARM's training record (base_commit.txt), its stub L3 checkpoint
  local su=$1 mode=$2 seed=$3 a=$4 ck=$5 c=runs/grpo_v2 ev=runs/eval/v2 rb=64 o x  # CKPT (a file name) and the rows of
  shift 5                                                                         # its plain read of SET at k K, each SET_K
  [ $mode = smoke ] && { c=runs/grpo_v2/smoke; ev=runs/eval/v2_smoke; }
  case $a in *nothink*) rb=0 ;; esac
  echo "mkdir -p \"\$SB/$c\" && echo 'dry training commit' > \"\$SB/$c/${a}_s${seed}.base_commit.txt\"" >> $su
  echo "mkdir -p \"\$SB/$c/${a}_s${seed}_L3/ckpts\" && : > \"\$SB/$c/${a}_s${seed}_L3/ckpts/$ck\"" >> $su
  for x in "$@"; do
    o=$ev/${x%_*}_${a}_s${seed}_plain_k${x##*_}
    echo "mkdir -p \"\$SB/$o\" && : > \"\$SB/$o/rows_plain_B$rb.jsonl\"" >> $su
  done; }

minisetup() {  # NAME [nosub|nopass|stale] -> a SETUP file: the mini subsets (10 rows each; the pool arms' nested ones
  local f=$H/setups/$1.sh m='$SB/runs/pools/mini' x  # too) and their pass record (each subset's md5); nosub: no
  mkdir -p $H/setups                                # 3-premise subset, nopass: no pass record, stale: the L3 subset
  { echo "mkdir -p \"$m\""                          # changed after its record
    for x in L2_mini L3_mini L3_3p_mini L2_mini250 L3_mini250 L3_3p_mini250 L2_mini500 L3_mini500 L3_3p_mini500; do
      echo "for i in 1 2 3 4 5 6 7 8 9 10; do echo \"{\\\"dry\\\": \\\"$x row \$i\\\"}\"; done > \"$m/$x.jsonl\""
    done
    [ "${2:-}" = nopass ] \
      || echo "for x in L2_mini L3_mini L3_3p_mini L2_mini250 L3_mini250 L3_3p_mini250 L2_mini500 L3_mini500 L3_3p_mini500; do echo \"PREFLIGHT PASSED \$(md5sum < \"$m/\$x.jsonl\" | cut -c1-32) \$x.jsonl\"; done > \"$m/preflight_pass.txt\""
    [ "${2:-}" = nosub ] && echo "rm \"$m/L3_3p_mini.jsonl\""  # after its pass record
    [ "${2:-}" = stale ] && echo "echo '{\"dry\": \"another row\"}' >> \"$m/L3_mini.jsonl\""
    echo true; } > $f
  echo $f; }
forksetup() {  # SETUP MODE SEED GUARDLINE [nojob]: append the reference grpo_think_vs_mini's L2-mini at SEED: its last
  local su=$1 c=runs/grpo_v2 l=runs/logs it=10 a=grpo_think_vs_mini_s$3  # checkpoint, its stage log (GUARDLINE last)
  [ $2 = smoke ] && { c=runs/grpo_v2/smoke; l=$c/logs; it=2; }    # and its line's exit line (nojob: not yet)
  { echo "mkdir -p \"\$SB/$c/${a}_L2/ckpts\" \"\$SB/$l\" && : > \"\$SB/$c/${a}_L2/ckpts/${a}_L2_it$it.pt\""
    printf 'printf "%%s\\n" "it %s: dry" %q > "$SB/%s/v2_%s_L2.log"\n' $it "$4" "$l" "$a"
    [ "${5:-}" = nojob ] || echo "echo '00:00Z GRPO ${a}_L2 exit 0: it $it: dry' > \"\$SB/$l/v2_$a.log\""; } >> $su; }
ckrsetup() {  # SETUP MODE SEED ARM IT...: append ARM's L3 rolling checkpoints at each IT (with donesetup: its finished L3)
  local su=$1 mode=$2 seed=$3 a=$4 c=runs/grpo_v2 i
  shift 4; [ $mode = smoke ] && c=runs/grpo_v2/smoke
  for i in "$@"; do
    echo "mkdir -p \"\$SB/$c/${a}_s${seed}_L3/ckpts\" && : > \"\$SB/$c/${a}_s${seed}_L3/ckpts/${a}_s${seed}_L3_it${i}_rolling.pt\"" >> $su
  done; }
rm -rf "$H/res" "$H/setups"; t0=$(date +%s); n=0
for mode in "${MODES[@]}"; do for seed in "${SEEDS[@]}"; do
  for arm in "${BASE[@]}" "${HARMS[@]}" "${EIH[@]}" "${EIB[@]}" bogus_arm "${CARMS[@]}" "${WARMS[@]}" "${EIS[@]}" \
             "${COMBO[@]}" "${PREF[@]}" "${FBREF[@]}" "${KREF[@]}" "${EKREF[@]}" "${MREF[@]}" "${CKREF[@]}"; do
    bash "$D/run.sh" "$H/line.sh.dry" $mode $seed $arm "$H/res/$mode/s$seed/$arm"; n=$((n+1))
  done
  for arm in "${GRPO[@]}"; do
    su=$(setup b_${mode}_s${seed}_$arm $mode $seed $arm yes "$OKLOG")
    bash "$D/run.sh" "$H/line.sh.dry" $mode $seed ${arm}_b "$H/res/$mode/s$seed/${arm}_b" $su; n=$((n+1))
  done
  for arm in "${GBASE[@]}"; do  # ARM_wb from ARM's L2
    su=$(setup wb_${mode}_s${seed}_$arm $mode $seed $arm yes "$OKLOG")
    bash "$D/run.sh" "$H/line.sh.dry" $mode $seed ${arm}_wb "$H/res/$mode/s$seed/${arm}_wb" $su; n=$((n+1))
  done
  for arm in "${PARMS[@]}"; do  # the L3 pilots, from the base arm's L2
    b=${arm%_*}
    su=$(setup p_${mode}_s${seed}_$arm $mode $seed $b yes "$OKLOG")
    bash "$D/run.sh" "$H/line.sh.dry" $mode $seed $arm "$H/res/$mode/s$seed/$arm" $su; n=$((n+1))
  done
  # the pilots' refusals as ARM_b's; and a missing 3-premise file (being rebuilt) for _3pb and the _vs3 arms
  su=$(setup pnockpt_${mode}_s$seed $mode $seed grpo_think_vs no "$OKLOG")
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_think_vs_rfb "$H/res/refuse/$mode/s$seed/nockpt_grpo_think_vs_rfb" $su
  su=$(setup prb_${mode}_s$seed $mode $seed grpo_think_vs yes "$RBLOG")
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_think_vs_ulb "$H/res/refuse/$mode/s$seed/rb_grpo_think_vs_ulb" $su
  su=$(setup pnolog_${mode}_s$seed $mode $seed grpo_think_vs yes none)
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_think_vs_3pb "$H/res/refuse/$mode/s$seed/nolog_grpo_think_vs_3pb" $su
  mbsetup() {  # NAME MODE SEED ARM GUARDLINE [ITERATIONS]: ARM's base L2 (setup), and ARM's L3 rolling checkpoints
    local su c=runs/grpo_v2 i; [ $2 = smoke ] && c=runs/grpo_v2/smoke
    su=$(setup $1 $2 $3 ${4%_mb} yes "$5")
    for i in ${6-25 30 35}; do [ $2 = smoke ] && continue
      echo "mkdir -p \"\$SB/$c/${4}_s${3}_L3/ckpts\" && : > \"\$SB/$c/${4}_s${3}_L3/ckpts/${4}_s${3}_L3_it${i}_rolling.pt\""
    done >> $su
    echo $su; }
  for arm in "${MBARMS[@]}"; do
    su=$(mbsetup mb_${mode}_s${seed}_$arm $mode $seed $arm "$OKLOG")
    bash "$D/run.sh" "$H/line.sh.dry" $mode $seed $arm "$H/res/$mode/s$seed/$arm" $su; n=$((n+1))
  done
  # ARM_mb: as ARM_3pb's refusals; a rolling checkpoint missing (its averaged reads refused); a rerun with the averaged
  # checkpoint there (no averaging)
  su=$(mbsetup mbnockpt_${mode}_s$seed $mode $seed grpo_think_vs_mb "$OKLOG"); sed -i '/_L2_it[0-9]*\.pt"$/d' $su
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_think_vs_mb "$H/res/refuse/$mode/s$seed/nockpt_grpo_think_vs_mb" $su
  su=$(mbsetup mbrb_${mode}_s$seed $mode $seed grpo_think_vs_mb "$RBLOG")
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_think_vs_mb "$H/res/refuse/$mode/s$seed/rb_grpo_think_vs_mb" $su
  su=$(mbsetup mbnolog_${mode}_s$seed $mode $seed grpo_nothink_vs_mb none)
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_nothink_vs_mb "$H/res/refuse/$mode/s$seed/nolog_grpo_nothink_vs_mb" $su
  su=$(mbsetup mbno3p_${mode}_s$seed $mode $seed grpo_think_vs_mb "$OKLOG")
  echo 'rm -f "$SB/runs/pools/curriculum_3p/L3_3p.jsonl"' >> $su
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_think_vs_mb "$H/res/refuse/$mode/s$seed/no3pL3_grpo_think_vs_mb" $su
  if [ $mode = normal ]; then
    su=$(mbsetup mbnoroll_s$seed $mode $seed grpo_think_vs_mb "$OKLOG" "25 35")
    bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_think_vs_mb "$H/res/mb/$mode/s$seed/noroll30_grpo_think_vs_mb" $su
    n=$((n+1))
  fi
  su=$(mbsetup mbavg_${mode}_s$seed $mode $seed grpo_think_vs_mb "$OKLOG")
  c=runs/grpo_v2; [ $mode = smoke ] && c=runs/grpo_v2/smoke; it=39; [ $mode = smoke ] && it=2
  a=grpo_think_vs_mb_avg_s${seed}_L3
  echo "mkdir -p \"\$SB/$c/$a/ckpts\" && : > \"\$SB/$c/$a/ckpts/${a}_it$it.pt\"" >> $su
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_think_vs_mb "$H/res/mb/$mode/s$seed/rerun_grpo_think_vs_mb" $su
  n=$((n+5))
  # ARM_pb's three, as ARM_rfb's
  su=$(setup pbnockpt_${mode}_s$seed $mode $seed grpo_think_vs no "$OKLOG")
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_think_vs_pb "$H/res/refuse/$mode/s$seed/nockpt_grpo_think_vs_pb" $su
  su=$(setup pbrb_${mode}_s$seed $mode $seed grpo_nothink_vs yes "$RBLOG")
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_nothink_vs_pb "$H/res/refuse/$mode/s$seed/rb_grpo_nothink_vs_pb" $su
  su=$(setup pbnolog_${mode}_s$seed $mode $seed grpo_think yes none)
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_think_pb "$H/res/refuse/$mode/s$seed/nolog_grpo_think_pb" $su
  n=$((n+3))
  for arm in "${FBARMS[@]}"; do  # ARM_fbb, from the base arm's L2
    b=${arm%_fbb}
    su=$(setup f_${mode}_s${seed}_$arm $mode $seed $b yes "$OKLOG")
    bash "$D/run.sh" "$H/line.sh.dry" $mode $seed $arm "$H/res/$mode/s$seed/$arm" $su; n=$((n+1))
  done
  # ARM_fbb's three, as ARM_rfb's
  su=$(setup fbnockpt_${mode}_s$seed $mode $seed grpo_think_vs no "$OKLOG")
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_think_vs_fbb "$H/res/refuse/$mode/s$seed/nockpt_grpo_think_vs_fbb" $su
  su=$(setup fbrb_${mode}_s$seed $mode $seed grpo_think_v yes "$RBLOG")
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_think_v_fbb "$H/res/refuse/$mode/s$seed/rb_grpo_think_v_fbb" $su
  su=$(setup fbnolog_${mode}_s$seed $mode $seed grpo_think_vs3 yes none)
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_think_vs3_fbb "$H/res/refuse/$mode/s$seed/nolog_grpo_think_vs3_fbb" $su
  n=$((n+3))
  for arm in "${KARMS[@]}"; do  # ARM_mb_kKNOB, from the base arm's L2
    b=${arm%_mb_k*}
    su=$(setup k_${mode}_s${seed}_$arm $mode $seed $b yes "$OKLOG")
    bash "$D/run.sh" "$H/line.sh.dry" $mode $seed $arm "$H/res/$mode/s$seed/$arm" $su; n=$((n+1))
  done
  # ARM_mb_kKNOB's refusals, as ARM_mb's: no L2 checkpoint, a rolled-back L2, no L2 log, no L3 3-premise file
  su=$(setup knockpt_${mode}_s$seed $mode $seed grpo_think_vs no "$OKLOG")
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_think_vs_mb_ki78 "$H/res/refuse/$mode/s$seed/nockpt_grpo_think_vs_mb_ki78" $su
  su=$(setup krb_${mode}_s$seed $mode $seed grpo_think_vs yes "$RBLOG")
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_think_vs_mb_kt12 "$H/res/refuse/$mode/s$seed/rb_grpo_think_vs_mb_kt12" $su
  su=$(setup knolog_${mode}_s$seed $mode $seed grpo_think_vs yes none)
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_think_vs_mb_kg64 "$H/res/refuse/$mode/s$seed/nolog_grpo_think_vs_mb_kg64" $su
  su=$(setup kno3p_${mode}_s$seed $mode $seed grpo_think_vs yes "$OKLOG")
  echo 'rm -f "$SB/runs/pools/curriculum_3p/L3_3p.jsonl"' >> $su
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_think_vs_mb_kx66 "$H/res/refuse/$mode/s$seed/no3pL3_grpo_think_vs_mb_kx66" $su
  n=$((n+4))
  # ARM_mb_ki78's L3 is it78: an it39 L3 checkpoint there does not end the stage; with it78 there, only the reads run
  c=runs/grpo_v2; it=39; [ $mode = smoke ] && { c=runs/grpo_v2/smoke; it=2; }
  a=grpo_think_vs_mb_ki78_s${seed}_L3
  su=$(setup ki39_${mode}_s$seed $mode $seed grpo_think_vs yes "$OKLOG")
  echo "mkdir -p \"\$SB/$c/$a/ckpts\" && : > \"\$SB/$c/$a/ckpts/${a}_it$it.pt\"" >> $su
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_think_vs_mb_ki78 "$H/res/knob/$mode/s$seed/it39_grpo_think_vs_mb_ki78" $su
  su=$(setup ki78_${mode}_s$seed $mode $seed grpo_think_vs yes "$OKLOG")
  echo "mkdir -p \"\$SB/$c/$a/ckpts\" && : > \"\$SB/$c/$a/ckpts/${a}_it$((2 * it)).pt\"" >> $su
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_think_vs_mb_ki78 "$H/res/knob/$mode/s$seed/it78_grpo_think_vs_mb_ki78" $su
  n=$((n+2))
  c=runs/grpo_v2; it=39; [ $mode = smoke ] && { c=runs/grpo_v2/smoke; it=2; }
  for arm in "${EKARMS[@]}"; do  # EI ARM_kKNOB, from the base arm's L2 (ARM_k3p: and the mix with its pass record)
    b=${arm%_k*}; mx=""; [ "${arm##*_k}" = 3p ] && mx=mix
    su=$(eisetup ek_${mode}_s${seed}_$arm $mode $seed $b yes $mx)
    bash "$D/run.sh" "$H/line.sh.dry" $mode $seed $arm "$H/res/$mode/s$seed/$arm" $su; n=$((n+1))
  done
  # EI ARM_kKNOB's refusals: no L2 checkpoint (its _r0.pt, the effort init, and a GRPO-named _it39.pt there); ARM_k3p
  # without the mix, without its pass record, with a pass record of another mix (its md5)
  su=$(eisetup eknockpt_${mode}_s$seed $mode $seed ei_think_v no mix)
  echo "mkdir -p \"\$SB/$c/ei_think_v_s${seed}_L2/ckpts\" && : > \"\$SB/$c/ei_think_v_s${seed}_L2/ckpts/ei_think_v_s${seed}_L2_it39.pt\"" >> $su
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed ei_think_v_k3p "$H/res/refuse/$mode/s$seed/nockpt_ei_think_v_k3p" $su
  for x in nomix nopass stale; do
    su=$(eisetup ek${x}_${mode}_s$seed $mode $seed ei_think_v yes ${x/nomix/})
    bash "$D/run.sh" "$H/line.sh.dry" $mode $seed ei_think_v_k3p "$H/res/refuse/$mode/s$seed/${x}_ei_think_v_k3p" $su
  done
  n=$((n+4))
  # EI ARM_kKNOB reruns (section J): ARM_kr2's stage ends at _r2.pt (a _r1.pt there does not end it; with _r2.pt only the
  # reads run, on _r2.pt); a finished pilot (its L3 checkpoint and the 3 plain reads there) runs nothing again; with one
  # read there, the other two; KNOB_FULL=1 on a finished pilot adds the other reads of ARM (tb72, the guided, hand14)
  # and on a fresh one runs the stage and all of ARM's reads; KNOB_FULL=1 for a GRPO knob arm as well
  rk=(t1320_16 t2130_8 charles173_64); [ $mode = smoke ] && rk=(t1320_2 t2130_2 charles173_2)
  su=$(eisetup ekr1_${mode}_s$seed $mode $seed ei_think_v yes)
  donesetup $su $mode $seed ei_think_v_kr2 ei_think_v_kr2_s${seed}_L3_r1.pt
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed ei_think_v_kr2 "$H/res/eiknob/$mode/s$seed/r1_ei_think_v_kr2" $su
  su=$(eisetup ekr2_${mode}_s$seed $mode $seed ei_think_v yes)
  donesetup $su $mode $seed ei_think_v_kr2 ei_think_v_kr2_s${seed}_L3_r2.pt
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed ei_think_v_kr2 "$H/res/eiknob/$mode/s$seed/r2_ei_think_v_kr2" $su
  for kn in "${EKNOBS[@]}"; do
    a=ei_think_v_k$kn; r=1; [ $kn = r2 ] && r=2; mx=""; [ $kn = 3p ] && mx=mix
    su=$(eisetup ekdone_${mode}_s${seed}_$kn $mode $seed ei_think_v yes $mx)
    donesetup $su $mode $seed $a ${a}_s${seed}_L3_r$r.pt "${rk[@]}"
    bash "$D/run.sh" "$H/line.sh.dry" $mode $seed $a "$H/res/eiknob/$mode/s$seed/done_$a" $su
    DRY_KNOB_FULL=1 bash "$D/run.sh" "$H/line.sh.dry" $mode $seed $a "$H/res/eiknob/$mode/s$seed/full_$a" $su
  done
  su=$(eisetup ekpart_${mode}_s$seed $mode $seed ei_think_v yes)
  donesetup $su $mode $seed ei_think_v_kt10 ei_think_v_kt10_s${seed}_L3_r1.pt ${rk[1]}
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed ei_think_v_kt10 "$H/res/eiknob/$mode/s$seed/part_ei_think_v_kt10" $su
  su=$(eisetup ekfresh_${mode}_s$seed $mode $seed ei_think_v yes mix)
  DRY_KNOB_FULL=1 bash "$D/run.sh" "$H/line.sh.dry" $mode $seed ei_think_v_k3p "$H/res/eiknob/$mode/s$seed/fresh_ei_think_v_k3p" $su
  su=$(setup gkfull_${mode}_s$seed $mode $seed grpo_think_vs yes "$OKLOG")
  donesetup $su $mode $seed grpo_think_vs_mb_kkl0 grpo_think_vs_mb_kkl0_s${seed}_L3_it$it.pt "${rk[@]}"
  DRY_KNOB_FULL=1 bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_think_vs_mb_kkl0 "$H/res/eiknob/$mode/s$seed/full_grpo_think_vs_mb_kkl0" $su
  n=$((n+15))
  # the mini runs (2026-10-16), two stages from the Stage-1 init, with the subsets and their pass record
  for arm in "${MARMS[@]}"; do
    su=$(minisetup mini_${mode}_s${seed}_$arm)
    bash "$D/run.sh" "$H/line.sh.dry" $mode $seed $arm "$H/res/$mode/s$seed/$arm" $su; n=$((n+1))
  done
  # the forks, L3-mini only, from the reference's finished L2-mini at the same seed
  for arm in "${MFORKS[@]}"; do
    su=$(minisetup fork_${mode}_s${seed}_$arm); forksetup $su $mode $seed "$OKLOG"
    bash "$D/run.sh" "$H/line.sh.dry" $mode $seed $arm "$H/res/$mode/s$seed/$arm" $su; n=$((n+1))
  done
  # their refusals: no 3-premise subset, no pass record, a stale one; KNOB_FULL=1 (with the subsets there); a fork whose
  # reference L2-mini rolled back, or never finished (no exit line: 180 waits, then FAILED)
  for x in nosub nopass stale; do
    su=$(minisetup mini${x}_${mode}_s$seed $x)
    bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_think_vs_mini_kt07 "$H/res/refuse/$mode/s$seed/${x}_grpo_think_vs_mini_kt07" $su
  done
  su=$(minisetup minifull_${mode}_s$seed)
  DRY_KNOB_FULL=1 bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_think_vs_mini "$H/res/refuse/$mode/s$seed/full_grpo_think_vs_mini" $su
  su=$(minisetup forkrb_${mode}_s$seed); forksetup $su $mode $seed "$RBLOG"
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_think_vs_mini_kkl1 "$H/res/refuse/$mode/s$seed/rb_grpo_think_vs_mini_kkl1" $su
  su=$(minisetup forkwait_${mode}_s$seed); forksetup $su $mode $seed "$OKLOG" nojob
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_think_vs_mini_kg64 "$H/res/refuse/$mode/s$seed/wait_grpo_think_vs_mini_kg64" $su
  # --ban_stop in L3-mini (the stub's exit 3, no checkpoint): the FAILED line
  su=$(minisetup ministop_${mode}_s$seed)
  DRY_GRPO_STOP=_L3 bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_think_vs_mini_kt13 "$H/res/refuse/$mode/s$seed/stop_grpo_think_vs_mini_kt13" $su
  n=$((n+7))
  # reruns (section N): L2-mini finished (only L3-mini and the reads run); both stages finished and two reads there (the
  # other reads run)
  c=runs/grpo_v2; it=10; rm2=(l2test_8 charles173_64); [ $mode = smoke ] && { c=runs/grpo_v2/smoke; it=2; rm2=(l2test_2 charles173_2); }
  a=grpo_think_vs_mini_klp20
  su=$(minisetup minil2_${mode}_s$seed)
  echo "mkdir -p \"\$SB/$c/${a}_s${seed}_L2/ckpts\" && : > \"\$SB/$c/${a}_s${seed}_L2/ckpts/${a}_s${seed}_L2_it$it.pt\"" >> $su
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed $a "$H/res/mini/$mode/s$seed/l2done_$a" $su
  a=grpo_think_vs_mini_kpool250
  su=$(minisetup minidone_${mode}_s$seed)
  echo "mkdir -p \"\$SB/$c/${a}_s${seed}_L2/ckpts\" && : > \"\$SB/$c/${a}_s${seed}_L2/ckpts/${a}_s${seed}_L2_it$it.pt\"" >> $su
  donesetup $su $mode $seed $a ${a}_s${seed}_L3_it$it.pt "${rm2[@]}"
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed $a "$H/res/mini/$mode/s$seed/done_$a" $su
  # MINI_ROLL (the time-resolved reads' hook): L2:4 and L3:2 there, L3:6 not (an alarm iteration: skipped)
  a=grpo_think_vs_mini; i2=4; i3=2; [ $mode = smoke ] && i2=2
  su=$(minisetup miniroll_${mode}_s$seed)
  echo "mkdir -p \"\$SB/$c/${a}_s${seed}_L2/ckpts\" \"\$SB/$c/${a}_s${seed}_L3/ckpts\"" >> $su
  echo ": > \"\$SB/$c/${a}_s${seed}_L2/ckpts/${a}_s${seed}_L2_it${i2}_rolling.pt\"" >> $su
  echo ": > \"\$SB/$c/${a}_s${seed}_L3/ckpts/${a}_s${seed}_L3_it${i3}_rolling.pt\"" >> $su
  DRY_MINI_ROLL="L2:$i2 L3:$i3 L3:6" bash "$D/run.sh" "$H/line.sh.dry" $mode $seed $a "$H/res/mini/$mode/s$seed/roll_$a" $su
  n=$((n+3))
  # ARM_ckreads (section Q): full L3 runs with their base arm's L2 and their last checkpoint (donesetup) and rolling ones:
  # kl0 (it10 / 20 / 30 there; its pilot reads there), t12 (it5, it15, it25, it30: it10 / it20 from the nearest earlier
  # one), i78 (every 5 to 75: it10 ... it70), grpo_think_vs_mb (ARM_mb), grpo_think_vs_fbb (ARM_fbb, from
  # grpo_think_vs's L2: no rolling one at all, its reads skipped); the shared L2 (grpo_think_vs_l2_ckreads) and the
  # Stage-1 base (s1g3_ckreads); refused without the finished L3, without the L2 checkpoint, without the L2 to read,
  # with KNOB_FULL=1
  it=39; [ $mode = smoke ] && it=2
  for x in kl0:10_20_30:39 t12:5_15_25_30:39 i78:5_10_15_20_25_30_35_40_45_50_55_60_65_70_75:78 mb:10_20_30:39 fbb:none:39; do
    k=${x%%:*}; its=${x#*:}; its=${its%:*}; last=${x##*:}; [ $mode = smoke ] && last=$((last * 2 / 39))
    case $k in mb) a=grpo_think_vs_mb ;; fbb) a=grpo_think_vs_fbb ;; *) a=grpo_think_vs_mb_k$k ;; esac
    su=$(setup ckr${k}_${mode}_s$seed $mode $seed grpo_think_vs yes "$OKLOG")
    if [ $k = kl0 ]; then donesetup $su $mode $seed $a ${a}_s${seed}_L3_it$last.pt "${rk[@]}"
    else donesetup $su $mode $seed $a ${a}_s${seed}_L3_it$last.pt; fi
    [ "$its" = none ] || ckrsetup $su $mode $seed $a ${its//_/ }
    bash "$D/run.sh" "$H/line.sh.dry" $mode $seed ${a}_ckreads "$H/res/ckreads/$mode/s$seed/${a}_ckreads" $su
  done
  su=$(setup ckrl2_${mode}_s$seed $mode $seed grpo_think_vs yes "$OKLOG")
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_think_vs_l2_ckreads "$H/res/ckreads/$mode/s$seed/grpo_think_vs_l2_ckreads" $su
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed s1g3_ckreads "$H/res/ckreads/$mode/s$seed/s1g3_ckreads"
  su=$(setup ckrnol3_${mode}_s$seed $mode $seed grpo_think_vs yes "$OKLOG")
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_think_vs_mb_kkl0_ckreads "$H/res/refuse/$mode/s$seed/nol3_grpo_think_vs_mb_kkl0_ckreads" $su
  su=$(setup ckrnockpt_${mode}_s$seed $mode $seed grpo_think_vs no "$OKLOG")
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_think_vs_mb_kkl0_ckreads "$H/res/refuse/$mode/s$seed/nockpt_grpo_think_vs_mb_kkl0_ckreads" $su
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_think_vs_l2_ckreads "$H/res/refuse/$mode/s$seed/nol2_grpo_think_vs_l2_ckreads"
  DRY_KNOB_FULL=1 bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_think_vs_mb_kkl0_ckreads "$H/res/refuse/$mode/s$seed/full_grpo_think_vs_mb_kkl0_ckreads"
  n=$((n+11))
  for x in L3 L2; do for arm in grpo_think_vs_3pb grpo_think_vs3 grpo_nothink_vs3; do
    su=$(setup no3p${x}_${mode}_s${seed}_$arm $mode $seed grpo_think_vs yes "$OKLOG")
    echo "rm -f \"\$SB/runs/pools/curriculum_3p/${x}_3p.jsonl\"" >> $su
    bash "$D/run.sh" "$H/line.sh.dry" $mode $seed $arm "$H/res/refuse/$mode/s$seed/no3p${x}_$arm" $su; n=$((n+1))
  done; done
  n=$((n+3))
  # ARM_b refusals: no L2 checkpoint; an L2 log at lr x0.5 (1 rollbacks); no L2 log
  su=$(setup nockpt_${mode}_s$seed $mode $seed grpo_think_h no "$OKLOG")
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_think_h_b "$H/res/refuse/$mode/s$seed/nockpt_grpo_think_h_b" $su
  su=$(setup rb_${mode}_s$seed $mode $seed grpo_think yes "$RBLOG")
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_think_b "$H/res/refuse/$mode/s$seed/rb_grpo_think_b" $su
  su=$(setup nolog_${mode}_s$seed $mode $seed grpo_think_vs yes none)
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_think_vs_b "$H/res/refuse/$mode/s$seed/nolog_grpo_think_vs_b" $su
  # the same three for ARM_wb (from ARM's L2)
  su=$(setup wbnockpt_${mode}_s$seed $mode $seed grpo_think_vs no "$OKLOG")
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_think_vs_wb "$H/res/refuse/$mode/s$seed/nockpt_grpo_think_vs_wb" $su
  su=$(setup wbrb_${mode}_s$seed $mode $seed grpo_think_vs yes "$RBLOG")
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_think_vs_wb "$H/res/refuse/$mode/s$seed/rb_grpo_think_vs_wb" $su
  su=$(setup wbnolog_${mode}_s$seed $mode $seed grpo_nothink_vs3 yes none)
  bash "$D/run.sh" "$H/line.sh.dry" $mode $seed grpo_nothink_vs3_wb "$H/res/refuse/$mode/s$seed/nolog_grpo_nothink_vs3_wb" $su
  n=$((n+6))
done; done
# the mini reference at seed 3: its init is S1g3_s0 whatever the seed (the sandbox has no seed-3 init)
for mode in "${MODES[@]}"; do
  su=$(minisetup mini3_$mode)
  bash "$D/run.sh" "$H/line.sh.dry" $mode 3 grpo_think_vs_mini "$H/res/$mode/s3/grpo_think_vs_mini" $su; n=$((n+1))
done
# seed 3 has no Stage-1 init in the sandbox; usage error
bash "$D/run.sh" "$H/line.sh.dry" normal 3 grpo_think "$H/res/refuse/normal/s3/grpo_think"
bash "$D/run.sh" "$H/line.sh.dry" normal "" "" "$H/res/refuse/normal/usage"
# the live run's GPU: grpo_think_h_b seed 1 on GPU 1 (ps evidence), normal
su=$(setup b_gpu1 normal 1 grpo_think_h yes "$OKLOG")
DRY_GPU=1 bash "$D/run.sh" "$H/line.sh.dry" normal 1 grpo_think_h_b "$H/res/gpu1/grpo_think_h_b_s1" $su
# waitfree with a busy GPU (another user's process): the stubbed nvidia-smi reports 23474 MiB and a pid
DRY_GPU_MEM=23474 DRY_GPU_PID=2116814 bash "$D/run.sh" "$H/line.sh.dry" normal 0 grpo_think "$H/res/busy/grpo_think_s0"
# queue.sh: two arms on GPU 2 after a wait line and a PRE_CMD, as the serrano queues were started
cat > $H/setups/queue.sh <<'EOF'
mkdir -p "$SB/runs/logs"; echo "00:00Z grpo_v2 line done (grpo_think seed 0)" > "$SB/runs/logs/v2_grpo_think_s0.log"
EOF
DRY_GPU=2 bash "$D/run.sh" queue normal - - "$H/res/queue/normal" $H/setups/queue.sh -- \
  2 "grpo_think_v:1 ei_nothink:2" "$H/sb/runs/logs/v2_grpo_think_s0.log" 'grpo_v2 line done (grpo_think seed 0)' 'echo pre ran'
n=$((n+7))
echo "$n dry runs in $(( $(date +%s) - t0 ))s under $H/res"
