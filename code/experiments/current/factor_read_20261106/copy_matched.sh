#!/usr/bin/env bash
# Amendment 3: the seven checkpoints to serrano, then to ynez, md5-checked. Originals untouched.
# temescal: this one file, read-only, under the coordinator's override; nothing else is done there.
set -u
export SSH_AUTH_SOCK=
S="ssh -i $HOME/.ssh/id_ynez -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10"
R=/scratch/cowifornia/nd-rl/knockout_danbase/runs
OUT=/scratch/cowifornia/nd-rl/factor_read/matched/ckpts
mkdir -p "$OUT"
fetch() {  # host path
  local h=$1 p=$2 f m r
  f=$(basename "$p")
  if [ "$h" = serrano ]; then
    m=$(md5sum "$p" | cut -d' ' -f1); cp -p "$p" "$OUT/$f"
  else
    m=$($S "$h" "md5sum $p" </dev/null | cut -d' ' -f1)
    rsync -a -e "$S" "$h:$p" "$OUT/$f" || echo "RSYNC FAIL $f"
  fi
  r=$(md5sum "$OUT/$f" | cut -d' ' -f1)
  if [ "$r" = "$m" ] && [ -n "$m" ]; then echo "ok $h $f $m"; else echo "MD5 MISMATCH $h $f source $m here $r"; fi
}
fetch serrano $R/dko_s0/seg3/ckpts/dko_s0_it60_snap.pt
fetch temescal $R/dko_s1/seg3/ckpts/dko_s1_it70_snap.pt
fetch ynez $R/dko_s2/seg2/ckpts/dko_s2_it100_snap.pt
fetch serrano $R/dd01_s0/seg1/ckpts/dd01_s0_it60_snap.pt
fetch ynez $R/dd01g0_s2/seg1/ckpts/dd01g0_s2_it90_snap.pt
fetch ynez $R/dd01_s1/seg3/ckpts/dd01_s1_it35_rolling.pt
fetch ynez $R/dd01_s2/seg3/ckpts/dd01_s2_it25_rolling.pt
echo COPY_DONE
