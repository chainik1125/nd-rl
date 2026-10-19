#!/bin/zsh
# Cost of a full exhaustion (what a cert_ge needs) vs |S|: invalid sequents have no proof, so no pruning.
P=~/repos/nd-takehome/.venv/bin/python
for thm in 'THM ( P > Q ) SEQ ( Q > P ) PRF' \
           'THM ( P > Q ) , ( Q v R ) SEQ ( P & R ) PRF' \
           'THM ( P > Q ) , ( Q v R ) SEQ ( P & ( R > Q ) ) PRF' \
           'THM ( P > Q ) , ( Q v R ) , ( R > P ) SEQ ( P & ( R > Q ) ) PRF' \
           'THM ( ( P > Q ) > P ) , ( P > ( ( R > S ) > R ) ) SEQ S PRF'; do
  /usr/bin/time -l $P growth.py "$thm" ${SECS:-300} 13 15 17 19 2>&1 | grep -E 'body_bound|maximum resident'
done
