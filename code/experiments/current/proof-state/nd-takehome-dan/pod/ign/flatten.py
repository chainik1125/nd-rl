#!/usr/bin/env python3
"""After `podpull <pod> artifacts/p2/<arm>` into an existing local dir, rsync nests <arm>/<arm>/; move the contents up."""
import os, shutil, sys
for d in sys.argv[1:]:
    inner = os.path.join(d, os.path.basename(d))
    if os.path.isdir(inner):
        for f in os.listdir(inner):
            shutil.move(os.path.join(inner, f), os.path.join(d, f))
        os.rmdir(inner)
    print(d, sum(1 for f in os.listdir(d) if f.startswith('round_')), 'rounds')
