#!/usr/bin/env python3
"""Frozen harness copy for one mid-trained arm: ab3/frozen_039 with only pretrain.pt replaced.

    python make_frozen.py --arm long|short|more [--ckpt ckpt/mt_<arm>.pt] [--out frozen/<arm>]

Copies frozen_039 (without __pycache__), replaces pretrain.pt with the mid-trained checkpoint, removes
data/SHA256SUMS so the harness recomputes the data hashes, and checks by sha256, file by file, that pretrain.pt is
the only difference (writes frozen/<arm>.check.json).
"""
import argparse, hashlib, json, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = '/scratch/cowifornia/nd-rl/ab3/frozen_039'


def files(root):
    out = {}
    for d, ds, fs in os.walk(root):
        ds[:] = [x for x in ds if x != '__pycache__']
        for f in fs:
            p = os.path.join(d, f)
            h = hashlib.sha256()
            with open(p, 'rb') as fh:
                for b in iter(lambda: fh.read(1 << 20), b''):
                    h.update(b)
            out[os.path.relpath(p, root)] = h.hexdigest()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    ckpt = a.ckpt or os.path.join(HERE, 'ckpt', f'mt_{a.arm}.pt')
    out = a.out or os.path.join(HERE, 'frozen', a.arm)
    if os.path.exists(out):
        sys.exit(f'{out} exists')
    shutil.copytree(SRC, out, ignore=shutil.ignore_patterns('__pycache__'), copy_function=shutil.copy2)
    os.remove(os.path.join(out, 'pretrain.pt'))
    shutil.copy2(ckpt, os.path.join(out, 'pretrain.pt'))
    if os.path.exists(os.path.join(out, 'data', 'SHA256SUMS')):
        os.remove(os.path.join(out, 'data', 'SHA256SUMS'))
    fa, fb = files(SRC), files(out)
    only_src = sorted(set(fa) - set(fb))
    only_out = sorted(set(fb) - set(fa))
    differ = sorted(k for k in fa if k in fb and fa[k] != fb[k])
    ok = differ == ['pretrain.pt'] and only_out == [] and only_src in ([], ['data/SHA256SUMS'])
    rep = {'arm': a.arm, 'src': SRC, 'out': out, 'ckpt': ckpt, 'pretrain_sha256': fb['pretrain.pt'], 'differ': differ,
           'only_in_src': only_src, 'only_in_copy': only_out, 'ok': ok}
    with open(out.rstrip('/') + '.check.json', 'w') as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep))
    if not ok:
        sys.exit('frozen copy differs in more than pretrain.pt')


if __name__ == '__main__':
    main()
