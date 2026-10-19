#!/usr/bin/env python3
"""Generate evaluate.py = lengthexp/evaluate.py with the goals edits (DESIGN.md 6) as checked text replacements (each
must apply exactly once); the diff goes to evaluate.diff. Test sets, k, T, max_new, seeds, the sampler
(lengthexp/sampler.py) and the checker (lengthexp/ndcheck.py) are lengthexp's.

Edits: the model is loaded with the goals tokenizer (tok_mode from the checkpoint; abs checkpoints load as in
lengthexp); every sample is stripped of its goal annotations before classify(); samples/*.jsonl.gz store the stripped
proof in `body` and the raw text in `raw`; every row gets `n_lines` (written lines of every sample, stripped) and
`ann` (per-sample groups / malformed groups, and for correct samples the consistency with the canonical annotation of
their own proof); summaries get aggregate annotation statistics; output goes to goals/eval/<label>/.

    python make_eval.py
"""
import difflib, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common_goals import LE, sha256_file

SRC = os.path.join(LE, 'evaluate.py')
EDITS = [
    ('''"""Evaluate one checkpoint at every test length: k samples per theorem, verified, with first-error lines.
''', '''"""GOALS COPY (ab3/goals/make_eval.py) of lengthexp/evaluate.py: samples are stripped of goal annotations before
verification; see make_eval.py for the edits. Evaluate one checkpoint at every test length: k samples per theorem,
verified, with first-error lines.
'''),
    ('''HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import (EVAL, FROZEN, K, LE, LENGTHS, MAX_NEW, SAMPLE_SEED, TEMPERATURE, TESTSETS, pass_at_k,
                    read_jsonl, sha256_file, sha256_str, unit_name, write_json, write_jsonl)
''', '''HERE = os.path.dirname(os.path.abspath(__file__))
LE_DIR = '/scratch/cowifornia/nd-rl/ab3/lengthexp'
sys.path.insert(0, LE_DIR)          # lengthexp: common.py, sampler.py, ndcheck.py
from common import (FROZEN, K, LENGTHS, MAX_NEW, SAMPLE_SEED, TEMPERATURE, TESTSETS, pass_at_k,  # lengthexp/common.py
                    read_jsonl, sha256_file, sha256_str, unit_name, write_json, write_jsonl)
sys.path.insert(0, HERE)            # goals: common_goals.py
LE = HERE
EVAL = os.path.join(HERE, 'eval')
TESTSETS = os.path.join(HERE, 'testsets')   # goals: frozen copy of lengthexp's final draw (02:27 UTC; = repr/testsets)
'''),
    ("run=f'lengthexp_eval_{a.label}'", "run=f'goals_eval_{a.label}'"),
    ('''def load_model(path, device):
    import torch
    from model import GPT
    from tokenizer import Tokenizer
    ck = torch.load(path, map_location='cpu', weights_only=True)
    tok = Tokenizer(ck.get('tok_mode', 'abs'))
    cfg = dict(ck.get('cfg') or dict(vocab=tok.vocab_size, n_layer=4, d=256, n_head=8))
    cfg.setdefault('vocab', tok.vocab_size)
    model = GPT(**cfg)
    state = {k.removeprefix('_orig_mod.').removeprefix('module.'): v for k, v in ck['state'].items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, tok, cfg
''', '''def load_model(path, device):
    from common_goals import load_model as lm          # goals: tokenizer from tok_mode (abs_goals or abs)
    model, tok, ck = lm(path, device)
    return model, tok, dict(model.cfg)
'''),
    ('''    from ndcheck import classify, sequent
    from sampler import sample_ids
''', '''    from ndcheck import classify, sequent
    from sampler import sample_ids
    import goals as GL                                  # goals: annotation strip / consistency
    acnt = collections.Counter()
'''),
    ('''        ok_lines, ok_starts, fails, ntok = [], [], [], []
        for i in range(a.k):
            o = ids[j * a.k + i]
            body = tok.decode(o)
            ntok.append(len(o))
''', '''        ok_lines, ok_starts, fails, ntok = [], [], [], []
        n_lines, ann = [], []
        for i in range(a.k):
            o = ids[j * a.k + i]
            raw = tok.decode(o)
            body, groups, n_bad = GL.read_annotations(raw)     # goals: verify the stripped proof
            n_groups = sum(len(v) for v in groups.values())
            n_lines.append(body.split().count(';'))
            ntok.append(len(o))
'''),
    ('''            if c['ok']:
                ok_lines.append(c['n_lines'])
                ok_starts.append(c['start'])
                kinds['ok'] += 1
''', '''            cs = GL.consistency(r['prompt'], raw) if c['ok'] else None
            ann.append([n_groups, n_bad] + ([int(cs['exact']), cs['matched'], cs['n_canon'], cs['extra'], cs['valid_goal']] if cs else []))
            acnt.update(samples=1, groups=n_groups, malformed=n_bad, with_malformed=int(n_bad > 0))
            if cs:
                acnt.update(ok=1, exact=int(cs['exact']), matched=cs['matched'], n_canon=cs['n_canon'], extra=cs['extra'],
                            valid_goal=cs['valid_goal'], n_written=cs['n_written'])
            if c['ok']:
                ok_lines.append(c['n_lines'])
                ok_starts.append(c['start'])
                kinds['ok'] += 1
'''),
    ('''                samples.append({'name': r['name'], 'i': i, 'body': body})
''', '''                samples.append({'name': r['name'], 'i': i, 'body': body, 'raw': raw})
'''),
    ('''                     'ok_starts': ok_starts, 'fails': fails, 'n_tok_mean': sum(ntok) / len(ntok)})
''', '''                     'ok_starts': ok_starts, 'fails': fails, 'n_tok_mean': sum(ntok) / len(ntok),
                     'n_lines': n_lines, 'ann': ann})
'''),
    ('''            'sample_secs': t_sample, 'check_secs': t_check, 'samples_per_sec': len(prompts) / max(t_sample, 1e-9)}
''', '''            'sample_secs': t_sample, 'check_secs': t_check, 'samples_per_sec': len(prompts) / max(t_sample, 1e-9),
            'annotation': {'groups_per_sample': acnt['groups'] / max(1, acnt['samples']),
                           'samples_with_malformed': acnt['with_malformed'] / max(1, acnt['samples']),
                           'correct_exact_canonical': acnt['exact'] / max(1, acnt['ok']),
                           'correct_canon_groups_matched': acnt['matched'] / max(1, acnt['n_canon']),
                           'correct_extra_groups_per_proof': acnt['extra'] / max(1, acnt['ok']),
                           'correct_written_groups_pending_goal': acnt['valid_goal'] / max(1, acnt['n_written']),
                           'counts': dict(acnt)}}
'''),
    ("    os.environ.setdefault('AB_WORKER', 'lengthexp')", "    os.environ.setdefault('AB_WORKER', 'goals')"),
    ('''        sys.path.insert(0, os.path.join(FROZEN, 'lib'))
        model, tok, mcfg = load_model(ckpt, 'cuda')
        meta = {'label': a.label, 'ckpt': ckpt, 'ckpt_sha256': ck_sha, 'model_cfg': mcfg, 'config': cfg_key,''',
     '''        model, tok, mcfg = load_model(ckpt, 'cuda')
        meta = {'label': a.label, 'ckpt': ckpt, 'ckpt_sha256': ck_sha, 'model_cfg': mcfg, 'tok_mode': tok.mode, 'config': cfg_key,'''),
]


def main():
    src = open(SRC).read()
    out = src
    for a, b in EDITS:
        if out.count(a) != 1:
            raise SystemExit(f'evaluate edit does not apply exactly once ({out.count(a)}x): {a[:80]!r}')
        out = out.replace(a, b)
    dst = os.path.join(HERE, 'evaluate.py')
    open(dst, 'w').write(out)
    diff = difflib.unified_diff(src.splitlines(), out.splitlines(), 'lengthexp/evaluate.py', 'evaluate.py', lineterm='', n=1)
    open(os.path.join(HERE, 'evaluate.diff'), 'w').write(f'# lengthexp/evaluate.py sha256 {sha256_file(SRC)}\n' + '\n'.join(diff) + '\n')
    print('wrote', dst, 'sha256', sha256_file(dst))


if __name__ == '__main__':
    main()
