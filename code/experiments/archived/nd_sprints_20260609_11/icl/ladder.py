"""ICL capability ladder: few-shot ND-proof completion across frontier APIs.

For each target sequent, prompt a model with K worked ND examples (in our exact
token format) then the target's `THM ... SEQ ... PRF`, ask it to complete the
proof through QED, and check the completion with the INDEPENDENT verifier.

A theorem's "ladder height" = the weakest model (by a fixed capability order)
that solves it. Our weak 3.3M transformer (+RL) sits at the bottom; RL uplift =
how far up the ladder it climbs.

Providers: openai, anthropic, xai (grok). Keys from env *_API_KEY_MATS.
Concurrency via threads. Results -> jsonl (one row per model x target x sample).

Usage:
  python -m icl.ladder --targets ../data/targets_textbook.jsonl \
     --shots data/icl_shots.jsonl --k-shot 50 --models gpt-4o-mini,claude-haiku \
     --samples 3 --out ../results/icl_textbook.jsonl
"""
import argparse
import json
import os
import sys
import time
import ssl
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import certifi
    _SSL = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL = ssl.create_default_context()
    _SSL.check_hostname = False
    _SSL.verify_mode = ssl.CERT_NONE

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nd.verify import verify_text

# model registry: name -> (provider, api_model_id, ladder_rank)
MODELS = {
    'gpt-4o-mini':   ('openai', 'gpt-4o-mini', 10),
    'gpt-4o':        ('openai', 'gpt-4o', 30),
    'gpt-4.1':       ('openai', 'gpt-4.1', 35),
    'o4-mini':       ('openai', 'o4-mini', 45),
    'grok-3-mini':   ('xai', 'grok-3-mini', 15),
    'grok-3':        ('xai', 'grok-3', 33),
    'claude-haiku':  ('anthropic', 'claude-haiku-4-5-20251001', 20),
    'claude-sonnet': ('anthropic', 'claude-sonnet-4-6', 40),
}

SYS = ("You are completing formal natural-deduction proofs written in a fixed "
       "token format. Output ONLY proof tokens continuing after 'PRF', ending "
       "with 'QED'. Use exactly the token vocabulary and line format of the "
       "examples: each line is 'N<i> [| ...] <formula> : <RULE> <refs> ;'. "
       "Do not explain. Do not use markdown.")


def build_prompt(shots, target_prompt, k):
    ex = []
    for s in shots[:k]:
        ex.append(s['text'])  # full THM..SEQ..PRF..QED example
    body = ("Here are worked natural-deduction proofs in the required format:\n\n"
            + "\n\n".join(ex)
            + "\n\nNow complete this proof. Output only tokens from 'PRF' "
            "onward (repeat the 'PRF' token first), ending with 'QED':\n\n"
            + target_prompt)
    return body


def _post(url, headers, payload, timeout=90):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method='POST')
    with urllib.request.urlopen(req, timeout=timeout, context=_SSL) as r:
        return json.loads(r.read().decode())


def call_model(name, prompt, temperature):
    provider, mid, _ = MODELS[name]
    if provider == 'openai':
        key = os.environ['OPENAI_API_KEY_MATS']
        payload = {'model': mid,
                   'messages': [{'role': 'system', 'content': SYS},
                                {'role': 'user', 'content': prompt}]}
        if mid.startswith('o'):
            payload['max_completion_tokens'] = 1200
        else:
            payload['max_tokens'] = 1200
            payload['temperature'] = temperature
        d = _post('https://api.openai.com/v1/chat/completions',
                  {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
                  payload)
        return d['choices'][0]['message']['content']
    if provider == 'xai':
        key = os.environ['GROK_API_KEY_MATS']
        payload = {'model': mid, 'temperature': temperature, 'max_tokens': 1200,
                   'messages': [{'role': 'system', 'content': SYS},
                                {'role': 'user', 'content': prompt}]}
        d = _post('https://api.x.ai/v1/chat/completions',
                  {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
                  payload)
        return d['choices'][0]['message']['content']
    if provider == 'anthropic':
        key = os.environ['ANTHROPIC_API_KEY_MATS']
        payload = {'model': mid, 'max_tokens': 1200, 'temperature': temperature,
                   'system': SYS,
                   'messages': [{'role': 'user', 'content': prompt}]}
        d = _post('https://api.anthropic.com/v1/messages',
                  {'x-api-key': key, 'anthropic-version': '2023-06-01',
                   'Content-Type': 'application/json'},
                  payload)
        return d['content'][0]['text']
    raise ValueError(provider)


def normalize_completion(target_prompt, text):
    """Build a full 'THM..QED' string from the target prompt + model output.
    Model is asked to emit from PRF; we splice its PRF-onward onto the prompt."""
    text = text.strip().replace('`', ' ')
    toks = text.split()
    # take from the last 'PRF' the model emitted, else assume it continued body
    if 'PRF' in toks:
        body = toks[toks.index('PRF') + 1:]
    else:
        body = toks
    if 'QED' in body:
        body = body[:body.index('QED') + 1]
    return target_prompt + ' ' + ' '.join(body)


def eval_one(name, rec, shots, k, samples, temperature):
    res = {'model': name, 'thm': rec['thm'], 'name': rec.get('name'),
           'rank': MODELS[name][2], 'solved': False, 'n_ok': 0,
           'errors': []}
    prompt = build_prompt(shots, rec['prompt'], k)
    for _ in range(samples):
        try:
            out = call_model(name, prompt, temperature)
        except Exception as e:
            res['errors'].append(str(e)[:120])
            time.sleep(2.0)
            continue
        full = normalize_completion(rec['prompt'], out)
        ok, reason, _ = verify_text(full)
        if ok:
            res['n_ok'] += 1
            res['solved'] = True
        else:
            res['errors'].append(reason[:40])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', required=True)
    ap.add_argument('--shots', required=True, help='jsonl of worked examples (text field)')
    ap.add_argument('--k-shot', type=int, default=50)
    ap.add_argument('--models', default='gpt-4o-mini,grok-3-mini,claude-haiku,gpt-4o,grok-3,claude-sonnet')
    ap.add_argument('--samples', type=int, default=3)
    ap.add_argument('--temperature', type=float, default=0.3)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    targets = [json.loads(l) for l in open(args.targets) if l.strip()]
    if args.limit:
        targets = targets[:args.limit]
    shots = [json.loads(l) for l in open(args.shots) if l.strip()]
    models = [m for m in args.models.split(',') if m in MODELS]
    print(f'{len(targets)} targets x {len(models)} models x {args.samples} samples, '
          f'{args.k_shot}-shot', file=sys.stderr)

    jobs = [(m, t) for m in models for t in targets]
    out = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(eval_one, m, t, shots, args.k_shot, args.samples,
                          args.temperature): (m, t) for m, t in jobs}
        done = 0
        for f in as_completed(futs):
            out.append(f.result())
            done += 1
            if done % 20 == 0:
                print(f'  {done}/{len(jobs)} ({time.time()-t0:.0f}s)', file=sys.stderr)
    with open(args.out, 'w') as fo:
        for r in out:
            fo.write(json.dumps(r) + '\n')
    # summary
    from collections import defaultdict
    by_model = defaultdict(lambda: [0, 0])
    for r in out:
        by_model[r['model']][0] += r['solved']
        by_model[r['model']][1] += 1
    for m in sorted(by_model, key=lambda x: MODELS[x][2]):
        s, n = by_model[m]
        print(f'{m} (rank {MODELS[m][2]}): {s}/{n} = {s/n:.3f}')


if __name__ == '__main__':
    main()
