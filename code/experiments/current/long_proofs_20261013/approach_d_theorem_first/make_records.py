"""Theorem-first pipeline, generation half: sample clause-family sequents (sample_cnf.sample), write a verified
(non-minimal) proof for each with prover.py, and emit {"text": "THM ... PRF ... QED"} records for certify.py.

    python make_records.py OUT.jsonl N SEED [--atoms 3-6] [--prems 3-7]
then
    python ../certify.py OUT.jsonl OUT_cert.jsonl --floor 20 --secs 60 --procs 2
"""
import argparse, json, random, time
import sample_cnf as SC
import prover
import certify as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('n', type=int); ap.add_argument('seed', type=int)
    ap.add_argument('--atoms', default='3-6'); ap.add_argument('--prems', default='3-7')
    a = ap.parse_args()
    ar = tuple(int(x) for x in a.atoms.split('-')); pr = tuple(int(x) for x in a.prems.split('-'))
    rng = random.Random(a.seed)
    t0 = time.time(); seen = set(); n = tries = fails = 0
    with open(a.out, 'w') as f:
        while n < a.n:
            tries += 1
            s = SC.sample(rng, ar, pr)
            if s is None:
                continue
            key = C.theorem_key(*s)
            if key in seen:
                continue
            seen.add(key)
            prompt = SC.prompt_of(*s)
            body = prover.prove_text(prompt)
            if body is None:
                fails += 1; continue
            f.write(json.dumps({'text': prompt + ' ' + body, 'n_prem': len(s[0]), 'gen_body': body.count(';') - len(s[0])}) + '\n')
            n += 1
    print(f'{n} records, {tries} draws, {fails} prover failures, {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
