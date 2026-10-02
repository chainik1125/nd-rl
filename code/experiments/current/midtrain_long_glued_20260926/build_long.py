#!/usr/bin/env python3
"""testsets/long.jsonl from the data agent's data/test_long.jsonl (held-out glued theorems at L_true 22/24/26).

    python build_long.py

Keeps every record (the data agent sized the set), in lengthexp's test-row layout with set = 'long'. Checks, by pkey
(premise-order-invariant class, recomputed from thm): disjoint from data/sft_long.jsonl and data/sft_short.jsonl,
from every lengthexp training file (RL pools incl. arm D's, train_cap6, heldout_cap6) and from testsets/std.jsonl and
matched.jsonl; and, when records carry `parts`, no part shared with the SFT records' parts. Writes
testsets/long_MANIFEST.json.
"""
import collections, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LE = '/scratch/cowifornia/nd-rl/ab3/lengthexp'
sys.path.insert(0, LE)
from common import TRAIN_FILES, iter_jsonl, pkey, thm_of, write_json, write_jsonl  # noqa: E402

KEEP = ('name', 'prompt', 'thm', 'L_true', 'source', 'schema', 'n_prem', 'family', 'label_kind', 'parts', 'part_L')


def sha(fn):
    return hashlib.sha256(open(fn, 'rb').read()).hexdigest()


def main():
    src = os.path.join(HERE, 'data', 'test_long.jsonl')
    recs = list(iter_jsonl(src))
    excl = {}
    for k, fn in list(TRAIN_FILES.items()) + [('sft_long', os.path.join(HERE, 'data', 'sft_long.jsonl')),
                                              ('sft_short', os.path.join(HERE, 'data', 'sft_short.jsonl')),
                                              ('test_std', os.path.join(HERE, 'testsets', 'std.jsonl')),
                                              ('test_matched', os.path.join(HERE, 'testsets', 'matched.jsonl'))]:
        excl[k] = {pkey(thm_of(r)) for r in iter_jsonl(fn)}
    sft_parts = set()
    for k in ('sft_long', 'sft_short'):
        for r in iter_jsonl(os.path.join(HERE, 'data', f'{k}.jsonl')):
            sft_parts |= set(r.get('parts') or [])
    rows, overlap, part_overlap = [], collections.Counter(), 0
    for r in recs:
        pk = pkey(thm_of(r))
        hit = [k for k, s in excl.items() if pk in s]
        for k in hit:
            overlap[k] += 1
        part_overlap += bool(set(r.get('parts') or []) & sft_parts)
        row = {k: r.get(k) for k in KEEP}
        row.update(pkey=pk, set='long', src_file='test_long.jsonl', source=r.get('source') or 'ab3_glued',
                   family=r.get('family') or 'andi')
        if row['L_true'] is None:                    # no exact label in the file: the constructed length (n_lines)
            row['L_true'] = r['n_lines']
            row['label_kind'] = row.get('label_kind') or 'constructed_n_lines'
        rows.append(row)
    dup = len(rows) - len({r['pkey'] for r in rows})
    by = collections.Counter(r['L_true'] for r in rows)
    man = {'src': src, 'src_sha256': sha(src), 'n': len(rows), 'by_L': dict(sorted(by.items())), 'overlap_by_pkey': dict(overlap),
           'records_sharing_a_part_with_sft': part_overlap, 'sft_parts_known': len(sft_parts), 'internal_duplicates': dup,
           'fields_in_source': sorted(recs[0].keys()) if recs else []}
    print(json.dumps(man, indent=1))
    assert not overlap and dup == 0, 'test_long overlaps training or test data'
    rows.sort(key=lambda r: (r['L_true'], r['name']))
    write_jsonl(os.path.join(HERE, 'testsets', 'long.jsonl'), rows)
    write_json(os.path.join(HERE, 'testsets', 'long_MANIFEST.json'), man)


if __name__ == '__main__':
    main()
